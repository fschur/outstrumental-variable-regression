"""Shared-decoder structural model and paired shift designs."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import math
import numpy as np
from .dgps import Dataset, _random_mlp

@dataclass(frozen=True)
class SharedDecoderScenarioSpec:
    """One member of the paired shared-decoder stress-test family."""

    scenario_id: int
    key: str
    label: str
    exposure_shift_scale: float
    latent_shift_scale: float


def _scenario_specs() -> tuple[SharedDecoderScenarioSpec, ...]:
    """Return the three scenarios in their paper-panel order."""

    return (
        SharedDecoderScenarioSpec(
            scenario_id=0,
            key="shared_decoder_no_ex",
            label="Outcome shifts",
            exposure_shift_scale=0.0,
            latent_shift_scale=0.0,
        ),
        SharedDecoderScenarioSpec(
            scenario_id=1,
            key="shared_decoder_additive_ex",
            label="Outcome and treatment shifts",
            exposure_shift_scale=0.3,
            latent_shift_scale=0.0,
        ),
        SharedDecoderScenarioSpec(
            scenario_id=2,
            key="shared_decoder_latent_outcome_exposure_shifts",
            label="Latent, outcome and treatment shifts",
            exposure_shift_scale=0.3,
            latent_shift_scale=0.3,
        ),
    )


@dataclass
class SharedDecoderAdditiveSCM:
    """Random nonlinear SCM with shared decoders and additive environment shifts.

    The equations are

    ``H = std(f_H(U_H)) + a_E``,
    ``X = std(f_X(H, U_X)) + b_E``, and
    ``Y = beta0' X + confound_scale * std(psi_E(H)) + eps``.

    The random maps and standardization constants do not depend on either shift
    scale.  In particular, setting ``latent_shift_scale=0`` reproduces the two
    specified shared-decoder settings.
    """

    d: int = 4
    d_h: int = 1
    n_environments: int = 5
    hidden: int = 32
    depth: int = 3
    noise_sd: float = 0.5
    confound_scale: float = 3.0
    signal_scale: float = 1.0
    shift_scale: float = 0.3
    latent_shift_scale: float = 0.0
    n_calib: int = 10_000
    seed: int = 0

    beta0_: np.ndarray = field(init=False, repr=False)
    _f_h: Callable[[np.ndarray], np.ndarray] = field(init=False, repr=False)
    _f_x: Callable[[np.ndarray], np.ndarray] = field(init=False, repr=False)
    _psi: tuple[Callable[[np.ndarray], np.ndarray], ...] = field(
        init=False, repr=False
    )
    _h_mean: np.ndarray = field(init=False, repr=False)
    _h_sd: np.ndarray = field(init=False, repr=False)
    _x_mean: np.ndarray = field(init=False, repr=False)
    _x_sd: np.ndarray = field(init=False, repr=False)
    _psi_mean: float = field(init=False, repr=False)
    _psi_sd: float = field(init=False, repr=False)
    shifts_: np.ndarray = field(init=False, repr=False)
    latent_shifts_: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.d < 2 or self.d_h != 1:
            raise ValueError("This experiment requires d >= 2 and d_h = 1.")
        if self.n_environments < 2 or self.hidden < 1 or self.depth < 1:
            raise ValueError("Invalid environment or network dimensions.")
        if (
            min(self.noise_sd, self.shift_scale, self.latent_shift_scale) < 0
            or self.confound_scale <= 0
        ):
            raise ValueError(
                "Noise/shift scales must be nonnegative and confounding positive."
            )

        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 32452843]))
        direction = rng.normal(size=self.d)
        self.beta0_ = (
            self.signal_scale
            * math.sqrt(self.d)
            * direction
            / np.linalg.norm(direction)
        )
        self._f_h = _random_mlp(rng, self.d_h, self.d_h, self.hidden, self.depth)
        self._f_x = _random_mlp(
            rng, self.d_h + self.d, self.d, self.hidden, self.depth
        )
        self._psi = tuple(
            _random_mlp(rng, self.d_h, 1, self.hidden, self.depth)
            for _ in range(self.n_environments)
        )

        shifts = rng.normal(size=(self.n_environments, self.d))
        shifts -= shifts.mean(axis=0, keepdims=True)
        shifts /= max(float(shifts.std()), np.finfo(float).eps)
        self.shifts_ = self.shift_scale * shifts

        # Use a separate deterministic stream so adding a_E cannot perturb any
        # same draw when latent_shift_scale=0.  Centering across
        # environments makes a_E a pure environment contrast.
        latent_shift_rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, 49979687])
        )
        latent_shifts = latent_shift_rng.normal(
            size=(self.n_environments, self.d_h)
        )
        latent_shifts -= latent_shifts.mean(axis=0, keepdims=True)
        latent_shifts /= max(
            float(latent_shifts.std()), np.finfo(float).eps
        )
        self.latent_shifts_ = self.latent_shift_scale * latent_shifts

        latent_noise = rng.normal(size=(self.n_calib, self.d_h))
        raw_h = self._f_h(latent_noise)
        self._h_mean = raw_h.mean(axis=0)
        self._h_sd = raw_h.std(axis=0) + 1e-8
        h = (raw_h - self._h_mean) / self._h_sd

        exposure_noise = rng.normal(size=(self.n_calib, self.d))
        raw_x = self._f_x(np.hstack([h, exposure_noise]))
        self._x_mean = raw_x.mean(axis=0)
        self._x_sd = raw_x.std(axis=0) + 1e-8

        environments = np.arange(self.n_calib, dtype=int) % self.n_environments
        rng.shuffle(environments)
        raw_psi = np.empty(self.n_calib)
        for environment, function in enumerate(self._psi):
            mask = environments == environment
            raw_psi[mask] = function(h[mask])[:, 0]
        self._psi_mean = float(raw_psi.mean())
        self._psi_sd = float(raw_psi.std() + 1e-8)

    def sample(self, n: int, seed: int | None = None) -> Dataset:
        if n <= 0:
            raise ValueError("n must be positive.")
        draw_seed = self.seed + 1 if seed is None else seed
        streams = np.random.SeedSequence(draw_seed).spawn(4)
        rng_e, rng_h, rng_x, rng_y = [np.random.default_rng(s) for s in streams]
        environment = rng_e.integers(0, self.n_environments, size=n)
        raw_h = self._f_h(rng_h.normal(size=(n, self.d_h)))
        h = (
            (raw_h - self._h_mean) / self._h_sd
            + self.latent_shifts_[environment]
        )
        raw_x = self._f_x(np.hstack([h, rng_x.normal(size=(n, self.d))]))
        x = (raw_x - self._x_mean) / self._x_sd + self.shifts_[environment]
        nuisance = np.empty(n)
        for level, function in enumerate(self._psi):
            mask = environment == level
            nuisance[mask] = function(h[mask])[:, 0]
        nuisance = self.confound_scale * (
            nuisance - self._psi_mean
        ) / self._psi_sd
        y = x @ self.beta0_ + nuisance + self.noise_sd * rng_y.normal(size=n)
        return Dataset(x, y, environment, self.beta0_.copy(), h)


def _seed(master: int, scenario: int, rep: int, stream: int) -> int:
    return int(
        np.random.SeedSequence([master, scenario, rep, stream]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


@dataclass(frozen=True)
class ReplicationSeeds:
    structural_seed: int
    data_seed: int
    estimator_seed: int


def _replication_seeds(master_seed: int, rep: int) -> ReplicationSeeds:
    """Seed plan paired across all shift scenarios and sample sizes."""

    return ReplicationSeeds(
        structural_seed=_seed(master_seed, 0, rep, 11),
        data_seed=_seed(master_seed, 0, rep, 23),
        estimator_seed=_seed(master_seed, 0, rep, 37),
    )
