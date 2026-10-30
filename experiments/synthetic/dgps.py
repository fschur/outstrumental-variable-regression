"""Data-generating processes used by the OVR experiments.

The active suite covers the paper's low-rank, clean-environment, and recoverable-
state sufficient regimes. Each
``sample(n)`` returns a ``Dataset`` containing ``X``, ``Y``, the environment
label ``T``, and the true ``beta0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class Dataset:
    X: np.ndarray
    Y: np.ndarray
    T: np.ndarray
    beta0: np.ndarray
    H: Optional[np.ndarray] = None  # latent confounder, when available (diagnostics)


def _onehot(T: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((T.shape[0], k))
    out[np.arange(T.shape[0]), T] = 1.0
    return out


def _random_mlp(
    rng: np.random.Generator,
    in_dim: int,
    out_dim: int,
    hidden: int = 32,
    depth: int = 2,
) -> Callable[[np.ndarray], np.ndarray]:
    """A randomly-initialized tanh MLP, used as a random nonlinear structural eq."""
    Ws: list[np.ndarray] = []
    bs: list[np.ndarray] = []
    d = in_dim
    for _ in range(depth):
        Ws.append(rng.normal(size=(d, hidden)) / np.sqrt(d))
        bs.append(rng.normal(size=hidden) * 0.1)
        d = hidden
    Ws.append(rng.normal(size=(d, out_dim)) / np.sqrt(d))
    bs.append(rng.normal(size=out_dim) * 0.1)

    def f(Z: np.ndarray) -> np.ndarray:
        a = Z
        for W, b in zip(Ws[:-1], bs[:-1]):
            a = np.tanh(a @ W + b)
        return a @ Ws[-1] + bs[-1]

    return f


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
@dataclass
class CleanEnvironmentSCM:
    """One unconfounded environment with nonlinear confounding elsewhere.

    The model permits T -> H, T -> X, T -> Y, H -> X, H -> Y, and X -> Y.
    Its outcome nuisance psi(H, T) is set to zero in environment ``t_star``,
    giving ``q_{t_star}(X) = 0``. For any reference t0 different from t_star,
    the contrast includes ``q_{t_star} - q_{t0} = -q_{t0}``; for t0=t_star,
    the reference nuisance is zero. The inclusion restriction therefore holds.

    Environment overlap and residual-variation positivity are separate
    requirements. A shared nonlinear map f_X(H) induces confounding, while
    additive environment shifts and Gaussian exposure noise retain common
    support. The parameters ``x_mean_scale`` and ``h_loc_scale`` control these
    shifts; ``confound_scale`` controls the outcome nuisance.
    """

    d: int = 4
    d_h: int = 2
    n_environments: int = 10
    t_star: int = 5  # the clean (unconfounded) environment; != reference 0
    hidden: int = 32
    depth: int = 2
    noise_sd: float = 0.5
    x_noise_sd: float = 0.5
    confound_scale: float = 3.0
    h_loc_scale: float = 0.3   # mild T -> H so H does not separate environments
    x_mean_scale: float = 0.5  # mild additive T -> X shift (overlap-preserving)
    n_calib: int = 4000
    seed: int = 0

    def sample(self, n: int, seed: Optional[int] = None) -> Dataset:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        S, d, d_h = self.n_environments, self.d, self.d_h
        beta0 = rng.normal(size=d)
        m_H = self.h_loc_scale * rng.normal(size=(S, d_h))          # T -> H (mild)
        f_X = _random_mlp(rng, d_h, d, self.hidden, self.depth)     # H -> X (shared core)
        mu_X = self.x_mean_scale * rng.normal(size=(S, d))          # T -> X (mild additive)
        psi = _random_mlp(rng, S + d_h, 1, self.hidden, self.depth)  # (T,H) -> Y term

        def _draw_H(Tv: np.ndarray) -> np.ndarray:
            return m_H[Tv] + rng.normal(size=(Tv.shape[0], d_h))

        Tc = rng.integers(0, S, size=self.n_calib)
        Hc = _draw_H(Tc)
        rawX = f_X(Hc)
        muX, sdX = rawX.mean(0), rawX.std(0) + 1e-8
        rawp = psi(np.hstack([_onehot(Tc, S), Hc]))[:, 0]
        mup, sdp = rawp.mean(), rawp.std() + 1e-8

        T = rng.integers(0, S, size=n)
        H = _draw_H(T)
        X = (f_X(H) - muX) / sdX + mu_X[T] + self.x_noise_sd * rng.normal(size=(n, d))
        psi_t = (psi(np.hstack([_onehot(T, S), H]))[:, 0] - mup) / sdp
        psi_t[T == self.t_star] = 0.0                               # clean environment
        eps = self.noise_sd * rng.normal(size=n)
        Y = X @ beta0 + self.confound_scale * psi_t + eps
        return Dataset(X=X, Y=Y, T=T, beta0=beta0, H=H)


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
@dataclass
class RecoverableStateSCM:
    """Exact recoverable-state regime with all environment arrows present.

    Let ``R > 0`` be a scalar confounding state and let ``V_T`` be a random
    direction on the unit sphere.  The structural equations are

        X = R V_T,
        H = R^2 + alpha_T R + U_H,
        Y = beta0^T X + H + d_T + eps.

    Environment ``t`` gives ``V_T`` density

        f_t(v) = 1 + tilt * a_t^T v

    relative to the uniform distribution on the sphere, where the unit vectors
    ``a_t`` are equally spaced on a circle and sum to zero.  Thus ``T -> X`` is
    present when ``tilt > 0`` while all environments retain common support.  In
    particular, for uniform environment probabilities,

        P(T=t | X=x) = (1 + tilt * a_t^T x/||x||) / n_environments

    and the exact overlap lower bound is ``(1 - tilt) / n_environments``.

    This is an exact instance of the paper's recoverable-state proposition:
    ``R = ||X||``, and, conditional on ``(R,T)``, the direction driving ``X`` is
    independent of ``U_H`` driving ``H``.  The outcome nuisance is

        q_t(x) = ||x||^2 + alpha_t ||x|| + d_t,

    so any nonconstant loading vector ``alpha`` makes the contrast injective in
    ``R``.  The class exposes population state, contrast, propensities, overlap,
    and residual-positivity quantities for simulation diagnostics.
    """

    d: int = 4
    n_environments: int = 4
    noise_sd: float = 0.5
    h_noise_sd: float = 0.5
    radius_mean: float = 1.0
    radius_log_sd: float = 0.45
    alpha_scale: float = 1.0
    direct_shift_scale: float = 0.5
    tilt: float = 0.5
    signal_scale: float = 1.0
    seed: int = 0

    beta0_: np.ndarray = field(init=False, repr=False)

    # Population quantities from the most recent draw.  Structural parameters
    # are deterministic given the public knobs, but retaining these arrays makes
    # downstream diagnostic code convenient.
    alpha_: np.ndarray = field(default_factory=lambda: np.empty(0), init=False, repr=False)
    direct_shift_: np.ndarray = field(default_factory=lambda: np.empty(0), init=False, repr=False)
    tilt_directions_: np.ndarray = field(default_factory=lambda: np.empty((0, 0)), init=False, repr=False)
    oracle_state_: np.ndarray = field(default_factory=lambda: np.empty(0), init=False, repr=False)
    oracle_contrast_: np.ndarray = field(default_factory=lambda: np.empty((0, 0)), init=False, repr=False)
    oracle_propensity_: np.ndarray = field(default_factory=lambda: np.empty((0, 0)), init=False, repr=False)
    oracle_residual_covariance_: np.ndarray = field(default_factory=lambda: np.empty((0, 0)), init=False, repr=False)
    metadata_: dict[str, float | int | bool | str] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._validate()
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 104729]))
        self.beta0_ = self.signal_scale * rng.normal(size=self.d)

    def _validate(self) -> None:
        if self.d < 2:
            raise ValueError("RecoverableStateSCM requires d >= 2 for sphere directions.")
        if self.n_environments < 2:
            raise ValueError("RecoverableStateSCM requires at least two environments.")
        if not 0.0 <= self.tilt < 1.0:
            raise ValueError("tilt must lie in [0, 1) to guarantee common support.")
        if self.radius_mean <= 0.0 or self.radius_log_sd < 0.0:
            raise ValueError("radius_mean must be positive and radius_log_sd nonnegative.")
        if self.alpha_scale <= 0.0:
            raise ValueError("alpha_scale must be positive so the contrast recovers R.")
        if self.noise_sd < 0.0 or self.h_noise_sd < 0.0:
            raise ValueError("noise standard deviations must be nonnegative.")

    def _environment_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return tilt directions, H loadings, and direct outcome shifts."""
        angles = 2.0 * np.pi * np.arange(self.n_environments) / self.n_environments
        directions = np.zeros((self.n_environments, self.d))
        directions[:, 0] = np.cos(angles)
        directions[:, 1] = np.sin(angles)
        alpha = self.alpha_scale * np.cos(angles)
        direct_shift = self.direct_shift_scale * np.linspace(-1.0, 1.0, self.n_environments)
        return directions, alpha, direct_shift

    def _sample_directions(
        self,
        rng: np.random.Generator,
        T: np.ndarray,
        directions: np.ndarray,
    ) -> np.ndarray:
        """Sample the linearly tilted sphere density by rejection sampling."""
        V = np.empty((T.shape[0], self.d))
        for t in range(self.n_environments):
            idx = np.flatnonzero(T == t)
            remaining = idx
            while remaining.size:
                proposal = rng.normal(size=(remaining.size, self.d))
                proposal /= np.linalg.norm(proposal, axis=1, keepdims=True)
                weight = 1.0 + self.tilt * np.sum(proposal * directions[t], axis=1)
                accept = rng.random(remaining.size) < weight / (1.0 + self.tilt)
                V[remaining[accept]] = proposal[accept]
                remaining = remaining[~accept]
        return V

    def oracle_state(self, X: np.ndarray) -> np.ndarray:
        """Recover ``R`` exactly from observed exposures."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(f"X must have shape (n, {self.d}).")
        return np.linalg.norm(X, axis=1)

    def oracle_nuisance(self, X: np.ndarray) -> np.ndarray:
        """Return ``q_t(X)`` for every observation and environment."""
        r = self.oracle_state(X)[:, None]
        _, alpha, direct_shift = self._environment_parameters()
        return r**2 + r * alpha[None, :] + direct_shift[None, :]

    def oracle_q(self, X: np.ndarray) -> np.ndarray:
        """Alias using the common oracle interface of the other exact DGPs."""
        return self.oracle_nuisance(X)

    def oracle_outcome_regression(self, X: np.ndarray) -> np.ndarray:
        """Return ``mu_t(X)=beta0^T X+q_t(X)`` for every environment."""
        X = np.asarray(X, dtype=float)
        causal_outcome = np.einsum("nd,d->n", X, self.beta0_)
        return causal_outcome[:, None] + self.oracle_nuisance(X)

    def oracle_contrast(self, X: np.ndarray, reference: int = 0) -> np.ndarray:
        """Return the population outcome contrast relative to ``reference``."""
        if not 0 <= reference < self.n_environments:
            raise ValueError("reference must index an existing environment.")
        q = self.oracle_nuisance(X)
        keep = np.arange(self.n_environments) != reference
        return q[:, keep] - q[:, [reference]]

    def oracle_propensity(self, X: np.ndarray) -> np.ndarray:
        """Return the exact ``P(T=t | X)`` under uniform environment sampling."""
        r = self.oracle_state(X)
        if np.any(r <= 0.0):
            raise ValueError("Propensities are undefined at X=0.")
        directions, _, _ = self._environment_parameters()
        V = np.asarray(X, dtype=float) / r[:, None]
        density = 1.0 + self.tilt * np.einsum("nd,sd->ns", V, directions)
        return density / density.sum(axis=1, keepdims=True)

    @property
    def overlap_lower_bound(self) -> float:
        """Exact infimum of ``P(T=t | X=x)`` over environments and support."""
        return (1.0 - self.tilt) / self.n_environments

    @property
    def oracle_residual_covariance(self) -> np.ndarray:
        """Population covariance of ``X-E[X | C(X),T]``.

        Since the contrast recovers ``R``, conditioning on ``(C,T)`` is the
        same as conditioning on ``(R,T)``.  For the tilted sphere,
        ``Cov(V | T=t) = I/d - tilt^2 a_t a_t^T/d^2``.
        """
        directions, _, _ = self._environment_parameters()
        mean_outer = directions.T @ directions / self.n_environments
        second_radius_moment = self.radius_mean**2 * np.exp(self.radius_log_sd**2)
        return second_radius_moment * (
            np.eye(self.d) / self.d - self.tilt**2 * mean_outer / self.d**2
        )

    @property
    def positivity_eigenvalue(self) -> float:
        """Exact minimum eigenvalue in the residual-positivity condition."""
        return float(np.linalg.eigvalsh(self.oracle_residual_covariance).min())

    @property
    def contrast_sufficiency_holds(self) -> bool:
        return self.cs_holds

    @property
    def cs_holds(self) -> bool:
        """Whether the contrast injectively recovers the confounding state."""
        return self.alpha_scale > 0.0 and self.n_environments >= 2

    @property
    def metadata(self) -> dict[str, float | int | bool | str]:
        """Theory-facing population quantities for experiment diagnostics."""
        return {
            "regime": "recoverable_confounding_state",
            "contrast_sufficiency": self.cs_holds,
            "state_dimension": 1,
            "contrast_rank": 1,
            "exposure_dimension": self.d,
            "n_environments": self.n_environments,
            "reference_environment": 0,
            "common_support": True,
            "overlap_lower_bound": self.overlap_lower_bound,
            "residual_positivity": True,
            "positivity_min_eigenvalue": self.positivity_eigenvalue,
        }

    def sample(self, n: int, seed: Optional[int] = None) -> Dataset:
        self._validate()
        if n <= 0:
            raise ValueError("n must be positive.")
        draw_seed = self.seed + 1 if seed is None else seed
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, draw_seed, 314159])
        )
        directions, alpha, direct_shift = self._environment_parameters()

        T = rng.integers(0, self.n_environments, size=n)
        # This normalization makes E[R] = radius_mean and keeps scale comparable
        # as radius_log_sd varies.
        R = self.radius_mean * np.exp(
            self.radius_log_sd * rng.normal(size=n) - 0.5 * self.radius_log_sd**2
        )
        V = self._sample_directions(rng, T, directions)
        X = R[:, None] * V

        U_H = self.h_noise_sd * rng.normal(size=n)
        H = R**2 + alpha[T] * R + U_H
        eps = self.noise_sd * rng.normal(size=n)
        Y = np.einsum("nd,d->n", X, self.beta0_) + H + direct_shift[T] + eps

        self.alpha_ = alpha
        self.direct_shift_ = direct_shift
        self.tilt_directions_ = directions
        self.oracle_state_ = R
        self.oracle_contrast_ = self.oracle_contrast(X)
        self.oracle_propensity_ = self.oracle_propensity(X)
        self.oracle_residual_covariance_ = self.oracle_residual_covariance
        self.metadata_ = self.metadata.copy()
        return Dataset(X=X, Y=Y, T=T, beta0=self.beta0_.copy(), H=H[:, None])


# --------------------------------------------------------------------------- #
@dataclass
class LowRankLatentLoadingSCM:
    """Exact latent-confounding SCM for the low-rank sufficient regime.

    The latent state has independent binary coordinates, and each coordinate is
    measured by its own exposure block::

        H_j ~ Bernoulli(pi),
        X^(j) | H_j=h ~ N(mu_{j,h}, sigma_h^2 I),
        T independent of (H, X),
        Y = beta0^T X + lambda_T^T H + eps.

    Consequently ``g_j(x) = P(H_j=1 | X^(j)=x^(j))`` is available in closed
    form and is invariant across environments.  The conditional nuisance and
    its contrast are exactly

        q_t(x) = lambda_t^T g(x),       C(x) = L g(x),

    where the rows of ``L`` are ``lambda_t-lambda_reference``.  Thus this is a
    causal SCM, rather than a reduced-form regression benchmark, and it matches
    the low-rank theorem literally.  The optional ``contrast_rank`` constructs
    an exact rank-deficient control; otherwise ``rank(L)=min(r, |S|-1)``.

    Unequal conditional variances make each posterior log-odds a quadratic
    function of its exposure block.  With ``block_dim >= 2``, conditioning on
    that scalar posterior leaves angular variation in every coordinate, which
    supplies the theorem's residual-positivity condition.  Since ``T`` is
    independent of ``(H,X)``, common support is exact and the environment-IV
    first stage is void.

    Use :meth:`warmup` for the paper's scalar two-environment example with
    loadings ``lambda_0=1`` and ``lambda_1=2``.
    """

    latent_rank: int = 2
    block_dim: int = 2
    n_environments: int = 4
    reference_environment: int = 0
    noise_sd: float = 0.5
    confound_scale: float = 1.0
    signal_scale: float = 1.0
    h_prob: float = 0.5
    mean_shift: float = 0.8
    x_sd0: float = 0.7
    x_sd1: float = 1.3
    reference_loading_scale: float = 1.0
    contrast_rank: Optional[int] = None
    min_contrast_singular_value: float = 1.0
    max_contrast_singular_value: float = 1.5
    loadings: Optional[np.ndarray] = field(default=None, repr=False)
    seed: int = 0

    beta0_: np.ndarray = field(init=False, repr=False)
    h_probs_: np.ndarray = field(init=False, repr=False)
    x_means_: np.ndarray = field(init=False, repr=False)
    x_sds_: np.ndarray = field(init=False, repr=False)
    loadings_: np.ndarray = field(init=False, repr=False)
    contrast_matrix_: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.latent_rank < 1:
            raise ValueError("latent_rank must be positive")
        if self.block_dim < 2:
            raise ValueError("block_dim must be at least 2 for residual positivity")
        if self.n_environments < 2:
            raise ValueError("n_environments must be at least 2")
        if not 0 <= self.reference_environment < self.n_environments:
            raise ValueError("reference_environment is outside the environment range")
        if not 0.0 < self.h_prob < 1.0:
            raise ValueError("h_prob must lie strictly between zero and one")
        if self.mean_shift <= 0.0:
            raise ValueError("mean_shift must be positive so H confounds X and Y")
        if self.x_sd0 <= 0.0 or self.x_sd1 <= 0.0:
            raise ValueError("conditional exposure standard deviations must be positive")
        if np.isclose(self.x_sd0, self.x_sd1):
            raise ValueError("x_sd0 and x_sd1 must differ to preserve residual positivity")
        if self.noise_sd < 0.0 or self.confound_scale <= 0.0:
            raise ValueError("noise_sd must be nonnegative and confound_scale positive")

        max_rank = min(self.latent_rank, self.n_environments - 1)
        requested_rank = max_rank if self.contrast_rank is None else self.contrast_rank
        if not 0 <= requested_rank <= max_rank:
            raise ValueError(f"contrast_rank must lie between 0 and {max_rank}")
        if self.min_contrast_singular_value <= 0.0:
            raise ValueError("min_contrast_singular_value must be positive")
        if self.max_contrast_singular_value < self.min_contrast_singular_value:
            raise ValueError(
                "max_contrast_singular_value must be no smaller than the minimum"
            )

        rng = np.random.default_rng(self.seed)
        self.beta0_ = self.signal_scale * rng.normal(size=self.d)
        self.h_probs_ = np.full(self.latent_rank, self.h_prob)

        # Each block has a randomly oriented mean shift.  H therefore causes X,
        # while the unequal variances make the exact posterior genuinely
        # nonlinear and leave within-posterior angular variation.
        directions = rng.normal(size=(self.latent_rank, self.block_dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        self.x_means_ = np.empty((self.latent_rank, 2, self.block_dim))
        self.x_means_[:, 0, :] = -self.mean_shift * directions
        self.x_means_[:, 1, :] = self.mean_shift * directions
        self.x_sds_ = np.column_stack(
            [
                np.full(self.latent_rank, self.x_sd0),
                np.full(self.latent_rank, self.x_sd1),
            ]
        )

        if self.loadings is not None:
            supplied = np.asarray(self.loadings, dtype=float)
            expected = (self.n_environments, self.latent_rank)
            if supplied.shape != expected:
                raise ValueError(f"loadings must have shape {expected}")
            if not np.all(np.isfinite(supplied)):
                raise ValueError("loadings must be finite")
            effective_loadings = supplied.copy()
        else:
            effective_loadings = self._random_loadings(rng, requested_rank)

        self.loadings_ = self.confound_scale * effective_loadings
        environments = self.nonreference_environments
        self.contrast_matrix_ = (
            self.loadings_[environments] - self.loadings_[self.reference_environment]
        )

    @classmethod
    def warmup(cls, **kwargs: object) -> "LowRankLatentLoadingSCM":
        """Construct ``Y=beta0^T X + T H + eps`` for environments 1 and 2.

        Internally environments are zero-indexed, but their loadings are exactly
        one and two, so the oracle contrast is ``P(H=1 | X)`` as in the paper.
        """
        reserved = {"latent_rank", "n_environments", "reference_environment", "loadings"}
        duplicated = reserved.intersection(kwargs)
        if duplicated:
            names = ", ".join(sorted(duplicated))
            raise TypeError(f"warmup fixes the following arguments: {names}")
        return cls(
            latent_rank=1,
            n_environments=2,
            reference_environment=0,
            loadings=np.array([[1.0], [2.0]]),
            **kwargs,
        )

    @property
    def d(self) -> int:
        """Observed exposure dimension."""
        return self.latent_rank * self.block_dim

    @property
    def nonreference_environments(self) -> np.ndarray:
        """Environment indices in the column order used by oracle contrasts."""
        return np.flatnonzero(np.arange(self.n_environments) != self.reference_environment)

    def _random_loadings(
        self, rng: np.random.Generator, requested_rank: int
    ) -> np.ndarray:
        """Construct loadings whose reference differences have exact rank."""
        n_differences = self.n_environments - 1
        if requested_rank == 0:
            differences = np.zeros((n_differences, self.latent_rank))
            row_basis = np.empty((self.latent_rank, 0))
        else:
            left, _ = np.linalg.qr(
                rng.normal(size=(n_differences, requested_rank)), mode="reduced"
            )
            row_basis, _ = np.linalg.qr(
                rng.normal(size=(self.latent_rank, requested_rank)), mode="reduced"
            )
            singular_values = (
                np.array([self.min_contrast_singular_value])
                if requested_rank == 1
                else np.linspace(
                    self.max_contrast_singular_value,
                    self.min_contrast_singular_value,
                    requested_rank,
                )
            )
            differences = (left * singular_values) @ row_basis.T

        # In rank-deficient controls, force the reference nuisance to contain a
        # component outside the contrast row space, so CS truly fails rather
        # than the rank theorem merely becoming inconclusive.
        if requested_rank < self.latent_rank:
            complete_basis, _ = np.linalg.qr(row_basis, mode="complete")
            missing_direction = complete_basis[:, requested_rank]
            reference_loading = self.reference_loading_scale * missing_direction
            if requested_rank:
                reference_loading += 0.25 * row_basis.sum(axis=1)
        else:
            reference_loading = rng.normal(size=self.latent_rank)
            norm = np.linalg.norm(reference_loading)
            reference_loading *= self.reference_loading_scale / max(norm, 1e-12)

        result = np.tile(reference_loading, (self.n_environments, 1))
        result[self.nonreference_environments] += differences
        return result

    def sample(self, n: int, seed: Optional[int] = None) -> Dataset:
        """Draw observations while keeping structural parameters fixed."""
        if n <= 0:
            raise ValueError("n must be positive")
        draw_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, draw_seed, 2718]))

        H = (rng.random(size=(n, self.latent_rank)) < self.h_probs_).astype(float)
        X = np.empty((n, self.d))
        for j in range(self.latent_rank):
            block = slice(j * self.block_dim, (j + 1) * self.block_dim)
            state = H[:, j].astype(int)
            means = self.x_means_[j, state]
            sds = self.x_sds_[j, state]
            X[:, block] = means + sds[:, None] * rng.normal(
                size=(n, self.block_dim)
            )

        T = rng.integers(0, self.n_environments, size=n)
        latent_outcome = np.sum(self.loadings_[T] * H, axis=1)
        causal_outcome = np.einsum("nd,d->n", X, self.beta0_)
        Y = causal_outcome + latent_outcome + self.noise_sd * rng.normal(size=n)
        return Dataset(X=X, Y=Y, T=T, beta0=self.beta0_.copy(), H=H)

    def oracle_posterior(self, X: np.ndarray) -> np.ndarray:
        """Return ``g(x)=E[H | X=x]`` in closed form."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(f"X must have shape (n, {self.d})")

        posterior = np.empty((X.shape[0], self.latent_rank))
        normal_constant = 0.5 * self.block_dim * np.log(2.0 * np.pi)
        for j in range(self.latent_rank):
            block = X[:, j * self.block_dim : (j + 1) * self.block_dim]
            log_weights = np.empty((X.shape[0], 2))
            for h in (0, 1):
                sd = self.x_sds_[j, h]
                squared_distance = np.sum(
                    (block - self.x_means_[j, h]) ** 2, axis=1
                )
                log_prior = np.log(
                    self.h_probs_[j] if h == 1 else 1.0 - self.h_probs_[j]
                )
                log_weights[:, h] = (
                    log_prior
                    - self.block_dim * np.log(sd)
                    - squared_distance / (2.0 * sd**2)
                    - normal_constant
                )
            log_weights -= log_weights.max(axis=1, keepdims=True)
            weights = np.exp(log_weights)
            posterior[:, j] = weights[:, 1] / weights.sum(axis=1)
        return posterior

    def oracle_q(self, X: np.ndarray) -> np.ndarray:
        """Return all population conditional nuisances ``q_t(X)``."""
        return np.einsum("nr,sr->ns", self.oracle_posterior(X), self.loadings_)

    def oracle_contrast(
        self, X: np.ndarray, reference: Optional[int] = None
    ) -> np.ndarray:
        """Return population outcome contrasts in increasing environment order."""
        reference = self.reference_environment if reference is None else reference
        if not 0 <= reference < self.n_environments:
            raise ValueError("reference is outside the environment range")
        q = self.oracle_q(X)
        environments = np.flatnonzero(np.arange(self.n_environments) != reference)
        return q[:, environments] - q[:, [reference]]

    def oracle_propensity(self, X: np.ndarray) -> np.ndarray:
        """Return the exact uniform ``P(T=t | X)`` for all environments."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(f"X must have shape (n, {self.d})")
        return np.full((X.shape[0], self.n_environments), 1.0 / self.n_environments)

    def oracle_outcome_regression(self, X: np.ndarray) -> np.ndarray:
        """Return ``mu_t(X)`` for every row and environment."""
        X = np.asarray(X, dtype=float)
        q = self.oracle_q(X)
        causal_outcome = np.einsum("nd,d->n", X, self.beta0_)
        return causal_outcome[:, None] + q

    @property
    def cs_holds(self) -> bool:
        """Whether the low-rank condition, and hence CS, holds exactly."""
        return np.linalg.matrix_rank(self.contrast_matrix_) == self.latent_rank

    @property
    def metadata(self) -> dict[str, object]:
        """Theory-facing metadata for experiment labels and diagnostics."""
        singular_values = np.linalg.svd(self.contrast_matrix_, compute_uv=False)
        return {
            "regime": "environment_varying_latent_loadings",
            "latent_rank": self.latent_rank,
            "exposure_dimension": self.d,
            "n_environments": self.n_environments,
            "reference_environment": self.reference_environment,
            "contrast_rank": int(np.linalg.matrix_rank(self.contrast_matrix_)),
            "contrast_singular_values": singular_values.copy(),
            "contrast_sufficiency": bool(self.cs_holds),
            "common_support": True,
            "overlap_lower_bound": 1.0 / self.n_environments,
            "residual_positivity": True,
        }
