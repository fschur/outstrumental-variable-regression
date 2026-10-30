"""Sufficient-setting definitions and the published seed protocol."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .dgps import CleanEnvironmentSCM, LowRankLatentLoadingSCM, RecoverableStateSCM

@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: int
    key: str
    label: str
    parameters: dict[str, Any]


def _scenario_specs() -> tuple[ScenarioSpec, ...]:
    low_rank = {
        "latent_rank": 2,
        "block_dim": 2,
        "n_environments": 4,
        "confound_scale": 2.0,
    }
    nonlinear_clean = {
        "d": 4,
        "d_h": 2,
        "n_environments": 5,
        "t_star": 3,
        "confound_scale": 3.0,
        "x_noise_sd": 1.0,
        "h_loc_scale": 0.3,
        "x_mean_scale": 0.5,
    }
    recoverable = {
        "d": 4,
        "n_environments": 4,
        "tilt": 0.5,
        "alpha_scale": 3.0,
        "radius_log_sd": 0.6,
    }
    return (
        ScenarioSpec(
            scenario_id=0,
            key="low_rank",
            label="Latent loadings",
            parameters=low_rank,
        ),
        ScenarioSpec(
            scenario_id=2,
            key="recoverable_state",
            label="Recoverable confounding state",
            parameters=recoverable,
        ),
        ScenarioSpec(
            scenario_id=4,
            key="nonlinear_clean_environment",
            label="One unconfounded environment",
            parameters=nonlinear_clean,
        ),
    )


def _make_dgp(spec: ScenarioSpec, seed: int) -> Any:
    """Instantiate a scenario without storing non-pickleable lambdas in tasks."""
    if spec.key == "low_rank":
        return LowRankLatentLoadingSCM(seed=seed, **spec.parameters)
    if spec.key == "recoverable_state":
        return RecoverableStateSCM(seed=seed, **spec.parameters)
    if spec.key == "nonlinear_clean_environment":
        return CleanEnvironmentSCM(seed=seed, **spec.parameters)
    raise KeyError(f"unknown scenario key {spec.key!r}")


def _seed(master_seed: int, scenario_id: int, rep: int, stream: int) -> int:
    """Stable seed independent of task order and worker count."""
    state = np.random.SeedSequence(
        [int(master_seed), int(scenario_id), int(rep), int(stream)]
    ).generate_state(1, dtype=np.uint32)
    return int(state[0])


@dataclass(frozen=True)
class ReplicationSeeds:
    """Seeds and provenance for one paired Monte Carlo replication."""

    structural_seed: int
    data_seed: int
    estimator_seed: int
    protocol: str


def _replication_seeds(
    master_seed: int, spec: ScenarioSpec, rep: int
) -> ReplicationSeeds:
    """Use one joint structure/observation stream for the clean-environment SCM.

    Its law varies across replications and stays fixed across sample sizes.
    The other two sufficient designs keep one structural law across replications.
    """

    data_seed = _seed(master_seed, spec.scenario_id, rep, 23)
    estimator_seed = _seed(master_seed, spec.scenario_id, rep, 37)
    if spec.key == "nonlinear_clean_environment":
        return ReplicationSeeds(
            structural_seed=data_seed,
            data_seed=data_seed,
            estimator_seed=estimator_seed,
            protocol="joint_structure_and_observation_per_replication",
        )

    return ReplicationSeeds(
        structural_seed=_seed(master_seed, spec.scenario_id, 0, 11),
        data_seed=data_seed,
        estimator_seed=estimator_seed,
        protocol="fixed_structure_across_replications",
    )
