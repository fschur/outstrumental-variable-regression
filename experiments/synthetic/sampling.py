"""Paper scenario catalog and exact paired sampling protocol."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from experiments.synthetic.shared_decoder import (
    SharedDecoderAdditiveSCM,
    _replication_seeds as _shared_replication_seeds,
)
from experiments.synthetic.designs import (
    ScenarioSpec,
    _make_dgp,
    _replication_seeds as _main_replication_seeds,
    _scenario_specs,
)

@dataclass(frozen=True)
class AblationScenario:
    scenario_id: int
    key: str
    label: str
    main_spec: ScenarioSpec | None = None
    exposure_shift_scale: float | None = None
    latent_shift_scale: float = 0.0


def _scenario_catalog() -> tuple[AblationScenario, ...]:
    main = {spec.key: spec for spec in _scenario_specs()}
    return (
        AblationScenario(0, "low_rank", "Latent loadings", main["low_rank"]),
        AblationScenario(
            1,
            "nonlinear_clean_environment",
            "One unconfounded environment",
            main["nonlinear_clean_environment"],
        ),
        AblationScenario(
            2,
            "recoverable_state",
            "Recoverable confounding state",
            main["recoverable_state"],
        ),
        AblationScenario(
            3,
            "shared_decoder_no_ex",
            "Outcome shifts",
            exposure_shift_scale=0.0,
        ),
        AblationScenario(
            4,
            "shared_decoder_additive_ex",
            "Outcome and treatment shifts",
            exposure_shift_scale=0.3,
        ),
        AblationScenario(
            5,
            "shared_decoder_latent_outcome_exposure_shifts",
            "Latent, outcome and treatment shifts",
            exposure_shift_scale=0.3,
            latent_shift_scale=0.3,
        ),
    )


@dataclass(frozen=True)
class ReplicationData:
    dgp: Any
    data: Any
    structural_seed: int
    data_seed: int
    estimator_seed: int
    seed_protocol: str


def _sample_scenario(
    scenario: AblationScenario,
    rep: int,
    n: int,
    main_seed: int,
    shared_seed: int,
) -> ReplicationData:
    if scenario.main_spec is not None:
        seeds = _main_replication_seeds(main_seed, scenario.main_spec, rep)
        dgp = _make_dgp(scenario.main_spec, seeds.structural_seed)
        if scenario.key == "nonlinear_clean_environment":
            data = dgp.sample(n)
        else:
            data = dgp.sample(n, seed=seeds.data_seed)
        return ReplicationData(
            dgp=dgp,
            data=data,
            structural_seed=seeds.structural_seed,
            data_seed=seeds.data_seed,
            estimator_seed=seeds.estimator_seed,
            seed_protocol=seeds.protocol,
        )

    if scenario.exposure_shift_scale is None:
        raise RuntimeError("shared-decoder scenario is missing its exposure shift scale")
    seeds = _shared_replication_seeds(shared_seed, rep)
    dgp = SharedDecoderAdditiveSCM(
        d=4,
        d_h=1,
        n_environments=5,
        shift_scale=scenario.exposure_shift_scale,
        latent_shift_scale=scenario.latent_shift_scale,
        seed=seeds.structural_seed,
    )
    data = dgp.sample(n, seeds.data_seed)
    return ReplicationData(
        dgp=dgp,
        data=data,
        structural_seed=seeds.structural_seed,
        data_seed=seeds.data_seed,
        estimator_seed=seeds.estimator_seed,
        seed_protocol="paired_shift_scenarios_per_replication",
    )
