"""Paired sampling for the six synthetic settings."""

from .sampling import _scenario_catalog, _sample_scenario


def sample_task(scenario, n, rep, options):
    """Generate the same observations and seeds for every comparison method."""
    spec = next(s for s in _scenario_catalog() if s.key == scenario)
    replication = _sample_scenario(spec, rep, n, options["main_seed"], options["shared_seed"])
    seeds = dict(estimator_seed=replication.estimator_seed,
                 structural_seed=replication.structural_seed, data_seed=replication.data_seed)
    return replication.data, seeds
