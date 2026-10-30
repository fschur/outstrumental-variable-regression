"""Run the six synthetic settings with OVR and the comparison methods."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import torch

from ovr import DirectOVR, SmoothedOVR
from experiments.common.baselines import OLS, EnvironmentFixedEffects, AnchorRegression
from experiments.common.config import paper_config_section
from experiments.common.linear_neural_benchmark import compare_linear_neural
from experiments.common.results import array_hash, estimate_row
from experiments.common.runner import run_tasks, write_estimates
from .sampling import _scenario_catalog
from .scenarios import sample_task


def default_sample_sizes():
    """Use the sample-size grid shown in the synthetic figures."""
    return paper_config_section('synthetic')['sample_sizes']


def run_replication(task):
    torch.set_num_threads(1)
    data, seeds = sample_task(task['scenario'], task['n'], task['rep'], task['sampling'])
    X, Y, E, target = data.X, data.Y, data.T, data.beta0
    seed = seeds['estimator_seed']
    common = dict(epochs=task['epochs'], resid_epochs=task['residual_epochs'],
                  hidden=task['hidden'], seed=seed)
    source = SmoothedOVR(**common).fit(X, Y, E)
    metadata = dict(dataset='synthetic', scenario=task['scenario'], n=len(X), rep=task['rep'],
                    data_hash=array_hash(X, Y, E), **seeds)
    rows = []
    for name, estimator in [('DOVR', DirectOVR(**common)), ('SOVR', source),
                            ('OLS', OLS()), ('Environment FE', EnvironmentFixedEffects()),
                            ('Anchor-g2', AnchorRegression(gamma=2))]:
        fit = estimator if name == 'SOVR' else estimator.fit(X, Y, E)
        rows.append(estimate_row(name, fit.coef_, target,
            dict(status='ok', gamma=getattr(fit, 'gamma_', None)), **metadata))
    row = compare_linear_neural(X, Y, E, target, seed=seed, **task['lnn'])[-1]
    # Store compact split evidence rather than all row indices.
    diagnostic = json.loads(row['diagnostics'])
    for key in list(diagnostic):
        if key.endswith('_indices'):
            diagnostic[key + '_hash'] = array_hash(np.asarray(diagnostic.pop(key), dtype=np.int64))
    row['diagnostics'] = json.dumps(diagnostic, sort_keys=True)
    row.update(metadata)
    rows.append(row)
    return dict(estimates=rows)


def main():
    synthetic = paper_config_section('synthetic')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-sizes', type=int, nargs='+')
    parser.add_argument('--reps', type=int)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or (args.reps is not None and args.reps < 1):
        parser.error('Workers and replications must be positive')
    sizes = args.sample_sizes or default_sample_sizes()
    reps = args.reps if args.reps is not None else synthetic['replications']
    lnn = paper_config_section('lnn')
    epochs, residual_epochs, hidden = 200, 300, 64
    if args.quick:
        sizes, reps, epochs, residual_epochs, hidden = [600], 2, 2, 2, 8
        lnn = dict(lnn, epochs=2, hidden=8, patience=2)
    if min(sizes) < 100:
        parser.error('Sample sizes must be at least 100')
    scenarios = [s.key for s in _scenario_catalog()]
    tasks = [dict(name=f'{s}_n{n}_rep{r}', scenario=s, n=n, rep=r,
                  sampling=dict(main_seed=synthetic['main_seed'], shared_seed=synthetic['shared_decoder_seed']),
                  epochs=epochs, residual_epochs=residual_epochs, hidden=hidden, lnn=lnn)
             for s in scenarios for n in sizes for r in range(reps)]
    results = run_tasks(tasks, run_replication, args.output_dir,
                        dict(quick=args.quick), args.workers)
    rows = [row for result in results for row in result['estimates']]
    if rows:
        write_estimates(rows, args.output_dir)


if __name__ == '__main__':
    main()
