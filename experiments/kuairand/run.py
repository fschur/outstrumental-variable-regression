"""KuaiRand train/reference splits with all reported methods.

The held-out random-insertion regression is a reference projection; it is not
known causal ground truth. All methods share training rows and the reference.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from experiments.common.baselines import OLS, EnvironmentFixedEffects, AnchorRegression
from experiments.common.config import paper_config_section
from experiments.common.linear_neural_benchmark import compare_linear_neural
from experiments.common.results import array_hash, estimate_row, clean_json
from experiments.common.runner import run_tasks, write_estimates
from .data import (KuaiRandConfig, X_COLUMNS, _make_direct_ovr, _make_smoothed_ovr,
                   prepare_kuairand_cohort, randomized_exposure_projection,
                   download_kuairand, resolve_source)


def stratified_half_split(
    frame: pd.DataFrame,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split rows within each user-policy cell with no outcome reuse."""
    rng = np.random.default_rng(seed)
    order = frame.loc[:, ["user_id", "environment"]].copy()
    order["_row"] = np.arange(frame.shape[0])
    order["_key"] = rng.random(frame.shape[0])
    order.sort_values(
        ["user_id", "environment", "_key"],
        kind="mergesort",
        inplace=True,
    )
    order["_position"] = order.groupby(
        ["user_id", "environment"], sort=False
    ).cumcount()
    order["_cell_size"] = order.groupby(
        ["user_id", "environment"], sort=False
    )["_row"].transform("size")
    order["_train"] = order["_position"].lt(order["_cell_size"] // 2)

    train = np.sort(order.loc[order["_train"], "_row"].to_numpy(int))
    reference = np.sort(
        order.loc[
            ~order["_train"] & order["environment"].eq(1), "_row"
        ].to_numpy(int)
    )
    if np.intersect1d(train, reference).size:
        raise RuntimeError("Training and randomized-reference outcomes overlap.")

    train_frame = frame.iloc[train]
    reference_frame = frame.iloc[reference]
    train_counts = (
        train_frame.groupby(["user_id", "environment"])
        .size()
        .unstack(fill_value=0)
    )
    if not np.array_equal(train_counts[0].to_numpy(), train_counts[1].to_numpy()):
        raise RuntimeError("Training policy counts are not balanced within user.")
    train_users = set(train_frame["user_id"])
    reference_users = set(reference_frame["user_id"])
    diagnostics = {
        "seed": seed,
        "training_rows": int(train.size),
        "training_rows_standard": int(
            train_frame["environment"].eq(0).sum()
        ),
        "training_rows_random": int(train_frame["environment"].eq(1).sum()),
        "reference_rows_random": int(reference.size),
        "training_users": int(train_frame["user_id"].nunique()),
        "reference_users": int(reference_frame["user_id"].nunique()),
        "shared_training_reference_users": int(
            len(train_users & reference_users)
        ),
        "outcome_overlap": 0,
        "minimum_training_rows_per_user_policy": int(train_counts.min().min()),
        "maximum_training_rows_per_user_policy": int(train_counts.max().max()),
    }
    return train, reference, diagnostics



def run_replication(frame, cfg, replication, lnn_config):
    import torch
    torch.set_num_threads(1)
    train, reference_rows, split = stratified_half_split(frame, cfg.seed+600000+replication)
    X = frame[list(X_COLUMNS)].to_numpy(float)
    Y = frame.outcome_log_watch_seconds.to_numpy(float)
    E = frame.environment.to_numpy(int)
    users = frame.user_id.to_numpy()
    reference, reference_se = randomized_exposure_projection(
        X[reference_rows], Y[reference_rows], E[reference_rows], users[reference_rows],
        frame.video_id.to_numpy()[reference_rows])
    X, Y, E, groups = X[train], Y[train], E[train], users[train]
    seed = cfg.seed+900000+10000*replication
    source = _make_smoothed_ovr(cfg, seed).fit(X, Y, E, groups=groups)
    methods = [('DOVR', _make_direct_ovr(cfg, seed).fit(X, Y, E, groups=groups)),
               ('SOVR', source), ('OLS', OLS().fit(X,Y,E)),
               ('Environment FE', EnvironmentFixedEffects().fit(X,Y,E)),
               ('Anchor-g2', AnchorRegression(gamma=2).fit(X,Y,E))]
    metadata = dict(dataset='kuairand', scenario='kuairand', n=len(X), rep=replication,
                    data_hash=array_hash(X,Y,E), estimator_seed=seed)
    rows = [estimate_row(label, model.coef_, reference, **metadata) for label, model in methods]
    row = compare_linear_neural(X,Y,E,reference,groups=groups,seed=seed,**lnn_config)[-1]
    diagnostic = json.loads(row['diagnostics'])
    for key in list(diagnostic):
        if key.endswith('_indices'):
            diagnostic[key+'_hash'] = array_hash(np.asarray(diagnostic.pop(key),dtype=np.int64))
    row['diagnostics'] = json.dumps(diagnostic,sort_keys=True)
    row.update(metadata)
    rows.append(row)
    contrast_users=set(groups[source.contrast_indices_])
    estimation_users=set(groups[source.estimation_indices_])
    split.update(rep=replication, user_overlap=len(contrast_users & estimation_users),
                 reference=reference.tolist(), reference_clustered_se=reference_se.tolist())
    return dict(estimates=rows,split=split)


def parse_args(argv=None):
    defaults=paper_config_section('kuairand')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data/kuairand'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/paper/kuairand'))
    parser.add_argument('--download',action='store_true',help='Download the official public release')
    parser.add_argument('--reps',type=int,default=defaults['replications'])
    parser.add_argument('--seed',type=int,default=defaults['seed'])
    parser.add_argument('--quick',action='store_true',help='Small pipeline check, not paper results')
    return parser.parse_args(argv)


def main():
    args=parse_args()
    if args.reps < 2:
        raise ValueError('At least two repetitions are needed to report Monte Carlo standard errors')
    source=download_kuairand(args.data_dir) if args.download else resolve_source(args.data_dir)
    cfg=KuaiRandConfig(seed=args.seed, max_users=100 if args.quick else None,
                      per_user_environment_cap=4 if args.quick else paper_config_section('kuairand')['per_user_environment_cap'],
                      ovr_hidden=8 if args.quick else 64,ovr_epochs=2 if args.quick else 120,
                      ovr_resid_epochs=2 if args.quick else 160)
    frame,metadata,scales=prepare_kuairand_cohort(source,cfg)
    lnn=paper_config_section('lnn')
    lnn['epochs']=paper_config_section('kuairand')['lnn_epochs']
    if args.quick:
        lnn.update(epochs=2,hidden=8,patience=2)
    tasks=[dict(name=f'rep{r}',rep=r) for r in range(2 if args.quick else args.reps)]
    config=dict(cohort=metadata,scales=scales,settings=asdict(cfg),lnn=lnn,quick=args.quick,
                cohort_hash=array_hash(frame.to_numpy()))
    results=run_tasks(tasks,lambda task:run_replication(frame,cfg,task['rep'],lnn),
                      args.output_dir,config)
    write_estimates([row for result in results for row in result['estimates']],args.output_dir)
    (args.output_dir/'reference_splits.json').write_text(json.dumps(clean_json([r['split'] for r in results]),indent=2)+'\n')


if __name__=='__main__':
    main()
