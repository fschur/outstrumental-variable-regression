"""Paired sensitivity sweep of the Smoothed OVR noise standard-deviation scale.

Fits each contrast ensemble once, then reuses it, the slope folds and the
second-stage seeds across all tau values. Outputs never modify paper artifacts.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from .run_ablation import (
    VariantSpec, _atomic_csv, _fit_ensemble_artifact, _make_engine,
    _make_split_plan, _variant_row,
)
from .sampling import _sample_scenario, _scenario_catalog


DEFAULT_TAUS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def fit_tau_sweep(scenario, replication, rep, n, taus=DEFAULT_TAUS, **settings):
    taus = tuple(float(t) for t in taus)
    if not taus or len(set(taus)) != len(taus) or any(t < 0 or not np.isfinite(t) for t in taus):
        raise ValueError("taus must be distinct finite nonnegative values")
    engine = _make_engine(
        jitter_scale=1.0, seed=replication.estimator_seed, **settings
    )
    data = replication.data
    engine._prepare(data.X, data.Y, data.T)
    plan = _make_split_plan(data.T, replication.estimator_seed, engine.n_folds)
    artifact = _fit_ensemble_artifact(
        engine, data.X, data.Y, data.T, plan, replication.estimator_seed
    )
    rows = []
    for step, tau in enumerate(taus):
        row = _variant_row(
            scenario=scenario, rep=rep, n=n, replication=replication,
            variant=VariantSpec(f"tau_{tau:g}", f"tau={tau:g}", step),
            artifact=artifact, plan=plan, engine=engine, jitter_scale=tau,
        )
        row["tau"] = tau
        rows.append(row)
    return rows


def _run(task):
    import torch
    torch.set_num_threads(1)
    key, rep, n, config = task
    scenario = next(s for s in _scenario_catalog() if s.key == key)
    replication = _sample_scenario(
        scenario, rep, n, config["main_seed"], config["shared_seed"]
    )
    return fit_tau_sweep(scenario, replication, rep, n, config["taus"], **config["settings"])


def summarize(raw):
    keys = ["scenario_id", "scenario_key", "scenario", "n", "tau"]
    metrics = ("coefficient_error", "gamma", "minimum_residual_eigenvalue",
               "truth_score_moment_norm", "effective_amplification", "jitter_trace_mean")
    rows = []
    for cell, group in raw.groupby(keys, sort=True):
        row = dict(zip(keys, cell))
        row["replications"] = len(group)
        for metric in metrics:
            row[metric + "_mean"] = group[metric].mean()
            row[metric + "_se"] = group[metric].sem()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_summary(raw):
    keys = ["scenario_key", "n", "rep"]
    ref = raw[raw.tau == 1][keys + ["coefficient_error"]].rename(
        columns={"coefficient_error": "error_tau1"})
    joined = raw.merge(ref, on=keys, validate="many_to_one")
    joined["difference_from_tau1"] = joined.coefficient_error - joined.error_tau1
    return joined.groupby(["scenario_key", "n", "tau"]).difference_from_tau1.agg(
        ["mean", "sem", "count"]).reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", default="1000,10000")
    parser.add_argument("--taus", default=",".join(map(str, DEFAULT_TAUS)))
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--contrast-epochs", type=int, default=200)
    parser.add_argument("--residual-epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--scenarios", default=",".join(s.key for s in _scenario_catalog()))
    parser.add_argument("--output-dir", type=Path, default=Path("results/paper/tau_sweep"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    sizes = [int(v) for v in args.sample_sizes.split(",")]
    taus = [float(v) for v in args.taus.split(",")]
    keys = args.scenarios.split(",")
    if (min(sizes) < 1 or args.reps < 1 or args.workers < 1 or
            set(keys) - {s.key for s in _scenario_catalog()} or
            len(set(taus)) != len(taus) or 1.0 not in taus or
            any(not np.isfinite(v) or v < 0 for v in taus)):
        parser.error("invalid sample sizes, counts, scenarios or taus (include tau=1)")
    settings = dict(n_folds=3, hidden=args.hidden, contrast_epochs=args.contrast_epochs,
                    residual_epochs=args.residual_epochs, strong_contrast_lambda=1e-2,
                    prediction_draws=8, validation_fraction=0.15, patience=15)
    root = Path(__file__).resolve().parents[2]
    config = dict(sample_sizes=sizes, taus=taus, reps=args.reps, scenarios=keys,
                  main_seed=20260807, shared_seed=20260808, settings=settings,
                  python=platform.python_version(), numpy=np.__version__,
                  sources={str(p.resolve().relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                      Path(__file__), Path(__file__).with_name("run_ablation.py"),
                      root / "src/ovr/_core.py", root / "src/ovr/smoothed.py")})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    raw_path = out / "raw_results.csv"
    rows = []
    if raw_path.exists():
        if not args.resume:
            parser.error("output already exists; use --resume or a new output directory")
        if json.loads(config_path.read_text()) != config:
            parser.error("resume configuration differs from the saved run")
        rows = pd.read_csv(raw_path).to_dict("records")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    done = set()
    if rows:
        for cell, group in pd.DataFrame(rows).groupby(["scenario_key", "rep", "n"]):
            if set(group.tau) != set(taus) or len(group) != len(taus):
                raise ValueError(f"incomplete checkpoint: {cell}")
            done.add(cell)
    tasks = [(k, r, n, config) for n in sizes for k in keys for r in range(args.reps)
             if (k, r, n) not in done]
    print(f"Running {len(tasks)} paired replications, {len(taus)} scales each", flush=True)
    with cf.ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(_run, task) for task in tasks]
        for count, future in enumerate(cf.as_completed(futures), 1):
            result = future.result()
            if not all(row["success"] for row in result):
                raise RuntimeError("Nonfinite estimate; refusing to summarize as a success")
            rows.extend(result)
            _atomic_csv(pd.DataFrame(rows), raw_path)
            print(f"{count}/{len(tasks)}: {result[0]['scenario_key']} n={result[0]['n']} rep={result[0]['rep']}", flush=True)
    raw = pd.DataFrame(rows).sort_values(["n", "scenario_id", "rep", "tau"])
    _atomic_csv(raw, raw_path)
    summary = summarize(raw)
    _atomic_csv(summary, out / "summary.csv")
    _atomic_csv(paired_summary(raw), out / "paired_vs_tau1.csv")
    print(f"Saved {len(raw)} fits in {out}", flush=True)


if __name__ == "__main__":
    main()
