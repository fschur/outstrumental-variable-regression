"""Angle analysis of the binary runs of the November 10 campaign.

Fixed on 10 November 2026, before any data of the campaign were collected.
Run from the repository root:

    PYTHONPATH=src MPLCONFIGDIR=/tmp/ovr-matplotlib python \
      -m experiments.causal_chambers.angle_analysis \
      --data RUN1/measurements.csv RUN2/measurements.csv ... --output-dir RESULTS

Data: the binary repetitions, each with 50 confounded and 20 randomized
measurements per block at 0, 20, 40 and 60 degrees.

Angle sets, always with 0 degrees as the baseline environment:
{0,20}, {0,40}, {0,60} and {0,20,40,60}.

Comparisons:
  equal size per angle: every set at n = 500, 1000, 2500, 5000 per angle;
  equal total size: {0,20,40,60} at n/2 per angle against each pair at n per
  angle, for totals 2,000, 5,000 and 10,000 (four angles at 500, 1250, 2500).

Methods: Direct and Smoothed OVR with the settings of campaign_analysis.py
(seeds 0,1,2); OLS on all angles of the set; OLS within each angle; OLS with
the observed H (privileged). Subsets are the same nested whole-block subsets
as in campaign_analysis.py, so the {0,60} fits reproduce the main analysis.

Target: the repetition's slope from the randomized measurements at 0 and 60
degrees, exactly as in the main analysis. In addition, the randomized
measurements at each angle give the slope of Y on X and the effect of H at
that angle; these are reported as a check, not used as targets.

Errors and failed fits are summarized as in campaign_analysis.py: errors in
percent of the repetition's reference; absolute errors averaged over seeds
within a repetition; a repetition counts only if all its seeds succeeded;
mean and standard error over these repetitions, with the numbers of attempted
and complete repetitions and of failed fits. Paired differences between
compared settings use the repetitions complete in both.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from ovr import DirectOVR, SmoothedOVR
from experiments.common.baselines import OLS
from .analyze_ovr import linear_fit
from .campaign_analysis import (file_hash, json_hash, observed_cell_design, paper_settings,
                                reference_targets, source_record, write_json)

ANGLES = (0., 20., 40., 60.)
SETS = {"0/20": (0., 20.), "0/40": (0., 40.), "0/60": (0., 60.), "0/20/40/60": ANGLES}
SIZES = {"0/20": (500, 1000, 2500, 5000), "0/40": (500, 1000, 2500, 5000),
         "0/60": (500, 1000, 2500, 5000), "0/20/40/60": (500, 1000, 1250, 2500, 5000)}
EQUAL_TOTAL = ((500, 1000), (1250, 2500), (2500, 5000))  # (four-angle n, pair n) per angle
SUBSET_SEED = 2026110601  # as in campaign_analysis.py
PER_BLOCK = 50


def load(path):
    path = Path(path).resolve()
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if str(manifest.get("status")).upper() != "DONE" or manifest["measurements_sha256"] != file_hash(path):
        raise ValueError(f"Not a completed, unchanged run: {path}")
    if manifest.get("design") != "binary" or [float(a) for a in manifest["angles"]] != [0., 60., 20., 40.]:
        raise ValueError(f"Expected a four-angle binary run: {path}")
    frame = pd.read_csv(path)
    frame["angle"] = np.asarray(manifest["angles"], float)[frame.E.to_numpy(int)]
    if not np.allclose(frame.angle, frame.pol_2, atol=1e-5, rtol=0):
        raise ValueError("E labels disagree with the polarizer angles")
    frame["block_id"] = frame.block_id.astype(str)
    replicate = manifest["replicate_id"]
    return frame, replicate, dict(path=str(path), measurements_sha256=manifest["measurements_sha256"],
                                  replicate_id=replicate, experiment_ids=manifest.get("experiment_ids"))


def block_order(frame, replicate):
    """The whole-block order of campaign_analysis.nested_subsets for this repetition."""
    training = frame.loc[frame.kind.eq("confounded")]
    counts = training.groupby(["block_id", "E"]).size().unstack(fill_value=0)
    if not (counts == PER_BLOCK).all().all():
        raise ValueError("Every block needs 50 confounded rows per angle")
    stable = int(hashlib.sha256(("binary/" + replicate).encode()).hexdigest()[:8], 16)
    return np.random.default_rng((SUBSET_SEED + stable) % (2**32)).permutation(sorted(counts.index)).tolist()


def subset(frame, order, angles, n):
    selected = order[:n // PER_BLOCK]
    part = frame.loc[frame.kind.eq("confounded") & frame.block_id.isin(selected)
                     & frame.angle.isin(angles)].reset_index(drop=True).copy()
    part["E"] = [angles.index(a) for a in part.angle]
    if part.groupby("E").size().tolist() != [n] * len(angles):
        raise RuntimeError("Subset sizes are inconsistent")
    return part


def main_reference(frame):
    """Reference of the main analysis: randomized rows at 0 and 60 degrees (E=0,1)."""
    common, _ = reference_targets(frame.loc[frame.E.isin([0, 1])])
    return common


def angle_checks(frame):
    """Slope of Y on X and effect of H at each angle, from the randomized rows."""
    reference = frame.loc[frame.kind.eq("reference")]
    rows = []
    for angle in ANGLES:
        part = reference.loc[reference.angle.eq(angle)]
        design = np.column_stack([part.X - 16., np.ones(len(part)), part.H])
        coefficient, covariance, _ = linear_fit(design, part.Y, part.block_id)
        rows.append(dict(angle=angle, slope=float(coefficient[0]), slope_cluster_se=float(np.sqrt(covariance[0, 0])),
                         h_effect=float(coefficient[2]), h_effect_cluster_se=float(np.sqrt(covariance[2, 2])),
                         n=len(part)))
    for row in rows:
        row["h_effect_relative_to_0deg"] = row["h_effect"] / rows[0]["h_effect"]
    return rows


def fit(part, angles, seed, n, replicate, settings, cache):
    signature = json_hash(dict(settings=settings, rows=part.row_uid.tolist(), angles=angles, seed=seed, n=n))
    if cache.exists():
        stored = json.loads(cache.read_text())
        if stored["identity_sha256"] != signature:
            raise ValueError(f"Cached fit has a different identity; use a new output directory: {cache}")
        return stored["estimates"]
    torch.set_num_threads(1)
    x, y, e, groups = part[["X"]].to_numpy(float), part.Y.to_numpy(float), part.E.to_numpy(int), part.block_id.to_numpy()
    rows = []
    for name, cls in (("DOVR", DirectOVR), ("SOVR", SmoothedOVR)):
        start = time.perf_counter()
        try:
            model = cls(seed=seed, **settings["common"]).fit(x, y, e, groups=groups)
            row = dict(method=name, status="ok", coefficient=float(model.coef_[0]), fitted_gamma=float(model.gamma_))
            if not np.isfinite(row["coefficient"]):
                raise FloatingPointError("Estimator returned a nonfinite coefficient")
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, AssertionError) as error:
            row = dict(method=name, status="failed", coefficient=None, error=f"{type(error).__name__}: {error}")
        rows.append(dict(row, seed=seed, fit_seconds=time.perf_counter() - start))
    if seed == 0:
        rows.append(dict(method="OLS", status="ok", seed=None, coefficient=float(OLS().fit(x, y, e).coef_[0])))
        for index, angle in enumerate(angles):
            mask = e == index
            rows.append(dict(method=f"OLS {angle:g}deg", status="ok", seed=None,
                             coefficient=float(OLS().fit(x[mask], y[mask], e[mask]).coef_[0])))
        cells = observed_cell_design(part)
        rows.append(dict(method="Oracle H", status="ok", seed=None,
                         coefficient=float(OLS().fit(np.column_stack([x, cells[:, 1:]]), y, e).coef_[0])))
    rows = [dict(row, replicate_id=replicate, angles="/".join(f"{a:g}" for a in angles), n_per_angle=n,
                 n_total=n * len(angles)) for row in rows]
    write_json(cache, dict(identity_sha256=signature, estimates=rows))
    return rows


def summarize(estimates):
    """Same rules as campaign_analysis.aggregate_results."""
    keys = ["angles", "n_per_angle", "n_total", "method"]
    estimates = estimates.copy()
    estimates["coefficient"] = estimates.coefficient.astype(float)
    estimates["signed_error_pct"] = 100 * (estimates.coefficient - estimates.reference) / estimates.reference
    per_rep = []
    for values, part in estimates.groupby(keys + ["replicate_id"], sort=False):
        good = part.loc[part.status.eq("ok") & np.isfinite(part.signed_error_pct)]
        # Do not average away a failed seed inside a repetition.
        complete = len(good) == len(part)
        per_rep.append(dict(zip(keys + ["replicate_id"], values), attempted=len(part), succeeded=len(good),
                            complete=complete,
                            abs_error_pct=float(good.signed_error_pct.abs().mean()) if complete else np.nan,
                            signed_error_pct=float(good.signed_error_pct.mean()) if complete else np.nan))
    per_rep = pd.DataFrame(per_rep)
    summary = []
    for values, part in per_rep.groupby(keys, sort=False):
        good = part.loc[part.complete]
        r = len(good)
        summary.append(dict(zip(keys, values), attempted_repetitions=len(part), complete_repetitions=r,
                            attempted_fits=int(part.attempted.sum()),
                            failed_fits=int(part.attempted.sum() - part.succeeded.sum()),
                            metric_scope="all planned fits" if r == len(part) else "conditional on fully successful repetitions",
                            mae_pct=float(good.abs_error_pct.mean()) if r else np.nan,
                            mae_se=float(good.abs_error_pct.std(ddof=1) / np.sqrt(r)) if r > 1 else np.nan,
                            signed_pct=float(good.signed_error_pct.mean()) if r else np.nan))
    complete = per_rep.loc[per_rep.complete].set_index(
        ["angles", "n_per_angle", "method", "replicate_id"]).abs_error_pct.sort_index()
    comparisons = []

    def compare(kind, a, b):
        for method in ("DOVR", "SOVR"):
            try:
                left, right = complete.loc[(a[0], a[1], method)], complete.loc[(b[0], b[1], method)]
            except KeyError:
                continue
            difference = (left - right).dropna()
            comparisons.append(dict(comparison=kind, method=method, setting_a=f"{a[0]} at {a[1]} per angle",
                                    setting_b=f"{b[0]} at {b[1]} per angle", mae_a=float(left.mean()),
                                    mae_b=float(right.mean()), paired_difference=float(difference.mean()),
                                    paired_difference_se=float(difference.std(ddof=1) / np.sqrt(len(difference)))
                                    if len(difference) > 1 else np.nan,
                                    paired_repetitions=int(len(difference))))
    for pair in ("0/20", "0/40", "0/60"):
        for n in (500, 1000, 2500, 5000):
            compare("equal size per angle", ("0/20/40/60", n), (pair, n))
        for n_four, n_pair in EQUAL_TOTAL:
            compare("equal total size", ("0/20/40/60", n_four), (pair, n_pair))
    return per_rep, pd.DataFrame(summary), pd.DataFrame(comparisons)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sizes", type=int, nargs="+", help="Restrict sizes per angle (for tests)")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = paper_settings()
    estimates, references, checks, provenance = [], [], [], []
    for path in args.data:
        frame, replicate, record = load(path)
        frame["row_uid"] = replicate + ":" + frame.flag.astype(str)
        provenance.append(record)
        reference = main_reference(frame)
        references.append(dict(replicate_id=replicate, **reference))
        checks.extend(dict(replicate_id=replicate, **row) for row in angle_checks(frame))
        order = block_order(frame, replicate)
        for name, angles in SETS.items():
            for n in SIZES[name]:
                if args.sizes and n not in args.sizes:
                    continue
                part = subset(frame, order, list(angles), n)
                for seed in args.seeds:
                    cache = output / "fits" / replicate / f"{name.replace('/', '-')}_n{n}_seed{seed}.json"
                    print(f"{replicate} {name} n={n} seed={seed}", flush=True)
                    for row in fit(part, list(angles), seed, n, replicate, settings, cache):
                        estimates.append(dict(row, reference=reference["coefficient"]))
    estimates = pd.DataFrame(estimates)
    per_rep, summary, comparisons = summarize(estimates)
    estimates.to_csv(output / "estimates.csv", index=False)
    per_rep.to_csv(output / "per_repetition.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    comparisons.to_csv(output / "comparisons.csv", index=False)
    pd.DataFrame(references).to_csv(output / "references.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "angle_checks.csv", index=False)
    write_json(output / "analysis_manifest.json", dict(
        protocol=__doc__, data=provenance, settings=settings, sources=source_record(),
        script_sha256=file_hash(Path(__file__)), seeds=args.seeds, sizes=args.sizes))
    print(f"Saved angle analysis to {output}")


if __name__ == "__main__":
    main()
