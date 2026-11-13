"""Frozen analysis for the physical outcome-shift pilot.

Run from the repository root, for example::

    PYTHONPATH=src python -m experiments.causal_chambers.analyze_ovr joined.csv \
        --center 100 --scale 40 --output-dir results/causal_chambers/pilot

The CSV must contain X,Y,E,H,block_id,kind. ``kind`` is ``confounded`` or
``reference``; E and H are binary. Acquisition blocks, not shuffled rows, are
the splitting and covariance units. Reference outcomes never enter OVR fits.
The center/scale must be the frozen acquisition values, not fitted to outcomes.
This script requires scipy in addition to the repository's dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student_t

from ovr import DirectOVR, SmoothedOVR


def rounded_posterior(x: np.ndarray, center: int, scale: int) -> np.ndarray:
    """P(H=1|X), using exact integer-rounding bin masses and equal H priors.

    The sign probability cancels, including at K=0 where both signs coincide.
    Only this oracle diagnostic receives the acquisition-law parameters.
    """
    k = np.abs(np.asarray(x, dtype=float) - center)
    if scale <= 0 or np.any(k > scale) or not np.allclose(k, np.round(k)):
        raise ValueError("X must be integer commands within center +/- scale.")
    lower = np.maximum(0.0, (k - 0.5) / scale)
    upper = np.minimum(1.0, (k + 0.5) / scale)
    mass0 = beta_distribution.cdf(upper, 2, 5) - beta_distribution.cdf(lower, 2, 5)
    mass1 = beta_distribution.cdf(upper, 5, 2) - beta_distribution.cdf(lower, 5, 2)
    if np.any(mass0 + mass1 <= 0):
        raise ValueError("Numerically zero acquisition probability in an observed bin.")
    return mass1 / (mass0 + mass1)


def clustered_covariance(design, residual, groups):
    """One-way acquisition-block sandwich with a finite-sample correction."""
    design, residual = np.asarray(design, float), np.asarray(residual, float)
    codes, labels = pd.factorize(np.asarray(groups), sort=False)
    n, p = design.shape
    g = len(labels)
    if g < 2 or n <= p:
        return np.full((p, p), np.nan), g
    scores = np.zeros((g, p))
    np.add.at(scores, codes, design * residual[:, None])
    bread = np.linalg.pinv(design.T @ design)
    correction = g / (g - 1) * (n - 1) / (n - p)
    covariance = correction * bread @ (scores.T @ scores) @ bread
    return (covariance + covariance.T) / 2, g


def linear_fit(design, y, groups):
    design, y = np.asarray(design, float), np.asarray(y, float)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Linear design is rank deficient; a slope is not identified.")
    coefficient = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficient
    covariance, n_blocks = clustered_covariance(design, residual, groups)
    return coefficient, covariance, n_blocks


def linear_row(name, design, frame, index=0, *, role="training"):
    coefficient, covariance, n_blocks = linear_fit(design, frame.Y, frame.block_id)
    value = float(coefficient[index])
    se = float(np.sqrt(max(covariance[index, index], 0)))
    critical = float(student_t.ppf(0.975, n_blocks - 1)) if n_blocks > 1 else np.nan
    return dict(method=name, seed=None, role=role, status="ok", coefficient=value,
                cluster_se=se, ci_low=value-critical*se, ci_high=value+critical*se,
                n=len(frame), n_blocks=n_blocks,
                interval_kind="block-cluster sandwich, t(G-1); approximate")


def environment_indicators(frame):
    return np.eye(2)[frame.E.to_numpy(int)]


def cell_indicators(frame):
    return np.eye(4)[2*frame.E.to_numpy(int) + frame.H.to_numpy(int)]


def linear_baselines(frame, posterior):
    x = frame.X.to_numpy(float) - frame.X.mean()
    env = environment_indicators(frame)
    rows = [
        linear_row("OLS", np.column_stack([x, np.ones(len(frame))]), frame),
        linear_row("Environment FE", np.column_stack([x, env]), frame),
        linear_row("Oracle H (E,H intercepts)",
                   np.column_stack([x, cell_indicators(frame)]), frame),
        linear_row("Oracle p (linear p-by-E nuisance)",
                   np.column_stack([x, env, env*posterior[:, None]]), frame),
    ]
    for e in (0, 1):
        subset = frame.loc[frame.E.eq(e)]
        xe = subset.X.to_numpy(float) - subset.X.mean()
        rows.append(linear_row(f"OLS E={e}",
                               np.column_stack([xe, np.ones(len(subset))]), subset))
    return rows


def reference_analysis(frame):
    """Common and environment-specific slopes; saturated E,H intercepts."""
    x = frame.X.to_numpy(float) - frame.X.mean()
    cells = cell_indicators(frame)
    common = linear_row("Randomized reference (common)", np.column_stack([x, cells]),
                        frame, role="reference")
    design = np.column_stack([x[:, None]*environment_indicators(frame), cells])
    coefficient, covariance, n_blocks = linear_fit(design, frame.Y, frame.block_id)
    rows = [common]
    for e in (0, 1):
        rows.append(linear_row(f"Randomized reference E={e}", design, frame,
                               index=e, role="reference"))
    difference = float(coefficient[1]-coefficient[0])
    se = float(np.sqrt(max(covariance[0, 0]+covariance[1, 1]-2*covariance[0, 1], 0)))
    critical = float(student_t.ppf(.975, n_blocks-1)) if n_blocks > 1 else np.nan
    diagnostics = dict(environment_slope_difference_e1_minus_e0=difference,
                       cluster_se=se, ci_low=difference-critical*se,
                       ci_high=difference+critical*se, n_blocks=n_blocks,
                       note="Small or nonsignificant differences do not establish equivalence.")
    return rows, diagnostics


def oracle_p_with_same_second_stage(direct, posterior):
    """Replace ONLY DirectOVR's contrast on its existing slope split/folds.

    Reuses the unmodified DirectOVR neural residualization routine, its raw
    X/Y units, nuisance settings, seeds and folds. No new learned first stage,
    no reference outcomes, and no analytically supplied treatment residuals.
    """
    indices = direct.estimation_indices_
    position = np.full(direct.n_, -1, dtype=int)
    position[indices] = np.arange(len(indices))
    folds = [position[fold] for fold in direct.second_stage_folds_]
    if any(np.any(fold < 0) for fold in folds):
        raise RuntimeError("DirectOVR folds do not match its estimation split.")
    xt, yt, _ = direct._crossfit_direct_residuals(
        direct.X_[indices], direct.Y_[indices], direct.T_[indices],
        posterior[indices, None], folds,
    )
    coefficient, sigma = direct._solve_beta(xt, yt)
    gamma = direct._gamma(sigma, direct.Sigma_T_)
    return coefficient, xt, yt, gamma


def neural_row(name, seed, coefficient, xt, yt, groups, gamma, n_contrast):
    coefficient = float(np.asarray(coefficient).ravel()[0])
    residual = yt - xt[:, 0]*coefficient
    covariance, n_blocks = clustered_covariance(xt, residual, groups)
    plugin_se = float(np.sqrt(max(covariance[0, 0], 0)))
    return dict(method=name, seed=seed, role="training", status="ok",
                coefficient=coefficient, cluster_se=None, ci_low=None, ci_high=None,
                plugin_cluster_se=plugin_se, n=len(xt), n_blocks=n_blocks,
                n_contrast=n_contrast, fitted_positivity_gamma=float(gamma),
                interval_kind="none; plugin_cluster_se ignores nuisance-learning uncertainty")


def fit_ovr(frame, posterior, seeds):
    x, y, e = frame[["X"]].to_numpy(float), frame.Y.to_numpy(float), frame.E.to_numpy(int)
    groups = frame.block_id.to_numpy()
    rows, contrasts = [], []
    for seed in seeds:
        direct = None
        for name, cls in (("Direct OVR", DirectOVR), ("Smoothed OVR", SmoothedOVR)):
            print(f"Fitting {name}, seed {seed}", flush=True)
            try:
                fitted = cls(seed=seed).fit(x, y, e, groups=groups)
                if name == "Direct OVR":
                    direct = fitted
                idx = fitted.estimation_indices_
                rows.append(neural_row(name, seed, fitted.coef_, fitted.residualized_X_,
                                       fitted.residualized_Y_, groups[idx], fitted.gamma_,
                                       fitted.n_contrast_))
                for i, value in zip(idx, fitted.estimated_contrast_[:, 0]):
                    contrasts.append(dict(method=name, seed=seed, row=int(i), X=float(x[i, 0]),
                                          E=int(e[i]), posterior=float(posterior[i]),
                                          estimated_contrast=float(value)))
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                rows.append(dict(method=name, seed=seed, role="training", status="failed",
                                 error=str(error)))
        if direct is not None:
            print(f"Fitting oracle-p neural diagnostic, seed {seed}", flush=True)
            try:
                coef, xt, yt, gamma = oracle_p_with_same_second_stage(direct, posterior)
                rows.append(neural_row("Oracle p (Direct OVR second stage)", seed, coef,
                                       xt, yt, groups[direct.estimation_indices_], gamma,
                                       n_contrast=0))
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                rows.append(dict(method="Oracle p (Direct OVR second stage)", seed=seed,
                                 role="training", status="failed", error=str(error)))
    return rows, contrasts


def validate_data(frame, center, scale):
    required = {"X", "Y", "E", "H", "block_id", "kind"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required-set(frame.columns))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Required columns contain missing values.")
    for column in ("X", "Y", "E", "H"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[["X", "Y", "E", "H"]].to_numpy(float)).all():
        raise ValueError("X,Y,E,H must be finite.")
    if set(frame.kind) != {"confounded", "reference"}:
        raise ValueError("Expected both confounded and reference rows, and no other kinds.")
    if set(frame.E) != {0, 1} or set(frame.H) != {0, 1}:
        raise ValueError("E and H must be binary with both levels present.")
    frame["E"], frame["H"] = frame.E.astype(int), frame.H.astype(int)
    rounded_posterior(frame.X.to_numpy(), center, scale)
    for kind, subset in frame.groupby("kind"):
        if len(subset.groupby(["E", "H"])) != 4:
            raise ValueError(f"All four E,H cells are required in {kind} data.")
        if not subset.groupby("block_id").E.nunique().eq(2).all():
            raise ValueError(f"Every {kind} acquisition block must include both environments.")
        if subset.block_id.nunique() < 2:
            raise ValueError(f"At least two acquisition blocks are required for {kind}.")
    train = frame.loc[frame.kind.eq("confounded")]
    reference = frame.loc[frame.kind.eq("reference")]
    if train.block_id.nunique() < 8:
        raise ValueError("At least eight confounded blocks are required for grouped OVR splitting.")
    if "led_3_ir" in frame and not np.allclose(frame.X, frame.led_3_ir):
        raise ValueError("X differs from the recorded led_3_ir command.")
    if "ir_3" in frame and not np.allclose(frame.Y, frame.ir_3):
        raise ValueError("Y differs from the recorded raw ir_3 outcome.")
    return train.reset_index(drop=True), reference.reset_index(drop=True)


def coefficient_plot(table, output):
    good = table.loc[table.status.eq("ok")].copy()
    reference = good.loc[good.method.eq("Randomized reference (common)")].iloc[0]
    methods = list(dict.fromkeys(good.method))
    fig, ax = plt.subplots(figsize=(9, max(5, .45*len(methods))))
    if np.isfinite(reference.ci_low) and np.isfinite(reference.ci_high):
        ax.axvspan(reference.ci_low, reference.ci_high, color="0.85", label="Reference 95% interval")
    ax.axvline(reference.coefficient, color="0.35", linestyle="--", linewidth=1)
    for position, method in enumerate(methods):
        selected = good.loc[good.method.eq(method)]
        offsets = np.linspace(-.14, .14, len(selected)) if len(selected)>1 else [0]
        for offset, (_, row) in zip(offsets, selected.iterrows()):
            color = "#0072B2" if "OVR" in method else "#8A4C0F"
            if pd.notna(row.ci_low) and pd.notna(row.ci_high):
                ax.errorbar(row.coefficient, position+offset,
                            xerr=[[row.coefficient-row.ci_low], [row.ci_high-row.coefficient]],
                            fmt="o", color=color, capsize=3, markersize=4)
            else:
                ax.plot(row.coefficient, position+offset, "o", color=color, markersize=4)
    ax.set_yticks(range(len(methods)), methods)
    ax.invert_yaxis()
    ax.set_xlabel("Slope: raw sensor units per LED command unit")
    ax.set_title("Physical pilot: coefficient estimates and randomized reference")
    ax.text(0, -.12, "OVR dots show fixed-seed fits of the same data, not independent physical replications.\n"
            "OVR intervals are omitted; other bars are approximate block-cluster intervals.",
            transform=ax.transAxes, fontsize=8, va="top")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--center", type=int, required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not repeat values")
    frame = pd.read_csv(args.data)
    training, reference = validate_data(frame, args.center, args.scale)
    posterior = rounded_posterior(training.X.to_numpy(), args.center, args.scale)
    # Fit and freeze training estimates before reading reference outcomes below.
    rows = linear_baselines(training, posterior)
    neural, contrasts = fit_ovr(training, posterior, args.seeds)
    rows.extend(neural)
    reference_rows, heterogeneity = reference_analysis(reference)
    rows.extend(reference_rows)
    table = pd.DataFrame(rows)
    reference_beta = reference_rows[0]["coefficient"]
    table["difference_from_reference"] = table.coefficient-reference_beta
    table["absolute_distance_from_reference"] = table.difference_from_reference.abs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir/"estimates.csv", index=False)
    pd.DataFrame(contrasts).to_csv(args.output_dir/"estimated_contrasts.csv", index=False)
    coefficient_plot(table, args.output_dir/"coefficients.png")
    summaries = []
    for method, subset in table.loc[table.role.eq("training")].groupby("method", sort=False):
        values = subset.loc[subset.status.eq("ok"), "coefficient"]
        summaries.append(dict(method=method, attempted=len(subset), succeeded=len(values),
                              mean_coefficient=values.mean(),
                              same_data_seed_sd=values.std(ddof=1) if len(values)>1 else None,
                              mean_absolute_reference_distance=float(np.mean(np.abs(values-reference_beta)))
                              if len(values) else None))
    pd.DataFrame(summaries).to_csv(args.output_dir/"method_summary.csv", index=False)
    notes = [
        "This is a controlled physical pilot with an engineered confounder assignment.",
        "DirectOVR/SmoothedOVR receive only raw X,Y,E and acquisition block IDs; defaults are unmodified.",
        "Oracle p uses exact rounded-bin probabilities for Beta(2,5)/Beta(5,2), with equal H priors.",
        "Neural oracle p reuses DirectOVR's coefficient sample, folds, nuisance learner and seeds.",
        "Repeated seeds describe splitting/optimization variability on the same physical data.",
        "Linear intervals use acquisition-block sandwich covariance and t(G-1) critical values; few blocks limit reliability.",
        "OVR plugin_cluster_se omits nuisance-learning uncertainty; no OVR coverage claim or interval is made.",
        "Distances to an estimated reference are not errors against exact ground truth.",
        "Reference uncertainty is shown separately; no joint OVR-reference confidence interval is claimed.",
        "Training and reference draws can share acquisition blocks; inference on their difference would require joint block resampling.",
        "A common causal-slope interpretation requires the independently checked physical response model.",
        "Finite fitted positivity diagnostics do not establish population positivity.",
        "No IV estimate is fit: the collection design makes E independent of X in the population.",
    ]
    summary = dict(input=str(args.data.resolve()), input_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                   center=args.center, scale=args.scale, seeds=args.seeds,
                   n_training=len(training), n_reference=len(reference),
                   training_blocks=training.block_id.nunique(), reference_blocks=reference.block_id.nunique(),
                   shared_acquisition_blocks=len(set(training.block_id) & set(reference.block_id)),
                   estimator_defaults={cls.__name__: {name: p.default for name, p in inspect.signature(cls).parameters.items()}
                                       for cls in (DirectOVR, SmoothedOVR)},
                   reference=reference_rows, reference_heterogeneity=heterogeneity,
                   methods=summaries, notes=notes, estimates=table.to_dict(orient="records"))
    (args.output_dir/"analysis.json").write_text(json.dumps(clean_json(summary), indent=2, allow_nan=False)+"\n")
    failed = table.loc[table.status.eq("failed")]
    if len(failed):
        warnings.warn(f"{len(failed)} fits failed; see estimates.csv. No replacement or fallback was applied.")
    print(table[["method", "seed", "status", "coefficient", "absolute_distance_from_reference"]].to_string(index=False))
    print(f"Saved analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
