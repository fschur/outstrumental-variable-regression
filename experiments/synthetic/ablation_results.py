"""Validate paired component-ablation fits and compute the table ratios."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from experiments.synthetic.paper_plot import PANELS as SCENARIOS
from experiments.synthetic.run_ablation import VARIANTS


def _successful(frame: pd.DataFrame) -> pd.Series:
    return frame["success"].astype(str).str.lower().isin(("true", "1"))


def load_replications(
    results_root: Path,
    sample_sizes: tuple[int, ...],
    expected_reps: int | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for n in sample_sizes:
        path = results_root / f"n{n}" / "raw_results.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if not frame["n"].eq(n).all():
            raise ValueError(f"sample sizes in {path} do not match n={n}")
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    expected_variants = {variant.key for variant in VARIANTS}
    expected_scenarios = {key for key, _ in SCENARIOS}
    raw = raw[
        raw["scenario_key"].isin(expected_scenarios)
        & raw["variant_key"].isin(expected_variants)
    ].copy()
    keys = ["scenario_key", "n", "rep", "variant_key"]
    duplicated = raw[raw.duplicated(keys, keep=False)]
    if not duplicated.empty:
        raise ValueError(f"duplicate ablation rows: {duplicated[keys].to_dict('records')}")
    failures = raw[~_successful(raw)]
    if not failures.empty:
        raise ValueError(
            "failed ablation rows: "
            + str(failures[keys + ["error_type", "error_message"]].to_dict("records"))
        )
    for cell, frame in raw.groupby(["scenario_key", "n", "rep"], sort=False):
        if set(frame["variant_key"]) != expected_variants:
            raise ValueError(f"incomplete variant block for {cell}")
        if frame["split_plan_hash"].nunique() != 1 or frame["fold_hash"].nunique() != 1:
            raise ValueError(f"unpaired split plan for {cell}")
        ensemble = frame[frame["variant_key"].isin(("ensemble", "smoothed"))]
        if ensemble["contrast_artifact_hash"].nunique() != 1:
            raise ValueError(f"tau=0/1 do not reuse the ensemble for {cell}")
    if expected_reps is not None:
        counts = raw.groupby(["scenario_key", "n", "variant_key"])["rep"].nunique()
        bad = counts[counts != expected_reps]
        if not bad.empty:
            raise ValueError(f"incomplete ablation cells: {bad.to_dict()}")
    return raw


def paired_step_effects(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    variant_keys = [variant.key for variant in VARIANTS]
    for (scenario, n), frame in raw.groupby(["scenario_key", "n"], sort=False):
        indexed = frame.set_index(["rep", "variant_key"]).sort_index()
        for transition, (old, new) in enumerate(zip(variant_keys[:-1], variant_keys[1:])):
            old_frame = indexed.xs(old, level="variant_key").sort_index()
            new_frame = indexed.xs(new, level="variant_key").sort_index()
            if not old_frame.index.equals(new_frame.index):
                raise ValueError(f"unpaired replications for {(scenario, n, old, new)}")
            old_error = old_frame["coefficient_error"].to_numpy(dtype=float)
            new_error = new_frame["coefficient_error"].to_numpy(dtype=float)
            old_moment = old_frame["ridge_adjusted_moment_norm"].to_numpy(dtype=float)
            new_moment = new_frame["ridge_adjusted_moment_norm"].to_numpy(dtype=float)
            old_amp = old_frame["effective_amplification"].to_numpy(dtype=float)
            new_amp = new_frame["effective_amplification"].to_numpy(dtype=float)
            log_error = np.log(new_error / old_error)
            log_moment = np.log(new_moment / old_moment)
            log_amplification = np.log(new_amp / old_amp)
            if not np.allclose(
                log_error,
                log_moment + log_amplification,
                rtol=1e-9,
                atol=1e-9,
            ):
                raise ValueError(f"moment decomposition failed for {(scenario, n, old, new)}")
            row: dict[str, object] = {
                "scenario_key": scenario,
                "n": int(n),
                "transition": transition,
                "old_variant": old,
                "new_variant": new,
                "n_pairs": int(log_error.size),
                "error_ratio_geometric_mean": float(np.exp(log_error.mean())),
                "new_wins_fraction": float((new_error < old_error).mean()),
                "log_error_ratio_mean": float(log_error.mean()),
                "log_moment_ratio_mean": float(log_moment.mean()),
                "log_amplification_ratio_mean": float(log_amplification.mean()),
            }
            for metric in ("minimum_residual_eigenvalue", "gamma", "contrast_rmse"):
                old_values = pd.to_numeric(old_frame[metric], errors="coerce").to_numpy()
                new_values = pd.to_numeric(new_frame[metric], errors="coerce").to_numpy()
                valid = np.isfinite(old_values) & np.isfinite(new_values) & (old_values > 0)
                row[f"{metric}_ratio_geometric_mean"] = (
                    float(np.exp(np.log(new_values[valid] / old_values[valid]).mean()))
                    if valid.any()
                    else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)
