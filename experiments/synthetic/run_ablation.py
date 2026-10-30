"""Controlled component ablation from Direct OVR to Smoothed OVR.

The five nested variants share observations, the outer contrast/slope split,
second-stage folds, nuisance-network seeds, and the nuisance-training code path.
The final two variants literally reuse the same fitted ensemble contrast and
differ only in whether disagreement-calibrated perturbations are applied.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from .sampling import AblationScenario, ReplicationData, _scenario_catalog, _sample_scenario
from ovr import SmoothedOVR
from ovr._core import (
    _disagreement_covariance,
    _onehot,
    _stratified_kfold_indices,
    _stratified_split_indices,
    _train_uncertainty_smoothed_regressor,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "ovr_component_ablation"
RIDGE = 1e-10


@dataclass(frozen=True)
class VariantSpec:
    key: str
    label: str
    step: int


VARIANTS = (
    VariantSpec("direct", "DOVR", 0),
    VariantSpec("regularized", "+ head regularization", 1),
    VariantSpec("early_stopped", "+ validation/early stopping", 2),
    VariantSpec("ensemble", "+ split ensemble ($\\tau=0$)", 3),
    VariantSpec("smoothed", "SOVR ($\\tau=1$)", 4),
)
VARIANT_BY_KEY = {variant.key: variant for variant in VARIANTS}


@dataclass(frozen=True)
class SplitPlan:
    contrast_indices: np.ndarray
    estimation_indices: np.ndarray
    member_a_indices: np.ndarray
    member_b_indices: np.ndarray
    local_folds: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class ContrastArtifact:
    contrast: np.ndarray
    disagreement: np.ndarray
    epochs_mean: float
    training_n: int
    validation_n: int
    first_stage_seconds: float
    artifact_hash: str


def _checksum(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _fold_checksum(folds: tuple[np.ndarray, ...]) -> str:
    pieces: list[np.ndarray] = []
    for fold_id, fold in enumerate(folds):
        pieces.append(np.asarray([fold_id, fold.size], dtype=np.int64))
        pieces.append(np.asarray(fold, dtype=np.int64))
    return _checksum(*pieces)


def _json_vector(value: np.ndarray) -> str:
    return json.dumps(np.asarray(value, dtype=float).ravel().tolist(), separators=(",", ":"))


def _make_engine(
    *,
    n_folds: int,
    hidden: int,
    contrast_epochs: int,
    residual_epochs: int,
    strong_contrast_lambda: float,
    jitter_scale: float,
    prediction_draws: int,
    validation_fraction: float,
    patience: int,
    seed: int,
) -> SmoothedOVR:
    return SmoothedOVR(
        reference=0,
        contrast_fraction=0.5,
        contrast_lambda=strong_contrast_lambda,
        jitter_scale=jitter_scale,
        prediction_draws=prediction_draws,
        contrast_validation_fraction=validation_fraction,
        contrast_patience=patience,
        n_folds=n_folds,
        hidden=hidden,
        depth=2,
        epochs=contrast_epochs,
        resid_epochs=residual_epochs,
        seed=seed,
    )


def _make_split_plan(
    T: np.ndarray, estimator_seed: int, n_folds: int
) -> SplitPlan:
    rng = np.random.default_rng(estimator_seed)
    contrast_indices, estimation_indices = _stratified_split_indices(T, 0.5, rng)
    member_a_local, member_b_local = _stratified_split_indices(
        T[contrast_indices], 0.5, rng
    )
    member_a_indices = contrast_indices[member_a_local]
    member_b_indices = contrast_indices[member_b_local]
    local_folds = tuple(
        _stratified_kfold_indices(T[estimation_indices], n_folds, rng)
    )
    return SplitPlan(
        contrast_indices=contrast_indices,
        estimation_indices=estimation_indices,
        member_a_indices=member_a_indices,
        member_b_indices=member_b_indices,
        local_folds=local_folds,
    )


def _evaluate_model_contrast(
    engine: SmoothedOVR,
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    return engine._contrast_np(model, X) * float(model._ysd)


def _single_artifact(
    engine: SmoothedOVR,
    model: Any,
    X_beta: np.ndarray,
    elapsed: float,
    training_n: int,
    validation_n: int,
    epochs: float,
) -> ContrastArtifact:
    contrast = _evaluate_model_contrast(engine, model, X_beta)
    disagreement = np.zeros_like(contrast)
    return ContrastArtifact(
        contrast=contrast,
        disagreement=disagreement,
        epochs_mean=float(epochs),
        training_n=int(training_n),
        validation_n=int(validation_n),
        first_stage_seconds=float(elapsed),
        artifact_hash=_checksum(contrast, disagreement),
    )


def _fit_contrast_artifacts(
    engine: SmoothedOVR,
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    plan: SplitPlan,
    weak_lambda: float,
    strong_lambda: float,
    estimator_seed: int,
) -> dict[str, ContrastArtifact]:
    X_beta = X[plan.estimation_indices]
    X_contrast = X[plan.contrast_indices]
    Y_contrast = Y[plan.contrast_indices]
    T_contrast = T[plan.contrast_indices]
    artifacts: dict[str, ContrastArtifact] = {}

    started = time.perf_counter()
    direct_model = engine._fit_joint(
        X_contrast,
        Y_contrast,
        T_contrast,
        weak_lambda,
        seed=estimator_seed + 1_000,
    )
    elapsed = time.perf_counter() - started
    artifacts["direct"] = _single_artifact(
        engine,
        direct_model,
        X_beta,
        elapsed,
        plan.contrast_indices.size,
        0,
        engine.epochs,
    )

    started = time.perf_counter()
    regularized_model = engine._fit_joint(
        X_contrast,
        Y_contrast,
        T_contrast,
        strong_lambda,
        seed=estimator_seed + 1_000,
    )
    elapsed = time.perf_counter() - started
    artifacts["regularized"] = _single_artifact(
        engine,
        regularized_model,
        X_beta,
        elapsed,
        plan.contrast_indices.size,
        0,
        engine.epochs,
    )

    started = time.perf_counter()
    stabilized_model = engine._fit_joint_member(
        X_contrast,
        Y_contrast,
        T_contrast,
        seed=estimator_seed + 1_000,
    )
    elapsed = time.perf_counter() - started
    artifacts["early_stopped"] = _single_artifact(
        engine,
        stabilized_model,
        X_beta,
        elapsed,
        int(stabilized_model._member_training_size),
        int(stabilized_model._member_validation_size),
        float(stabilized_model._epochs_trained),
    )

    ensemble = _fit_ensemble_artifact(engine, X, Y, T, plan, estimator_seed)
    artifacts["ensemble"] = ensemble
    artifacts["smoothed"] = ensemble
    return artifacts


def _fit_ensemble_artifact(
    engine: SmoothedOVR,
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    plan: SplitPlan,
    estimator_seed: int,
) -> ContrastArtifact:
    """Fit the two contrast learners once for all smoothing scales."""
    X_beta = X[plan.estimation_indices]
    started = time.perf_counter()
    model_a = engine._fit_joint_member(
        X[plan.member_a_indices],
        Y[plan.member_a_indices],
        T[plan.member_a_indices],
        seed=estimator_seed + 1_000,
    )
    model_b = engine._fit_joint_member(
        X[plan.member_b_indices],
        Y[plan.member_b_indices],
        T[plan.member_b_indices],
        seed=estimator_seed + 2_000,
    )
    ensemble_seconds = time.perf_counter() - started
    contrast_a = _evaluate_model_contrast(engine, model_a, X_beta)
    contrast_b = _evaluate_model_contrast(engine, model_b, X_beta)
    contrast = 0.5 * (contrast_a + contrast_b)
    disagreement = contrast_a - contrast_b
    ensemble = ContrastArtifact(
        contrast=contrast,
        disagreement=disagreement,
        epochs_mean=float(
            np.mean([model_a._epochs_trained, model_b._epochs_trained])
        ),
        training_n=int(
            model_a._member_training_size + model_b._member_training_size
        ),
        validation_n=int(
            model_a._member_validation_size + model_b._member_validation_size
        ),
        first_stage_seconds=float(ensemble_seconds),
        artifact_hash=_checksum(contrast, disagreement),
    )
    return ensemble


def _crossfit_residuals(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    contrast: np.ndarray,
    disagreement: np.ndarray,
    local_folds: tuple[np.ndarray, ...],
    n_environments: int,
    jitter_scale: float,
    prediction_draws: int,
    hidden: int,
    depth: int,
    residual_epochs: int,
    residual_lr: float,
    residual_weight_decay: float,
    second_stage_seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    n, d = X.shape
    residualized_X = np.zeros((n, d))
    residualized_Y = np.zeros(n)
    onehot = _onehot(T, n_environments)
    targets = np.hstack([X, Y[:, None]])
    all_indices = np.arange(n)
    jitter_covariances: list[np.ndarray] = []
    for fold_id, test_indices in enumerate(local_folds):
        train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
        uncertainty = _disagreement_covariance(disagreement[train_indices])
        jitter_covariance = jitter_scale**2 * uncertainty
        train_features = np.hstack([contrast[train_indices], onehot[train_indices]])
        test_features = np.hstack([contrast[test_indices], onehot[test_indices]])
        predict = _train_uncertainty_smoothed_regressor(
            train_features,
            targets[train_indices],
            contrast_dim=contrast.shape[1],
            jitter_covariance=jitter_covariance,
            prediction_draws=prediction_draws,
            hidden=hidden,
            depth=depth,
            epochs=residual_epochs,
            lr=residual_lr,
            weight_decay=residual_weight_decay,
            seed=second_stage_seed + fold_id,
        )
        fitted = predict(test_features)
        residualized_X[test_indices] = X[test_indices] - fitted[:, :d]
        residualized_Y[test_indices] = Y[test_indices] - fitted[:, d]
        jitter_covariances.append(jitter_covariance)
    return residualized_X, residualized_Y, tuple(jitter_covariances)


def _contrast_error(
    artifact: ContrastArtifact,
    dgp: Any,
    X_beta: np.ndarray,
) -> tuple[float, float]:
    if not hasattr(dgp, "oracle_contrast"):
        return float("nan"), float("nan")
    truth = np.asarray(dgp.oracle_contrast(X_beta), dtype=float)
    if truth.ndim == 1:
        truth = truth[:, None]
    error = artifact.contrast - truth
    rmse = float(np.sqrt(np.mean(error**2)))
    scale = max(float(np.sqrt(np.mean(truth**2))), np.finfo(float).eps)
    return rmse, rmse / scale


def _variant_row(
    *,
    scenario: AblationScenario,
    rep: int,
    n: int,
    replication: ReplicationData,
    variant: VariantSpec,
    artifact: ContrastArtifact,
    plan: SplitPlan,
    engine: SmoothedOVR,
    jitter_scale: float,
) -> dict[str, Any]:
    data = replication.data
    X_beta = data.X[plan.estimation_indices]
    Y_beta = data.Y[plan.estimation_indices]
    T_beta = data.T[plan.estimation_indices]
    started = time.perf_counter()
    residualized_X, residualized_Y, jitter_covariances = _crossfit_residuals(
        X=X_beta,
        Y=Y_beta,
        T=T_beta,
        contrast=artifact.contrast,
        disagreement=artifact.disagreement,
        local_folds=plan.local_folds,
        n_environments=int(np.unique(data.T).size),
        jitter_scale=jitter_scale,
        prediction_draws=engine.prediction_draws,
        hidden=engine.hidden,
        depth=engine.depth,
        residual_epochs=engine.resid_epochs,
        residual_lr=engine.resid_lr,
        residual_weight_decay=engine.resid_weight_decay,
        second_stage_seed=replication.estimator_seed + 20_000,
    )
    second_stage_seconds = time.perf_counter() - started
    n_beta = X_beta.shape[0]
    sigma = residualized_X.T @ residualized_X / n_beta
    moment_xy = residualized_X.T @ residualized_Y / n_beta
    regularized_sigma = sigma + RIDGE * np.eye(sigma.shape[0])
    beta_hat = np.linalg.solve(regularized_sigma, moment_xy)
    beta0 = np.asarray(data.beta0, dtype=float)
    coefficient_error = float(np.linalg.norm(beta_hat - beta0))
    beta_norm = max(float(np.linalg.norm(beta0)), np.finfo(float).eps)
    ridge_adjusted_moment = moment_xy - regularized_sigma @ beta0
    moment_norm = float(np.linalg.norm(ridge_adjusted_moment))
    truth_score = residualized_Y - residualized_X @ beta0
    truth_score_moment = residualized_X.T @ truth_score / n_beta
    truth_score_norm = float(np.linalg.norm(truth_score_moment))
    amplification = coefficient_error / max(moment_norm, np.finfo(float).tiny)
    eigenvalues = np.linalg.eigvalsh(0.5 * (sigma + sigma.T))
    minimum_eigenvalue = float(eigenvalues.min())
    condition_number = (
        float(eigenvalues.max() / minimum_eigenvalue)
        if minimum_eigenvalue > 1e-12
        else float("inf")
    )
    sigma_environment = engine._sigma_T(
        X_beta, T_beta, int(np.unique(data.T).size)
    )
    gamma = engine._gamma(sigma, sigma_environment)
    worst_case_bound = moment_norm / max(
        minimum_eigenvalue + RIDGE, np.finfo(float).tiny
    )
    bound_slack = worst_case_bound / max(coefficient_error, np.finfo(float).tiny)
    contrast_rmse, contrast_relative_rmse = _contrast_error(
        artifact, replication.dgp, X_beta
    )
    uncertainty = _disagreement_covariance(artifact.disagreement)
    jitter_trace = float(np.mean([np.trace(value) for value in jitter_covariances]))
    identity_error = float(
        np.linalg.norm(
            regularized_sigma @ (beta_hat - beta0) - ridge_adjusted_moment
        )
    )
    return {
        "scenario_id": scenario.scenario_id,
        "scenario_key": scenario.key,
        "scenario": scenario.label,
        "rep": rep,
        "n": n,
        "variant_key": variant.key,
        "variant": variant.label,
        "variant_step": variant.step,
        "success": bool(np.isfinite(beta_hat).all()),
        "structural_seed": replication.structural_seed,
        "data_seed": replication.data_seed,
        "estimator_seed": replication.estimator_seed,
        "seed_protocol": replication.seed_protocol,
        "beta0": _json_vector(beta0),
        "beta_hat": _json_vector(beta_hat),
        "coefficient_error": coefficient_error,
        "coefficient_error_normalized": coefficient_error / beta_norm,
        "ridge_adjusted_moment_norm": moment_norm,
        "truth_score_moment_norm": truth_score_norm,
        "effective_amplification": amplification,
        "minimum_residual_eigenvalue": minimum_eigenvalue,
        "gamma": gamma,
        "condition_number": condition_number,
        "residual_covariance_trace": float(np.trace(sigma)),
        "worst_case_bound": worst_case_bound,
        "worst_case_bound_slack": bound_slack,
        "moment_identity_error": identity_error,
        "contrast_rmse": contrast_rmse,
        "contrast_relative_rmse": contrast_relative_rmse,
        "contrast_epochs_mean": artifact.epochs_mean,
        "contrast_training_n": artifact.training_n,
        "contrast_validation_n": artifact.validation_n,
        "contrast_disagreement_rmse": float(
            np.sqrt(np.mean(artifact.disagreement**2))
        ),
        "uncertainty_trace": float(np.trace(uncertainty)),
        "jitter_trace_mean": jitter_trace,
        "first_stage_seconds": artifact.first_stage_seconds,
        "second_stage_seconds": second_stage_seconds,
        "contrast_split_n": int(plan.contrast_indices.size),
        "estimation_split_n": int(plan.estimation_indices.size),
        "split_plan_hash": _checksum(
            plan.contrast_indices,
            plan.estimation_indices,
            plan.member_a_indices,
            plan.member_b_indices,
        ),
        "fold_hash": _fold_checksum(plan.local_folds),
        "contrast_artifact_hash": artifact.artifact_hash,
        "error_type": None,
        "error_message": None,
        "traceback": None,
    }


def fit_ablation_variants(
    *,
    scenario: AblationScenario,
    rep: int,
    n: int,
    replication: ReplicationData,
    n_folds: int,
    hidden: int,
    contrast_epochs: int,
    residual_epochs: int,
    weak_contrast_lambda: float,
    strong_contrast_lambda: float,
    jitter_scale: float,
    prediction_draws: int,
    validation_fraction: float,
    patience: int,
) -> list[dict[str, Any]]:
    data = replication.data
    engine = _make_engine(
        n_folds=n_folds,
        hidden=hidden,
        contrast_epochs=contrast_epochs,
        residual_epochs=residual_epochs,
        strong_contrast_lambda=strong_contrast_lambda,
        jitter_scale=jitter_scale,
        prediction_draws=prediction_draws,
        validation_fraction=validation_fraction,
        patience=patience,
        seed=replication.estimator_seed,
    )
    engine._prepare(data.X, data.Y, data.T)
    plan = _make_split_plan(data.T, replication.estimator_seed, n_folds)
    artifacts = _fit_contrast_artifacts(
        engine,
        data.X,
        data.Y,
        data.T,
        plan,
        weak_contrast_lambda,
        strong_contrast_lambda,
        replication.estimator_seed,
    )
    rows = []
    for variant in VARIANTS:
        rows.append(
            _variant_row(
                scenario=scenario,
                rep=rep,
                n=n,
                replication=replication,
                variant=variant,
                artifact=artifacts[variant.key],
                plan=plan,
                engine=engine,
                jitter_scale=jitter_scale if variant.key == "smoothed" else 0.0,
            )
        )
    return rows


def _failed_rows(
    scenario: AblationScenario,
    rep: int,
    n: int,
    replication: ReplicationData,
    exc: BaseException,
) -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": scenario.scenario_id,
            "scenario_key": scenario.key,
            "scenario": scenario.label,
            "rep": rep,
            "n": n,
            "variant_key": variant.key,
            "variant": variant.label,
            "variant_step": variant.step,
            "success": False,
            "structural_seed": replication.structural_seed,
            "data_seed": replication.data_seed,
            "estimator_seed": replication.estimator_seed,
            "seed_protocol": replication.seed_protocol,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
        for variant in VARIANTS
    ]


def _run_rep(task: tuple[Any, ...]) -> list[dict[str, Any]]:
    import torch
    torch.set_num_threads(1)
    (
        scenario,
        rep,
        n,
        main_seed,
        shared_seed,
        n_folds,
        hidden,
        contrast_epochs,
        residual_epochs,
        weak_contrast_lambda,
        strong_contrast_lambda,
        jitter_scale,
        prediction_draws,
        validation_fraction,
        patience,
    ) = task
    replication = _sample_scenario(scenario, rep, n, main_seed, shared_seed)
    try:
        return fit_ablation_variants(
            scenario=scenario,
            rep=rep,
            n=n,
            replication=replication,
            n_folds=n_folds,
            hidden=hidden,
            contrast_epochs=contrast_epochs,
            residual_epochs=residual_epochs,
            weak_contrast_lambda=weak_contrast_lambda,
            strong_contrast_lambda=strong_contrast_lambda,
            jitter_scale=jitter_scale,
            prediction_draws=prediction_draws,
            validation_fraction=validation_fraction,
            patience=patience,
        )
    except Exception as exc:
        return _failed_rows(scenario, rep, n, replication, exc)


SUMMARY_METRICS = (
    "coefficient_error",
    "coefficient_error_normalized",
    "ridge_adjusted_moment_norm",
    "truth_score_moment_norm",
    "effective_amplification",
    "minimum_residual_eigenvalue",
    "gamma",
    "condition_number",
    "residual_covariance_trace",
    "worst_case_bound",
    "worst_case_bound_slack",
    "contrast_rmse",
    "contrast_relative_rmse",
    "contrast_epochs_mean",
    "contrast_disagreement_rmse",
    "uncertainty_trace",
    "jitter_trace_mean",
    "first_stage_seconds",
    "second_stage_seconds",
)


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (scenario_key, scenario, variant_key, variant, step), group in raw.groupby(
        ["scenario_key", "scenario", "variant_key", "variant", "variant_step"],
        sort=False,
    ):
        success = group[group["success"].astype(str).str.lower().isin(("true", "1"))]
        row: dict[str, Any] = {
            "scenario_key": scenario_key,
            "scenario": scenario,
            "variant_key": variant_key,
            "variant": variant,
            "variant_step": int(step),
            "n_reps": int(len(group)),
            "n_success": int(len(success)),
        }
        for metric in SUMMARY_METRICS:
            values = pd.to_numeric(success.get(metric), errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_se"] = (
                float(values.std(ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else np.nan
            )
            row[f"{metric}_median"] = (
                float(values.median()) if len(values) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _config(args: argparse.Namespace, scenarios: tuple[AblationScenario, ...]) -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        versions["torch"] = None
    return {
        "arguments": {k: v for k, v in vars(args).items() if k not in {"output_dir", "workers", "resume"}},
        "variants": [variant.__dict__ for variant in VARIANTS],
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "key": scenario.key,
                "label": scenario.label,
                "exposure_shift_scale": scenario.exposure_shift_scale,
                "latent_shift_scale": scenario.latent_shift_scale,
            }
            for scenario in scenarios
        ],
        "controls": {
            "shared_outer_split": True,
            "shared_second_stage_folds": True,
            "shared_second_stage_seeds": True,
            "shared_nuisance_training_code": True,
            "ensemble_reused_for_tau_zero_and_one": True,
        },
        "versions": versions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--main-seed", type=int, default=20260807)
    parser.add_argument("--shared-seed", type=int, default=20260808)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--contrast-epochs", type=int, default=200)
    parser.add_argument("--residual-epochs", type=int, default=300)
    parser.add_argument("--weak-contrast-lambda", type=float, default=1e-4)
    parser.add_argument("--strong-contrast-lambda", type=float, default=1e-2)
    parser.add_argument("--jitter-scale", type=float, default=1.0)
    parser.add_argument("--prediction-draws", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scenarios",
        type=str,
        default=",".join(scenario.key for scenario in _scenario_catalog()),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.n = 600
        args.reps = 2
        args.contrast_epochs = 2
        args.residual_epochs = 2
        args.hidden = 8
    if args.n <= 0 or args.reps <= 0 or args.workers <= 0:
        parser.error("n, reps, and workers must be positive")

    requested = tuple(value.strip() for value in args.scenarios.split(",") if value.strip())
    catalog = {scenario.key: scenario for scenario in _scenario_catalog()}
    unknown = sorted(set(requested) - set(catalog))
    if unknown:
        parser.error(f"unknown scenarios: {unknown}")
    scenarios = tuple(catalog[key] for key in requested)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_results.csv"
    summary_path = args.output_dir / "summary.csv"
    config_path = args.output_dir / "config.json"

    existing = pd.DataFrame()
    complete: set[tuple[str, int]] = set()
    if args.resume and raw_path.exists():
        saved = json.loads(config_path.read_text())
        if saved != _config(args, scenarios):
            parser.error("Resume configuration differs; use a new output directory")
        existing = pd.read_csv(raw_path)
        expected = {variant.key for variant in VARIANTS}
        for (scenario_key, rep), frame in existing.groupby(["scenario_key", "rep"]):
            if expected == set(frame["variant_key"]):
                complete.add((str(scenario_key), int(rep)))

    tasks = [
        (
            scenario,
            rep,
            args.n,
            args.main_seed,
            args.shared_seed,
            args.n_folds,
            args.hidden,
            args.contrast_epochs,
            args.residual_epochs,
            args.weak_contrast_lambda,
            args.strong_contrast_lambda,
            args.jitter_scale,
            args.prediction_draws,
            args.validation_fraction,
            args.patience,
        )
        for scenario in scenarios
        for rep in range(args.reps)
        if (scenario.key, rep) not in complete
    ]
    config_path.write_text(
        json.dumps(_config(args, scenarios), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    collected = existing.to_dict("records") if not existing.empty else []

    def checkpoint(new_rows: list[dict[str, Any]], done: int) -> None:
        collected.extend(new_rows)
        frame = pd.DataFrame(collected).sort_values(
            ["scenario_id", "rep", "variant_step"], kind="stable"
        )
        _atomic_csv(frame, raw_path)
        print(f"  {done}/{len(tasks)} replications complete", flush=True)

    print(
        f"Running {len(tasks)} controlled replications "
        f"({len(scenarios)} scenarios x {args.reps} requested) on "
        f"{args.workers} worker(s)",
        flush=True,
    )
    if args.workers == 1:
        for done, task in enumerate(tasks, 1):
            checkpoint(_run_rep(task), done)
    else:
        try:
            executor = cf.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=mp.get_context("spawn")
            )
        except PermissionError:
            print("Process workers unavailable; falling back to one worker.", flush=True)
            for done, task in enumerate(tasks, 1):
                checkpoint(_run_rep(task), done)
        else:
            with executor:
                futures = [executor.submit(_run_rep, task) for task in tasks]
                for done, future in enumerate(cf.as_completed(futures), 1):
                    checkpoint(future.result(), done)

    raw = pd.DataFrame(collected).sort_values(
        ["scenario_id", "rep", "variant_step"], kind="stable"
    )
    summary = summarize(raw)
    _atomic_csv(raw, raw_path)
    _atomic_csv(summary, summary_path)
    print(
        summary[
            [
                "scenario",
                "variant",
                "coefficient_error_mean",
                "coefficient_error_se",
                "ridge_adjusted_moment_norm_mean",
                "effective_amplification_mean",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"Outputs written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
