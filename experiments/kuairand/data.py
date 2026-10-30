"""KuaiRand-Pure cohort preparation and held-out reference regression.

The two environments are standard recommendations and uniform random video
insertions. X contains six pre-viewing video attributes and Y is capped log
watch time. User IDs determine grouped splits; no user-level predictors are
supplied to the methods. The reference is a linear projection under random
assignment, not known causal ground truth.
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ovr import (
    DirectOVR,
    SmoothedOVR,
)


KUAI_PURE_URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
ARCHIVE_NAME = "KuaiRand-Pure.tar.gz"
ARCHIVE_MD5 = "0820331067a3784d9691136f772b35a7"
EXTRACTED_DIR = "KuaiRand-Pure"
RANDOM_LOG = "log_random_4_22_to_5_08_pure.csv"
STANDARD_LOG = "log_standard_4_22_to_5_08_pure.csv"
HISTORICAL_LOG = "log_standard_4_08_to_4_21_pure.csv"
USER_FEATURES = "user_features_pure.csv"
VIDEO_FEATURES = "video_features_basic_pure.csv"

LOG_COLUMNS = (
    "user_id",
    "video_id",
    "date",
    "hourmin",
    "time_ms",
    "play_time_ms",
    "duration_ms",
    "is_rand",
    "tab",
)
VIDEO_COLUMNS = (
    "video_id",
    "video_type",
    "upload_dt",
    "upload_type",
    "video_duration",
    "server_width",
    "server_height",
    "tag",
)
USER_COLUMNS = ("user_id",)

X_COLUMNS = (
    "x_log_duration",
    "x_log_upload_age",
    "x_log_aspect_ratio",
    "x_long_import",
    "x_web_upload",
    "x_tag_count",
)
X_LABELS = (
    "Log duration",
    "Log upload age",
    "Log aspect ratio",
    "Long import",
    "Web upload",
    "Tag count",
)


@dataclass(frozen=True)
class KuaiRandConfig:
    tab: int = 1
    minimum_interactions_per_environment: int = 2
    per_user_environment_cap: int = 100000
    max_users: int | None = None
    watch_time_cap_seconds: float = 300.0
    contrast_fraction: float = 0.5
    contrast_lambda: float = 1e-2
    ovr_folds: int = 3
    ovr_hidden: int = 64
    ovr_depth: int = 2
    ovr_epochs: int = 120
    ovr_resid_epochs: int = 160
    ovr_batch_size: int = 512
    seed: int = 20260808


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_kuairand(data_dir: Path) -> Path:
    """Download and extract the official public KuaiRand-Pure release."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / ARCHIVE_NAME
    if not archive.exists() or _md5(archive) != ARCHIVE_MD5:
        temporary = archive.with_suffix(".part")
        urllib.request.urlretrieve(KUAI_PURE_URL, temporary)
        if _md5(temporary) != ARCHIVE_MD5:
            raise ValueError("Downloaded KuaiRand-Pure archive failed its official MD5 check.")
        temporary.replace(archive)
    extracted = data_dir / EXTRACTED_DIR
    if not (extracted / "data" / RANDOM_LOG).exists():
        with tarfile.open(archive, "r:gz") as source:
            root = data_dir.resolve()
            for member in source.getmembers():
                destination = (data_dir / member.name).resolve()
                if root not in destination.parents and destination != root:
                    raise ValueError(f"Unsafe archive path: {member.name}")
            source.extractall(data_dir)
    return extracted / "data"


def resolve_source(data_dir: Path) -> Path:
    candidates = (data_dir / EXTRACTED_DIR / "data", data_dir / "data", data_dir)
    for candidate in candidates:
        required = (RANDOM_LOG, STANDARD_LOG, HISTORICAL_LOG, USER_FEATURES, VIDEO_FEATURES)
        if all((candidate / filename).exists() for filename in required):
            return candidate
    raise FileNotFoundError(
        f"No KuaiRand-Pure files found under {data_dir}; pass --download to retrieve them."
    )


def _read_sources(
    source: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random = pd.read_csv(source / RANDOM_LOG, usecols=list(LOG_COLUMNS))
    standard = pd.read_csv(source / STANDARD_LOG, usecols=list(LOG_COLUMNS))
    historical = pd.read_csv(
        source / HISTORICAL_LOG, usecols=["user_id", "video_id", "tab"]
    )
    videos = pd.read_csv(source / VIDEO_FEATURES, usecols=list(VIDEO_COLUMNS))
    users = pd.read_csv(source / USER_FEATURES, usecols=list(USER_COLUMNS))
    if not random["is_rand"].eq(1).all() or not standard["is_rand"].eq(0).all():
        raise ValueError("KuaiRand policy indicators do not match the source filenames.")
    return random, standard, historical, videos, users


def _tag_count(value: Any) -> int:
    if pd.isna(value) or str(value).strip() in {"", "-1"}:
        return 0
    return min(len([part for part in str(value).split(",") if part.strip()]), 6)


def _standardize(frame: pd.DataFrame, columns: Iterable[str]) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    result = frame.copy()
    scales: dict[str, dict[str, float]] = {}
    for column in columns:
        mean = float(result[column].mean())
        sd = float(result[column].std(ddof=0))
        if not np.isfinite(sd) or sd < 1e-8:
            raise ValueError(f"Feature {column} has no usable variation.")
        result[column] = (result[column] - mean) / sd
        scales[column] = {"mean": mean, "sd": sd}
    return result, scales


def _equal_user_policy_sample(frame: pd.DataFrame, cfg: KuaiRandConfig) -> pd.DataFrame:
    counts = frame.groupby(["user_id", "environment"]).size().unstack(fill_value=0)
    if set(counts.columns) != {0, 1}:
        raise ValueError("Both logging policies must be present before user matching.")
    eligible = counts.loc[
        counts.min(axis=1).ge(cfg.minimum_interactions_per_environment)
    ].copy()
    eligible["take"] = np.minimum(
        eligible.min(axis=1).to_numpy(), cfg.per_user_environment_cap
    ).astype(int)
    if cfg.max_users is not None and eligible.shape[0] > cfg.max_users:
        rng = np.random.default_rng(cfg.seed + 101)
        retained = rng.choice(eligible.index.to_numpy(), cfg.max_users, replace=False)
        eligible = eligible.loc[np.sort(retained)]
    sampled = frame.loc[frame["user_id"].isin(eligible.index)].copy()
    rng = np.random.default_rng(cfg.seed + 102)
    sampled["_sample_key"] = rng.random(sampled.shape[0])
    sampled = sampled.sort_values(
        ["user_id", "environment", "_sample_key"], kind="mergesort"
    )
    sampled["_within_user_policy"] = sampled.groupby(
        ["user_id", "environment"], sort=False
    ).cumcount()
    sampled = sampled.join(eligible[["take"]], on="user_id")
    sampled = sampled.loc[sampled["_within_user_policy"].lt(sampled["take"])].copy()
    sampled.drop(columns=["_sample_key", "_within_user_policy", "take"], inplace=True)
    policy_counts = sampled["environment"].value_counts().sort_index()
    if policy_counts.size != 2 or policy_counts.iloc[0] != policy_counts.iloc[1]:
        raise RuntimeError("Equal within-user policy sampling failed.")
    return sampled.sort_values(["user_id", "time_ms"], kind="mergesort").reset_index(drop=True)


def prepare_kuairand_cohort(
    source: Path, cfg: KuaiRandConfig
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, float]]]:
    random, standard, historical, videos, users = _read_sources(source)
    flow: dict[str, int] = {
        "random_contemporaneous_raw": int(random.shape[0]),
        "standard_contemporaneous_raw": int(standard.shape[0]),
        "historical_standard_raw": int(historical.shape[0]),
    }
    random = random.loc[random["tab"].eq(cfg.tab)].copy()
    standard = standard.loc[standard["tab"].eq(cfg.tab)].copy()
    flow["random_same_interface"] = int(random.shape[0])
    flow["standard_same_interface"] = int(standard.shape[0])

    videos = videos.copy()
    videos["upload_dt"] = pd.to_datetime(videos["upload_dt"], errors="coerce")
    valid_video = (
        videos["video_type"].eq("NORMAL")
        & videos["video_duration"].gt(0)
        & videos["server_width"].gt(0)
        & videos["server_height"].gt(0)
        & videos["upload_dt"].notna()
    )
    videos = videos.loc[valid_video].copy()
    upload_age = (pd.Timestamp("2022-04-22") - videos["upload_dt"]).dt.days.clip(lower=0)
    videos["x_log_duration"] = np.log1p(videos["video_duration"] / 1000.0)
    videos["x_log_upload_age"] = np.log1p(upload_age)
    videos["x_log_aspect_ratio"] = np.log(
        videos["server_width"] / videos["server_height"]
    )
    videos["x_long_import"] = videos["upload_type"].eq("LongImport").astype(float)
    videos["x_web_upload"] = videos["upload_type"].eq("Web").astype(float)
    videos["x_tag_count"] = videos["tag"].map(_tag_count).astype(float)

    historical = historical.loc[historical["tab"].eq(cfg.tab)].merge(
        videos.loc[:, ["video_id", *X_COLUMNS]],
        on="video_id",
        how="inner",
        validate="many_to_one",
    )
    historical_users = historical["user_id"].unique()
    flow["historical_same_interface_valid_metadata"] = int(historical.shape[0])
    flow["historical_profile_users"] = int(len(historical_users))
    common_items = (
        set(random["video_id"])
        & set(standard["video_id"])
        & set(videos["video_id"])
    )
    random = random.loc[
        random["video_id"].isin(common_items)
        & random["duration_ms"].gt(0)
        & random["play_time_ms"].ge(0)
    ].copy()
    standard = standard.loc[
        standard["video_id"].isin(common_items)
        & standard["duration_ms"].gt(0)
        & standard["play_time_ms"].ge(0)
    ].copy()
    common_users = (
        set(random["user_id"])
        & set(standard["user_id"])
        & set(historical_users)
    )
    random = random.loc[random["user_id"].isin(common_users)].copy()
    standard = standard.loc[standard["user_id"].isin(common_users)].copy()
    flow["shared_valid_items"] = int(len(common_items))
    flow["shared_users_before_minimum"] = int(len(common_users))
    flow["random_on_common_support"] = int(random.shape[0])
    flow["standard_on_common_support"] = int(standard.shape[0])

    random["environment"] = 1
    standard["environment"] = 0
    frame = pd.concat([standard, random], ignore_index=True)
    frame = frame.merge(
        videos.loc[:, ["video_id", *X_COLUMNS]],
        on="video_id",
        how="inner",
        validate="many_to_one",
    )
    frame = frame.merge(users, on="user_id", how="inner", validate="many_to_one")
    frame = _equal_user_policy_sample(frame, cfg)

    watch_seconds = (frame["play_time_ms"] / 1000.0).clip(
        lower=0.0, upper=cfg.watch_time_cap_seconds
    )
    frame["outcome_log_watch_seconds"] = np.log1p(watch_seconds)

    if not np.isfinite(frame.loc[:, list(X_COLUMNS) + ["outcome_log_watch_seconds"]].to_numpy(float)).all():
        raise ValueError("Prepared KuaiRand cohort contains non-finite analysis values.")
    frame, x_scales = _standardize(frame, X_COLUMNS)
    scales = x_scales
    policy_counts = frame["environment"].value_counts().sort_index()
    user_policy_counts = frame.groupby(["user_id", "environment"]).size().unstack()
    if not np.array_equal(user_policy_counts[0].to_numpy(), user_policy_counts[1].to_numpy()):
        raise RuntimeError("Retained users do not have equal policy weights.")
    metadata = {
        "flow": flow,
        "analysis_rows": int(frame.shape[0]),
        "rows_per_environment": {str(int(k)): int(v) for k, v in policy_counts.items()},
        "analysis_users": int(frame["user_id"].nunique()),
        "analysis_items": int(frame["video_id"].nunique()),
        "items_by_environment": {
            str(int(k)): int(v)
            for k, v in frame.groupby("environment")["video_id"].nunique().items()
        },
        "users_by_environment": {
            str(int(k)): int(v)
            for k, v in frame.groupby("environment")["user_id"].nunique().items()
        },
        "environment_order": ["Standard recommender", "Uniform random insertion"],
        "x_columns": list(X_COLUMNS),
        "outcome": "log1p(min(play_time_seconds, 300))",
        "raw_common_item_count": int(len(common_items)),
    }
    keep_columns = (
        ["user_id", "video_id", "environment", "date", "outcome_log_watch_seconds"]
        + list(X_COLUMNS)
    )
    return frame.loc[:, keep_columns].copy(), metadata, scales


def _make_direct_ovr(
    cfg: KuaiRandConfig, seed: int
) -> DirectOVR:
    """Build the theory-aligned estimator without smoothing or early stopping."""
    return DirectOVR(
        reference=0,
        contrast_fraction=cfg.contrast_fraction,
        contrast_lambda=1e-4,
        n_folds=cfg.ovr_folds,
        hidden=cfg.ovr_hidden,
        depth=cfg.ovr_depth,
        epochs=cfg.ovr_epochs,
        batch_size=cfg.ovr_batch_size,
        resid_epochs=cfg.ovr_resid_epochs,
        seed=seed,
    )


def _make_smoothed_ovr(
    cfg: KuaiRandConfig, seed: int
) -> SmoothedOVR:
    return SmoothedOVR(
        reference=0,
        contrast_fraction=cfg.contrast_fraction,
        contrast_lambda=cfg.contrast_lambda,
        jitter_scale=1.0,
        prediction_draws=8,
        contrast_validation_fraction=0.0,
        contrast_patience=0,
        n_folds=cfg.ovr_folds,
        hidden=cfg.ovr_hidden,
        depth=cfg.ovr_depth,
        epochs=cfg.ovr_epochs,
        batch_size=cfg.ovr_batch_size,
        resid_epochs=cfg.ovr_resid_epochs,
        seed=seed,
    )


def _fit_linear(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    design = np.column_stack([np.ones(X.shape[0]), X])
    coefficient = np.linalg.lstsq(design, Y, rcond=1e-10)[0]
    residual = Y - design @ coefficient
    return coefficient[1:], float(coefficient[0]), residual


def _one_way_cluster_covariance(
    design: np.ndarray,
    residual: np.ndarray,
    clusters: np.ndarray,
    bread: np.ndarray,
) -> np.ndarray:
    """Finite-sample-corrected one-way cluster sandwich covariance."""
    codes, uniques = pd.factorize(np.asarray(clusters), sort=False)
    scores = (
        pd.DataFrame(design * residual[:, None])
        .assign(_cluster=codes)
        .groupby("_cluster", sort=False)
        .sum()
        .to_numpy()
    )
    meat = scores.T @ scores
    n, p, g = design.shape[0], design.shape[1], uniques.size
    correction = (g / max(g - 1, 1)) * ((n - 1) / max(n - p, 1))
    return correction * bread @ meat @ bread


def randomized_exposure_projection(
    X: np.ndarray,
    Y: np.ndarray,
    E: np.ndarray,
    users: np.ndarray,
    videos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomized-exposure-arm slope and two-way user/video clustered SEs.

    The slope is an associational linear projection. Random insertion removes
    assignment confounding by user state, but does not randomize the individual
    video attributes or rule out omitted item-level attributes.
    """
    mask = E == 1
    Xr, Yr = X[mask], Y[mask]
    user_clusters, video_clusters = users[mask], videos[mask]
    beta, intercept, residual = _fit_linear(Xr, Yr)
    design = np.column_stack([np.ones(Xr.shape[0]), Xr])
    bread = np.linalg.pinv(design.T @ design)
    pair_clusters = pd.MultiIndex.from_arrays([user_clusters, video_clusters])
    covariance = (
        _one_way_cluster_covariance(design, residual, user_clusters, bread)
        + _one_way_cluster_covariance(design, residual, video_clusters, bread)
        - _one_way_cluster_covariance(design, residual, pair_clusters, bread)
    )
    covariance = (covariance + covariance.T) / 2.0
    return beta, np.sqrt(np.clip(np.diag(covariance)[1:], 0.0, None))
