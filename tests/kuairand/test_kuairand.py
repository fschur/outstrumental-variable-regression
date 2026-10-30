from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from experiments.kuairand.data import (
    HISTORICAL_LOG,
    RANDOM_LOG,
    STANDARD_LOG,
    USER_FEATURES,
    VIDEO_FEATURES,
    X_COLUMNS,
    KuaiRandConfig,
    prepare_kuairand_cohort,
    randomized_exposure_projection,
)
from experiments.kuairand.run import stratified_half_split
from ovr import (
    DirectOVR,
    SmoothedOVR,
)


def _write_fixture(directory: Path, n_users: int = 18) -> None:
    videos = []
    for video in range(6):
        videos.append(
            {
                "video_id": video,
                "author_id": 100 + video,
                "video_type": "NORMAL",
                "upload_dt": f"2022-04-{1 + video:02d}",
                "upload_type": "LongImport" if video % 3 == 0 else "Web",
                "visible_status": 0,
                "video_duration": 10000 + 1000 * video,
                "server_width": 720,
                "server_height": 1280 if video % 2 else 720,
                "music_id": video,
                "music_type": 9,
                "tag": "1,2" if video % 2 else "3",
            }
        )
    users = []
    for user in range(n_users):
        users.append(
            {
                "user_id": user,
                "user_active_degree": (
                    "low_active" if user % 5 == 0 else ("full_active" if user % 2 else "high_active")
                ),
                "is_lowactive_period": user % 2,
                "is_live_streamer": user % 3 == 0,
                "is_video_author": user % 4 == 0,
                "follow_user_num": 10 + user,
                "follow_user_num_range": "(10,50]",
                "fans_user_num": 5 + user,
                "fans_user_num_range": "[1,10)",
                "friend_user_num": 2 + user,
                "friend_user_num_range": "[1,5)",
                "register_days": 100 + user,
                "register_days_range": "91-180",
                **{f"onehot_feat{j}": 0 for j in range(18)},
            }
        )
    def rows(environment: int) -> list[dict]:
        output = []
        for user in range(n_users):
            for repeat in range(4):
                video = (user + repeat + environment) % 6
                duration = 10000 + 1000 * video
                output.append(
                    {
                        "user_id": user,
                        "video_id": video,
                        "date": 20220422 + repeat,
                        "hourmin": 900 + 100 * repeat,
                        "time_ms": 100000 * user + repeat,
                        "is_click": 1,
                        "is_like": 0,
                        "is_follow": 0,
                        "is_comment": 0,
                        "is_forward": 0,
                        "is_hate": 0,
                        "long_view": 0,
                        "play_time_ms": 1000 + 100 * user + 50 * video,
                        "duration_ms": duration,
                        "profile_stay_time": 0,
                        "comment_stay_time": 0,
                        "is_profile_enter": 0,
                        "is_rand": environment,
                        "tab": 1,
                    }
                )
        return output
    pd.DataFrame(rows(1)).to_csv(directory / RANDOM_LOG, index=False)
    pd.DataFrame(rows(0)).to_csv(directory / STANDARD_LOG, index=False)
    historical = pd.DataFrame(rows(0)).groupby("user_id", sort=False).head(1)
    historical.to_csv(directory / HISTORICAL_LOG, index=False)
    pd.DataFrame(videos).to_csv(directory / VIDEO_FEATURES, index=False)
    pd.DataFrame(users).to_csv(directory / USER_FEATURES, index=False)


def _pairwise_sandwich(X, Y, users, videos=None):
    """Independent reference using QR and observation-pair influence products.

    The implementation under test instead sums scores by cluster and uses the
    normal-equation pseudoinverse. Repeated user/video pairs here ensure the
    intersection correction is different from an observation-level correction.
    """
    design = np.column_stack([np.ones(len(Y)), X])
    q, r = np.linalg.qr(design, mode="reduced")
    coefficient = np.linalg.solve(r, q.T @ Y)
    residual = Y - design @ coefficient
    influence = np.linalg.solve(r, q.T).T * residual[:, None]
    same_user = users[:, None] == users[None, :]
    partitions = [(same_user, np.unique(users).size, 1)]
    if videos is not None:
        same_video = videos[:, None] == videos[None, :]
        pairs = np.column_stack([users, videos])
        partitions.extend([
            (same_video, np.unique(videos).size, 1),
            (same_user & same_video, np.unique(pairs, axis=0).shape[0], -1),
        ])
    n, p = design.shape
    weights = np.zeros((n, n))
    for same_cluster, n_clusters, sign in partitions:
        correction = n_clusters / (n_clusters - 1) * (n - 1) / (n - p)
        weights += sign * correction * same_cluster
    covariance = np.einsum("ij,ia,jb->ab", weights, influence, influence)
    return coefficient[1:], covariance[1:, 1:]


class KuaiRandExperimentTests(unittest.TestCase):
    def test_cohort_is_user_balanced_and_uses_only_pre_exposure_features(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary)
            _write_fixture(source)
            cfg = KuaiRandConfig(
                per_user_environment_cap=3,
                minimum_interactions_per_environment=2,
                max_users=None,
            )
            frame, metadata, _ = prepare_kuairand_cohort(source, cfg)
        counts = frame.groupby(["user_id", "environment"]).size().unstack()
        np.testing.assert_array_equal(counts[0], counts[1])
        self.assertTrue((counts[0] == 3).all())
        self.assertEqual(metadata["rows_per_environment"], {"0": 54, "1": 54})
        self.assertTrue(np.isfinite(frame.loc[:, list(X_COLUMNS)].to_numpy()).all())
        forbidden = ("play", "click", "like", "show", "watch")
        self.assertFalse(any(token in column for column in X_COLUMNS for token in forbidden))

    def test_heldout_reference_split_is_balanced_and_outcome_disjoint(self):
        frame = pd.DataFrame(
            [
                {"user_id": user, "environment": environment}
                for user in range(5)
                for environment in (0, 1)
                for _ in range(4)
            ]
        )
        train, reference, diagnostics = stratified_half_split(frame, seed=23)

        self.assertEqual(np.intersect1d(train, reference).size, 0)
        training = frame.iloc[train]
        counts = training.groupby(["user_id", "environment"]).size().unstack()
        np.testing.assert_array_equal(counts[0], counts[1])
        self.assertTrue((counts == 2).to_numpy().all())
        self.assertTrue(frame.iloc[reference]["environment"].eq(1).all())
        self.assertEqual(reference.size, 10)
        self.assertEqual(diagnostics["outcome_overlap"], 0)
        self.assertEqual(diagnostics["minimum_training_rows_per_user_policy"], 2)

    def test_randomized_projection_recovers_linear_slope_with_two_way_clustering(self):
        rng = np.random.default_rng(9)
        users = np.repeat(np.arange(500), 6)
        E = np.tile(np.array([0, 0, 0, 1, 1, 1]), 500)
        X = rng.normal(size=(users.size, 3))
        videos = np.tile(np.array([0, 1, 2, 0, 1, 2]), 500)
        beta0 = np.array([0.4, -0.2, 0.1])
        user_effect = 0.5 * rng.normal(size=500)[users]
        Y = X @ beta0 + user_effect + rng.normal(scale=0.01, size=users.size)
        beta, se = randomized_exposure_projection(X, Y, E, users, videos)
        np.testing.assert_allclose(beta, beta0, atol=0.05)
        self.assertTrue(np.all(se > 0))

    def test_two_way_standard_errors_match_pairwise_sandwich(self):
        rng = np.random.default_rng(142)
        users = np.repeat(np.array([10, 30, 80, 110, 250, 1000]), 12)
        videos = np.tile(np.repeat(np.array([3, 9, 18, 27]), 3), 6)
        X = rng.normal(size=(users.size, 2))
        user_effect = np.repeat(rng.normal(size=6), 12)
        video_effect = np.tile(np.repeat(rng.normal(size=4), 3), 6)
        Y = 3. + X @ np.array([.4, -.7]) + user_effect + video_effect
        Y += (.2 + np.abs(X[:, 0])) * rng.normal(size=users.size)
        expected_beta, expected_covariance = _pairwise_sandwich(X, Y, users, videos)
        self.assertTrue(np.all(np.diag(expected_covariance) > 0))

        # Unrelated standard-policy outcomes must not affect the reference fit
        # or any of its cluster counts and finite-sample corrections.
        extra_X = rng.normal(size=(9, 2))
        beta, se = randomized_exposure_projection(
            np.vstack([X, extra_X]), np.r_[Y, np.full(9, 1e6)],
            np.r_[np.ones(len(Y), dtype=int), np.zeros(9, dtype=int)],
            np.r_[users, np.arange(2000, 2009)],
            np.r_[videos, np.arange(3000, 3009)],
        )
        np.testing.assert_allclose(beta, expected_beta, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(se, np.sqrt(np.diag(expected_covariance)),
                                   rtol=1e-11, atol=1e-12)

    def test_identical_cluster_partitions_reduce_to_one_way_standard_errors(self):
        rng = np.random.default_rng(143)
        users = np.repeat(np.arange(8), 10)
        X = rng.normal(size=(users.size, 2))
        Y = 2. + X @ np.array([.3, -.2]) + np.repeat(rng.normal(size=8), 10)
        Y += rng.normal(scale=.3, size=users.size)
        expected_beta, covariance = _pairwise_sandwich(X, Y, users)
        # Different labels, same partition: user + video - intersection is one
        # cluster correction, not twice the one-way variance.
        beta, se = randomized_exposure_projection(
            X, Y, np.ones(len(Y), dtype=int), users, 100 + 7 * users,
        )
        np.testing.assert_allclose(beta, expected_beta, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(se, np.sqrt(np.diag(covariance)),
                                   rtol=1e-11, atol=1e-12)

    def test_ovr_estimators_keep_users_disjoint(self):
        rng = np.random.default_rng(17)
        users = np.repeat(np.arange(30), 8)
        E = np.tile(np.array([0, 0, 0, 0, 1, 1, 1, 1]), 30)
        X = rng.normal(size=(users.size, 2))
        H = rng.normal(size=30)[users]
        Y = X @ np.array([0.3, -0.2]) + E * H + rng.normal(scale=0.2, size=users.size)
        common = dict(
            contrast_fraction=0.5,
            n_folds=3,
            hidden=8,
            depth=1,
            epochs=3,
            resid_epochs=3,
            seed=3,
        )
        estimators = (
            DirectOVR(**common),
            SmoothedOVR(
                contrast_validation_fraction=0.0,
                contrast_patience=0,
                prediction_draws=2,
                **common,
            ),
        )
        for estimator in estimators:
            with self.subTest(estimator=type(estimator).__name__):
                estimator.fit(X, Y, E, groups=users)
                contrast_users = set(users[estimator.contrast_indices_])
                estimation_users = set(users[estimator.estimation_indices_])
                self.assertFalse(contrast_users & estimation_users)
                fold_users = [
                    set(users[fold]) for fold in estimator.second_stage_folds_
                ]
                for i in range(len(fold_users)):
                    for j in range(i):
                        self.assertFalse(fold_users[i] & fold_users[j])

        smoothed = estimators[1]
        self.assertFalse(
            set(users[smoothed.contrast_indices_a_])
            & set(users[smoothed.contrast_indices_b_])
        )


if __name__ == "__main__":
    unittest.main()
