"""Focused tests for uncertainty-smoothed split OVR."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.synthetic.dgps import LowRankLatentLoadingSCM
from ovr import (
    DirectOVR,
    SmoothedOVR,
)


class _CountingSmoothedSplit(SmoothedOVR):
    """Record the complete member samples passed to both first-stage fits."""

    def _fit_joint_member(self, X, Y, T, seed):  # type: ignore[override]
        calls = getattr(self, "first_stage_inputs_", [])
        calls.append(
            (
                np.asarray(X).copy(),
                np.asarray(Y).copy(),
                np.asarray(T).copy(),
                int(seed),
            )
        )
        self.first_stage_inputs_ = calls
        return super()._fit_joint_member(X, Y, T, seed)


def _fast_estimator(seed: int = 7, **kwargs) -> _CountingSmoothedSplit:
    defaults = dict(
        contrast_fraction=0.5,
        contrast_lambda=1e-2,
        jitter_scale=1.25,
        prediction_draws=3,
        contrast_validation_fraction=0.2,
        contrast_patience=2,
        n_folds=3,
        hidden=10,
        depth=1,
        epochs=4,
        batch_size=256,
        resid_epochs=5,
        seed=seed,
    )
    defaults.update(kwargs)
    return _CountingSmoothedSplit(**defaults)


def _covariance_of_ensemble_error(disagreement: np.ndarray) -> np.ndarray:
    disagreement = np.asarray(disagreement, dtype=float)
    centered = disagreement - disagreement.mean(axis=0, keepdims=True)
    return centered.T @ centered / (4.0 * (disagreement.shape[0] - 1))


class UncertaintySmoothedEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dgp = LowRankLatentLoadingSCM(
            latent_rank=2,
            block_dim=2,
            n_environments=4,
            seed=121,
        )
        cls.data = cls.dgp.sample(256, seed=122)

    def test_three_disjoint_samples_and_exact_ensemble(self) -> None:
        est = _fast_estimator().fit(self.data.X, self.data.Y, self.data.T)
        idx_a = est.contrast_indices_a_
        idx_b = est.contrast_indices_b_
        idx_beta = est.estimation_indices_
        assert idx_a is not None and idx_b is not None and idx_beta is not None

        self.assertEqual(est.first_stage_fit_count_, 2)
        self.assertEqual(len(est.first_stage_inputs_), 2)
        self.assertEqual(np.intersect1d(idx_a, idx_b).size, 0)
        self.assertEqual(np.intersect1d(idx_a, idx_beta).size, 0)
        self.assertEqual(np.intersect1d(idx_b, idx_beta).size, 0)
        np.testing.assert_array_equal(
            np.sort(np.concatenate([idx_a, idx_b, idx_beta])),
            np.arange(self.data.X.shape[0]),
        )
        np.testing.assert_array_equal(est.contrast_indices_, np.sort(np.r_[idx_a, idx_b]))

        environments = np.unique(self.data.T)
        for idx in (idx_a, idx_b, idx_beta):
            np.testing.assert_array_equal(np.unique(self.data.T[idx]), environments)
        np.testing.assert_array_equal(est.first_stage_inputs_[0][0], self.data.X[idx_a])
        np.testing.assert_array_equal(est.first_stage_inputs_[1][0], self.data.X[idx_b])

        component_a, component_b = est.predict_contrast_components(
            self.data.X[idx_beta]
        )
        assert est.estimated_contrast_components_ is not None
        np.testing.assert_allclose(
            component_a, est.estimated_contrast_components_[0], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            component_b, est.estimated_contrast_components_[1], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            est.estimated_contrast_, 0.5 * (component_a + component_b)
        )
        np.testing.assert_allclose(
            est.predict_contrast(self.data.X[idx_beta]), est.estimated_contrast_
        )

    def test_disagreement_covariance_and_fold_local_jitter(self) -> None:
        est = _fast_estimator(seed=11).fit(
            self.data.X, self.data.Y, self.data.T
        )
        disagreement = est.contrast_disagreement_
        covariance = est.contrast_disagreement_covariance_
        assert disagreement is not None and covariance is not None
        np.testing.assert_allclose(
            covariance,
            _covariance_of_ensemble_error(disagreement),
            rtol=1e-7,
            atol=1e-10,
        )
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) >= -1e-12))
        self.assertGreater(float(np.trace(covariance)), 0.0)

        assert est.fold_disagreement_covariances_ is not None
        assert est.jitter_covariances_ is not None
        assert est.second_stage_folds_ is not None
        assert est.estimation_indices_ is not None
        for fold, fold_covariance, jitter_covariance in zip(
            est.second_stage_folds_,
            est.fold_disagreement_covariances_,
            est.jitter_covariances_,
        ):
            is_test = np.isin(est.estimation_indices_, fold)
            expected = _covariance_of_ensemble_error(disagreement[~is_test])
            np.testing.assert_allclose(
                fold_covariance, expected, rtol=1e-6, atol=1e-9
            )
            np.testing.assert_allclose(
                jitter_covariance,
                est.jitter_scale**2 * expected,
                rtol=1e-6,
                atol=1e-9,
            )

    def test_estimation_outcomes_cannot_change_first_stage(self) -> None:
        est_a = _fast_estimator(seed=17).fit(
            self.data.X, self.data.Y, self.data.T
        )
        assert est_a.estimation_indices_ is not None
        changed_y = self.data.Y.copy()
        changed_y[est_a.estimation_indices_] += np.linspace(
            -200.0, 200.0, est_a.estimation_indices_.size
        )
        est_b = _fast_estimator(seed=17).fit(
            self.data.X, changed_y, self.data.T
        )
        np.testing.assert_array_equal(
            est_a.estimation_indices_, est_b.estimation_indices_
        )
        components_a = est_a.predict_contrast_components(self.data.X)
        components_b = est_b.predict_contrast_components(self.data.X)
        for first, second in zip(components_a, components_b):
            np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)

    def test_fit_is_deterministic_and_diagnostics_are_finite(self) -> None:
        est_a = _fast_estimator(seed=23).fit(
            self.data.X, self.data.Y, self.data.T
        )
        est_b = _fast_estimator(seed=23).fit(
            self.data.X, self.data.Y, self.data.T
        )
        np.testing.assert_allclose(est_a.coef_, est_b.coef_, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            est_a.estimated_contrast_,
            est_b.estimated_contrast_,
            rtol=0.0,
            atol=0.0,
        )
        self.assertTrue(np.isfinite(est_a.coef_).all())
        self.assertTrue(np.isfinite(est_a.gamma_))
        self.assertTrue(np.isfinite(est_a.cond_))
        assert est_a.Sigma_ is not None and est_a.residualized_X_ is not None
        expected_sigma = (
            est_a.residualized_X_.T @ est_a.residualized_X_ / est_a.n_estimation_
        )
        np.testing.assert_allclose(est_a.Sigma_, expected_sigma)
        self.assertAlmostEqual(
            est_a.minimum_eigenvalue_,
            float(np.linalg.eigvalsh(0.5 * (est_a.Sigma_ + est_a.Sigma_.T)).min()),
        )
        self.assertTrue(np.isfinite(est_a.contrast_validation_losses_).all())
        self.assertTrue(
            all(1 <= epoch <= est_a.epochs for epoch in est_a.contrast_epochs_trained_)
        )
        self.assertEqual(est_a.contrast_dim_, np.unique(self.data.T).size - 1)

    def test_direct_and_smoothed_use_the_same_slope_split(self) -> None:
        direct = DirectOVR(
            contrast_fraction=0.5,
            n_folds=3,
            hidden=10,
            depth=1,
            epochs=4,
            batch_size=256,
            resid_epochs=5,
            seed=31,
        ).fit(self.data.X, self.data.Y, self.data.T)
        smoothed = _fast_estimator(seed=31).fit(
            self.data.X, self.data.Y, self.data.T
        )
        np.testing.assert_array_equal(
            direct.estimation_indices_, smoothed.estimation_indices_
        )
        np.testing.assert_allclose(direct.Sigma_T_, smoothed.Sigma_T_)

    def test_zero_jitter_is_supported(self) -> None:
        est = _fast_estimator(
            seed=29,
            jitter_scale=0.0,
            prediction_draws=1,
            contrast_validation_fraction=0.0,
            contrast_patience=0,
        ).fit(self.data.X, self.data.Y, self.data.T)
        assert est.jitter_covariances_ is not None
        for covariance in est.jitter_covariances_:
            np.testing.assert_array_equal(covariance, np.zeros_like(covariance))
        self.assertTrue(np.isfinite(est.coef_).all())
        self.assertTrue(np.isnan(est.contrast_validation_losses_).all())


if __name__ == "__main__":
    unittest.main()
