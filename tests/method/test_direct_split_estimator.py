"""Focused tests for the Algorithm-1 direct OVR estimator."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.synthetic.dgps import LowRankLatentLoadingSCM
from ovr import DirectOVR


class _CountingDirectSplit(DirectOVR):
    """Test helper recording exactly which observations reach the first stage."""

    def _fit_joint(self, X, Y, T, lam, seed):  # type: ignore[override]
        self.first_stage_calls_ = getattr(self, "first_stage_calls_", 0) + 1
        self.first_stage_X_ = np.asarray(X).copy()
        self.first_stage_Y_ = np.asarray(Y).copy()
        self.first_stage_T_ = np.asarray(T).copy()
        return super()._fit_joint(X, Y, T, lam, seed)


def _fast_estimator(seed: int = 7) -> _CountingDirectSplit:
    return _CountingDirectSplit(
        contrast_fraction=0.5,
        contrast_lambda=1e-4,
        n_folds=3,
        hidden=12,
        depth=1,
        epochs=4,
        batch_size=256,
        resid_epochs=5,
        seed=seed,
    )


class DirectSplitEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dgp = LowRankLatentLoadingSCM(
            latent_rank=2,
            block_dim=2,
            n_environments=4,
            seed=91,
        )
        cls.data = cls.dgp.sample(192, seed=92)

    def test_splits_are_disjoint_stratified_and_use_full_contrast(self) -> None:
        est = _fast_estimator().fit(self.data.X, self.data.Y, self.data.T)
        contrast_idx = est.contrast_indices_
        estimation_idx = est.estimation_indices_
        assert contrast_idx is not None and estimation_idx is not None

        self.assertEqual(est.first_stage_calls_, 1)
        np.testing.assert_array_equal(est.first_stage_X_, self.data.X[contrast_idx])
        np.testing.assert_array_equal(est.first_stage_Y_, self.data.Y[contrast_idx])
        np.testing.assert_array_equal(est.first_stage_T_, self.data.T[contrast_idx])
        self.assertEqual(np.intersect1d(contrast_idx, estimation_idx).size, 0)
        np.testing.assert_array_equal(
            np.sort(np.concatenate([contrast_idx, estimation_idx])),
            np.arange(self.data.X.shape[0]),
        )

        environments = np.unique(self.data.T)
        np.testing.assert_array_equal(np.unique(self.data.T[contrast_idx]), environments)
        np.testing.assert_array_equal(np.unique(self.data.T[estimation_idx]), environments)
        assert est.second_stage_folds_ is not None
        np.testing.assert_array_equal(
            np.sort(np.concatenate(est.second_stage_folds_)), estimation_idx
        )
        for fold in est.second_stage_folds_:
            np.testing.assert_array_equal(np.unique(self.data.T[fold]), environments)

        self.assertEqual(est.contrast_dim_, environments.size - 1)
        assert est.estimated_contrast_ is not None
        self.assertEqual(
            est.estimated_contrast_.shape,
            (estimation_idx.size, environments.size - 1),
        )

    def test_estimation_outcomes_do_not_enter_contrast_fit(self) -> None:
        est_a = _fast_estimator(seed=13).fit(
            self.data.X, self.data.Y, self.data.T
        )
        assert est_a.estimation_indices_ is not None
        changed_Y = self.data.Y.copy()
        changed_Y[est_a.estimation_indices_] += np.linspace(
            -100.0, 100.0, est_a.estimation_indices_.size
        )

        est_b = _fast_estimator(seed=13).fit(self.data.X, changed_Y, self.data.T)
        np.testing.assert_array_equal(
            est_a.contrast_indices_, est_b.contrast_indices_
        )
        np.testing.assert_allclose(
            est_a.predict_contrast(self.data.X),
            est_b.predict_contrast(self.data.X),
            rtol=0.0,
            atol=0.0,
        )

    def test_exact_regime_smoke_fit_is_finite(self) -> None:
        est = _fast_estimator(seed=19).fit(
            self.data.X, self.data.Y, self.data.T
        )
        self.assertEqual(est.coef_.shape, self.data.beta0.shape)
        self.assertTrue(np.isfinite(est.coef_).all())
        self.assertTrue(np.isfinite(est.gamma_))
        self.assertTrue(np.isfinite(est.cond_))
        assert est.Sigma_ is not None
        expected_sigma = est.residualized_X_.T @ est.residualized_X_ / est.n_estimation_
        np.testing.assert_allclose(est.Sigma_, expected_sigma)
        self.assertAlmostEqual(
            est.minimum_eigenvalue_,
            float(np.linalg.eigvalsh(0.5 * (est.Sigma_ + est.Sigma_.T)).min()),
        )
        assert est.residualized_X_ is not None and est.residualized_Y_ is not None
        self.assertEqual(est.residualized_X_.shape[0], est.n_estimation_)
        self.assertEqual(est.residualized_Y_.shape[0], est.n_estimation_)


if __name__ == "__main__":
    unittest.main()
