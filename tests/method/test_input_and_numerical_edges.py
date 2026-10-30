"""Public input contracts and degenerate or nonfinite numerical calculations."""
import unittest
from unittest.mock import patch

import numpy as np

from ovr import DirectOVR, SmoothedOVR
from ovr._core import _OVRCore, _disagreement_covariance, _psd_square_root


class InputAndNumericalEdgeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(61)
        self.X = rng.normal(size=(120, 2))
        self.Y = self.X[:, 0] + rng.normal(size=120)
        self.E = np.tile(np.arange(3), 40)

    @staticmethod
    def _estimator(cls, **kwargs):
        return cls(hidden=4, epochs=1, resid_epochs=1, seed=17, **kwargs)

    def test_nonfinite_observations_are_rejected_before_training(self):
        for cls in (DirectOVR, SmoothedOVR):
            for variable in ("X", "Y"):
                for value in (np.nan, np.inf, -np.inf):
                    with self.subTest(estimator=cls.__name__, variable=variable,
                                      value=value):
                        X, Y = self.X.copy(), self.Y.copy()
                        if variable == "X":
                            X[0, 0] = value
                        else:
                            Y[0] = value
                        estimator = self._estimator(cls)
                        with patch.object(estimator, "_prepare") as prepare:
                            with self.assertRaisesRegex(ValueError, "finite"):
                                estimator.fit(X, Y, self.E)
                        prepare.assert_not_called()

    def test_wrong_dimensions_and_mismatched_lengths_are_rejected(self):
        cases = (
            (self.X[:, 0], self.Y, self.E, "two-dimensional"),
            (self.X, self.Y[:-1], self.E, "same number of rows"),
            (self.X, self.Y, self.E[:-1], "same number of rows"),
        )
        for cls in (DirectOVR, SmoothedOVR):
            for X, Y, E, message in cases:
                with self.subTest(estimator=cls.__name__, message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        self._estimator(cls).fit(X, Y, E)

    def test_invalid_environment_labels_are_rejected(self):
        cases = (
            (self.E + .25, "integers"),
            (self.E * 2, "consecutive integers"),
            (self.E - 1, "consecutive integers"),
            (np.zeros_like(self.E), "at least two environments"),
        )
        for cls in (DirectOVR, SmoothedOVR):
            for E, message in cases:
                with self.subTest(estimator=cls.__name__, message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        self._estimator(cls).fit(self.X, self.Y, E)

    def test_insufficient_environment_samples_are_rejected(self):
        # One observation cannot be split; four leave only two estimation rows,
        # insufficient for the default three residualization folds.
        for cls in (DirectOVR, SmoothedOVR):
            for counts, message in (((1, 12), "at least two observations"),
                                    ((4, 4), "n_folds=3")):
                E = np.repeat(np.arange(2), counts)
                with self.subTest(estimator=cls.__name__, counts=counts):
                    with self.assertRaisesRegex(ValueError, message):
                        self._estimator(cls).fit(self.X[:len(E)], self.Y[:len(E)], E)

    def test_invalid_groups_and_impossible_group_split_are_rejected(self):
        missing = np.arange(len(self.E), dtype=float)
        missing[0] = np.nan
        cases = (
            (self.E[:-1], "one label for every observation"),
            (np.zeros_like(self.E), "at least two groups"),
            (missing, "finite labels"),
            # Each environment confined to one group: no valid disjoint split.
            (self.E, "group-disjoint split"),
        )
        for cls in (DirectOVR, SmoothedOVR):
            for groups, message in cases:
                with self.subTest(estimator=cls.__name__, message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        self._estimator(cls).fit(self.X, self.Y, self.E, groups=groups)

    def test_invalid_split_and_smoothing_parameters_are_rejected(self):
        for cls in (DirectOVR, SmoothedOVR):
            for settings in ({"contrast_fraction": 0.}, {"contrast_fraction": 1.},
                             {"n_folds": 1}, {"reference": 3}):
                with self.subTest(estimator=cls.__name__, settings=settings):
                    with self.assertRaises(ValueError):
                        self._estimator(cls, **settings).fit(self.X, self.Y, self.E)
        for settings in ({"jitter_scale": -1.}, {"prediction_draws": 0},
                         {"contrast_validation_fraction": 1.}):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    self._estimator(SmoothedOVR, **settings).fit(self.X, self.Y, self.E)

    def test_contrast_prediction_requires_fit_and_matching_feature_count(self):
        for cls in (DirectOVR, SmoothedOVR):
            with self.subTest(estimator=cls.__name__):
                estimator = self._estimator(cls)
                with self.assertRaisesRegex(RuntimeError, "fit"):
                    estimator.predict_contrast(self.X)
                estimator.fit(self.X, self.Y, self.E)
                with self.assertRaisesRegex(ValueError, "shape"):
                    estimator.predict_contrast(self.X[:, :1])

    def test_zero_residual_variation_does_not_report_positive_diagnostics(self):
        beta, covariance = _OVRCore._solve_beta(np.zeros((8, 2)), np.arange(8.))
        np.testing.assert_array_equal(beta, np.zeros(2))
        np.testing.assert_array_equal(covariance, np.zeros((2, 2)))
        self.assertEqual(_OVRCore._gamma(covariance, np.eye(2)), 0.)
        self.assertEqual(_OVRCore._minimum_eigenvalue(covariance), 0.)
        self.assertEqual(_OVRCore._cond(covariance), np.inf)

    def test_collinear_residuals_match_the_ridge_solution(self):
        x = np.tile(np.array([-2., -1., 1., 2.]), 3)
        residual_X = np.column_stack([x, x])
        residual_Y = 3. * x
        beta, covariance = _OVRCore._solve_beta(residual_X, residual_Y)
        # Only the sum of the two coefficients is identified. The existing
        # 1e-10 ridge splits it equally; neither coordinate has a causal target.
        second_moment = np.mean(x**2)
        expected = 3. * second_moment / (2. * second_moment + 1e-10)
        np.testing.assert_allclose(beta, [expected, expected], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(residual_X @ beta, residual_Y, atol=1e-9)
        self.assertEqual(_OVRCore._cond(covariance), np.inf)

    def test_zero_and_rank_deficient_disagreement_covariances_are_supported(self):
        constant = np.tile(np.array([2., -3.]), (6, 1))
        np.testing.assert_array_equal(_disagreement_covariance(constant),
                                      np.zeros((2, 2)))
        np.testing.assert_array_equal(_psd_square_root(np.zeros((2, 2))),
                                      np.zeros((2, 2)))
        # Columns are perfectly dependent. No inverse or positive eigenvalue
        # floor should invent noise in the unobserved direction.
        z = np.arange(6.)
        covariance = _disagreement_covariance(np.column_stack([z, 2. * z]))
        factor = _psd_square_root(covariance)
        self.assertTrue(np.isfinite(factor).all())
        np.testing.assert_allclose(factor @ factor.T, covariance, atol=1e-12)
        # Eigensolver roundoff is square-rooted when constructing the factor.
        np.testing.assert_allclose(factor @ np.array([-2., 1.]), 0., atol=1e-7)

    def test_nonfinite_nuisance_predictions_are_rejected(self):
        for cls, target in (
            (DirectOVR, "ovr.direct._train_regressor"),
            (SmoothedOVR, "ovr.smoothed._train_uncertainty_smoothed_regressor"),
        ):
            with self.subTest(estimator=cls.__name__):
                predict = lambda features: np.full((len(features), 3), np.nan)
                with patch(target, return_value=predict):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        self._estimator(cls).fit(self.X, self.Y, self.E)

    def test_genuine_moment_overflow_is_rejected(self):
        residual_X = np.full((8, 2), 1e200)
        residual_Y = np.ones(8)
        self.assertTrue(np.isfinite(residual_X).all())
        # These finite inputs really overflow when squared. Suppress warnings
        # only for this intentional failure; the estimator must reject it by
        # inspecting values, independently of the backend's warning behavior.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            with self.assertRaisesRegex(ValueError, "finite"):
                _OVRCore._solve_beta(residual_X, residual_Y)

    def test_genuine_coefficient_overflow_is_rejected(self):
        # Both moments are finite, but the regularized solution is
        # 1e303 / (1e-10 + 1e-10) = 5e312, outside float64's range.
        residual_X = np.full((8, 1), 1e-5)
        residual_Y = np.full(8, 1e308)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            with self.assertRaisesRegex(ValueError, "coefficients must be finite"):
                _OVRCore._solve_beta(residual_X, residual_Y)


if __name__ == "__main__":
    unittest.main()
