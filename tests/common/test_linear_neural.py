"""Learning, data separation, model selection, and original-unit predictions."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.common.baselines import LinearNeuralRegression
from experiments.common.linear_neural_benchmark import compare_linear_neural


class LinearNeuralTests(unittest.TestCase):
    def data(self, n=360):
        rng = np.random.default_rng(19)
        X = rng.normal(size=(n, 2))
        E = np.arange(n) % 3
        Y = 1.5*X[:, 0]+2*X[:, 1]**2+.7*E+rng.normal(scale=.1, size=n)
        return X, Y, E

    def test_learns_nonlinear_outcome_and_predicts_in_original_units(self):
        X, Y, E = self.data(600)
        train, test = np.arange(450), np.arange(450, 600)
        fit = LinearNeuralRegression(hidden=24, epochs=140, lr=.01, batch_size=256,
                                     weight_decays=(0.,), initializations=("fe",), seed=3).fit(X[train], Y[train], E[train])
        pred = fit.predict(X[test], E[test])
        mse = np.mean((pred-Y[test])**2)
        self.assertLess(mse, 1.)
        self.assertLess(mse, .2*np.mean((Y[test]-Y[train].mean())**2))
        xmu, xsd, ymu, ysd = fit.scaling_
        with torch.no_grad():
            xs = torch.tensor((X[test]-xmu)/xsd, dtype=torch.float32)
            es = torch.tensor(E[test, None] == fit.environments_[None, :], dtype=torch.float32)
            f = fit.model_.nuisance(torch.cat([xs, es], dim=1)).numpy().ravel()*ysd+ymu
        np.testing.assert_allclose(pred, (X[test]-xmu) @ fit.coef_+f, rtol=1e-5, atol=2e-6)

    def test_group_splits_and_refit_selection_preservation(self):
        X, Y, E = self.data()
        groups = np.repeat(np.arange(120), 3)
        fit = LinearNeuralRegression(hidden=8, epochs=4, weight_decays=(0., .01), seed=7).fit(X, Y, E, groups)
        self.assertEqual(len(np.intersect1d(groups[fit.training_indices_], groups[fit.validation_indices_])), 0)
        best = min(fit.validation_results_, key=lambda r: r["validation_mse"])
        self.assertEqual(fit.selected_, best)
        before = fit.predict(X, E)
        torch_state = torch.get_rng_state().clone()
        fit.refit(X, Y, E)
        np.testing.assert_array_equal(before, fit.predict(X, E))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        with self.assertRaisesRegex(ValueError, "unobserved environment"):
            fit.predict(X[:2], np.full(2, 99))

    def test_heldout_rows_and_groups_never_enter_selection(self):
        X, Y, E = self.data()
        groups = np.repeat(np.arange(120), 3)
        rows = compare_linear_neural(X, Y, E, np.ones(2), groups=groups, epochs=4,
                                      hidden=8, weight_decays=(0.,), seed=3)
        nn = rows[-1]
        diag = json.loads(nn["diagnostics"])
        test = np.asarray(diag["outer_test_indices"])
        for name in ("outer_training_indices", "selection_training_indices", "selection_validation_indices"):
            indices = np.asarray(diag[name])
            self.assertEqual(len(np.intersect1d(test, indices)), 0)
            self.assertEqual(len(np.intersect1d(groups[test], groups[indices])), 0)
        self.assertEqual(nn["n_final_training"], len(X))
        self.assertLess(nn["n_selection_rows"], len(X))
        self.assertEqual(len({r["prediction_split_hash"] for r in rows}), 1)
        self.assertEqual(len({r["data_hash"] for r in rows}), 1)

    def test_changing_scoring_target_cannot_change_fitted_coefficients_or_predictions(self):
        X, Y, E = self.data(120)
        opts = dict(epochs=3, hidden=8, weight_decays=(0.,), initializations=("fe",), seed=3)
        a = compare_linear_neural(X, Y, E, np.zeros(2), **opts)
        b = compare_linear_neural(X, Y, E, np.array([99., -99.]), **opts)
        for x, y in zip(a, b):
            self.assertEqual(x["beta_hat"], y["beta_hat"])
            self.assertEqual(x["prediction_mse"], y["prediction_mse"])
            self.assertNotEqual(x["coefficient_error"], y["coefficient_error"])

    def test_scaling_for_candidate_selection_uses_training_rows_only(self):
        X, Y, E = self.data(120)
        fit = LinearNeuralRegression(hidden=8, epochs=3, weight_decays=(0.,), initializations=("fe",), seed=1)
        original = fit._train
        calls = []
        def record(x, y, e, **kwargs):
            result = original(x, y, e, **kwargs)
            calls.append((x.copy(), result[1], kwargs.get("validation")))
            return result
        with patch.object(fit, "_train", side_effect=record):
            fit.fit(X, Y, E)
        self.assertEqual(len(calls), 2)
        training, scaling, validation = calls[0]
        np.testing.assert_allclose(scaling[0], training.mean(0))
        self.assertIsNotNone(validation)
        self.assertEqual(len(training)+len(validation[0]), len(X))


if __name__ == "__main__":
    unittest.main()
