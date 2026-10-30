"""Coefficient recovery with learned and exact nuisance functions."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiments.common.baselines import OLS
from experiments.synthetic.dgps import Dataset, LowRankLatentLoadingSCM, RecoverableStateSCM
from ovr import DirectOVR, SmoothedOVR


def _finite_state_data(n, seed, beta=(1.25, -.75)):
    """A noiseless confounded model with eight possible treatment vectors.

    H ~ Bernoulli(1/2), independent signs S_j with P(S_j=1)=3/4,
    X_j=(1+H)S_j, E uniform on {0,1,2}, and Y=beta^T X+3(E+1)H.
    The state H=|X_1|-1 is recoverable from every nonzero contrast.
    E[X|H,E]=(1+H)/2 in each coordinate and
    Cov(X|H,E)=3(1+H)^2 I/4, so residual positivity holds.
    """
    rng = np.random.default_rng(seed)
    state = rng.integers(0, 2, size=n)
    signs = np.where(rng.random((n, 2)) < .75, 1., -1.)
    X = (1 + state[:, None]) * signs
    E = rng.integers(0, 3, size=n)
    beta0 = np.asarray(beta, dtype=float)
    Y = np.einsum("nd,d->n", X, beta0) + 3 * (E + 1) * state
    return Dataset(X=X, Y=Y, T=E, beta0=beta0)


class _ExactContrast(torch.nn.Module):
    """Known contrast in the standardized units expected by the first stage."""

    def __init__(self, reference):
        super().__init__()
        self.loadings = torch.tensor([3. * (e - reference)
                                      for e in range(3) if e != reference])
        # Nontrivial, exactly representable scales exercise input/output units.
        self._xmu = torch.tensor([.25, -.5])
        self._xsd = torch.tensor([2., .5])
        self._ysd = 4.
        # Reporting fields read by SmoothedOVR after fitting its two members.
        self._best_validation_loss = 0.
        self._epochs_trained = self._best_epoch = 0
        self._early_stopped = False
        self._member_training_size = self._member_validation_size = 0

    def contrast(self, standardized_X):
        X = standardized_X * self._xsd + self._xmu
        state = X[:, 0].abs() - 1.
        return state[:, None] * self.loadings / self._ysd


class CoefficientRecoveryTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)

    def _check_recovery(self, dgp, minimum_ols_error):
        # Consecutive seeds vary the data, sample splits and network training.
        # These are regression checks, not a Monte Carlo consistency study.
        for data_seed, fit_seed in ((92, 19), (93, 20), (94, 21)):
            data = dgp.sample(12000, seed=data_seed)
            ols = OLS().fit(data.X, data.Y, data.T)
            ols_error = np.linalg.norm(ols.coef_ - data.beta0)
            with self.subTest(data_seed=data_seed):
                self.assertGreater(ols_error, minimum_ols_error)
            for estimator in (DirectOVR, SmoothedOVR):
                with self.subTest(estimator=estimator.__name__, data_seed=data_seed,
                                  fit_seed=fit_seed):
                    fit = estimator(hidden=32, epochs=200, resid_epochs=300,
                                    seed=fit_seed).fit(data.X, data.Y, data.T)
                    error = np.linalg.norm(fit.coef_ - data.beta0)
                    self.assertTrue(np.isfinite(fit.coef_).all())
                    # Allow finite-sample and optimization error, but reject a
                    # regression to the substantially biased unadjusted fit.
                    self.assertLess(error, .35)
                    self.assertLess(error, ols_error/2)

    def test_low_rank_recovery_across_seeds(self):
        # Full contrast rank, environment overlap and residual positivity hold.
        dgp = LowRankLatentLoadingSCM(
            latent_rank=1, block_dim=2, n_environments=3,
            loadings=np.array([[1.], [2.], [3.]]), confound_scale=2., seed=91,
        )
        self.assertTrue(dgp.cs_holds)
        self._check_recovery(dgp, minimum_ols_error=.7)

    def test_recoverable_state_recovery_across_seeds(self):
        # The contrast recovers radius; angular variation supplies positivity.
        # Stronger state loadings make the OLS bias detectable at this size.
        dgp = RecoverableStateSCM(
            d=2, n_environments=3, alpha_scale=3., tilt=.75, seed=91,
        )
        self.assertTrue(dgp.contrast_sufficiency_holds)
        self.assertGreater(dgp.overlap_lower_bound, 0.)
        self.assertGreater(dgp.positivity_eigenvalue, 0.)
        self._check_recovery(dgp, minimum_ols_error=.25)

    def test_finite_state_recovery_with_learned_nuisances(self):
        # Actual neural learners, including default smoothing and early stopping.
        # A finite support and no outcome noise permit a tighter accuracy check.
        # Lighter head regularization targets accuracy in this special case;
        # the noisy examples above still exercise each estimator's defaults.
        for seed in (101, 102, 103):
            data = _finite_state_data(6000, seed)
            ols = OLS().fit(data.X, data.Y, data.T)
            self.assertGreater(np.linalg.norm(ols.coef_ - data.beta0), .4)
            for estimator in (DirectOVR, SmoothedOVR):
                with self.subTest(estimator=estimator.__name__, seed=seed):
                    fit = estimator(hidden=16, depth=1, epochs=300,
                                    resid_epochs=400, contrast_lambda=1e-4,
                                    seed=seed).fit(data.X, data.Y, data.T)
                    self.assertTrue(np.isfinite(fit.coef_).all())
                    self.assertLess(np.linalg.norm(fit.coef_ - data.beta0), .05)

    def test_exact_nuisances_recover_coefficients_to_numerical_precision(self):
        # Replace only the nuisance learners. Production standardization,
        # contrast assembly, cross-fitting, residualization and solving still run.
        # With no outcome noise, Y-m_Y = beta^T(X-m_X) holds observation by
        # observation, so no sampling tolerance is needed. Allow the solver's
        # 1e-10 numerical ridge. This is not a test of neural-network accuracy.
        for beta in ((1.25, -.75), (-.5, 2.)):
            data = _finite_state_data(600, seed=107, beta=beta)
            for reference in (0, 2):
                nonreference = [e for e in range(3) if e != reference]
                loadings = 3. * (np.asarray(nonreference) - reference)

                def predict(features):
                    state = features[:, 0] / loadings[0]
                    mean_X = np.repeat(((1. + state) / 2.)[:, None], 2, axis=1)
                    nuisance = state * np.einsum("ne,e->n", features[:, 2:],
                                                3. * np.arange(1, 4))
                    mean_Y = np.einsum("nd,d->n", mean_X, data.beta0) + nuisance
                    return np.column_stack([mean_X, mean_Y])

                for estimator, first_stage, second_stage in (
                    (DirectOVR, "_fit_joint", "ovr.direct._train_regressor"),
                    (SmoothedOVR, "_fit_joint_member",
                     "ovr.smoothed._train_uncertainty_smoothed_regressor"),
                ):
                    with self.subTest(estimator=estimator.__name__,
                                      reference=reference, beta=beta):
                        fit = estimator(reference=reference, seed=108)
                        with patch.object(fit, first_stage,
                                          return_value=_ExactContrast(reference)), \
                                patch(second_stage, return_value=predict):
                            fit.fit(data.X, data.Y, data.T)
                        np.testing.assert_allclose(fit.coef_, data.beta0,
                                                   rtol=0., atol=1e-8)
                        state = np.abs(data.X[fit.estimation_indices_, 0]) - 1.
                        np.testing.assert_allclose(fit.estimated_contrast_,
                                                   state[:, None] * loadings,
                                                   rtol=0., atol=1e-12)


if __name__ == '__main__':
    unittest.main()
