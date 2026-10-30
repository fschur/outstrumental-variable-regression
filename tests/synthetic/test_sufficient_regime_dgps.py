"""Structural checks for the three exact sufficient-regime DGPs."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.synthetic.dgps import (
    LowRankLatentLoadingSCM,
    RecoverableStateSCM,
)


class SufficientRegimeDGPTests(unittest.TestCase):
    def test_low_rank_loading_oracles_and_rank_control(self) -> None:
        dgp = LowRankLatentLoadingSCM(
            latent_rank=2, block_dim=2, n_environments=4, seed=11
        )
        data = dgp.sample(512, seed=101)
        q = dgp.oracle_q(data.X)
        contrast = dgp.oracle_contrast(data.X)

        np.testing.assert_allclose(
            contrast,
            q[:, dgp.nonreference_environments] - q[:, [dgp.reference_environment]],
        )
        np.testing.assert_allclose(
            dgp.oracle_propensity(data.X), 1.0 / dgp.n_environments
        )
        self.assertEqual(np.linalg.matrix_rank(dgp.contrast_matrix_), 2)
        self.assertTrue(dgp.cs_holds)

        other_draw = dgp.sample(128, seed=102)
        np.testing.assert_allclose(data.beta0, other_draw.beta0)

        deficient = LowRankLatentLoadingSCM(
            latent_rank=3,
            block_dim=2,
            n_environments=3,
            contrast_rank=2,
            seed=12,
        )
        self.assertEqual(np.linalg.matrix_rank(deficient.contrast_matrix_), 2)
        self.assertFalse(deficient.cs_holds)

    def test_scalar_warmup_contrast_is_latent_posterior(self) -> None:
        dgp = LowRankLatentLoadingSCM.warmup(block_dim=3, seed=21)
        data = dgp.sample(512, seed=103)
        np.testing.assert_allclose(
            dgp.oracle_contrast(data.X)[:, 0],
            dgp.oracle_posterior(data.X)[:, 0],
        )

    def test_recoverable_state_overlap_and_positivity(self) -> None:
        dgp = RecoverableStateSCM(d=4, n_environments=4, tilt=0.5, seed=41)
        data = dgp.sample(1024, seed=106)

        np.testing.assert_allclose(dgp.oracle_state(data.X), dgp.oracle_state_)
        propensity = dgp.oracle_propensity(data.X)
        np.testing.assert_allclose(propensity.sum(axis=1), 1.0)
        self.assertGreaterEqual(
            float(propensity.min()) + 1e-12, dgp.overlap_lower_bound
        )
        self.assertGreater(dgp.positivity_eigenvalue, 0.0)
        self.assertTrue(dgp.contrast_sufficiency_holds)

        q = dgp.oracle_nuisance(data.X)
        np.testing.assert_allclose(
            dgp.oracle_contrast(data.X), q[:, 1:] - q[:, [0]]
        )
        other_draw = dgp.sample(128, seed=107)
        np.testing.assert_allclose(data.beta0, other_draw.beta0)


if __name__ == "__main__":
    unittest.main()
