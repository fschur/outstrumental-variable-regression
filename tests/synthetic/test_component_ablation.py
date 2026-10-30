from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from experiments.synthetic.ablation_results import paired_step_effects
from experiments.synthetic.run_ablation import (
    VARIANTS,
    _sample_scenario,
    _scenario_catalog,
    fit_ablation_variants,
)
from ovr import SmoothedOVR


class ComponentAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        scenario = _scenario_catalog()[0]
        replication = _sample_scenario(
            scenario,
            rep=0,
            n=240,
            main_seed=20260807,
            shared_seed=20260808,
        )
        cls.replication = replication
        cls.rows = fit_ablation_variants(
            scenario=scenario,
            rep=0,
            n=240,
            replication=replication,
            n_folds=3,
            hidden=10,
            contrast_epochs=3,
            residual_epochs=4,
            weak_contrast_lambda=1e-4,
            strong_contrast_lambda=1e-2,
            jitter_scale=1.0,
            prediction_draws=3,
            validation_fraction=0.15,
            patience=2,
        )

    def test_catalog_matches_the_six_reported_designs(self) -> None:
        catalog = _scenario_catalog()
        self.assertEqual(
            tuple(scenario.key for scenario in catalog),
            (
                "low_rank",
                "nonlinear_clean_environment",
                "recoverable_state",
                "shared_decoder_no_ex",
                "shared_decoder_additive_ex",
                "shared_decoder_latent_outcome_exposure_shifts",
            ),
        )
        replacement = catalog[-1]
        self.assertEqual(replacement.label, "Latent, outcome and treatment shifts")
        self.assertEqual(replacement.exposure_shift_scale, 0.3)
        self.assertEqual(replacement.latent_shift_scale, 0.3)

    def test_all_variants_share_the_split_plan_and_folds(self) -> None:
        self.assertEqual(len(self.rows), len(VARIANTS))
        self.assertEqual(len({row["split_plan_hash"] for row in self.rows}), 1)
        self.assertEqual(len({row["fold_hash"] for row in self.rows}), 1)
        self.assertEqual(
            {row["variant_key"] for row in self.rows},
            {variant.key for variant in VARIANTS},
        )

    def test_tau_zero_and_one_reuse_the_exact_ensemble(self) -> None:
        by_key = {row["variant_key"]: row for row in self.rows}
        ensemble = by_key["ensemble"]
        smoothed = by_key["smoothed"]
        self.assertEqual(
            ensemble["contrast_artifact_hash"], smoothed["contrast_artifact_hash"]
        )
        self.assertEqual(ensemble["first_stage_seconds"], smoothed["first_stage_seconds"])
        self.assertEqual(ensemble["contrast_rmse"], smoothed["contrast_rmse"])
        self.assertEqual(ensemble["uncertainty_trace"], smoothed["uncertainty_trace"])
        self.assertEqual(ensemble["jitter_trace_mean"], 0.0)
        self.assertGreater(smoothed["jitter_trace_mean"], 0.0)

    def test_recorded_moment_decomposition_is_exact(self) -> None:
        for row in self.rows:
            self.assertTrue(row["success"])
            self.assertLess(row["moment_identity_error"], 1e-10)
            self.assertAlmostEqual(
                row["coefficient_error"],
                row["ridge_adjusted_moment_norm"]
                * row["effective_amplification"],
                places=12,
            )
            self.assertGreaterEqual(row["worst_case_bound_slack"], 1.0 - 1e-8)

    def test_smoothed_endpoint_matches_the_public_estimator(self) -> None:
        estimator = SmoothedOVR(
            contrast_fraction=0.5,
            contrast_lambda=1e-2,
            jitter_scale=1.0,
            prediction_draws=3,
            contrast_validation_fraction=0.15,
            contrast_patience=2,
            n_folds=3,
            hidden=10,
            depth=2,
            epochs=3,
            resid_epochs=4,
            seed=self.replication.estimator_seed,
        ).fit(
            self.replication.data.X,
            self.replication.data.Y,
            self.replication.data.T,
        )
        row = next(row for row in self.rows if row["variant_key"] == "smoothed")
        beta = np.asarray(json.loads(row["beta_hat"]), dtype=float)
        np.testing.assert_allclose(beta, estimator.coef_, rtol=0.0, atol=1e-12)
        self.assertAlmostEqual(row["gamma"], estimator.gamma_, places=12)
        self.assertAlmostEqual(
            row["minimum_residual_eigenvalue"],
            estimator.minimum_eigenvalue_,
            places=12,
        )

    def test_paired_step_summary_preserves_the_log_decomposition(self) -> None:
        frame = pd.DataFrame(self.rows)
        second = frame.copy()
        second["rep"] = 1
        second["coefficient_error"] *= 1.05
        second["ridge_adjusted_moment_norm"] *= 1.05
        combined = pd.concat([frame, second], ignore_index=True).sample(
            frac=1.0, random_state=17
        )
        effects = paired_step_effects(combined)
        self.assertEqual(len(effects), len(VARIANTS) - 1)
        np.testing.assert_allclose(
            effects["log_error_ratio_mean"],
            effects["log_moment_ratio_mean"]
            + effects["log_amplification_ratio_mean"],
            rtol=1e-10,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
