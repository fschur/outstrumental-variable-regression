import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from experiments.synthetic.run_tau_sweep import fit_tau_sweep
from experiments.synthetic.sampling import _sample_scenario, _scenario_catalog
from ovr import SmoothedOVR


class TauSweepTests(unittest.TestCase):
    def test_cli_records_sources_from_another_working_directory(self):
        root = Path(__file__).resolve().parents[2]
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root/'src'), str(root)]))
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([
                sys.executable, '-m', 'experiments.synthetic.run_tau_sweep',
                '--sample-sizes', '120', '--reps', '1', '--workers', '1',
                '--contrast-epochs', '1', '--residual-epochs', '1', '--hidden', '4',
                '--taus', '0,1', '--scenarios', 'low_rank', '--output-dir', 'results',
            ], cwd=directory, env=env, check=True, capture_output=True, text=True)
            config = json.loads((Path(directory)/'results/config.json').read_text())
            self.assertIn('src/ovr/_core.py', config['sources'])
            self.assertTrue((Path(directory)/'results/raw_results.csv').is_file())

    def test_scales_share_ensemble_and_match_public_estimator(self):
        scenario = _scenario_catalog()[0]
        replication = _sample_scenario(scenario, 0, 240, 20260807, 20260808)
        settings = dict(n_folds=3, hidden=10, contrast_epochs=3, residual_epochs=4,
                        strong_contrast_lambda=1e-2, prediction_draws=8,
                        validation_fraction=0.15, patience=2)
        rows = fit_tau_sweep(scenario, replication, 0, 240, (0, 0.5, 1, 2), **settings)
        for key in ("split_plan_hash", "fold_hash", "contrast_artifact_hash"):
            self.assertEqual(len({row[key] for row in rows}), 1)
        base_trace = rows[2]["jitter_trace_mean"]
        for row in rows:
            self.assertAlmostEqual(row["jitter_trace_mean"], row["tau"]**2 * base_trace)
            self.assertLess(row["moment_identity_error"], 1e-10)
        for tau in (0, 1, 2):
            fitted = SmoothedOVR(
                jitter_scale=tau, seed=replication.estimator_seed, hidden=10,
                epochs=3, resid_epochs=4, contrast_patience=2).fit(
                    replication.data.X, replication.data.Y, replication.data.T)
            row = next(row for row in rows if row["tau"] == tau)
            np.testing.assert_allclose(json.loads(row["beta_hat"]), fitted.coef_, atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
