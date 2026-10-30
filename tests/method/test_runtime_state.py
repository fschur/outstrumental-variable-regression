"""Library imports and fitted estimators preserve their caller's runtime state."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ovr import DirectOVR, SmoothedOVR


class RuntimeStateTests(unittest.TestCase):
    def test_import_preserves_environment_threads_random_state_and_warnings(self):
        script = '''
import os, warnings
import numpy as np
import torch
# Initialize dependencies before checking the package's own import effects.
torch.set_num_threads(2)
state = torch.get_rng_state().clone()
environment = dict(os.environ)
filters = list(warnings.filters)
import ovr
assert os.environ == environment
assert torch.get_num_threads() == 2
assert torch.equal(torch.get_rng_state(), state)
assert warnings.filters == filters
with warnings.catch_warnings(record=True) as caught:
    warnings.warn("overflow encountered in matmul", RuntimeWarning)
assert len(caught) == 1
'''
        env = dict(os.environ)
        env.pop('KMP_DUPLICATE_LIB_OK', None)
        env['OMP_NUM_THREADS'] = '2'
        root = Path(__file__).resolve().parents[2]
        env['PYTHONPATH'] = str(root/'src')
        subprocess.run([sys.executable, '-c', script], env=env, check=True,
                       capture_output=True, text=True)

    def test_fit_preserves_random_streams_and_thread_count(self):
        rng = np.random.default_rng(82)
        X = rng.normal(size=(240, 2))
        E = np.tile(np.arange(3), 80)
        Y = X @ np.array([1., -1.]) + (E+1)*X[:, 0]**2 + rng.normal(size=240)
        for estimator in (DirectOVR, SmoothedOVR):
            with self.subTest(estimator=estimator.__name__), torch.random.fork_rng(devices=[]):
                settings = dict(hidden=8, epochs=2, resid_epochs=2, seed=17)
                estimates = []
                for incoming_seed in (11, 29):
                    torch.set_rng_state(torch.Generator().manual_seed(incoming_seed).get_state())
                    state = torch.get_rng_state().clone()
                    numpy_state = np.random.get_state()
                    threads = torch.get_num_threads()
                    # This API also seeds accelerators, which a CPU fit must not do.
                    with patch('torch.manual_seed', side_effect=AssertionError('global seed')):
                        fitted = estimator(**settings).fit(X, Y, E)
                    self.assertTrue(torch.equal(torch.get_rng_state(), state))
                    current_numpy = np.random.get_state()
                    np.testing.assert_array_equal(current_numpy[1], numpy_state[1])
                    self.assertEqual(current_numpy[0], numpy_state[0])
                    self.assertEqual(current_numpy[2:], numpy_state[2:])
                    self.assertEqual(torch.get_num_threads(), threads)
                    estimates.append(fitted.coef_)
                np.testing.assert_array_equal(*estimates)

    def test_failed_fit_preserves_random_state(self):
        X = np.random.default_rng(31).normal(size=(120, 2))
        E = np.tile(np.arange(3), 40)
        for estimator in (DirectOVR, SmoothedOVR):
            with self.subTest(estimator=estimator.__name__):
                state = torch.get_rng_state().clone()
                with patch('torch.optim.Adam', side_effect=RuntimeError('optimizer failed')):
                    with self.assertRaisesRegex(RuntimeError, 'optimizer failed'):
                        estimator(hidden=8, epochs=2).fit(X, X[:, 0], E)
                self.assertTrue(torch.equal(torch.get_rng_state(), state))


if __name__ == '__main__':
    unittest.main()
