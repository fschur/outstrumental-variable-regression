import unittest

import numpy as np

from experiments.common.baselines import AnchorRegression, OLS


class AnchorRegressionTests(unittest.TestCase):
    def test_gamma_one_equals_ols(self) -> None:
        rng = np.random.default_rng(17)
        X = rng.normal(size=(400, 4))
        E = rng.integers(0, 4, size=400)
        beta = np.array([0.4, -0.7, 0.2, 1.1])
        Y = np.sum(X * beta, axis=1) + E + rng.normal(size=400)

        ols = OLS().fit(X, Y, E)
        anchor = AnchorRegression(gamma=1.0).fit(X, Y, E)

        np.testing.assert_allclose(anchor.coef_, ols.coef_, rtol=1e-12, atol=1e-12)



if __name__ == "__main__":
    unittest.main()
