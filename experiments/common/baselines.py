"""OLS, environment fixed effects, Anchor regression, and LNN.

All estimators expose fit(X, Y, E) and coef_. LNN jointly fits a linear term in X
and a neural network in (X, E). Anchor regression uses environment indicators;
strength one gives OLS and the paper uses strength two.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .linear_neural import LinearNeuralRegression


@dataclass
class OLS:
    """Ordinary least squares of Y on X (intercept included), ignoring T."""

    beta_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray, T: np.ndarray | None = None) -> "OLS":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).ravel()
        Xc = X - X.mean(0)
        Yc = Y - Y.mean()
        self.beta_ = np.linalg.lstsq(Xc, Yc, rcond=None)[0]
        return self

    @property
    def coef_(self) -> np.ndarray:
        assert self.beta_ is not None
        return self.beta_


@dataclass
class EnvironmentFixedEffects:
    """Environment fixed effects.

    Residualize X and Y on the environment (per-environment demeaning) and regress the
    residualized outcome on the residualized covariates.
    """

    beta_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray, T: np.ndarray) -> "EnvironmentFixedEffects":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).ravel()
        T = np.asarray(T).astype(int).ravel()
        Xt = np.empty_like(X)
        Yt = np.empty_like(Y)
        for t in np.unique(T):
            m = T == t
            Xt[m] = X[m] - X[m].mean(0)
            Yt[m] = Y[m] - Y[m].mean()
        self.beta_ = np.linalg.lstsq(Xt, Yt, rcond=None)[0]
        return self

    @property
    def coef_(self) -> np.ndarray:
        assert self.beta_ is not None
        return self.beta_


def _env_project(Z: np.ndarray, T: np.ndarray) -> np.ndarray:
    """P_A Z with A = onehot(T): replace each row by its environment mean."""
    Z = np.asarray(Z, dtype=float)
    flat = Z.ndim == 1
    if flat:
        Z = Z[:, None]
    out = np.empty_like(Z)
    for t in np.unique(T):
        m = T == t
        out[m] = Z[m].mean(0)
    return out[:, 0] if flat else out


@dataclass
class AnchorRegression:
    """Anchor regression (Rothenhausler et al., 2021) with anchor A = onehot(T).

    Solves  argmin_b  E[((I - P_A)(Y - X b))^2] + gamma * E[((P_A)(Y - X b))^2],
    implemented via the anchor transform Z -> Z + (sqrt(gamma) - 1) P_A Z applied
    to X and Y, followed by OLS (with intercept).  gamma = 1 is OLS; large gamma
    approaches the environment-IV solution.
    """

    gamma: float = 2.0
    beta_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray, T: np.ndarray) -> "AnchorRegression":
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).ravel()
        T = np.asarray(T).astype(int).ravel()
        w = np.sqrt(self.gamma) - 1.0
        Xa = X + w * _env_project(X, T)
        Ya = Y + w * _env_project(Y, T)
        Xc = Xa - Xa.mean(0)
        Yc = Ya - Ya.mean()
        self.beta_ = np.linalg.lstsq(Xc, Yc, rcond=None)[0]
        return self

    @property
    def coef_(self) -> np.ndarray:
        assert self.beta_ is not None
        return self.beta_
