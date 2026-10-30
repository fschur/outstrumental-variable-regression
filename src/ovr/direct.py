"""Direct Outstrumental Variable Regression."""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np
import torch.nn as nn
from ._core import (
    _stratified_split_indices,
    _stratified_kfold_indices,
    _validate_groups,
    _grouped_stratified_split_indices,
    _grouped_stratified_kfold_indices,
    _onehot,
    _train_regressor,
    _OVRCore,
)


class DirectOVR(_OVRCore):
    """Direct OVR with independent contrast and slope samples.

    This estimator deliberately has no representation or tuning selection.  It
    first makes an environment-stratified split into a contrast sample
    ``I_C`` and an independent slope sample ``I_beta``.  A single joint model
    ``mu_e(x) = f(x) + c_e(x)`` is fit on ``I_C``.  Its complete
    ``K - 1`` dimensional contrast is then evaluated on ``I_beta`` and held fixed
    while the regressions of ``X`` and ``Y`` on ``(C_hat(X), E)`` are cross-fit
    within ``I_beta``.  Finally, the coefficient is computed from the pooled
    held-out residuals.

    ``contrast_lambda`` is a fixed first-stage regularization constant, not a
    data-adaptive scan.  The reported positivity and conditioning quantities are
    diagnostics only: they never select a representation, replace the full
    contrast, or trigger a fixed-effects fallback.

    Parameters
    ----------
    reference : int
        Reference environment.  Environment labels must be consecutive integers
        ``0, ..., K - 1``.
    contrast_fraction : float
        Fraction placed in ``I_C`` within each environment.
    contrast_lambda : float
        Fixed weight decay on the non-reference contrast heads.
    n_folds : int
        Number of environment-stratified residualization folds in ``I_beta``.
    hidden, depth, epochs, lr, batch_size : int, float
        First-stage joint-network settings.
    resid_epochs, resid_lr, resid_weight_decay : int, float
        Second-stage nuisance-regression settings.
    seed : int
        Seed used for both splitting and learner initialization.
    """

    representation_name: str = "direct_split"

    def __init__(
        self,
        reference: int = 0,
        contrast_fraction: float = 0.5,
        contrast_lambda: float = 1e-4,
        n_folds: int = 3,
        hidden: int = 64,
        depth: int = 2,
        epochs: int = 200,
        lr: float = 5e-3,
        batch_size: int = 256,
        resid_epochs: int = 300,
        resid_lr: float = 1e-2,
        resid_weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> None:
        self.reference = int(reference)
        self.contrast_fraction = float(contrast_fraction)
        self.contrast_lambda = float(contrast_lambda)
        self.n_folds = int(n_folds)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.resid_epochs = int(resid_epochs)
        self.resid_lr = float(resid_lr)
        self.resid_weight_decay = float(resid_weight_decay)
        self.seed = int(seed)

        # Fitted state is initialized explicitly to keep the estimator state
        # transparent and independent of a framework-level fit template.
        self.beta_: Optional[np.ndarray] = None
        self.lambda_: Optional[float] = None
        self.gamma_: Optional[float] = None
        self.cond_: Optional[float] = None
        self.minimum_eigenvalue_: Optional[float] = None
        self.Sigma_: Optional[np.ndarray] = None
        self.contrast_dim_: Optional[int] = None
        self.contrast_model_: Optional[nn.Module] = None
        self.contrast_indices_: Optional[np.ndarray] = None
        self.estimation_indices_: Optional[np.ndarray] = None
        self.second_stage_folds_: Optional[list[np.ndarray]] = None
        self.estimated_contrast_: Optional[np.ndarray] = None
        self.residualized_X_: Optional[np.ndarray] = None
        self.residualized_Y_: Optional[np.ndarray] = None
        self.second_stage_predictors_: Optional[list[Callable]] = None
        self.n_contrast_: Optional[int] = None
        self.n_estimation_: Optional[int] = None
        self.groups_: Optional[np.ndarray] = None
        self.grouped_splitting_: bool = False

    def _validate_direct_split_input(
        self, X: np.ndarray, Y: np.ndarray, T: np.ndarray
    ) -> None:
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if Y.ndim != 1 or T.ndim != 1:
            raise ValueError("Y and E/T must be one-dimensional arrays.")
        if X.shape[0] != Y.shape[0] or X.shape[0] != T.shape[0]:
            raise ValueError("X, Y, and E/T must contain the same number of rows.")
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError("X and Y must contain only finite values.")
        if not 0.0 < self.contrast_fraction < 1.0:
            raise ValueError("contrast_fraction must lie strictly between zero and one.")
        if self.contrast_lambda < 0.0:
            raise ValueError("contrast_lambda must be nonnegative.")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least two.")
        labels = np.unique(T)
        expected = np.arange(labels.size)
        if not np.array_equal(labels, expected):
            raise ValueError(
                "Environment labels must be consecutive integers 0, ..., K - 1."
            )
        if labels.size < 2:
            raise ValueError("OVR requires at least two environments.")
        if self.reference < 0 or self.reference >= labels.size:
            raise ValueError("reference is not an observed environment.")

    def _crossfit_direct_residuals(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        contrast: np.ndarray,
        local_folds: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, list[Callable]]:
        """Cross-fit second-stage residuals with the full fixed contrast."""
        n, d = X.shape
        residualized_X = np.zeros((n, d))
        residualized_Y = np.zeros(n)
        predictors: list[Callable] = []
        all_local = np.arange(n)
        onehot = _onehot(T, self.n_environments_)
        targets = np.hstack([X, Y[:, None]])

        for fold_id, test_idx in enumerate(local_folds):
            train_idx = np.setdiff1d(all_local, test_idx, assume_unique=True)
            feat_train = np.hstack([contrast[train_idx], onehot[train_idx]])
            feat_test = np.hstack([contrast[test_idx], onehot[test_idx]])
            predict = _train_regressor(
                feat_train,
                targets[train_idx],
                hidden=self.hidden,
                depth=self.depth,
                epochs=self.resid_epochs,
                lr=self.resid_lr,
                weight_decay=self.resid_weight_decay,
                seed=self.seed + 10_000 + fold_id,
            )
            fitted = predict(feat_test)
            residualized_X[test_idx] = X[test_idx] - fitted[:, :d]
            residualized_Y[test_idx] = Y[test_idx] - fitted[:, d]
            predictors.append(predict)

        return residualized_X, residualized_Y, predictors

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ):
        X_arr = np.asarray(X, dtype=float)
        Y_arr = np.asarray(Y, dtype=float).ravel()
        T_raw = np.asarray(T)
        if not np.issubdtype(T_raw.dtype, np.integer):
            if not np.all(np.equal(T_raw, np.floor(T_raw))):
                raise ValueError("Environment labels must be integers.")
        T_arr = T_raw.astype(int).ravel()
        self._validate_direct_split_input(X_arr, Y_arr, T_arr)
        self._prepare(X_arr, Y_arr, T_arr)

        groups_arr = None if groups is None else _validate_groups(groups, X_arr.shape[0])
        self.groups_ = groups_arr
        self.grouped_splitting_ = groups_arr is not None

        rng = np.random.default_rng(self.seed)
        if groups_arr is None:
            contrast_idx, estimation_idx = _stratified_split_indices(
                self.T_, self.contrast_fraction, rng
            )
        else:
            contrast_idx, estimation_idx = _grouped_stratified_split_indices(
                self.T_, groups_arr, self.contrast_fraction, rng
            )
        estimation_counts = np.bincount(
            self.T_[estimation_idx], minlength=self.n_environments_
        )
        if int(estimation_counts.min()) < self.n_folds:
            raise ValueError(
                "Each environment in the estimation split must have at least "
                f"n_folds={self.n_folds} observations; observed counts are "
                f"{estimation_counts.tolist()}."
            )

        # The generated regressor is learned once, exclusively on I_C.
        contrast_model = self._fit_joint(
            self.X_[contrast_idx],
            self.Y_[contrast_idx],
            self.T_[contrast_idx],
            self.contrast_lambda,
            seed=self.seed + 1_000,
        )
        # The joint model is trained on standardized Y, whereas C is defined on
        # the original outcome scale.  Restoring that scale is immaterial for
        # residualization but necessary for interpretable oracle diagnostics.
        contrast = (
            self._contrast_np(contrast_model, self.X_[estimation_idx])
            * float(contrast_model._ysd)
        )

        X_beta = self.X_[estimation_idx]
        Y_beta = self.Y_[estimation_idx]
        T_beta = self.T_[estimation_idx]
        if groups_arr is None:
            local_folds = _stratified_kfold_indices(T_beta, self.n_folds, rng)
        else:
            local_folds = _grouped_stratified_kfold_indices(
                T_beta, groups_arr[estimation_idx], self.n_folds, rng
            )
        residualized_X, residualized_Y, predictors = self._crossfit_direct_residuals(
            X_beta, Y_beta, T_beta, contrast, local_folds
        )
        beta, Sigma = self._solve_beta(residualized_X, residualized_Y)

        # These are reporting diagnostics only.  No threshold is applied.
        Sigma_T = self._sigma_T(X_beta, T_beta, self.n_environments_)
        self.beta_ = beta
        self.lambda_ = self.contrast_lambda
        self.gamma_ = self._gamma(Sigma, Sigma_T)
        self.cond_ = self._cond(Sigma)
        self.minimum_eigenvalue_ = self._minimum_eigenvalue(Sigma)
        self.Sigma_ = Sigma
        self.Sigma_T_ = Sigma_T
        self.contrast_dim_ = int(contrast.shape[1])
        self.contrast_model_ = contrast_model
        self.contrast_indices_ = contrast_idx
        self.estimation_indices_ = estimation_idx
        self.second_stage_folds_ = [estimation_idx[idx] for idx in local_folds]
        self.estimated_contrast_ = contrast
        self.residualized_X_ = residualized_X
        self.residualized_Y_ = residualized_Y
        self.second_stage_predictors_ = predictors
        self.n_contrast_ = int(contrast_idx.size)
        self.n_estimation_ = int(estimation_idx.size)
        return self

    def predict_contrast(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the first-split contrast model on new exposures."""
        if self.contrast_model_ is None:
            raise RuntimeError("call fit() before predict_contrast().")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.d_:
            raise ValueError(f"X must have shape (n, {self.d_}).")
        return self._contrast_np(self.contrast_model_, X_arr) * float(
            self.contrast_model_._ysd
        )
