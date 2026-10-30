"""Smoothed Outstrumental Variable Regression."""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn
from .direct import DirectOVR
from ._core import (
    _MultiHeadContrast,
    _initialize_network,
    _stratified_split_indices,
    _stratified_kfold_indices,
    _validate_groups,
    _grouped_stratified_split_indices,
    _grouped_stratified_kfold_indices,
    _onehot,
    _disagreement_covariance,
    _train_uncertainty_smoothed_regressor,
)


class SmoothedOVR(DirectOVR):
    """Direct OVR smoothed by estimated contrast uncertainty.

    The contrast sample is split again, within environment, into two disjoint
    samples.  A separate joint contrast model is trained on each sample and their
    predictions on the independent slope sample are averaged.  For two
    independent, equally accurate fits, ``Cov(C1 - C2) / 4`` estimates the error
    covariance of that average.  Each second-stage fold estimates this covariance
    from its own training rows and uses it for anisotropic Gaussian jitter during
    both nuisance training and prediction.

    This is a practical generated-regressor stabilization: it retains the full
    ``K - 1`` contrast and neither estimates nor assumes an intrinsic rank.  It
    never uses an oracle contrast or the true coefficient.  The two first stages
    may additionally use stronger head regularization and deterministic early
    stopping on an environment-balanced validation loss.

    Parameters are as in :class:`DirectOVR`, except for:

    contrast_lambda : float
        Head weight decay used by both contrast fits.  The default is deliberately
        stronger than the direct estimator's ``1e-4``.
    jitter_scale : float
        Multiplier ``kappa`` on the estimated uncertainty standard deviation.
        Zero gives an unsmoothed two-model ensemble.
    prediction_draws : int
        Number of deterministic antithetic jitter draws used to average each
        held-out nuisance prediction.
    contrast_validation_fraction : float
        Within-member fraction reserved for early stopping.  Set to zero to fit
        for exactly ``epochs`` iterations on the whole member split.
    contrast_patience, contrast_min_delta : int, float
        Deterministic early-stopping controls.  A patience of zero disables early
        stopping and uses the entire member split.
    """

    representation_name: str = "uncertainty_smoothed_split"

    def __init__(
        self,
        reference: int = 0,
        contrast_fraction: float = 0.5,
        contrast_lambda: float = 1e-2,
        jitter_scale: float = 1.0,
        prediction_draws: int = 8,
        contrast_validation_fraction: float = 0.15,
        contrast_patience: int = 15,
        contrast_min_delta: float = 1e-5,
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
        super().__init__(
            reference=reference,
            contrast_fraction=contrast_fraction,
            contrast_lambda=contrast_lambda,
            n_folds=n_folds,
            hidden=hidden,
            depth=depth,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            resid_epochs=resid_epochs,
            resid_lr=resid_lr,
            resid_weight_decay=resid_weight_decay,
            seed=seed,
        )
        self.jitter_scale = float(jitter_scale)
        self.prediction_draws = int(prediction_draws)
        self.contrast_validation_fraction = float(contrast_validation_fraction)
        self.contrast_patience = int(contrast_patience)
        self.contrast_min_delta = float(contrast_min_delta)

        self.contrast_models_: Optional[tuple[nn.Module, nn.Module]] = None
        self.contrast_indices_a_: Optional[np.ndarray] = None
        self.contrast_indices_b_: Optional[np.ndarray] = None
        self.estimated_contrast_components_: Optional[
            tuple[np.ndarray, np.ndarray]
        ] = None
        self.contrast_disagreement_: Optional[np.ndarray] = None
        self.contrast_disagreement_covariance_: Optional[np.ndarray] = None
        self.contrast_disagreement_mean_: Optional[np.ndarray] = None
        self.contrast_disagreement_rmse_: Optional[float] = None
        self.uncertainty_eigenvalues_: Optional[np.ndarray] = None
        self.fold_disagreement_covariances_: Optional[list[np.ndarray]] = None
        self.jitter_covariances_: Optional[list[np.ndarray]] = None
        self.contrast_validation_losses_: Optional[tuple[float, float]] = None
        self.contrast_epochs_trained_: Optional[tuple[int, int]] = None
        self.contrast_best_epochs_: Optional[tuple[int, int]] = None
        self.contrast_early_stopped_: Optional[tuple[bool, bool]] = None
        self.contrast_member_training_sizes_: Optional[tuple[int, int]] = None
        self.contrast_member_validation_sizes_: Optional[tuple[int, int]] = None
        self.first_stage_fit_count_: int = 0

    def _validate_uncertainty_smoothed_input(
        self, X: np.ndarray, Y: np.ndarray, T: np.ndarray
    ) -> None:
        self._validate_direct_split_input(X, Y, T)
        if self.jitter_scale < 0.0:
            raise ValueError("jitter_scale must be nonnegative.")
        if self.prediction_draws < 1:
            raise ValueError("prediction_draws must be positive.")
        if not 0.0 <= self.contrast_validation_fraction < 1.0:
            raise ValueError(
                "contrast_validation_fraction must lie in [0, 1)."
            )
        if self.contrast_patience < 0:
            raise ValueError("contrast_patience must be nonnegative.")
        if self.contrast_min_delta < 0.0:
            raise ValueError("contrast_min_delta must be nonnegative.")
        if self.epochs < 1 or self.resid_epochs < 1:
            raise ValueError("epochs and resid_epochs must be positive.")

    def _fit_joint_member(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        seed: int,
        groups: Optional[np.ndarray] = None,
    ) -> _MultiHeadContrast:
        """Fit one contrast member, optionally with balanced early stopping."""
        use_early_stopping = (
            self.contrast_validation_fraction > 0.0
            and self.contrast_patience > 0
        )
        if not use_early_stopping:
            model = super()._fit_joint(
                X, Y, T, self.contrast_lambda, seed=seed
            )
            model._best_validation_loss = float("nan")  # type: ignore[attr-defined]
            model._best_epoch = int(self.epochs)  # type: ignore[attr-defined]
            model._epochs_trained = int(self.epochs)  # type: ignore[attr-defined]
            model._early_stopped = False  # type: ignore[attr-defined]
            model._member_training_size = int(X.shape[0])  # type: ignore[attr-defined]
            model._member_validation_size = 0  # type: ignore[attr-defined]
            return model

        rng = np.random.default_rng(seed + 17)
        if groups is None:
            train_idx, validation_idx = _stratified_split_indices(
                T, 1.0 - self.contrast_validation_fraction, rng
            )
        else:
            train_idx, validation_idx = _grouped_stratified_split_indices(
                T,
                groups,
                1.0 - self.contrast_validation_fraction,
                rng,
            )
        X_train, Y_train, T_train = X[train_idx], Y[train_idx], T[train_idx]
        X_validation, Y_validation, T_validation = (
            X[validation_idx],
            Y[validation_idx],
            T[validation_idx],
        )

        xmu, xsd = X_train.mean(0), X_train.std(0) + 1e-8
        ymu, ysd = Y_train.mean(), Y_train.std() + 1e-8
        Xs = torch.tensor((X_train - xmu) / xsd, dtype=torch.float32)
        Ys = torch.tensor((Y_train - ymu) / ysd, dtype=torch.float32)
        Tt = torch.tensor(T_train, dtype=torch.long)
        Xv = torch.tensor((X_validation - xmu) / xsd, dtype=torch.float32)
        Yv = torch.tensor((Y_validation - ymu) / ysd, dtype=torch.float32)
        Tv = torch.tensor(T_validation, dtype=torch.long)

        model = _initialize_network(
            lambda: _MultiHeadContrast(
                X.shape[1], self.n_environments_, self.reference, self.hidden, self.depth
            ), seed,
        )
        params = [
            {"params": model.trunk.parameters(), "weight_decay": 0.0},
            {"params": model.shared_head.parameters(), "weight_decay": 0.0},
            {
                "params": model.contrast_heads.parameters(),
                "weight_decay": self.contrast_lambda,
            },
        ]
        optimizer = torch.optim.Adam(params, lr=self.lr)
        loss_fn = nn.MSELoss()
        batch_size = min(self.batch_size, Xs.shape[0])
        generator = torch.Generator().manual_seed(seed)
        best_loss = float("inf")
        best_epoch = 0
        best_state: Optional[dict[str, torch.Tensor]] = None
        stale_epochs = 0
        epochs_trained = 0

        for epoch in range(self.epochs):
            model.train()
            permutation = torch.randperm(Xs.shape[0], generator=generator)
            for start in range(0, Xs.shape[0], batch_size):
                idx = permutation[start : start + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(model(Xs[idx], Tt[idx]), Ys[idx])
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                squared_error = (model(Xv, Tv) - Yv) ** 2
                per_environment = [
                    squared_error[Tv == environment].mean()
                    for environment in range(self.n_environments_)
                ]
                validation_loss = float(torch.stack(per_environment).mean())
            epochs_trained = epoch + 1
            if validation_loss < best_loss - self.contrast_min_delta:
                best_loss = validation_loss
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.contrast_patience:
                    break

        if best_state is None:
            raise RuntimeError("early stopping failed to record a contrast model.")
        model.load_state_dict(best_state)
        model.eval()
        model._xmu = torch.tensor(xmu, dtype=torch.float32)  # type: ignore[attr-defined]
        model._xsd = torch.tensor(xsd, dtype=torch.float32)  # type: ignore[attr-defined]
        model._ymu = float(ymu)  # type: ignore[attr-defined]
        model._ysd = float(ysd)  # type: ignore[attr-defined]
        model._best_validation_loss = best_loss  # type: ignore[attr-defined]
        model._best_epoch = best_epoch  # type: ignore[attr-defined]
        model._epochs_trained = epochs_trained  # type: ignore[attr-defined]
        model._early_stopped = epochs_trained < self.epochs  # type: ignore[attr-defined]
        model._member_training_size = int(train_idx.size)  # type: ignore[attr-defined]
        model._member_validation_size = int(validation_idx.size)  # type: ignore[attr-defined]
        return model

    def _crossfit_uncertainty_smoothed_residuals(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        contrast: np.ndarray,
        disagreement: np.ndarray,
        local_folds: list[np.ndarray],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        list[Callable],
        list[np.ndarray],
        list[np.ndarray],
    ]:
        """Cross-fit nuisances with fold-local disagreement calibration."""
        n, d = X.shape
        residualized_X = np.zeros((n, d))
        residualized_Y = np.zeros(n)
        predictors: list[Callable] = []
        fold_covariances: list[np.ndarray] = []
        jitter_covariances: list[np.ndarray] = []
        all_local = np.arange(n)
        onehot = _onehot(T, self.n_environments_)
        targets = np.hstack([X, Y[:, None]])
        contrast_dim = contrast.shape[1]

        for fold_id, test_idx in enumerate(local_folds):
            train_idx = np.setdiff1d(all_local, test_idx, assume_unique=True)
            disagreement_covariance = _disagreement_covariance(
                disagreement[train_idx]
            )
            jitter_covariance = (
                self.jitter_scale**2 * disagreement_covariance
            )
            feat_train = np.hstack([contrast[train_idx], onehot[train_idx]])
            feat_test = np.hstack([contrast[test_idx], onehot[test_idx]])
            predict = _train_uncertainty_smoothed_regressor(
                feat_train,
                targets[train_idx],
                contrast_dim=contrast_dim,
                jitter_covariance=jitter_covariance,
                prediction_draws=self.prediction_draws,
                hidden=self.hidden,
                depth=self.depth,
                epochs=self.resid_epochs,
                lr=self.resid_lr,
                weight_decay=self.resid_weight_decay,
                seed=self.seed + 20_000 + fold_id,
            )
            fitted = predict(feat_test)
            residualized_X[test_idx] = X[test_idx] - fitted[:, :d]
            residualized_Y[test_idx] = Y[test_idx] - fitted[:, d]
            predictors.append(predict)
            fold_covariances.append(disagreement_covariance)
            jitter_covariances.append(jitter_covariance)

        return (
            residualized_X,
            residualized_Y,
            predictors,
            fold_covariances,
            jitter_covariances,
        )

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
        self._validate_uncertainty_smoothed_input(X_arr, Y_arr, T_arr)
        self._prepare(X_arr, Y_arr, T_arr)

        groups_arr = None if groups is None else _validate_groups(groups, X_arr.shape[0])
        self.groups_ = groups_arr
        self.grouped_splitting_ = groups_arr is not None

        rng = np.random.default_rng(self.seed)
        if groups_arr is None:
            contrast_pool_idx, estimation_idx = _stratified_split_indices(
                self.T_, self.contrast_fraction, rng
            )
            member_a_local, member_b_local = _stratified_split_indices(
                self.T_[contrast_pool_idx], 0.5, rng
            )
        else:
            contrast_pool_idx, estimation_idx = _grouped_stratified_split_indices(
                self.T_, groups_arr, self.contrast_fraction, rng
            )
            member_a_local, member_b_local = _grouped_stratified_split_indices(
                self.T_[contrast_pool_idx],
                groups_arr[contrast_pool_idx],
                0.5,
                rng,
            )
        member_a_idx = contrast_pool_idx[member_a_local]
        member_b_idx = contrast_pool_idx[member_b_local]

        estimation_counts = np.bincount(
            self.T_[estimation_idx], minlength=self.n_environments_
        )
        if int(estimation_counts.min()) < self.n_folds:
            raise ValueError(
                "Each environment in the estimation split must have at least "
                f"n_folds={self.n_folds} observations; observed counts are "
                f"{estimation_counts.tolist()}."
            )

        self.first_stage_fit_count_ = 0
        if groups_arr is None:
            model_a = self._fit_joint_member(
                self.X_[member_a_idx],
                self.Y_[member_a_idx],
                self.T_[member_a_idx],
                seed=self.seed + 1_000,
            )
        else:
            model_a = self._fit_joint_member(
                self.X_[member_a_idx],
                self.Y_[member_a_idx],
                self.T_[member_a_idx],
                seed=self.seed + 1_000,
                groups=groups_arr[member_a_idx],
            )
        self.first_stage_fit_count_ += 1
        if groups_arr is None:
            model_b = self._fit_joint_member(
                self.X_[member_b_idx],
                self.Y_[member_b_idx],
                self.T_[member_b_idx],
                seed=self.seed + 2_000,
            )
        else:
            model_b = self._fit_joint_member(
                self.X_[member_b_idx],
                self.Y_[member_b_idx],
                self.T_[member_b_idx],
                seed=self.seed + 2_000,
                groups=groups_arr[member_b_idx],
            )
        self.first_stage_fit_count_ += 1

        X_beta = self.X_[estimation_idx]
        Y_beta = self.Y_[estimation_idx]
        T_beta = self.T_[estimation_idx]
        contrast_a = self._contrast_np(model_a, X_beta) * float(model_a._ysd)
        contrast_b = self._contrast_np(model_b, X_beta) * float(model_b._ysd)
        contrast = 0.5 * (contrast_a + contrast_b)
        disagreement = contrast_a - contrast_b
        if groups_arr is None:
            local_folds = _stratified_kfold_indices(T_beta, self.n_folds, rng)
        else:
            local_folds = _grouped_stratified_kfold_indices(
                T_beta, groups_arr[estimation_idx], self.n_folds, rng
            )
        (
            residualized_X,
            residualized_Y,
            predictors,
            fold_covariances,
            jitter_covariances,
        ) = self._crossfit_uncertainty_smoothed_residuals(
            X_beta,
            Y_beta,
            T_beta,
            contrast,
            disagreement,
            local_folds,
        )
        beta, Sigma = self._solve_beta(residualized_X, residualized_Y)

        Sigma_T = self._sigma_T(X_beta, T_beta, self.n_environments_)
        global_uncertainty = _disagreement_covariance(disagreement)
        self.beta_ = beta
        self.lambda_ = self.contrast_lambda
        self.gamma_ = self._gamma(Sigma, Sigma_T)
        self.cond_ = self._cond(Sigma)
        self.minimum_eigenvalue_ = self._minimum_eigenvalue(Sigma)
        self.Sigma_ = Sigma
        self.Sigma_T_ = Sigma_T
        self.contrast_dim_ = int(contrast.shape[1])
        self.contrast_model_ = None
        self.contrast_models_ = (model_a, model_b)
        self.contrast_indices_ = contrast_pool_idx
        self.contrast_indices_a_ = member_a_idx
        self.contrast_indices_b_ = member_b_idx
        self.estimation_indices_ = estimation_idx
        self.second_stage_folds_ = [estimation_idx[idx] for idx in local_folds]
        self.estimated_contrast_ = contrast
        self.estimated_contrast_components_ = (contrast_a, contrast_b)
        self.contrast_disagreement_ = disagreement
        self.contrast_disagreement_covariance_ = global_uncertainty
        self.contrast_disagreement_mean_ = disagreement.mean(axis=0)
        self.contrast_disagreement_rmse_ = float(
            np.sqrt(np.mean(disagreement**2))
        )
        self.uncertainty_eigenvalues_ = np.linalg.eigvalsh(global_uncertainty)
        self.fold_disagreement_covariances_ = fold_covariances
        self.jitter_covariances_ = jitter_covariances
        self.contrast_validation_losses_ = (
            float(model_a._best_validation_loss),
            float(model_b._best_validation_loss),
        )
        self.contrast_epochs_trained_ = (
            int(model_a._epochs_trained),
            int(model_b._epochs_trained),
        )
        self.contrast_best_epochs_ = (
            int(model_a._best_epoch),
            int(model_b._best_epoch),
        )
        self.contrast_early_stopped_ = (
            bool(model_a._early_stopped),
            bool(model_b._early_stopped),
        )
        self.contrast_member_training_sizes_ = (
            int(model_a._member_training_size),
            int(model_b._member_training_size),
        )
        self.contrast_member_validation_sizes_ = (
            int(model_a._member_validation_size),
            int(model_b._member_validation_size),
        )
        self.residualized_X_ = residualized_X
        self.residualized_Y_ = residualized_Y
        self.second_stage_predictors_ = predictors
        self.n_contrast_ = int(contrast_pool_idx.size)
        self.n_estimation_ = int(estimation_idx.size)
        return self

    def predict_contrast_components(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate both independently trained contrast models."""
        if self.contrast_models_ is None:
            raise RuntimeError("call fit() before predict_contrast_components().")
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.d_:
            raise ValueError(f"X must have shape (n, {self.d_}).")
        model_a, model_b = self.contrast_models_
        contrast_a = self._contrast_np(model_a, X_arr) * float(model_a._ysd)
        contrast_b = self._contrast_np(model_b, X_arr) * float(model_b._ysd)
        return contrast_a, contrast_b

    def predict_contrast(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the equally weighted two-model contrast ensemble."""
        contrast_a, contrast_b = self.predict_contrast_components(X)
        return 0.5 * (contrast_a + contrast_b)
