"""Private implementation shared by the public OVR estimators.

Only the estimators used by the paper's reproducibility package live here.  The
public import surface is defined in :mod:`ovr`.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


def _initialize_network(factory: Callable[[], nn.Module], seed: int) -> nn.Module:
    """Initialize a CPU network without advancing the caller's random state."""
    # Linear layers use the default CPU generator during construction. Scope
    # that state to initialization; training uses explicit local generators.
    # Seeding a CPU generator also leaves accelerator random states untouched.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        return factory()


# --------------------------------------------------------------------------- #
# small neural-network building blocks
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    """A plain multi-layer perceptron with ReLU activations."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MultiHeadContrast(nn.Module):
    """Explicit joint model ``mu_t(x) = f(x) + c_t(x)``, with ``c_ref = 0``.

    The shared head is trained on every environment and is not penalized.  Only the
    non-reference contrast heads are penalized.  Consequently ``contrast(X)`` is
    exactly the vector of learned contrast heads rather than a difference between
    independently regularized outcome heads.  This closes the finite-sample signal-
    leakage channel that becomes especially damaging when many environments are
    present.
    """

    def __init__(self, d: int, n_environments: int, ref: int, hidden: int, depth: int):
        super().__init__()
        self.ref = ref
        self.n_environments = n_environments
        self.nonref = [t for t in range(n_environments) if t != ref]
        self.trunk = _MLP(d, hidden, hidden, depth)
        self.shared_head = nn.Linear(hidden, 1)
        self.contrast_heads = nn.ModuleDict(
            {str(t): nn.Linear(hidden, 1) for t in self.nonref}
        )

    def _feat(self, X: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.trunk(X))

    def contrast(self, X: torch.Tensor) -> torch.Tensor:
        h = self._feat(X)
        if not self.nonref:
            return torch.zeros((X.shape[0], 0), dtype=X.dtype, device=X.device)
        return torch.cat([self.contrast_heads[str(t)](h) for t in self.nonref], dim=1)

    def forward(self, X: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        h = self._feat(X)
        out = self.shared_head(h).squeeze(1)
        for t in self.nonref:
            mask = T == t
            if bool(mask.any()):
                out = out.clone()
                out[mask] = out[mask] + self.contrast_heads[str(t)](h[mask]).squeeze(1)
        return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _stratified_split_indices(
    labels: np.ndarray,
    first_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices within every label, retaining each label on both sides."""
    labels = np.asarray(labels).ravel()
    first_parts: list[np.ndarray] = []
    second_parts: list[np.ndarray] = []
    for label in np.unique(labels):
        label_idx = rng.permutation(np.flatnonzero(labels == label))
        if label_idx.size < 2:
            raise ValueError(
                "Sample splitting requires at least two observations in every "
                f"environment; environment {label!r} has {label_idx.size}."
            )
        n_first = int(round(first_fraction * label_idx.size))
        n_first = min(max(n_first, 1), label_idx.size - 1)
        first_parts.append(label_idx[:n_first])
        second_parts.append(label_idx[n_first:])
    return (
        np.sort(np.concatenate(first_parts)),
        np.sort(np.concatenate(second_parts)),
    )


def _stratified_kfold_indices(
    labels: np.ndarray, k: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Return K folds that preserve every observed label in every fold."""
    labels = np.asarray(labels).ravel()
    folds: list[list[np.ndarray]] = [[] for _ in range(k)]
    for label in np.unique(labels):
        label_idx = np.flatnonzero(labels == label)
        for fold, chunk in zip(folds, np.array_split(rng.permutation(label_idx), k)):
            fold.append(chunk)
    return [np.sort(np.concatenate(parts)) for parts in folds]


def _validate_groups(groups: np.ndarray, n: int) -> np.ndarray:
    """Validate row-level cluster labels used for leakage-safe splitting."""
    groups_arr = np.asarray(groups).ravel()
    if groups_arr.shape[0] != n:
        raise ValueError("groups must contain one label for every observation.")
    if np.unique(groups_arr).size < 2:
        raise ValueError("grouped sample splitting requires at least two groups.")
    if groups_arr.dtype.kind in "fc" and not np.isfinite(groups_arr).all():
        raise ValueError("groups must contain only finite labels.")
    return groups_arr


def _grouped_stratified_split_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    first_fraction: float,
    rng: np.random.Generator,
    attempts: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Split entire groups while approximately balancing every environment.

    The routine samples several group partitions and keeps the valid partition
    whose environment counts are closest to the requested fraction.  This is
    intentionally simple and deterministic for a fixed ``rng``; it is designed
    for repeated-observation studies in which most groups occur in each
    environment (for example, users under two logging policies).
    """
    labels_arr = np.asarray(labels).ravel()
    groups_arr = _validate_groups(groups, labels_arr.shape[0])
    unique_groups, group_inverse = np.unique(groups_arr, return_inverse=True)
    n_first_groups = int(round(first_fraction * unique_groups.size))
    n_first_groups = min(max(n_first_groups, 1), unique_groups.size - 1)
    observed_labels = np.unique(labels_arr)
    total_counts = np.array(
        [np.sum(labels_arr == label) for label in observed_labels], dtype=float
    )
    target = first_fraction * total_counts
    best: Optional[tuple[float, np.ndarray, np.ndarray]] = None
    for _ in range(max(int(attempts), 1)):
        selected = rng.permutation(unique_groups.size)[:n_first_groups]
        first_mask = np.isin(group_inverse, selected)
        first_idx = np.flatnonzero(first_mask)
        second_idx = np.flatnonzero(~first_mask)
        first_counts = np.array(
            [np.sum(labels_arr[first_idx] == label) for label in observed_labels],
            dtype=float,
        )
        second_counts = total_counts - first_counts
        if np.any(first_counts == 0) or np.any(second_counts == 0):
            continue
        relative_error = np.max(
            np.abs(first_counts - target) / np.maximum(total_counts, 1.0)
        )
        candidate = (float(relative_error), first_idx, second_idx)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError(
            "Could not construct a group-disjoint split containing every "
            "environment on both sides."
        )
    return np.sort(best[1]), np.sort(best[2])


def _grouped_stratified_kfold_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    k: int,
    rng: np.random.Generator,
    attempts: int = 128,
) -> list[np.ndarray]:
    """Return approximately environment-balanced folds with disjoint groups."""
    labels_arr = np.asarray(labels).ravel()
    groups_arr = _validate_groups(groups, labels_arr.shape[0])
    unique_groups, group_inverse = np.unique(groups_arr, return_inverse=True)
    if unique_groups.size < k:
        raise ValueError(
            f"grouped {k}-fold splitting requires at least {k} distinct groups."
        )
    observed_labels = np.unique(labels_arr)
    total_counts = np.array(
        [np.sum(labels_arr == label) for label in observed_labels], dtype=float
    )
    target = total_counts / float(k)
    best: Optional[tuple[float, list[np.ndarray]]] = None
    for _ in range(max(int(attempts), 1)):
        group_chunks = np.array_split(rng.permutation(unique_groups.size), k)
        folds = [
            np.flatnonzero(np.isin(group_inverse, chunk)) for chunk in group_chunks
        ]
        counts = np.array(
            [
                [np.sum(labels_arr[fold] == label) for label in observed_labels]
                for fold in folds
            ],
            dtype=float,
        )
        if np.any(counts == 0):
            continue
        imbalance = np.max(np.abs(counts - target[None, :]) / np.maximum(target, 1.0))
        if best is None or imbalance < best[0]:
            best = (float(imbalance), folds)
    if best is None:
        raise ValueError(
            "Could not construct group-disjoint folds containing every environment."
        )
    return [np.sort(fold) for fold in best[1]]


def _onehot(T: np.ndarray, n_environments: int) -> np.ndarray:
    out = np.zeros((T.shape[0], n_environments))
    out[np.arange(T.shape[0]), T] = 1.0
    return out


def _train_regressor(
    feat: np.ndarray,
    targ: np.ndarray,
    hidden: int,
    depth: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a multi-output MLP feat -> targ; return a numpy predict function."""
    fmu, fsd = feat.mean(0), feat.std(0) + 1e-8
    tmu, tsd = targ.mean(0), targ.std(0) + 1e-8
    F = torch.tensor((feat - fmu) / fsd, dtype=torch.float32)
    Z = torch.tensor((targ - tmu) / tsd, dtype=torch.float32)
    net = _initialize_network(lambda: _MLP(F.shape[1], Z.shape[1], hidden, depth), seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(F), Z)
        loss.backward()
        opt.step()
    net.eval()

    def predict(new_feat: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            Fn = torch.tensor((new_feat - fmu) / fsd, dtype=torch.float32)
            out = net(Fn).numpy()
        return out * tsd + tmu

    return predict


def _disagreement_covariance(disagreement: np.ndarray) -> np.ndarray:
    """Estimate the covariance of an averaged two-model contrast error.

    If two independent first-stage fits have errors with the same covariance,
    one quarter of the covariance of their difference estimates the error
    covariance of their average.  The symmetrization and eigenvalue clipping
    only remove floating-point violations of positive semidefiniteness.
    """
    disagreement = np.asarray(disagreement, dtype=float)
    if disagreement.ndim != 2:
        raise ValueError("disagreement must be a two-dimensional array.")
    n, m = disagreement.shape
    if n < 2:
        return np.zeros((m, m))
    centered = disagreement - disagreement.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (4.0 * (n - 1))
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _psd_square_root(covariance: np.ndarray) -> np.ndarray:
    """Return the symmetric positive-semidefinite square root of a matrix."""
    covariance = np.asarray(covariance, dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def _train_uncertainty_smoothed_regressor(
    feat: np.ndarray,
    targ: np.ndarray,
    contrast_dim: int,
    jitter_covariance: np.ndarray,
    prediction_draws: int,
    hidden: int,
    depth: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a nuisance MLP while integrating over contrast uncertainty.

    Only the first ``contrast_dim`` feature coordinates are perturbed.  A fresh
    anisotropic Gaussian perturbation is drawn in every training epoch.  At
    prediction time the fitted nuisance is averaged over deterministic,
    antithetic perturbations from the same covariance.  This prevents the
    nuisance network from treating first-stage disagreement directions as exact
    generated regressors.
    """
    feat = np.asarray(feat, dtype=float)
    targ = np.asarray(targ, dtype=float)
    jitter_covariance = np.asarray(jitter_covariance, dtype=float)
    if feat.ndim != 2 or targ.ndim != 2:
        raise ValueError("feat and targ must be two-dimensional arrays.")
    if contrast_dim < 1 or contrast_dim > feat.shape[1]:
        raise ValueError("contrast_dim is incompatible with feat.")
    if jitter_covariance.shape != (contrast_dim, contrast_dim):
        raise ValueError("jitter_covariance has the wrong shape.")
    if prediction_draws < 1:
        raise ValueError("prediction_draws must be positive.")

    factor = _psd_square_root(jitter_covariance)
    factor_t = torch.tensor(factor, dtype=torch.float32)
    fmu = feat.mean(axis=0)
    # Standardize relative to the distribution actually shown to the learner.
    # In particular, uncertainty-only directions do not get divided by an
    # arbitrarily small empirical standard deviation.
    fvar = feat.var(axis=0)
    fvar[:contrast_dim] += np.diag(jitter_covariance)
    fsd = np.sqrt(np.maximum(fvar, 0.0)) + 1e-8
    tmu, tsd = targ.mean(axis=0), targ.std(axis=0) + 1e-8

    raw_feat = torch.tensor(feat, dtype=torch.float32)
    fmu_t = torch.tensor(fmu, dtype=torch.float32)
    fsd_t = torch.tensor(fsd, dtype=torch.float32)
    target_t = torch.tensor((targ - tmu) / tsd, dtype=torch.float32)
    net = _initialize_network(lambda: _MLP(feat.shape[1], targ.shape[1], hidden, depth), seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed + 1)
    has_jitter = bool(np.max(np.abs(factor)) > 0.0)

    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        epoch_feat = raw_feat
        if has_jitter:
            noise = torch.randn(
                (feat.shape[0], contrast_dim), generator=generator
            ) @ factor_t.T
            epoch_feat = raw_feat.clone()
            epoch_feat[:, :contrast_dim] += noise
        standardized = (epoch_feat - fmu_t) / fsd_t
        loss = loss_fn(net(standardized), target_t)
        loss.backward()
        opt.step()
    net.eval()

    def _raw_prediction(new_feat: np.ndarray) -> np.ndarray:
        new_tensor = torch.tensor(new_feat, dtype=torch.float32)
        standardized = (new_tensor - fmu_t) / fsd_t
        with torch.no_grad():
            prediction = net(standardized).numpy()
        return prediction * tsd + tmu

    def predict(new_feat: np.ndarray) -> np.ndarray:
        new_feat = np.asarray(new_feat, dtype=float)
        if new_feat.ndim != 2 or new_feat.shape[1] != feat.shape[1]:
            raise ValueError(
                f"new_feat must have shape (n, {feat.shape[1]})."
            )
        if not has_jitter:
            return _raw_prediction(new_feat)

        # Antithetic draws keep the average perturbation exactly zero.  An odd
        # draw count above one includes the unperturbed contrast as its first
        # member.  With a single requested draw, use one genuine perturbation.
        rng = np.random.default_rng(seed + 2)
        if prediction_draws == 1:
            noise = rng.normal(size=(new_feat.shape[0], contrast_dim)) @ factor.T
            perturbed = new_feat.copy()
            perturbed[:, :contrast_dim] += noise
            return _raw_prediction(perturbed)
        prediction_sum = np.zeros((new_feat.shape[0], targ.shape[1]))
        draws_used = 0
        if prediction_draws % 2 == 1:
            prediction_sum += _raw_prediction(new_feat)
            draws_used += 1
        for _ in range(prediction_draws // 2):
            noise = rng.normal(size=(new_feat.shape[0], contrast_dim)) @ factor.T
            plus = new_feat.copy()
            minus = new_feat.copy()
            plus[:, :contrast_dim] += noise
            minus[:, :contrast_dim] -= noise
            prediction_sum += _raw_prediction(plus) + _raw_prediction(minus)
            draws_used += 2
        return prediction_sum / draws_used

    return predict


# --------------------------------------------------------------------------- #
# shared base estimator
# --------------------------------------------------------------------------- #
class _OVRCore:
    """Shared first-stage fitting and linear-algebra helpers."""

    def _fit_joint(
        self, X: np.ndarray, Y: np.ndarray, T: np.ndarray, lam: float, seed: int
    ) -> _MultiHeadContrast:
        xmu, xsd = X.mean(0), X.std(0) + 1e-8
        ymu, ysd = Y.mean(), Y.std() + 1e-8
        Xs = torch.tensor((X - xmu) / xsd, dtype=torch.float32)
        Ys = torch.tensor((Y - ymu) / ysd, dtype=torch.float32)
        Tt = torch.tensor(T, dtype=torch.long)
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
                "weight_decay": float(lam),
            },
        ]
        optimizer = torch.optim.Adam(params, lr=self.lr)
        loss_fn = nn.MSELoss()
        model.train()
        n = Xs.shape[0]
        batch_size = min(self.batch_size, n)
        generator = torch.Generator().manual_seed(seed)
        for _ in range(self.epochs):
            permutation = torch.randperm(n, generator=generator)
            for start in range(0, n, batch_size):
                idx = permutation[start : start + batch_size]
                optimizer.zero_grad()
                loss = loss_fn(model(Xs[idx], Tt[idx]), Ys[idx])
                loss.backward()
                optimizer.step()
        model.eval()
        model._xmu = torch.tensor(xmu, dtype=torch.float32)  # type: ignore[attr-defined]
        model._xsd = torch.tensor(xsd, dtype=torch.float32)  # type: ignore[attr-defined]
        model._ymu = float(ymu)  # type: ignore[attr-defined]
        model._ysd = float(ysd)  # type: ignore[attr-defined]
        return model

    def _contrast_np(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            standardized = (
                torch.tensor(X, dtype=torch.float32) - model._xmu
            ) / model._xsd
            return model.contrast(standardized).numpy()

    @staticmethod
    def _sigma_T(X: np.ndarray, T: np.ndarray, n_environments: int) -> np.ndarray:
        """Return the within-environment covariance of X."""
        residualized = X.copy()
        for environment in range(n_environments):
            mask = T == environment
            if mask.any():
                residualized[mask] -= X[mask].mean(0)
        return residualized.T @ residualized / X.shape[0]

    @staticmethod
    def _gamma(sigma: np.ndarray, sigma_T: np.ndarray) -> float:
        """Return the scale-free residual-variation diagnostic."""
        eigenvalues, eigenvectors = np.linalg.eigh(sigma_T)
        eigenvalues = np.clip(eigenvalues, 1e-12, None)
        inverse_sqrt = (
            eigenvectors * (eigenvalues ** -0.5)
        ) @ eigenvectors.T
        normalized = inverse_sqrt @ sigma @ inverse_sqrt
        normalized = 0.5 * (normalized + normalized.T)
        return float(np.linalg.eigvalsh(normalized).min())

    @staticmethod
    def _cond(sigma: np.ndarray) -> float:
        eigenvalues = np.linalg.eigvalsh(0.5 * (sigma + sigma.T))
        largest, smallest = float(eigenvalues.max()), float(eigenvalues.min())
        return largest / smallest if smallest > 1e-12 else float("inf")

    @staticmethod
    def _minimum_eigenvalue(sigma: np.ndarray) -> float:
        """Return the smallest eigenvalue of a symmetrized covariance matrix."""
        return float(np.linalg.eigvalsh(0.5 * (sigma + sigma.T)).min())

    @staticmethod
    def _solve_beta(
        residualized_X: np.ndarray, residualized_Y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(residualized_X).all() or not np.isfinite(residualized_Y).all():
            raise ValueError("Residualized X and Y must contain only finite values.")
        n = residualized_X.shape[0]
        sigma = residualized_X.T @ residualized_X / n
        moment = residualized_X.T @ residualized_Y / n
        if not np.isfinite(sigma).all() or not np.isfinite(moment).all():
            raise ValueError("Residual moments must be finite; check for numerical overflow.")
        beta = np.linalg.solve(
            sigma + 1e-10 * np.eye(sigma.shape[0]), moment
        )
        if not np.isfinite(beta).all():
            raise ValueError("Estimated coefficients must be finite; check for numerical overflow.")
        return beta, sigma

    def _prepare(self, X: np.ndarray, Y: np.ndarray, T: np.ndarray) -> None:
        self.X_ = np.asarray(X, dtype=float)
        self.Y_ = np.asarray(Y, dtype=float).ravel()
        self.T_ = np.asarray(T).astype(int).ravel()
        self.n_, self.d_ = self.X_.shape
        self.n_environments_ = int(self.T_.max()) + 1
        self.m_ = self.n_environments_ - 1
        self.Sigma_T_ = self._sigma_T(self.X_, self.T_, self.n_environments_)

    @property
    def coef_(self) -> np.ndarray:
        assert self.beta_ is not None, "call fit() first"
        return self.beta_
