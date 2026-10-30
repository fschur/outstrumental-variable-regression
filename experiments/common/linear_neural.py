"""Joint linear-plus-neural outcome baseline: Y = X beta + f(X, E) + error.

Training and model selection use observed X, Y, E only. The linear coefficient
is scored against the simulation target after fitting, exactly as other slopes.
The nuisance can contain linear effects; no causal identification restriction is
imposed on this predictive baseline. Its fitted decomposition depends on training.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ovr._core import (
    _MLP, _stratified_split_indices, _grouped_stratified_split_indices,
)


class _LinearNeuralNet(nn.Module):
    def __init__(self, d, environments, hidden, depth, initial):
        super().__init__()
        self.beta = nn.Parameter(torch.as_tensor(initial, dtype=torch.float32).clone())
        self.nuisance = _MLP(d+environments, 1, hidden=hidden, depth=depth)

    def forward(self, X, environment_indicators):
        return X @ self.beta+self.nuisance(torch.cat([X, environment_indicators], dim=1)).squeeze(1)


class LinearNeuralRegression:
    """Jointly train a shared linear slope and an MLP of X and one-hot E.

    A group-disjoint (when groups are supplied) validation split selects the
    network weight decay, initialization, and number of epochs by outcome MSE.
    The selected configuration is then refit on every supplied training row.
    Standardization and FE initialization use only each fit's training rows.
    The common linear slope is unpenalized; only neural parameters get decay.
    """

    def __init__(self, *, hidden=64, depth=2, epochs=200, lr=1e-3,
                 batch_size=1024, weight_decays=(0., 1e-4, 1e-2),
                 initializations=("fe", "zero"), validation_fraction=.2,
                 patience=30, seed=0):
        if min(hidden, depth, epochs, batch_size, patience) < 1 or lr <= 0:
            raise ValueError("Network and training settings must be positive.")
        if not weight_decays or any(x < 0 or not np.isfinite(x) for x in weight_decays):
            raise ValueError("Weight decays must be finite and nonnegative.")
        if not initializations or set(initializations)-{"fe", "zero"}:
            raise ValueError("Initialization must be 'fe' or 'zero'.")
        if not 0 < validation_fraction < .5:
            raise ValueError("validation_fraction must be between zero and one half.")
        self.hidden, self.depth, self.epochs = hidden, depth, epochs
        self.lr, self.batch_size, self.weight_decays = lr, batch_size, tuple(weight_decays)
        self.initializations, self.validation_fraction = tuple(initializations), validation_fraction
        self.patience, self.seed = patience, seed

    @staticmethod
    def _arrays(X, Y, E):
        X, Y, E = np.asarray(X, float), np.asarray(Y, float).ravel(), np.asarray(E).ravel()
        if X.ndim != 2 or len(X) != len(Y) or len(E) != len(Y) or not len(Y):
            raise ValueError("X, Y, and E must have matching, nonempty rows.")
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError("X and Y must be finite.")
        if E.dtype.kind in 'fc' and not np.isfinite(E).all():
            raise ValueError("Environment labels must be finite.")
        return X, Y, E

    def _train(self, X, Y, E, *, decay, initialization, epochs, validation=None):
        xmu, xsd = X.mean(0), np.maximum(X.std(0), 1e-8)
        ymu, ysd = float(Y.mean()), max(float(Y.std()), 1e-8)
        xs, ys = (X-xmu)/xsd, (Y-ymu)/ysd
        indicators = (E[:, None] == self.environments_[None, :]).astype(np.float32)
        initial = np.zeros(X.shape[1])
        if initialization == "fe":
            rx, ry = xs.copy(), ys.copy()
            for e in self.environments_:
                mask = E == e
                rx[mask] -= rx[mask].mean(0)
                ry[mask] -= ry[mask].mean()
            initial = np.linalg.lstsq(rx, ry, rcond=None)[0]
        xt, yt, et = map(lambda a: torch.tensor(a, dtype=torch.float32), (xs, ys, indicators))
        if validation is not None:
            vx, vy, ve = validation
            vxt = torch.tensor((vx-xmu)/xsd, dtype=torch.float32)
            vet = torch.tensor(ve[:, None] == self.environments_[None, :], dtype=torch.float32)
        # Identical starting network and minibatches across validation candidates.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            model = _LinearNeuralNet(X.shape[1], len(self.environments_), self.hidden, self.depth, initial)
            optimizer = torch.optim.Adam([
                {"params": [model.beta], "weight_decay": 0.},
                {"params": model.nuisance.parameters(), "weight_decay": decay},
            ], lr=self.lr)
            generator = torch.Generator().manual_seed(self.seed+1)
            best_loss, best_epoch, best_state = float("inf"), 0, None
            for epoch in range(1, epochs+1):
                model.train()
                order = torch.randperm(len(X), generator=generator)
                for start in range(0, len(X), self.batch_size):
                    ix = order[start:start+self.batch_size]
                    optimizer.zero_grad()
                    loss = torch.mean((model(xt[ix], et[ix])-yt[ix])**2)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss.")
                    loss.backward()
                    optimizer.step()
                if validation is not None:
                    model.eval()
                    with torch.no_grad():
                        prediction = model(vxt, vet).numpy()*ysd+ymu
                    val_loss = float(np.mean((prediction-vy)**2))
                    if val_loss < best_loss:
                        best_loss, best_epoch = val_loss, epoch
                        best_state = copy.deepcopy(model.state_dict())
                    if epoch-best_epoch >= self.patience:
                        break
            if validation is not None:
                if best_state is None:
                    raise FloatingPointError("No finite validation checkpoint.")
                model.load_state_dict(best_state)
            else:
                best_epoch = epochs
            model.eval()
        return model, (xmu, xsd, ymu, ysd), best_loss, best_epoch, epoch

    def fit(self, X, Y, E, groups=None):
        X, Y, E = self._arrays(X, Y, E)
        self.environments_ = np.unique(E)
        if groups is not None and len(groups) != len(Y):
            raise ValueError("groups must match the data rows.")
        rng = np.random.default_rng(self.seed+7000)
        if groups is None:
            training, validation = _stratified_split_indices(E, 1-self.validation_fraction, rng)
        else:
            training, validation = _grouped_stratified_split_indices(
                E, groups, 1-self.validation_fraction, rng)
        self.training_indices_, self.validation_indices_ = training, validation
        self.validation_results_ = []
        best = None
        for initialization in self.initializations:
            for decay in self.weight_decays:
                _, _, loss, epoch, trained = self._train(
                    X[training], Y[training], E[training], decay=decay, initialization=initialization,
                    epochs=self.epochs, validation=(X[validation], Y[validation], E[validation]))
                candidate = dict(initialization=initialization, weight_decay=decay,
                                 validation_mse=loss, selected_epochs=epoch, trained_epochs=trained)
                self.validation_results_.append(candidate)
                if best is None or loss < best["validation_mse"]:
                    best = candidate
        self.selected_ = best.copy()
        self.model_, self.scaling_, _, _, _ = self._train(
            X, Y, E, decay=best["weight_decay"], initialization=best["initialization"],
            epochs=best["selected_epochs"])
        _, xsd, _, ysd = self.scaling_
        self.beta_ = self.model_.beta.detach().numpy().astype(float)*ysd/xsd
        self.diagnostics_ = dict(status="ok", selected_initialization=best["initialization"],
                                 selected_weight_decay=best["weight_decay"], selected_epochs=best["selected_epochs"],
                                 validation_mse=best["validation_mse"], n_training=len(training),
                                 n_validation=len(validation), n_final_training=len(X),
                                 n_selection_rows=len(X),
                                 validation_candidates=self.validation_results_, coefficient_identified=False,
                                 model="Y = beta'X + MLP(X, one-hot(E))", selection_target="validation outcome MSE")
        return self

    def refit(self, X, Y, E):
        """Refit selected settings on new training rows, without retuning.

        Useful after an independent prediction evaluation: the benchmark saves
        those predictions first, then refits on all n rows for coefficient scoring.
        Validation indices continue to refer to the original selection sample.
        """
        if not hasattr(self, "selected_"):
            raise AttributeError("Select settings with fit before refitting.")
        X, Y, E = self._arrays(X, Y, E)
        if X.shape[1] != len(self.coef_) or not np.array_equal(np.unique(E), self.environments_):
            raise ValueError("Refitting must retain the exposure dimension and environments.")
        best = self.selected_
        self.model_, self.scaling_, _, _, _ = self._train(
            X, Y, E, decay=best["weight_decay"], initialization=best["initialization"],
            epochs=best["selected_epochs"])
        _, xsd, _, ysd = self.scaling_
        self.beta_ = self.model_.beta.detach().numpy().astype(float)*ysd/xsd
        self.diagnostics_["n_final_training"] = len(X)
        return self

    @property
    def coef_(self):
        if not hasattr(self, "beta_"):
            raise AttributeError("Fit the model first.")
        return self.beta_

    def predict(self, X, E):
        X, E = np.asarray(X, float), np.asarray(E).ravel()
        if X.ndim != 2 or X.shape[1] != len(self.coef_) or len(E) != len(X) or not np.isfinite(X).all():
            raise ValueError("Prediction dimensions or values are invalid.")
        if not np.isin(E, self.environments_).all():
            raise ValueError("Cannot predict for an unobserved environment.")
        xmu, xsd, ymu, ysd = self.scaling_
        xs = torch.tensor((X-xmu)/xsd, dtype=torch.float32)
        indicators = torch.tensor(E[:, None] == self.environments_[None, :], dtype=torch.float32)
        with torch.no_grad():
            return self.model_(xs, indicators).numpy().astype(float)*ysd+ymu


__all__ = ["LinearNeuralRegression"]
