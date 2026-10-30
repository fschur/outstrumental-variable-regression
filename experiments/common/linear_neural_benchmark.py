"""Matched coefficient and independently held-out prediction evaluation."""

import time

import numpy as np

from .baselines import OLS, EnvironmentFixedEffects, LinearNeuralRegression
from .results import estimate_row, array_hash
from ovr._core import _stratified_split_indices, _grouped_stratified_split_indices


def _linear_prediction(fit, X, Y, E, test_X, test_E, fixed_effects):
    residual = Y-X @ fit.coef_
    if fixed_effects:
        intercepts = {e: residual[E == e].mean() for e in np.unique(E)}
        return test_X @ fit.coef_+np.array([intercepts[e] for e in test_E])
    return test_X @ fit.coef_+residual.mean()


def compare_linear_neural(X, Y, E, target, *, groups=None, seed=0, epochs=200,
                          hidden=64, batch_size=1024, weight_decays=(0., 1e-4, 1e-2),
                          initializations=("fe", "zero"), patience=30):
    """Target is read only when scoring the final coefficients.

    Prediction: reserve 20% of rows (or entire groups); select hyperparameters
    inside the other 80%. Compare all predictors on exactly the reserved rows.
    Coefficients: refit the selected neural model on all n rows, matching the
    paper's OLS/FE coefficient protocol. Test outcomes never select settings.
    """
    X, Y, E = np.asarray(X, float), np.asarray(Y, float), np.asarray(E)
    rng = np.random.default_rng(seed+91000)
    if groups is None:
        train, test = _stratified_split_indices(E, .8, rng)
    else:
        train, test = _grouped_stratified_split_indices(E, groups, .8, rng)
    metadata = dict(source="Baseline", data_hash=array_hash(X, Y, E),
                    prediction_split_hash=array_hash(train, test),
                    prediction_training_n=len(train), prediction_test_n=len(test),
                    coefficient_training_n=len(X), oracle_inputs=False)
    rows = []
    for label, cls in (("OLS", OLS), ("Environment FE", EnvironmentFixedEffects)):
        prediction_fit = cls().fit(X[train], Y[train], E[train])
        pred = _linear_prediction(prediction_fit, X[train], Y[train], E[train], X[test], E[test], label == "Environment FE")
        full = cls().fit(X, Y, E)
        rows.append(estimate_row(label, full.coef_, target, dict(status="ok"), **metadata,
                                 prediction_mse=float(np.mean((pred-Y[test])**2))))
    started = time.perf_counter()
    fit = LinearNeuralRegression(hidden=hidden, epochs=epochs, batch_size=batch_size,
                                 weight_decays=weight_decays, initializations=initializations,
                                 patience=patience, seed=seed)
    fit.fit(X[train], Y[train], E[train], None if groups is None else np.asarray(groups)[train])
    # Preserve outer-test predictions before using all n rows for the slope fit.
    pred = fit.predict(X[test], E[test])
    prediction_beta = fit.coef_.copy()
    selection_train, selection_validation = train[fit.training_indices_], train[fit.validation_indices_]
    fit.refit(X, Y, E)
    diagnostics = dict(fit.diagnostics_,
                       outer_training_indices=train.tolist(), outer_test_indices=test.tolist(),
                       selection_training_indices=selection_train.tolist(),
                       selection_validation_indices=selection_validation.tolist(),
                       prediction_beta=prediction_beta.tolist())
    row = estimate_row("LNN", fit.coef_, target, diagnostics, **metadata,
                       fit_seconds=time.perf_counter()-started,
                       prediction_mse=float(np.mean((pred-Y[test])**2)))
    rows.append(row)
    fe_mse = rows[1]["prediction_mse"]
    for row in rows:
        row["prediction_mse_vs_fe"] = row["prediction_mse"]/max(fe_mse, 1e-20)
        row["prediction_normalized_mse"] = row["prediction_mse"]/max(float(np.var(Y[test])), 1e-20)
    return rows
