"""A fourth-cumulant specialization of Kivva et al. (2025), Theorem 3.4.

This is an independently derived implementation, not the authors' software or
a port of their general moment-order selection algorithm. Primary source:
https://arxiv.org/html/2506.11756v1#S3.SS3

After centering within each environment and absorbing the treatment loading
into U, the sufficient model is

    X = U + V;  Y_e = beta*X + gamma_e*U + W_e.

U, V, W_e are mutually independent, U and V have invariant distributions,
W_e has invariant variance, and gamma_0 != gamma_1. We additionally require
kappa_4(V) != 0 and gamma_0 + gamma_1 != 0. Theorem 3.4 assumes the stronger
invariance of the entire outcome-noise distribution. Here fourth moments
suffice for the displayed algebra; no latent values enter the estimator.

Let r = (Var(Y_1)-Var(Y_0))/(Cov(X_1,Y_1)-Cov(X_0,Y_0)), a=gamma_1-gamma_0,
and b=gamma_0+gamma_1. Then r=2*beta+b and Z_e=r*X_e-2*Y_e equals
a*U+b*V-2*W_0 for e=0 and -a*U+b*V-2*W_1 for e=1. Consequently,

    A_e = cum(X_e,X_e,X_e,Z_e) = (+/-a)*kappa_4(U)+b*kappa_4(V),
    B_e = cum(Z_e,Z_e,Z_e,X_e) = (+/-a^3)*kappa_4(U)+b^3*kappa_4(V).

Thus (B_0+B_1)/(A_0+A_1)=b^2, and sign(b) is the sign of
Cov(X_0,Z_0)+Cov(X_1,Z_1). Return beta=(r-b)/2. This uses the sum branch
directly, which is available for the prespecified discrete-uniform treatment
noise. It does not estimate a non-Gaussian moment order from observed outcomes.

The optical experiment only approximately satisfies the physical assumptions:
heteroskedastic sensor noise, outcome curvature, or simultaneous changes can
bias this estimator. Finite-sample invalid moment ratios are reported as
failures, not clipped, replaced, or selected using the randomized reference.

Run `python .../kivva_baseline.py --self-test` for exact finite-support checks
and a repeated-sampling check. A CSV can be scored with --csv and --out.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np


METHOD_NAME = "Kivva gamma-shift (fourth cumulants)"


def _cum31(a, b):
    """Joint fourth cumulant of centered a,a,a,b."""
    return float(np.mean(a ** 3 * b) - 3 * np.mean(a ** 2) * np.mean(a * b))


def estimate_gamma_shift_fourth(x, y, environment):
    """Estimate the raw-unit slope from only X,Y,E; return failures explicitly.

    Result keys are method, status, coefficient, diagnostics, and reason on
    failure. There is no confidence interval: acquisition-block resampling
    and independent-reference uncertainty belong in the calling analysis.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    environment = np.asarray(environment).reshape(-1)
    result = {"method": METHOD_NAME, "status": "failed", "coefficient": None,
              "diagnostics": {}}

    def fail(reason):
        result["reason"] = reason
        return result

    if not (len(x) == len(y) == len(environment)):
        return fail("X, Y, E lengths differ.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return fail("Non-finite X or Y.")
    labels = np.unique(environment)
    if len(labels) != 2:
        return fail("Exactly two environments are required.")
    xs, ys = [], []
    for label in labels:
        mask = environment == label
        if mask.sum() < 5:
            return fail("Each environment needs at least five observations.")
        xs.append(x[mask] - x[mask].mean())
        ys.append(y[mask] - y[mask].mean())
    # One common scale for each variable; never separately scale environments.
    xscale = float(np.sqrt(np.mean([np.mean(a * a) for a in xs])))
    yscale = float(np.sqrt(np.mean([np.mean(a * a) for a in ys])))
    if xscale <= 0 or yscale <= 0:
        return fail("X or Y has zero within-environment variance.")
    xs = [a / xscale for a in xs]
    ys = [a / yscale for a in ys]
    varx = np.array([np.mean(a * a) for a in xs])
    vary = np.array([np.mean(a * a) for a in ys])
    covxy = np.array([np.mean(a * b) for a, b in zip(xs, ys)])
    covariance_shift = float(covxy[1] - covxy[0])
    variance_shift = float(vary[1] - vary[0])
    diagnostics = result["diagnostics"]
    diagnostics.update(environment_labels=labels.tolist(), n=[len(a) for a in xs],
                       x_scale=xscale, y_scale=yscale,
                       standardized_treatment_variances=varx.tolist(),
                       standardized_treatment_variance_difference=float(varx[1] - varx[0]),
                       standardized_outcome_variance_shift=variance_shift,
                       standardized_treatment_outcome_covariance_shift=covariance_shift,
                       required_noise_cumulant_order=4,
                       interval="None; point estimate only.")
    tolerance = 1e-12  # Numerical singularity guard, not a fitted statistical threshold.
    if abs(covariance_shift) <= tolerance:
        return fail("Treatment-outcome covariance shift is numerically zero.")
    r = variance_shift / covariance_shift
    zs = [r * a - 2 * b for a, b in zip(xs, ys)]
    As = np.array([_cum31(a, z) for a, z in zip(xs, zs)])
    Bs = np.array([_cum31(z, a) for a, z in zip(xs, zs)])
    covxz_sum = float(sum(np.mean(a * z) for a, z in zip(xs, zs)))
    asum, bsum = float(As.sum()), float(Bs.sum())
    diagnostics.update(standardized_r=r, cumulant_X3Z_by_environment=As.tolist(),
                       cumulant_Z3X_by_environment=Bs.tolist(),
                       cumulant_X3Z_sum=asum, cumulant_Z3X_sum=bsum,
                       covariance_XZ_sum=covxz_sum)
    if not np.isfinite([r, asum, bsum, covxz_sum]).all():
        return fail("Non-finite moment calculation.")
    if abs(asum) <= tolerance * max(1.0, float(np.max(np.abs(As)))):
        return fail("Fourth-cumulant sum is numerically zero; this specialization is unidentified.")
    if abs(covxz_sum) <= tolerance:
        return fail("Cannot determine the sign of the summed confounder loading.")
    ratio = bsum / asum
    diagnostics["estimated_squared_loading_sum"] = ratio
    if ratio <= 0:
        return fail("Estimated squared-loading ratio is nonpositive.")
    b = float(np.sign(covxz_sum) * np.sqrt(ratio))
    coefficient = float((r - b) * yscale / (2 * xscale))
    if not np.isfinite(coefficient):
        return fail("Non-finite coefficient.")
    result.update(status="ok", coefficient=coefficient)
    diagnostics["standardized_loading_sum"] = b
    return result


def _exact_data(beta, alpha, gamma, latent, treatment_noise, outcome_noise,
                outcome_noise_scale=(1.0, 1.0), latent_scale=(1.0, 1.0),
                treatment_noise_scale=(1.0, 1.0)):
    """Equal-weight finite-support population; repeated values encode masses."""
    rows = []
    for e in range(2):
        for u, v, w in product(latent, treatment_noise, outcome_noise):
            u = u * latent_scale[e]
            x = 16 + alpha * u + v * treatment_noise_scale[e]
            y = beta * x + gamma[e] * u + outcome_noise_scale[e] * w + 100 * e
            rows.append((x, y, e))
    return np.asarray(rows)


def self_test():
    """Check identities, sign handling, sampling, and known assumption failures."""
    cases = [
        dict(beta=83.0, alpha=4.0, gamma=(954.0, 220.0), latent=(-.5, .5),
             treatment_noise=tuple(range(-6, 7)), outcome_noise=(-3., 3.)),
        dict(beta=-1.3, alpha=-2.0, gamma=(.8, 2.5), latent=(-1., 0., 2.),
             treatment_noise=(-3., -1., 0., 4.), outcome_noise=(-.7, .7)),
        dict(beta=.7, alpha=1.0, gamma=(-3.0, -1.0), latent=(-1., 1.),
             treatment_noise=tuple(range(-4, 5)), outcome_noise=(-.3, .3)),
    ]
    exact = []
    for parameters in cases:
        rows = _exact_data(**parameters)
        result = estimate_gamma_shift_fourth(*rows.T)
        assert result["status"] == "ok", result
        assert abs(result["coefficient"] - parameters["beta"]) < 1e-9, result
        exact.append(dict(beta=parameters["beta"], estimate=result["coefficient"]))
    # Environment order and affine outcome/treatment units cannot change the
    # underlying raw-unit coefficient except by their specified scale factor.
    rows = _exact_data(**cases[0])
    transformed = estimate_gamma_shift_fourth(rows[:, 0] * 3 + 19,
                                               rows[:, 1] * 2 - 31, 1 - rows[:, 2])
    assert abs(transformed["coefficient"] - 83 * 2 / 3) < 1e-9, transformed
    equal = dict(cases[0], gamma=(500., 500.))
    failed = estimate_gamma_shift_fourth(*_exact_data(**equal).T)
    assert failed["status"] == "failed", failed
    zero_fourth = dict(cases[0], treatment_noise=(0., 0., 0., 0., -np.sqrt(3.), np.sqrt(3.)))
    failed_fourth = estimate_gamma_shift_fourth(*_exact_data(**zero_fourth).T)
    assert failed_fourth["status"] == "failed", failed_fourth
    zero_sum = dict(cases[0], gamma=(500., -500.))
    failed_sum = estimate_gamma_shift_fourth(*_exact_data(**zero_sum).T)
    assert failed_sum["status"] == "failed", failed_sum

    # Violating invariance need not trigger a numerical failure. Record that a
    # finite coefficient is not a certificate of valid physical assumptions.
    violations = {}
    for name, changes in {
        "changed_outcome_noise_variance": {"outcome_noise_scale": (1., 80.)},
        "changed_latent_distribution": {"latent_scale": (1., 2.)},
        "changed_treatment_noise_variance": {"treatment_noise_scale": (1., 2.)},
        "changed_latent_and_treatment_noise": {"latent_scale": (1., 2.), "treatment_noise_scale": (1., 2.)},
    }.items():
        result = estimate_gamma_shift_fourth(*_exact_data(**cases[0], **changes).T)
        assert result["status"] == "failed" or abs(result["coefficient"] - 83) > .1
        violations[name] = dict(status=result["status"], coefficient=result["coefficient"])

    # Eight independent samples at substantial n verify sample behavior while
    # exact population checks above establish algebra without Monte Carlo luck.
    estimates = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        pieces = []
        for e, gamma in enumerate((3., 1.)):
            n = 60000
            h = rng.choice([-.5, .5], size=n)
            v = rng.integers(-6, 7, size=n)
            x = 4 * h + v
            y = .8 * x + gamma * h + rng.normal(0, .25, size=n)
            pieces.append(np.column_stack([x, y, np.full(n, e)]))
        rows = np.vstack(pieces)
        result = estimate_gamma_shift_fourth(*rows.T)
        assert result["status"] == "ok", result
        estimates.append(result["coefficient"])
    assert np.mean(np.abs(np.asarray(estimates) - .8)) < .06, estimates
    return {"exact_population_cases": exact, "affine_and_environment_relabeling": "passed",
            "zero_shift_failure": "passed", "assumption_violation_examples": violations,
            "zero_fourth_treatment_cumulant_failure": "passed", "zero_loading_sum_failure": "passed",
            "sampling_n_per_environment": 60000, "sampling_repetitions": len(estimates),
            "sampling_estimates": estimates,
            "sampling_mean_absolute_error": float(np.mean(np.abs(np.asarray(estimates) - .8)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.self_test:
        output = self_test()
    elif args.csv:
        import pandas as pd
        data = pd.read_csv(args.csv)
        if "kind" in data:
            data = data.loc[data.kind.eq("confounded")]
        output = estimate_gamma_shift_fourth(data.X, data.Y, data.E)
    else:
        parser.error("Supply --self-test or --csv.")
    rendered = json.dumps(output, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
