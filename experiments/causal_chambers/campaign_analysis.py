"""Offline physical-campaign analysis using the existing paper estimators.

Example (run from the repository root)::

    PYTHONPATH=src python -m experiments.causal_chambers.campaign_analysis \
      --data RUN1/measurements.csv RUN2/measurements.csv \
      --calibration-data FULL_GRID/measurements.csv --output-dir RESULTS

Add completed acquisitions to --data and rerun: matching per-fit checkpoints
are reused. No hardware/API access, estimator replacement, or reference tuning.
See campaign_analysis_audit.txt for the exact configuration and interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from scipy.stats import t as student_t
import torch

from ovr import DirectOVR, SmoothedOVR
from ovr._core import _grouped_stratified_split_indices
from experiments.common.baselines import OLS, EnvironmentFixedEffects, AnchorRegression
from experiments.common.config import paper_config_section
from experiments.common.linear_neural_benchmark import compare_linear_neural
from experiments.common.results import clean_json
from .analyze_ovr import linear_fit, oracle_p_with_same_second_stage

ROOT = Path(__file__).resolve().parents[2]
METHODS = ["OLS", "Environment FE", "Anchor-g2", "LNN", "OLS E=0", "OLS E=1",
           "DOVR", "SOVR", "Oracle H", "Oracle-score DML", "Oracle-score linear nuisance"]
CORE_FILES = ["src/ovr/direct.py", "src/ovr/smoothed.py", "src/ovr/_core.py",
              "experiments/common/baselines.py", "experiments/common/linear_neural.py",
              "experiments/common/linear_neural_benchmark.py", "configs/paper.json",
              "experiments/causal_chambers/analyze_ovr.py",
              "experiments/causal_chambers/campaign_analysis.py"]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(clean_json(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, sort_keys=True,
                                   allow_nan=False)+"\n")
    temporary.replace(path)


def source_record():
    paths = list(CORE_FILES)
    if (ROOT/"experiments/causal_chambers/kivva_baseline.py").exists():
        paths.append("experiments/causal_chambers/kivva_baseline.py")
    current = {name: file_hash(ROOT/name) for name in paths}
    comparison = {}
    for name in paths:
        submitted = ROOT/"context/anonymous_submitted_code"/name
        if submitted.exists():
            comparison[name] = dict(sha256=file_hash(submitted),
                                    byte_identical=current[name] == file_hash(submitted))
    packages = {name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scipy", "torch", "matplotlib")}
    return dict(current_sha256=current, submitted=comparison, packages=packages)


def paper_settings():
    synthetic, lnn = paper_config_section("synthetic"), paper_config_section("lnn")
    common = dict(epochs=synthetic["contrast_epochs"],
                  resid_epochs=synthetic["residual_epochs"], hidden=64)
    defaults = {cls.__name__: {name: parameter.default
                for name, parameter in inspect.signature(cls).parameters.items()}
                for cls in (DirectOVR, SmoothedOVR)}
    return dict(protocol="existing synthetic-paper protocol", common=common,
                lnn=lnn, source_defaults=defaults, anchor_gamma=2)


def assignment_law(path, manifest, design, center, scale):
    """Read the frozen physical assignment table; derive only the original binary law."""
    table_path = path.parent/"assignment_law.csv"
    if table_path.exists():
        digest = file_hash(table_path)
        if manifest.get("assignment_sha256", digest) != digest:
            raise ValueError(f"Assignment table hash mismatch: {table_path}")
        law = pd.read_csv(table_path)
        if law.duplicated(["X", "H"]).any():
            raise ValueError("Duplicate X,H entries in frozen assignment table")
        pivot = law.pivot(index="X", columns="H", values="probability").sort_index()
        if pivot.isna().any().any():
            raise ValueError("Assignment table must include zero-probability cells explicitly")
        xs, hs, joint = pivot.index.to_numpy(float), pivot.columns.to_numpy(int), pivot.to_numpy(float)
        provenance = dict(path=str(table_path.resolve()), sha256=digest)
    elif design == "binary":
        xs, hs = np.arange(center-scale, center+scale+1, dtype=float), np.array([0, 1])
        k = np.abs(xs-center)
        lower, upper = np.maximum(0, (k-.5)/scale), np.minimum(1, (k+.5)/scale)
        masses = np.column_stack([beta_distribution.cdf(upper, a, b)-beta_distribution.cdf(lower, a, b)
                                  for a, b in ((2, 5), (5, 2))])
        sign = np.where(xs > center, .8, np.where(xs < center, .2, 1.))
        joint = .5*masses*sign[:, None]
        provenance = dict(formula="original rounded Beta(2,5)/Beta(5,2), H prior .5, sign probability .8")
    else:
        raise ValueError(f"{design} requires its frozen assignment_law.csv")
    if not np.isfinite(joint).all() or np.min(joint) < 0 or not np.isclose(joint.sum(), 1):
        raise ValueError("Invalid frozen joint assignment law")
    if not np.array_equal(hs, np.arange(len(hs))):
        raise ValueError("H labels must be consecutive integers starting at zero")
    px, ph = joint.sum(axis=1), joint.sum(axis=0)
    if np.min(px) <= 0 or np.min(ph) <= 0:
        raise ValueError("The frozen support must have positive X and H marginals")
    posterior = joint/px[:, None]
    coordinates = np.array([-1., float(manifest["optical_midpoint"]), 1.]) if design == "ternary" else hs.astype(float)
    score = posterior@coordinates
    # Rounding removes floating arithmetic differences in mathematically equal scores.
    score_groups = np.round(score, 12)
    conditional_x = np.array([np.sum(xs[score_groups == s]*px[score_groups == s]) /
                              np.sum(px[score_groups == s]) for s in score_groups])
    residual = xs-conditional_x
    variance = float(np.sum(px*residual**2))
    if variance <= 0:
        raise ValueError("The frozen oracle score has no treatment residual variation")
    reds = manifest.get("red_levels", [32, 192] if len(hs) == 2 else [32, 112, 192])
    if len(reds) != len(hs):
        raise ValueError("Red-level metadata and assignment H support disagree")
    return dict(design=design, xs=xs, hs=hs, joint=joint, px=px, ph=ph,
                posterior=posterior, score=score, coordinates=coordinates,
                conditional_x=conditional_x, residual=residual, residual_variance=variance,
                red_levels=reds, angles=manifest.get("angles", [0., 60.]),
                fixed_controls=manifest.get("fixed_controls", {}), provenance=provenance,
                score_interpretation="calibration-conditioned optical-mean score under the working rank-one model"
                if design == "ternary" else "known assignment posterior P(H=1|X)")


def load_inputs(paths, requested_design, center, scale):
    acquisitions, seen_ids, seen_files = [], set(), set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in seen_files:
            raise ValueError(f"Repeated data file: {path}")
        seen_files.add(path)
        manifest_path = path.parent/"manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        digest = file_hash(path)
        if manifest.get("measurements_sha256", digest) != digest:
            raise ValueError(f"Measurement hash mismatch: {path}")
        schedule_path = path.parent/"schedule.csv"
        if schedule_path.exists() and manifest.get("schedule_sha256", file_hash(schedule_path)) != file_hash(schedule_path):
            raise ValueError(f"Schedule hash mismatch: {schedule_path}")
        frame = pd.read_csv(path)
        rows_other_angles = 0
        if len(manifest.get("angles", [0., 60.])) > 2:
            # Binary runs of the November 10 campaign also measure 20 and 40 degrees
            # (E=2,3). The main analysis uses only the first two angles, 0 and 60 degrees.
            keep = frame.E.isin([0, 1])
            rows_other_angles = int((~keep).sum())
            frame = frame.loc[keep].reset_index(drop=True)
        required = {"X", "Y", "E", "H", "kind", "block_id", "flag"}
        if not required.issubset(frame) or frame[list(required)].isna().any().any():
            raise ValueError(f"Missing required columns or values in {path}")
        if not frame.flag.is_unique or not np.isfinite(frame[["X", "Y", "E", "H"]].to_numpy(float)).all():
            raise ValueError(f"Repeated flags or nonfinite analysis values in {path}")
        if set(frame.kind) != {"confounded", "reference"} or set(frame.E) != {0, 1}:
            raise ValueError("Each acquisition requires both kinds and two environments")
        if not np.array_equal(frame[["X", "E", "H"]], frame[["X", "E", "H"]].astype(int)):
            raise ValueError("X,E,H must be integers")
        for field, value in (("center", center), ("scale", scale)):
            if field in manifest and manifest[field] != value:
                raise ValueError(f"Frozen {field} differs from requested analysis value")
        if "led_3_ir" in frame and not np.allclose(frame.X, frame.led_3_ir, atol=1e-8, rtol=0):
            raise ValueError("X is not the recorded LED command")
        if "ir_3" in frame and not np.allclose(frame.Y, frame.ir_3, atol=1e-8, rtol=0):
            raise ValueError("Y is not the raw physical sensor outcome")
        frame["E"], frame["H"] = frame.E.astype(int), frame.H.astype(int)
        design = manifest.get("design", "binary") if requested_design == "auto" else requested_design
        if "design" in manifest and design != manifest["design"]:
            raise ValueError("Requested design disagrees with frozen manifest")
        law = assignment_law(path, manifest, design, center, scale)
        if rows_other_angles:
            law["angles"] = list(law["angles"][:2])
        if len(law["angles"]) != 2:
            raise ValueError("The campaign estimator expects exactly two physical environment angles")
        if "pol_2" in frame and not np.allclose(frame.pol_2, np.asarray(law["angles"])[frame.E], atol=1e-5, rtol=0):
            raise ValueError("Environment labels disagree with the frozen physical polarizer angles")
        if not set(frame.X).issubset(law["xs"]) or not set(frame.H).issubset(law["hs"]):
            raise ValueError("Measurements fall outside the frozen assignment support")
        if "red" in frame and not np.allclose(frame.red, np.asarray(law["red_levels"])[frame.H], atol=1e-5, rtol=0):
            raise ValueError("Hidden-state labels disagree with the recorded red-source command")
        if "replicate_id" in frame:
            ids = frame.replicate_id.astype(str).unique()
            if len(ids) != 1:
                raise ValueError("Each acquisition CSV must contain one replicate_id")
            rep = ids[0]
        else:
            rep = str(manifest.get("replicate_id", manifest.get("rep_id", path.parent.name)))
        unique_id = design+"/"+rep
        if unique_id in seen_ids:
            raise ValueError(f"Repeated physical replicate identifier: {unique_id}")
        seen_ids.add(unique_id)
        frame["block_id"] = frame.block_id.astype(str)
        frame["row_uid"] = [unique_id+":"+str(v) for v in frame.flag]
        for kind, part in frame.groupby("kind"):
            counts = part.groupby(["block_id", "E"]).size().unstack(fill_value=0)
            if not (counts[0] == counts[1]).all() or (counts[0] == 0).any():
                raise ValueError(f"Every {kind} block must contain balanced observations from both environments")
        acquisition = dict(replicate_id=rep, design=design, frame=frame, law=law,
                           manifest=manifest, provenance=dict(path=str(path), measurements_sha256=digest,
                           manifest_sha256=file_hash(manifest_path) if manifest_path.exists() else None,
                           experiment_id=manifest.get("experiment_id", manifest.get("experiment_ids")),
                           **(dict(rows_other_angles=rows_other_angles) if rows_other_angles else {})))
        acquisitions.append(acquisition)
    return acquisitions


def nested_subsets(frame, sizes, subset_seed, rep):
    training = frame.loc[frame.kind.eq("confounded")].copy()
    counts = training.groupby(["block_id", "E"]).size().unstack(fill_value=0)
    if counts[0].nunique() != 1:
        raise ValueError("Whole-block sample sizes require equal confounded counts per block")
    per_block = int(counts[0].iloc[0])
    stable_rep_seed = int(hashlib.sha256(rep.encode()).hexdigest()[:8], 16)
    seed = (subset_seed+stable_rep_seed) % (2**32)
    order = np.random.default_rng(seed).permutation(sorted(counts.index)).tolist()
    subsets = []
    for n in sizes:
        if n > int(counts[0].sum()):
            continue
        if n % per_block:
            raise ValueError(f"n/E={n} cannot be formed from complete {per_block}-row/environment blocks")
        selected = order[:n//per_block]
        if len(selected) < 8:
            raise ValueError("Nested subset has fewer than eight blocks for grouped OVR")
        part = training.loc[training.block_id.isin(selected)].reset_index(drop=True)
        if part.groupby("E").size().tolist() != [n, n]:
            raise RuntimeError("Nested sample-size selection is inconsistent")
        subsets.append((n, part, dict(subset_seed=seed, whole_block_order=order,
                                     selected_blocks=selected, rows_per_environment_per_block=per_block,
                                     selected_row_sha256=json_hash(part.row_uid.tolist()))))
    return subsets


def block_summary(frame, indices):
    part = frame.iloc[np.asarray(indices, dtype=int)]
    return dict(n_rows=len(part), blocks=sorted(part.block_id.unique().tolist()),
                row_sha256=json_hash(part.row_uid.tolist()))


def assert_group_disjoint(frame, a, b):
    if set(frame.iloc[a].block_id) & set(frame.iloc[b].block_id):
        raise ValueError("Estimator split reuses acquisition blocks")


def audit_ovr(model, frame):
    contrast, estimation = model.contrast_indices_, model.estimation_indices_
    assert_group_disjoint(frame, contrast, estimation)
    if set(contrast) | set(estimation) != set(range(len(frame))):
        raise ValueError("OVR outer split does not partition the input")
    folds = model.second_stage_folds_
    if sorted(np.concatenate(folds).tolist()) != sorted(estimation.tolist()):
        raise ValueError("Residualization folds do not partition the coefficient sample")
    for j, fold in enumerate(folds):
        for other in folds[:j]:
            assert_group_disjoint(frame, fold, other)
    audit = dict(contrast=block_summary(frame, contrast), estimation=block_summary(frame, estimation),
                 residualization_folds=[block_summary(frame, fold) for fold in folds], group_overlap=0)
    if isinstance(model, SmoothedOVR):
        members = [model.contrast_indices_a_, model.contrast_indices_b_]
        assert_group_disjoint(frame, *members)
        if sorted(np.concatenate(members).tolist()) != sorted(contrast.tolist()):
            raise ValueError("Smoothed members do not partition the contrast sample")
        audit["members"] = []
        for j, (indices, network) in enumerate(zip(members, model.contrast_models_)):
            train, validation = _grouped_stratified_split_indices(
                frame.E.to_numpy()[indices], frame.block_id.to_numpy()[indices],
                1-model.contrast_validation_fraction, np.random.default_rng(model.seed+1000*(j+1)+17))
            ti, vi = indices[train], indices[validation]
            assert_group_disjoint(frame, ti, vi)
            if len(ti) != network._member_training_size or len(vi) != network._member_validation_size:
                raise ValueError("Replayed validation split disagrees with fitted member sizes")
            np.testing.assert_allclose(network._xmu.numpy(), frame.iloc[ti][["X"]].mean().to_numpy(), rtol=1e-6)
            np.testing.assert_allclose(network._ymu, frame.iloc[ti].Y.mean(), rtol=1e-12)
            audit["members"].append(dict(member=block_summary(frame, indices), training=block_summary(frame, ti),
                                          validation=block_summary(frame, vi),
                                          validation_evidence="deterministic source-helper replay checked against fitted sizes and scaling"))
    else:
        np.testing.assert_allclose(model.contrast_model_._xmu.numpy(),
                                   frame.iloc[contrast][["X"]].mean().to_numpy(), rtol=1e-6)
    return audit


def audit_lnn(diagnostic, frame):
    mapping = {key: np.asarray(diagnostic[key], dtype=int) for key in diagnostic if key.endswith("_indices")}
    for a, b in (("outer_training_indices", "outer_test_indices"),
                 ("selection_training_indices", "selection_validation_indices")):
        assert_group_disjoint(frame, mapping[a], mapping[b])
    if not (set(mapping["selection_training_indices"]) | set(mapping["selection_validation_indices"])) <= set(mapping["outer_training_indices"]):
        raise ValueError("LNN selection enters its held-out prediction data")
    return {key: block_summary(frame, indices) for key, indices in mapping.items()}


def source_score(frame, law):
    lookup = dict(zip(law["xs"], law["score"]))
    return frame.X.map(lookup).to_numpy(float)


def observed_cell_design(frame):
    codes = frame.E.astype(str)+":"+frame.H.astype(str)
    return pd.get_dummies(codes, dtype=float).to_numpy()


def relevance_diagnostic(frame):
    x, e = frame.X.to_numpy(float), frame.E.to_numpy(float)
    design = np.column_stack([e, np.ones(len(e))])
    coefficient, covariance, groups = linear_fit(design, x, frame.block_id)
    residual = x-design@coefficient
    total = float(np.sum((x-x.mean())**2))
    rss = float(np.sum(residual**2))
    f = max(total-rss, 0)/(rss/(len(x)-2)) if rss > 0 else None
    robust = float(coefficient[0]**2/covariance[0, 0]) if covariance[0, 0] > 0 else None
    return dict(applicable=False, first_stage_slope=float(coefficient[0]),
                first_stage_r_squared=1-rss/total, ordinary_first_stage_f=f,
                block_robust_wald_f=robust, n_blocks=groups,
                reason="E has zero population treatment relevance by design and changes the outcome pathway; no IV coefficient is reported.",
                finite_sample_note="A realized null first-stage F need not be near zero.")


def fit_task(frame, law, settings, seed, n_per_env, replicate_id):
    torch.set_num_threads(1)
    x, y, e, groups = (frame[["X"]].to_numpy(float), frame.Y.to_numpy(float),
                       frame.E.to_numpy(int), frame.block_id.to_numpy())
    score = source_score(frame, law)
    rows, split_audits, contrast_rows = [], {}, []
    direct = None

    def record(method, function):
        start = time.perf_counter()
        try:
            values = function()
            if values.get("coefficient") is None or not np.isfinite(values["coefficient"]):
                raise FloatingPointError("Estimator returned a nonfinite coefficient")
            row = dict(method=method, status="ok", **values)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, AssertionError) as error:
            row = dict(method=method, status="failed", coefficient=None,
                       error=type(error).__name__+": "+str(error))
        row.update(seed=seed, n_per_environment=n_per_env, replicate_id=replicate_id,
                   design=law["design"], fit_seconds=time.perf_counter()-start)
        rows.append(row)

    for name, cls in (("OLS", OLS), ("Environment FE", EnvironmentFixedEffects),
                      ("Anchor-g2", lambda: AnchorRegression(gamma=2))):
        record(name, lambda cls=cls: dict(coefficient=float(cls().fit(x, y, e).coef_[0])))
    for environment in (0, 1):
        mask = e == environment
        record(f"OLS E={environment}", lambda mask=mask: dict(coefficient=float(OLS().fit(x[mask], y[mask], e[mask]).coef_[0])))
    cells = observed_cell_design(frame)
    record("Oracle H", lambda: dict(coefficient=float(OLS().fit(np.column_stack([x, cells[:, 1:]]), y, e).coef_[0]),
                                     oracle_inputs="observed H; saturated E,H intercepts"))
    record("Oracle-score linear nuisance", lambda: dict(
        coefficient=float(OLS().fit(np.column_stack([x, e, score, e*score]), y, e).coef_[0]),
        oracle_inputs=law["score_interpretation"]))
    for name, cls in (("DOVR", DirectOVR), ("SOVR", SmoothedOVR)):
        start = time.perf_counter()
        try:
            model = cls(seed=seed, **settings["common"]).fit(x, y, e, groups=groups)
            split_audits[name] = audit_ovr(model, frame)
            if name == "DOVR":
                direct = model
            values = model.predict_contrast(law["xs"][:, None]) if name == "DOVR" else np.mean(
                model.predict_contrast_components(law["xs"][:, None]), axis=0)
            for xi, value in zip(law["xs"], values[:, 0]):
                contrast_rows.append(dict(method=name, seed=seed, X=float(xi), contrast=float(value)))
            row = dict(method=name, status="ok", coefficient=float(model.coef_[0]),
                       fitted_gamma=float(model.gamma_), n_estimation=model.n_estimation_,
                       n_contrast=model.n_contrast_, oracle_inputs=False,
                       residual_variance=float(model.Sigma_[0, 0]))
            if name == "SOVR":
                row.update(contrast_disagreement_rmse=float(model.contrast_disagreement_rmse_),
                           contrast_best_epochs=model.contrast_best_epochs_,
                           contrast_epochs_trained=model.contrast_epochs_trained_)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError, AssertionError) as error:
            row = dict(method=name, status="failed", coefficient=None,
                       error=type(error).__name__+": "+str(error))
        row.update(seed=seed, n_per_environment=n_per_env, replicate_id=replicate_id,
                   design=law["design"], fit_seconds=time.perf_counter()-start)
        rows.append(row)
    if direct is not None:
        def fit_oracle():
            coef, _, _, gamma = oracle_p_with_same_second_stage(direct, score)
            return dict(coefficient=float(coef[0]), fitted_gamma=float(gamma),
                        n_estimation=direct.n_estimation_, oracle_inputs=law["score_interpretation"],
                        split_audit_shared_with="DOVR")
        record("Oracle-score DML", fit_oracle)
    else:
        rows.append(dict(method="Oracle-score DML", status="failed", coefficient=None,
                         seed=seed, n_per_environment=n_per_env, replicate_id=replicate_id,
                         design=law["design"], error="Direct OVR did not return an auditable split"))

    def fit_lnn():
        # The existing benchmark only uses this fixed placeholder for scoring;
        # its fitting/selection receive no physical reference outcomes or target.
        result = compare_linear_neural(x, y, e, np.zeros(1), groups=groups,
                                       seed=seed, **settings["lnn"])[-1]
        diagnostic = json.loads(result["diagnostics"])
        split_audits["LNN"] = audit_lnn(diagnostic, frame)
        for key in list(diagnostic):
            if key.endswith("_indices"):
                diagnostic.pop(key)
        return dict(coefficient=float(json.loads(result["beta_hat"])[0]),
                    prediction_mse=result["prediction_mse"], diagnostics=diagnostic,
                    coefficient_identified_by_prediction_objective=False)
    record("LNN", fit_lnn)
    if law["design"] == "uniform":
        from .kivva_baseline import estimate_gamma_shift_fourth
        start = time.perf_counter()
        result = estimate_gamma_shift_fourth(x, y, e)
        result.update(seed=seed, n_per_environment=n_per_env, replicate_id=replicate_id,
                      design=law["design"], fit_seconds=time.perf_counter()-start,
                      implementation="independently derived fourth-cumulant specialization; not official author code",
                      applicability="prespecified additive-uniform assignment; physical noise/linearity assumptions remain approximate")
        rows.append(result)
    return dict(estimates=rows, split_audits=split_audits, contrasts=contrast_rows,
                relevance=relevance_diagnostic(frame))


def reference_targets(frame):
    reference = frame.loc[frame.kind.eq("reference")].copy()
    x = reference.X.to_numpy(float)-reference.X.mean()
    cells = observed_cell_design(reference)
    design = np.column_stack([x, cells])
    coefficient, covariance, g = linear_fit(design, reference.Y, reference.block_id)
    se = float(np.sqrt(max(covariance[0, 0], 0)))
    critical = float(student_t.ppf(.975, g-1))
    common = dict(target="randomized_reference", coefficient=float(coefficient[0]),
                  cluster_se=se, ci_low=float(coefficient[0]-critical*se),
                  ci_high=float(coefficient[0]+critical*se), n=len(reference), n_blocks=g,
                  shared_training_blocks=len(set(reference.block_id) & set(frame.loc[frame.kind.eq("confounded"), "block_id"])))
    env = np.eye(2)[reference.E.to_numpy(int)]
    coefficient, covariance, _ = linear_fit(np.column_stack([x[:, None]*env, cells]), reference.Y, reference.block_id)
    delta = float(coefficient[1]-coefficient[0])
    delta_se = float(np.sqrt(max(covariance[0, 0]+covariance[1, 1]-2*covariance[0, 1], 0)))
    return common, dict(environment_slopes=coefficient[:2].tolist(), difference_e1_minus_e0=delta,
                        difference_cluster_se=delta_se, difference_ci=[delta-critical*delta_se, delta+critical*delta_se])


def calibration_functionals(path, law, bootstrap_seed, bootstrap_draws=1999):
    """Finite-support physical functionals; no polynomial response assumption.

    The randomized-H score functional averages causal response secants in the
    paired designs. The observational-score functional also reflects the
    physical H distribution conditional on treatment. Their difference
    diagnoses departures from the working mean model. Neither is asserted
    to be generic OVR's target when its physical assumptions fail.
    """
    path = Path(path).resolve()
    frame = pd.read_csv(path)
    manifest_path = path.parent/"manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("measurements_sha256", file_hash(path)) != file_hash(path):
        raise ValueError("Factorial calibration measurement hash mismatch")
    if "Y" not in frame and "ir_3" in frame:
        frame["Y"] = frame.ir_3
    if "pol_2" in frame:
        mapped = np.full(len(frame), -1, dtype=int)
        for e, angle in enumerate(law["angles"]):
            mapped[np.isclose(frame.pol_2, angle, atol=1e-5, rtol=0)] = e
        frame["E"] = mapped
    elif manifest.get("angles") != law["angles"]:
        raise ValueError("Calibration requires physical pol_2 or manifest angles matching the target environments")
    for column, value in law["fixed_controls"].items():
        if column in frame and not np.allclose(frame[column], value, atol=1e-5, rtol=0):
            raise ValueError(f"Calibration uses different fixed controls: {column}")
    required = {"X", "Y", "E", "red", "block_id"}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError("Factorial calibration requires finite X,Y,E,red,block_id")
    if not np.isfinite(frame[["X", "Y", "E", "red"]].to_numpy(float)).all():
        raise ValueError("Nonfinite values in factorial calibration")
    selected = frame.loc[frame.X.isin(law["xs"]) & frame.red.isin(law["red_levels"]) & frame.E.isin([0, 1])].copy()
    means = selected.groupby(["block_id", "E", "red", "X"]).Y.mean().unstack(["E", "red", "X"])
    columns = pd.MultiIndex.from_product([[0, 1], law["red_levels"], law["xs"]], names=["E", "red", "X"])
    means = means.reindex(columns=columns)
    if means.isna().any().any() or len(means) < 2:
        raise ValueError("Every factorial block must contain all requested E,red,X cells; no interpolation is applied")
    px, ph, xs = law["px"], law["ph"], law["xs"]
    mean_x = float(xs@px)
    variance_x = float(np.sum(px*(xs-mean_x)**2))
    ols_weights, observational_weights, causal_weights = [], [], []
    for environment, red, x in columns:
        i = int(np.flatnonzero(xs == x)[0])
        h = law["red_levels"].index(red)
        ols_weights.append(.5*ph[h]*px[i]*(x-mean_x)/variance_x)
        observational_weights.append(.5*law["joint"][i, h]*law["residual"][i]/law["residual_variance"])
        causal_weights.append(.5*px[i]*ph[h]*law["residual"][i]/law["residual_variance"])
    ols_weights, observational_weights, causal_weights = map(np.asarray, (ols_weights, observational_weights, causal_weights))
    weights = np.column_stack([ols_weights, causal_weights, observational_weights,
                               ols_weights-causal_weights, observational_weights-causal_weights,
                               ols_weights-observational_weights])
    block_targets = means.to_numpy()@weights
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_indices = rng.integers(0, len(means), (bootstrap_draws, len(means)))
    draws = block_targets[bootstrap_indices].mean(axis=1)
    labels = ["calibrated_randomized_projection", "calibrated_causal_score", "calibrated_observational_score",
              "randomized_minus_causal_score", "observational_minus_causal_score",
              "randomized_minus_observational_score"]
    rows = []
    for j, label in enumerate(labels):
        value, se = float(block_targets[:, j].mean()), float(block_targets[:, j].std(ddof=1)/np.sqrt(len(means)))
        ci = np.quantile(draws[:, j], [.025, .975])
        rows.append(dict(target=label, coefficient=value, cluster_se=se, ci_low=float(ci[0]), ci_high=float(ci[1]),
                         n_blocks=len(means), uncertainty="percentile bootstrap of complete physical calibration blocks"))
    overall = means.mean(axis=0)
    contrasts, contrast_blocks = [], []
    for i, x in enumerate(xs):
        conditional = [sum(law["posterior"][i, h]*overall.loc[(e, red, x)]
                           for h, red in enumerate(law["red_levels"])) for e in (0, 1)]
        block_contrast = sum(law["posterior"][i, h]*(means[(1, red, x)]-means[(0, red, x)])
                             for h, red in enumerate(law["red_levels"])).to_numpy()
        contrast_blocks.append(block_contrast)
        ci = np.quantile(block_contrast[bootstrap_indices].mean(axis=1), [.025, .975])
        contrasts.append(dict(X=float(x), calibrated_conditional_mean_e0=float(conditional[0]),
                              calibrated_conditional_mean_e1=float(conditional[1]),
                              calibrated_contrast=float(conditional[1]-conditional[0]), oracle_score=float(law["score"][i]),
                              cluster_se=float(block_contrast.std(ddof=1)/np.sqrt(len(means))),
                              ci_low=float(ci[0]), ci_high=float(ci[1])))
    paired_differences = []
    center = (xs.min()+xs.max())/2
    for i, x in enumerate(xs):
        if x <= center:
            continue
        mirror = np.flatnonzero(np.isclose(xs, 2*center-x))
        if len(mirror) != 1:
            continue
        difference = contrast_blocks[i]-contrast_blocks[int(mirror[0])]
        ci = np.quantile(difference[bootstrap_indices].mean(axis=1), [.025, .975])
        paired_differences.append(dict(magnitude=float(x-center), difference=float(difference.mean()),
                                       cluster_se=float(difference.std(ddof=1)/np.sqrt(len(means))),
                                       ci_low=float(ci[0]), ci_high=float(ci[1])))
    return dict(source=str(path), sha256=file_hash(path), design=law["design"],
                targets=rows, contrasts=contrasts, paired_contrast_differences=paired_differences,
                block_targets=block_targets.tolist(),
                note="Causal-score weights randomize H using its marginal prior; observational-score weights retain P(H|X). Their difference diagnoses physical mean-model departures. Equality to generic OVR requires the working physical model.")


def score_results(rows, references, calibrations):
    scored = []
    for row in rows:
        row = dict(row)
        if row["status"] == "ok" and (row.get("coefficient") is None or not np.isfinite(row["coefficient"])):
            row.update(status="failed", error="Nonfinite coefficient returned with status ok")
        targets = [references[(row["design"], row["replicate_id"])]]
        calibration = calibrations.get(row["design"])
        if calibration:
            targets.extend(calibration["targets"][:3])
        for target in targets:
            if not np.isfinite(target["coefficient"]):
                raise ValueError("Cannot score against a nonfinite evaluation target")
            value = row.get("coefficient")
            error = float(value-target["coefficient"]) if row["status"] == "ok" and value is not None else np.nan
            scored.append(dict(**row, target=target["target"], target_coefficient=target["coefficient"],
                               target_cluster_se=target["cluster_se"], signed_error=error,
                               absolute_error=abs(error), squared_error=error**2))
    return pd.DataFrame(scored)


def aggregate_results(scored):
    key = ["design", "n_per_environment", "target", "method", "replicate_id"]
    per_rep = []
    for values, part in scored.groupby(key, sort=False):
        finite = np.isfinite(part[["coefficient", "signed_error", "absolute_error", "squared_error"]].to_numpy(float)).all(axis=1)
        good = part.loc[part.status.eq("ok") & finite]
        # Do not average away a failed seed inside a physical replicate.
        complete = len(good) == len(part)
        item = dict(zip(key, values), attempted=len(part), succeeded=len(good), complete=complete)
        item.update(mae=float(good.absolute_error.mean()) if complete else np.nan,
                    mse=float(good.squared_error.mean()) if complete else np.nan,
                    signed_bias=float(good.signed_error.mean()) if complete else np.nan,
                    mean_coefficient=float(good.coefficient.mean()) if complete else np.nan,
                    seed_sd=float(good.coefficient.std(ddof=1)) if len(good)>1 else np.nan)
        per_rep.append(item)
    per_rep = pd.DataFrame(per_rep)
    summary = []
    group_key = key[:-1]
    for values, part in per_rep.groupby(group_key, sort=False):
        good = part.loc[part.complete]
        r = len(good)
        summary.append(dict(zip(group_key, values), attempted_physical_replicates=len(part), complete_physical_replicates=r,
                            attempted_fits=int(part.attempted.sum()), successful_fits=int(part.succeeded.sum()),
                            failed_fits=int(part.attempted.sum()-part.succeeded.sum()),
                            metric_scope="all planned fits" if r == len(part) else "conditional on fully successful physical replicates",
                            mae=float(good.mae.mean()), rmse=float(np.sqrt(good.mse.mean())),
                            signed_bias=float(good.signed_bias.mean()),
                            mae_se_across_physical_replicates=float(good.mae.std(ddof=1)/np.sqrt(r)) if r>1 else np.nan,
                            rmse_delta_se=float(good.mse.std(ddof=1)/np.sqrt(r)/(2*np.sqrt(good.mse.mean())))
                            if r>1 and good.mse.mean()>0 else np.nan,
                            mean_within_replicate_seed_sd=float(good.seed_sd.mean()),
                            inference_note="Replication variation includes each run's noisy reference; shared calibrated-target uncertainty is separate."))
    paired = []
    for values, part in per_rep.groupby(["design", "n_per_environment", "target"], sort=False):
        matrix = part.loc[part.complete].pivot(index="replicate_id", columns="method", values="mae")
        for method in ("DOVR", "SOVR"):
            if method not in matrix:
                continue
            for comparator in matrix.columns:
                if comparator == method:
                    continue
                differences = (matrix[method]-matrix[comparator]).dropna()
                paired.append(dict(zip(["design", "n_per_environment", "target"], values),
                                   method=method, comparator=comparator, n_paired_replicates=len(differences),
                                   mae_difference=float(differences.mean()),
                                   paired_se=float(differences.std(ddof=1)/np.sqrt(len(differences)))
                                   if len(differences)>1 else np.nan))
    return per_rep, pd.DataFrame(summary), pd.DataFrame(paired)


def plot_errors(summary, output):
    main = ["DOVR", "SOVR", "OLS", "Environment FE", "Anchor-g2", "LNN",
            "Kivva gamma-shift (fourth cumulants)"]
    for design, part in summary.groupby("design", sort=False):
        targets = part.target.unique()
        fig, axes = plt.subplots(1, len(targets), figsize=(5.5*len(targets), 4.5), squeeze=False)
        for ax, target in zip(axes[0], targets):
            for method in main:
                selected = part.loc[part.target.eq(target) & part.method.eq(method)].sort_values("n_per_environment")
                selected = selected.loc[selected.failed_fits.eq(0)]
                if len(selected):
                    ax.errorbar(selected.n_per_environment, selected.mae,
                                yerr=selected.mae_se_across_physical_replicates if selected.mae_se_across_physical_replicates.notna().any() else None,
                                marker="o", capsize=3, label=method)
            ax.set_title(target.replace("_", " "))
            ax.set_xlabel("Confounded observations per environment")
            ax.set_ylabel("Mean absolute discrepancy (raw slope units)")
            ax.set_xscale("log")
            ax.grid(alpha=.2)
        axes[0, -1].legend(fontsize=8)
        title = design+": seeds averaged within physical acquisitions; bars show replication SE"
        if part.failed_fits.gt(0).any():
            title += "\nPoints with failed fits omitted; failure counts are reported in summary.csv"
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output/f"errors_{design}.png", dpi=160)
        plt.close(fig)


def plot_contrasts(contrast_rows, calibrations, output):
    data = pd.DataFrame(contrast_rows)
    if not len(data):
        return
    for design, part in data.groupby("design", sort=False):
        part = part.loc[part.n_per_environment.eq(part.n_per_environment.max())]
        center = (part.X.min()+part.X.max())/2
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        for method, selected in part.groupby("method", sort=False):
            per_rep = selected.groupby(["replicate_id", "X"]).contrast.mean().unstack("X")
            average, se = per_rep.mean(), per_rep.std(ddof=1)/np.sqrt(len(per_rep))
            axes[0].plot(average.index, average, marker="o", label=method)
            if len(per_rep)>1:
                axes[0].fill_between(average.index, average-se, average+se, alpha=.15)
            differences = pd.DataFrame({float(x-center): per_rep[x]-per_rep[2*center-x]
                                        for x in per_rep.columns if x>center and 2*center-x in per_rep})
            dmean, dse = differences.mean(), differences.std(ddof=1)/np.sqrt(len(differences))
            axes[1].plot(dmean.index, dmean, marker="o", label=method)
            if len(differences)>1:
                axes[1].fill_between(dmean.index, dmean-dse, dmean+dse, alpha=.15)
        if design in calibrations:
            curve = pd.DataFrame(calibrations[design]["contrasts"])
            axes[0].plot(curve.X, curve.calibrated_contrast, color="black", linestyle="--",
                         label="Independent physical calibration")
            axes[0].fill_between(curve.X, curve.ci_low, curve.ci_high, color="black", alpha=.1)
            paired = pd.DataFrame(calibrations[design]["paired_contrast_differences"])
            axes[1].plot(paired.magnitude, paired.difference, color="black", linestyle="--",
                         label="Independent physical calibration")
            axes[1].fill_between(paired.magnitude, paired.ci_low, paired.ci_high, color="black", alpha=.1)
        axes[0].set(xlabel="Treatment command X", ylabel="Contrast: conditional mean E=1 minus E=0")
        axes[1].set(xlabel="Magnitude K", ylabel="Contrast at center+K minus center-K")
        axes[1].axhline(0, color="0.5", linewidth=.8)
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle(design+": learned contrasts and physical working-model check\n"
                     "Colored bands: one replication SE; gray bands: pointwise calibration 95% intervals")
        fig.tight_layout()
        fig.savefig(output/f"contrasts_{design}.png", dpi=160)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--calibration-data", type=Path)
    parser.add_argument("--design", choices=["auto", "binary", "ternary", "uniform"], default="auto")
    parser.add_argument("--center", type=int, default=16)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--n-per-env", type=int, nargs="+", default=[500, 1000, 2500, 5000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--subset-seed", type=int, default=2026110601)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.scale <= 0 or min(args.n_per_env) <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive scale/sample sizes and distinct seeds required")
    settings, sources = paper_settings(), source_record()
    acquisitions = load_inputs(args.data, args.design, args.center, args.scale)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, all_contrasts, relevance, provenance, refs, reference_diagnostics, calibrations = [], [], [], [], {}, [], {}
    for acquisition in acquisitions:
        rep, design, frame, law = (acquisition[k] for k in ("replicate_id", "design", "frame", "law"))
        print(f"Physical acquisition {design}/{rep}: {len(frame)} rows", flush=True)
        provenance.append(dict(design=design, replicate_id=rep, **acquisition["provenance"],
                               assignment=law["provenance"], oracle_score_interpretation=law["score_interpretation"]))
        numerical_law = {key: law[key] for key in ("xs", "hs", "joint", "score", "red_levels", "angles", "fixed_controls")}
        if design in calibrations and calibrations[design]["numerical_law_sha256"] != json_hash(numerical_law):
            raise ValueError("Replicates of a design have different assignment laws")
        if args.calibration_data and design not in calibrations:
            calibrated = calibration_functionals(args.calibration_data, law, args.subset_seed)
            calibrated["law"] = law
            calibrated["numerical_law_sha256"] = json_hash(numerical_law)
            calibrations[design] = calibrated
        for n, subset, subset_audit in nested_subsets(frame, sorted(set(args.n_per_env)), args.subset_seed, design+"/"+rep):
            slug = re.sub(r"[^a-zA-Z0-9_-]", "_", rep)+"_"+hashlib.sha256(rep.encode()).hexdigest()[:8]
            for seed in args.seeds:
                cache = output/"fits"/design/slug/f"n{n}_seed{seed}.json"
                identity = dict(settings=settings, source_sha256=sources["current_sha256"],
                                data=acquisition["provenance"], assignment=law["provenance"],
                                subset=subset_audit, seed=seed, n_per_environment=n, replicate_id=rep, design=design)
                signature = json_hash(identity)
                if cache.exists():
                    result = json.loads(cache.read_text())
                    if result.get("identity_sha256") != signature:
                        raise ValueError(f"Cached fit provenance changed; use a new output directory: {cache}")
                else:
                    print(f"  fitting n/E={n}, seed={seed}", flush=True)
                    result = fit_task(subset, law, settings, seed, n, rep)
                    result.update(identity_sha256=signature, identity=identity)
                    write_json(cache, result)
                rows.extend(result["estimates"])
                all_contrasts.extend([dict(row, design=design, replicate_id=rep, n_per_environment=n)
                                      for row in result["contrasts"]])
                if seed == args.seeds[0]:
                    relevance.append(dict(design=design, replicate_id=rep, n_per_environment=n, **result["relevance"]))
        reference, diagnostic = reference_targets(frame)
        refs[(design, rep)] = reference
        reference_diagnostics.append(dict(design=design, replicate_id=rep, **diagnostic))
    if not rows:
        raise ValueError("No requested whole-block sample size is available")
    scored = score_results(rows, refs, calibrations)
    per_rep, summary, paired = aggregate_results(scored)
    scored.to_csv(output/"estimates.csv", index=False)
    per_rep.to_csv(output/"per_physical_replicate.csv", index=False)
    summary.to_csv(output/"summary.csv", index=False)
    paired.to_csv(output/"paired_method_differences.csv", index=False)
    pd.DataFrame(all_contrasts).to_csv(output/"estimated_contrasts.csv", index=False)
    pd.DataFrame(relevance).to_csv(output/"iv_relevance.csv", index=False)
    pd.DataFrame([dict(design=d, replicate_id=r, **value) for (d, r), value in refs.items()]).to_csv(output/"references.csv", index=False)
    write_json(output/"reference_heterogeneity.json", reference_diagnostics)
    write_json(output/"calibration_targets.json", calibrations)
    write_json(output/"analysis_manifest.json", dict(settings=settings, sources=sources, inputs=provenance,
               requested_n_per_environment=args.n_per_env, seeds=args.seeds, subset_seed=args.subset_seed,
               n_physical_acquisitions=len(acquisitions),
               notes=["Main estimators use only raw X,Y,E and block IDs; no reference tuning.",
                      "Fitting seeds are not physical replications; scores are averaged within acquisition first.",
                      "No OVR plug-in confidence interval or coverage claim is made.",
                      "Reference and shared calibration uncertainty do not disappear when averaging fits.",
                      "Fitted gamma is a finite-sample diagnostic, not proof of population positivity.",
                      "Calibrated oracle-score targets require the stated physical working model for an OVR interpretation."]))
    plot_errors(summary, output)
    plot_contrasts(all_contrasts, calibrations, output)
    print(summary[["design", "n_per_environment", "target", "method", "mae", "rmse", "complete_physical_replicates"]].to_string(index=False))
    print(f"Saved campaign analysis to {output}")


if __name__ == "__main__":
    main()
