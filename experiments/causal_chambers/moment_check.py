"""Moment estimator of Kivva et al. (fourth-order case) on all three designs.

Post-hoc, as in data/reaction_review_20261107/reproduce_moment_check.py: for the
binary and three-level designs the estimator is outside its assumptions, since X
is not linear in H plus independent noise. For the additive-uniform design the
main analysis already contains these estimates; they are recomputed and checked.
Uses the same whole-block subsets as campaign_analysis.py. No hardware access.

Run from the repository root:
  PYTHONPATH=src /path/to/python -m experiments.causal_chambers.moment_check \
    --data-root experiments/causal_chambers/data/final_campaign_20261110 \
    --analysis-dir experiments/causal_chambers/data/final_campaign_20261110_analysis \
    --out experiments/causal_chambers/data/final_campaign_20261110_moment_check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .campaign_analysis import file_hash, load_inputs, nested_subsets
from .kivva_baseline import estimate_gamma_shift_fourth

MOMENT = "Kivva gamma-shift (fourth cumulants)"
SUBSET_SEED = 2026110601


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 1000, 2500, 5000])
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, inputs = [], []
    for design in ("binary", "ternary", "uniform"):
        results = args.analysis_dir / design
        refs = pd.read_csv(results / "references.csv")
        refs = refs[refs.target.eq("randomized_reference")].set_index("replicate_id")
        recorded = pd.read_csv(results / "estimates.csv")
        paths = sorted(args.data_root.glob(f"{design}_rep*/measurements.csv"))
        for acquisition in load_inputs(paths, design, 16, 8):
            rep = acquisition["replicate_id"]
            inputs.append(acquisition["provenance"])
            reference = float(refs.loc[rep, "coefficient"])
            for n, frame, _ in nested_subsets(acquisition["frame"], args.sizes, SUBSET_SEED, f"{design}/{rep}"):
                result = estimate_gamma_shift_fourth(frame.X, frame.Y, frame.E)
                coefficient = result.get("coefficient")
                if design == "uniform":
                    old = recorded.loc[recorded.replicate_id.eq(rep) & recorded.n_per_environment.eq(n)
                                       & recorded.seed.eq(0) & recorded.target.eq("randomized_reference")
                                       & recorded.method.eq(MOMENT)].iloc[0]
                    assert result["status"] == old.status
                    assert np.isclose(coefficient, old.coefficient, rtol=0, atol=1e-10)
                error = np.nan if coefficient is None else coefficient - reference
                rows.append(dict(design=design, replicate_id=rep, n_per_environment=n,
                                 outside_assumptions=design != "uniform", status=result["status"],
                                 coefficient=coefficient, reference=reference, signed_error=error,
                                 absolute_error=abs(error), percent_error=100 * abs(error) / abs(reference),
                                 reason=result.get("reason")))
    data = pd.DataFrame(rows)
    data.to_csv(args.out / "moment_estimates.csv", index=False)
    summary = data.groupby(["design", "n_per_environment"]).agg(
        repetitions=("replicate_id", "size"), failures=("status", lambda s: int(s.ne("ok").sum())),
        percent_mae=("percent_error", "mean"), percent_mae_se=("percent_error", "sem"),
        min_signed_error=("signed_error", "min"), max_signed_error=("signed_error", "max")).reset_index()
    summary.to_csv(args.out / "moment_summary.csv", index=False)
    here = Path(__file__).resolve().parent
    (args.out / "provenance.json").write_text(json.dumps(dict(
        note="Post-hoc; binary and three-level calls are outside the estimator's assumptions.",
        code={name: file_hash(here / name) for name in ("moment_check.py", "kivva_baseline.py", "campaign_analysis.py")},
        inputs=inputs), indent=2, default=str) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
