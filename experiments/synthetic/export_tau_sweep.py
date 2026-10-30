"""Export the paired tau sweep as a vector figure sized for the appendix."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .sampling import _scenario_catalog


def export(summary_path: Path, output_path: Path, *, quick=False) -> None:
    if output_path.name != 'tau_sweep.pdf':
        raise ValueError('The paper smoothing figure is exported as tau_sweep.pdf')
    summary = pd.read_csv(summary_path)
    catalog = _scenario_catalog()
    scales = [0, 0.25, 0.5, 1, 2, 4]
    sizes = [600] if quick else [1000, 10000]
    expected = {(s.key, n, t) for s in catalog for n in sizes for t in scales}
    cells = list(zip(summary.scenario_key, summary.n, summary.tau))
    if set(cells) != expected or len(cells) != len(expected):
        raise ValueError("Expected the complete six-setting, two-sample-size tau sweep")
    if not (summary.replications == (2 if quick else 10)).all():
        raise ValueError("Expected ten replications per cell")
    metrics = summary[["coefficient_error_mean", "coefficient_error_se"]].to_numpy()
    if not np.isfinite(metrics).all():
        raise ValueError("Nonfinite summary statistics")
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.45), layout="constrained")
    titles = ("(a) Latent loadings", "(b) One unconfounded\nenvironment",
              "(c) Recoverable\nconfounding state", "(d) Outcome shifts",
              "(e) Outcome and\ntreatment shifts", "(f) Latent, outcome\nand treatment shifts")
    handles = []
    for ax, scenario, title in zip(axes.flat, catalog, titles):
        for n, color, marker in zip(sizes, ("#0072B2", "#D55E00"), ("o", "s")):
            group = summary[(summary.scenario_key == scenario.key) & (summary.n == n)].sort_values("tau")
            curve = ax.errorbar(group.tau, group.coefficient_error_mean,
                                yerr=group.coefficient_error_se, color=color,
                                marker=marker, markersize=3, linewidth=1.1,
                                elinewidth=0.8, capsize=1.8, label=rf"$n={n:,}$")
            if scenario == catalog[0]:
                handles.append(curve)
        ax.axvline(1, color="0.55", linestyle=":", linewidth=0.8, zorder=0)
        ax.set_xscale("symlog", linthresh=0.25)
        ax.set_xticks(scales, labels=["0", ".25", ".5", "1", "2", "4"])
        ax.set_title(title, loc="left", pad=5)
        ax.grid(axis="y", color="0.9", linewidth=0.6)
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)
    fig.supxlabel(r"Smoothing scale $\tau$", fontsize=9)
    fig.supylabel(r"$\|\widehat\beta-\beta_0\|_2$", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    export(args.summary, args.output, quick=args.quick)
