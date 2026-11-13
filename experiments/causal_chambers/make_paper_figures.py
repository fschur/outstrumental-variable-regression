"""Paper figures for the light-tunnel experiment.

Computes the plotted values from saved measurements and analysis outputs (no
refits, no hardware) and writes three PDFs:
  chamber_physics.pdf      effects of X and H on Y from a grid run
  chamber_sample_size.pdf  error versus training size for the three designs
  chamber_angles.pdf       angle analysis of the binary design
The plotted values are written as CSV files to --values-out.

Usage (from the repository root; defaults are the campaign of 10 November 2026):
  python experiments/causal_chambers/make_paper_figures.py --out OUT_DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.lines
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import t

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"  # analysis outputs (not in the repository)
CAMPAIGN = HERE / "measurements" / "final_campaign_20261110"

# Environment colors from Fig. 1 of the paper (muone, mutwo, muthree).
ANGLE_STYLE = {
    0.0: dict(color="#009E73", marker="o", label="0°"),
    60.0: dict(color="#D55E00", marker="^", label="60°"),
}
# Method styles of the synthetic figures (experiments/synthetic/paper_plot.py) for OLS and the OVR
# variants; the methods that only appear here get colors that the synthetic figures do not use.
MOMENT = "Kivva gamma-shift (fourth cumulants)"
METHOD_STYLE = {
    "OLS": dict(color="#D55E00", marker="s", ls="--", label="OLS"),
    "OLS E=0": dict(color="#A0A0A0", marker="v", ls=":", label="OLS, 0° only"),
    "OLS E=1": dict(color="#555555", marker="^", ls="-.", label="OLS, 60° only"),
    MOMENT: dict(color="#E69F00", marker="P", ls="-", label="Moment estimator"),
    "DOVR": dict(color="#56B4E9", marker="o", ls="--", label=r"Direct OVR", mfc="white"),
    "SOVR": dict(color="#0072B2", marker="o", ls="-", label=r"Smoothed OVR"),
}
# Curve look of the synthetic figures, scaled from their 7.4-inch width to the 5.5-inch width here:
# lines with markers for the means and bands of +/- one standard error.
CURVE_RC = {"axes.edgecolor": "#777777", "xtick.color": "#444444", "ytick.color": "#444444",
            "ytick.major.size": 0}
CURVE_SCALE = 5.5 / 7.4
# Angle sets of the angle analysis; Okabe-Ito colors plus distinct markers and line styles.
SET_STYLE = {
    "0/20": dict(color="#E69F00", marker="v", ls="--", label="0° and 20°"),
    "0/40": dict(color="#009E73", marker="s", ls="-.", label="0° and 40°"),
    "0/60": dict(color="#0072B2", marker="o", ls="-", label="0° and 60°"),
    "0/20/40/60": dict(color="#CC79A7", marker="D", ls=":", label="All four angles"),
}


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Helvetica",
        "mathtext.it": "Helvetica:italic",
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": .6,
        "xtick.major.width": .6,
        "ytick.major.width": .6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })


def physics_values(grid):
    """Coefficient of X and effect of H on Y for each angle and H, over blocks of a grid run."""
    raw = pd.read_csv(grid)
    raw = raw[raw.pol_2.isin(ANGLE_STYLE) & raw.led_3_ir.between(8, 24)]
    keys = ["block_id", "pol_2", "red", "led_3_ir"]
    cells = raw.groupby(keys).ir_3.mean()
    levels = [sorted(raw[c].unique()) for c in keys]
    blocks, angles, reds, doses = levels
    values = cells.reindex(pd.MultiIndex.from_product(levels, names=keys)).to_numpy()
    values = values.reshape(*(len(x) for x in levels))
    if not np.isfinite(values).all():
        raise ValueError("The grid run is not complete in every block")
    xc = np.asarray(doses) - np.mean(doses)
    slopes = values @ xc / (xc @ xc)
    source = values.mean(axis=-1)
    effects = source - source[:, :, :1]
    critical = t.ppf(.975, len(blocks) - 1)
    mean = lambda a: a.mean(axis=0)
    se = lambda a: a.std(axis=0, ddof=1) / np.sqrt(len(a))
    base = mean(effects)[0]
    scales = mean(effects) @ base / (base @ base)
    rows = []
    for ai, angle in enumerate(angles):
        for hi, red in enumerate(reds):
            rows.append(dict(angle=angle, red=red, slope=mean(slopes)[ai, hi], slope_se=se(slopes)[ai, hi],
                             slope_ci95_radius=critical * se(slopes)[ai, hi],
                             source_effect=mean(effects)[ai, hi], source_effect_se=se(effects)[ai, hi],
                             source_effect_ci95_radius=critical * se(effects)[ai, hi],
                             anchored_scale=scales[ai], blocks=len(blocks)))
    return pd.DataFrame(rows)


def physics_figure(frame, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(3.45, 1.42), constrained_layout=True)
    reds = sorted(frame.red.unique())
    offsets = {0.0: -5, 60.0: 5}
    for angle, style in ANGLE_STYLE.items():
        part = frame[frame.angle == angle].sort_values("red")
        left.errorbar(part.red + offsets[angle], part.slope, yerr=part.slope_ci95_radius,
                      fmt=style["marker"], color=style["color"], ms=2.8, elinewidth=.7,
                      capsize=1.2, capthick=.6, label=style["label"])
    left.axhline(frame.slope.mean(), color="0.5", lw=.6, ls=":", zorder=0)
    left.set_xticks(reds)
    left.set_xlabel(r"Brightness setting $H$")
    left.set_ylabel(r"Coefficient of $X$")
    left.set_ylim(79.8, 85.6)
    left.set_title(r"(b) Effect of $X$ on $Y$", loc="left")

    # Effect of H at 60 degrees against the 0-degree effect times the fitted factor.
    base = frame[frame.angle == 0.0].sort_values("red")
    scale = float(frame[frame.angle == 60.0].anchored_scale.iloc[0])
    right.plot(base.red, scale * base.source_effect, ls="--", lw=.7,
               color=ANGLE_STYLE[60.0]["color"], zorder=1)
    right.text(176, scale * base.source_effect.iloc[-1] + 110, rf"${scale:.2f}\times$ 0° values",
               color=ANGLE_STYLE[60.0]["color"], fontsize=6, ha="center", va="bottom")
    for angle, style in ANGLE_STYLE.items():
        part = frame[frame.angle == angle].sort_values("red")
        if angle == 0.0:
            right.plot(part.red, part.source_effect, lw=.7, color=style["color"], zorder=1)
        right.errorbar(part.red, part.source_effect, yerr=part.source_effect_ci95_radius,
                       fmt=style["marker"], color=style["color"], ms=2.8, elinewidth=.7,
                       capsize=1.2, capthick=.6, zorder=2, label=style["label"])
    right.set_xticks(reds)
    right.set_yticks([0, 300, 600, 900])
    right.set_xlabel(r"Brightness setting $H$")
    right.set_ylabel(r"Change in mean of $Y$")
    right.set_title(r"(c) Effect of $H$ on $Y$", loc="left")
    # One legend for both panels; the first entry is a text-only label.
    handles, labels = left.get_legend_handles_labels()
    blank = matplotlib.lines.Line2D([], [], ls="none", marker="")
    labels[-1] += "  (angle between polarizers)"
    fig.legend([blank] + handles, [r"$E$:"] + labels,
               loc="outside upper center", ncol=3, frameon=False, handlelength=1.0,
               handletextpad=.3, columnspacing=1.0, borderaxespad=.1)
    fig.savefig(out / "chamber_physics.pdf")
    plt.close(fig)


def sample_size_values(analysis):
    """Mean and standard error over repetitions of the seed-averaged absolute error, in %."""
    rows = []
    for design in ("binary", "ternary", "uniform"):
        rep = pd.read_csv(analysis / design / "per_physical_replicate.csv")
        rep = rep[rep.target.eq("randomized_reference")]
        refs = pd.read_csv(analysis / design / "references.csv")
        refs = refs[refs.target.eq("randomized_reference")][["replicate_id", "coefficient"]]
        rep = rep.merge(refs.rename(columns={"coefficient": "reference"}), on="replicate_id",
                        validate="many_to_one")
        if not rep.complete.all():
            raise ValueError(f"Incomplete repetitions in {design}; summarize them explicitly")
        rep["percent_mae"] = 100 * rep.mae / rep.reference.abs()
        for (n, method), group in rep.groupby(["n_per_environment", "method"]):
            rows.append(dict(design=design, n_per_environment=n, method=method, repetitions=len(group),
                             percent_mae=group.percent_mae.mean(), percent_mae_se=group.percent_mae.sem(),
                             mean_reference=group.reference.mean()))
    return pd.DataFrame(rows)


def curve(ax, x, mean, se, style):
    """Mean with markers and a band of +/- one standard error, as in the synthetic figures."""
    ax.fill_between(x, mean - se, mean + se, color=style["color"], alpha=.10, linewidth=0, zorder=1)
    ax.plot(x, mean, color=style["color"], marker=style["marker"], ls=style["ls"],
            lw=1.35 * CURVE_SCALE, ms=3.7 * CURVE_SCALE, mew=.8 * CURVE_SCALE,
            mfc=style.get("mfc", style["color"]), label=style["label"], zorder=3)


def size_axis(ax, sizes, upper):
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s / 1000:g}k" for s in sizes])
    ax.minorticks_off()
    ax.set_xlim(sizes[0] / 1.18, sizes[-1] * 1.18)
    ax.set_ylim(0, upper)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.grid(axis="y", color="#DFE3E7", linewidth=.55 * CURVE_SCALE, zorder=0)
    ax.set_axisbelow(True)


def sample_size_figure(frame, out):
    titles = {"binary": "(a) Binary assignment", "ternary": "(b) Three-level assignment",
              "uniform": "(c) Additive assignment"}
    with plt.rc_context(CURVE_RC):
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.55), constrained_layout=True)
        for ax, design in zip(axes, titles):
            part = frame[frame.design == design]
            for method, style in METHOD_STYLE.items():
                rows = part[part.method == method].sort_values("n_per_environment")
                if not rows.empty:
                    curve(ax, rows.n_per_environment, rows.percent_mae, rows.percent_mae_se, style)
            shown = part[part.method.isin(METHOD_STYLE)]
            size_axis(ax, sorted(part.n_per_environment.unique()),
                      (shown.percent_mae + shown.percent_mae_se).max() * 1.08)
            ax.set_title(titles[design], loc="left")
            ax.set_xlabel("Training size per environment")
        axes[0].set_ylabel("Error (% of reference)")
        legend(fig, axes, ncol=6)
        fig.savefig(out / "chamber_sample_size.pdf")
        plt.close(fig)


def angle_values(angle_analysis):
    summary = pd.read_csv(angle_analysis / "summary.csv")
    if (summary.failed_fits > 0).any() or (summary.complete_repetitions < summary.attempted_repetitions).any():
        raise ValueError("Failed fits in the angle analysis; summarize them explicitly")
    checks = pd.read_csv(angle_analysis / "angle_checks.csv")
    per_angle = checks.groupby("angle").agg(
        relative_effect=("h_effect_relative_to_0deg", "mean"), relative_effect_se=("h_effect_relative_to_0deg", "sem"),
        slope=("slope", "mean"), slope_se=("slope", "sem"), repetitions=("replicate_id", "size")).reset_index()
    # t interval over repetitions, as for the grid blocks in chamber_physics.pdf.
    per_angle["relative_effect_ci95_radius"] = (t.ppf(.975, per_angle.repetitions - 1)
                                                * per_angle.relative_effect_se)
    return summary, per_angle


def angle_figure(summary, per_angle, out):
    with plt.rc_context(CURVE_RC):
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.55), constrained_layout=True)
        left = axes[0]
        grid = np.linspace(0, 65, 200)
        left.plot(grid, np.cos(np.radians(grid)) ** 2, color="0.6", lw=.7, ls="--", zorder=1)
        left.text(29, .93, r"$\cos^2$ of the angle", color="0.4", fontsize=6, ha="left", va="bottom")
        left.errorbar(per_angle.angle, per_angle.relative_effect, yerr=per_angle.relative_effect_ci95_radius,
                      fmt="o", color="0.15", ms=2.8, elinewidth=.7, capsize=1.2, capthick=.6, zorder=2)
        left.set_xticks([0, 20, 40, 60])
        left.set_xticklabels(["0°", "20°", "40°", "60°"])
        left.set_ylim(0, 1.08)
        left.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        left.grid(axis="y", color="#DFE3E7", linewidth=.55 * CURVE_SCALE, zorder=0)
        left.set_axisbelow(True)
        left.set_xlabel("Angle between polarizers")
        left.set_ylabel(r"Effect of $H$, relative to 0°")
        left.set_title(r"(a) Effect of $H$ on $Y$", loc="left")
        shown = summary[summary.method.isin(["DOVR", "SOVR"])]
        for ax, method, title in [(axes[1], "DOVR", "(b) Direct OVR"), (axes[2], "SOVR", "(c) Smoothed OVR")]:
            part = summary[summary.method == method]
            for angles, style in SET_STYLE.items():
                rows = part[part.angles == angles].sort_values("n_per_angle")
                curve(ax, rows.n_per_angle, rows.mae_pct, rows.mae_se, style)
            size_axis(ax, [500, 1000, 2500, 5000], (shown.mae_pct + shown.mae_se).max() * 1.08)
            ax.set_title(title, loc="left")
            ax.set_xlabel("Training size per angle")
        axes[1].set_ylabel("Error (% of reference)")
        legend(fig, axes[1:], ncol=4)
        fig.savefig(out / "chamber_angles.pdf")
        plt.close(fig)


def legend(fig, axes, ncol):
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="outside upper center", ncol=ncol, frameon=False,
               handlelength=1.7, handletextpad=.4, columnspacing=.9)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", type=Path, default=CAMPAIGN / "grid_before" / "measurements.csv")
    parser.add_argument("--analysis", type=Path, default=DATA / "final_campaign_20261110_analysis")
    parser.add_argument("--angle-analysis", default=str(DATA / "final_campaign_20261110_angle_analysis"),
                        help="angle-analysis directory, or 'none'")
    parser.add_argument("--values-out", type=Path, default=DATA / "final_campaign_20261110_paper_figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.values_out.mkdir(parents=True, exist_ok=True)
    setup_style()
    physics = physics_values(args.grid)
    physics.to_csv(args.values_out / "physics_plot_data.csv", index=False)
    physics_figure(physics, args.out)
    sizes = sample_size_values(args.analysis)
    sizes.to_csv(args.values_out / "sample_size_summary.csv", index=False)
    sample_size_figure(sizes, args.out)
    if args.angle_analysis != "none":
        summary, per_angle = angle_values(Path(args.angle_analysis))
        per_angle.to_csv(args.values_out / "angle_plot_data.csv", index=False)
        angle_figure(summary, per_angle, args.out)


if __name__ == "__main__":
    main()
