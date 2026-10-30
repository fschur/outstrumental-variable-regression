"""Compact paper figure, independent of manuscript sources and local scripts."""
from __future__ import annotations

import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, MaxNLocator

STYLES = {
    'DOVR': ('Direct OVR', '#56B4E9', 'o', '--'),
    'SOVR': ('Smoothed OVR', '#0072B2', 'o', '-'),
    'LNN': ('LNN', '#7B3294', 'D', '-'),
    'OLS': ('OLS', '#D55E00', 's', '--'),
    'Environment FE': ('FE', '#CC79A7', '^', ':'),
    'Anchor-g2': (r'Anchor ($\gamma=2$)', '#009E73', 'v', '-.'),
}
PANELS = [
    ('low_rank', '(a) Latent loadings\n'),
    ('nonlinear_clean_environment', '(b) One unconfounded\nenvironment'),
    ('recoverable_state', '(c) Recoverable\nconfounding state'),
    ('shared_decoder_no_ex', '(d) Outcome shifts\n'),
    ('shared_decoder_additive_ex', '(e) Outcome and\ntreatment shifts'),
    ('shared_decoder_latent_outcome_exposure_shifts', '(f) Latent, outcome\nand treatment shifts'),
]
# Short figure titles; retain the full setting names above for table exports.
PLOT_TITLES = {
    'low_rank': '(a) Latent loadings',
    'nonlinear_clean_environment': '(b) One unconf. env.',
    'recoverable_state': '(c) Recoverable state',
    'shared_decoder_no_ex': '(d) Outcome shifts',
    'shared_decoder_additive_ex': '(e) Outc. + trt. shifts',
    'shared_decoder_latent_outcome_exposure_shifts': '(f) Latent + outc. + trt.',
}
STYLE = {
    'font.family': 'sans-serif', 'font.size': 8.5, 'axes.titlesize': 8.1,
    'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .55,
    'axes.edgecolor': '#777777', 'xtick.color': '#444444', 'ytick.color': '#444444',
    'xtick.major.width': .55, 'ytick.major.width': 0,
    'xtick.major.size': 3, 'ytick.major.size': 0, 'savefig.facecolor': 'white',
    'pgf.texsystem': 'pdflatex', 'pgf.rcfonts': False,
    'pgf.preamble': r'\usepackage{helvet}\providecommand{\methodabbr}{OVR}',
}


def validate_summary(frame, sample_sizes, replications=None, require_lnn=True):
    methods = set(STYLES) if require_lnn else set(frame.method)
    expected = {(s, m, n) for s, _ in PANELS for m in methods for n in sample_sizes}
    keys = ['scenario', 'method', 'n']
    if frame.duplicated(keys).any() or set(map(tuple, frame[keys].values)) != expected:
        raise ValueError('Summary does not match the six-panel method/sample-size grid')
    if replications is not None and not frame.n_reps.eq(replications).all():
        raise ValueError('Replication counts differ from the requested paper protocol')
    if frame.coefficient_error_mean.isna().any() or (frame.coefficient_error_mean < 0).any():
        raise ValueError('Coefficient errors must be finite and nonnegative')
    if frame.loc[frame.n_reps.gt(1), 'coefficient_error_se'].isna().any():
        raise ValueError('Monte Carlo standard errors are missing')


def make_figure(frame, sample_sizes, *, tex_labels=False, include_direct=False,
                independent_y=True):
    """Plot unchanged means and +/- one MC SE; Direct OVR is supplementary."""
    methods = {m: style for m, style in STYLES.items() if include_direct or m != 'DOVR'}
    frame = frame[frame.method.isin(methods)]
    style = dict(STYLE)
    if independent_y:
        style.update({'font.size': 9.5, 'axes.titlesize': 10,
                      'xtick.labelsize': 9, 'ytick.labelsize': 9})
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.45) if independent_y else (5.5, 2.35),
                                 sharey=not independent_y, sharex=True)
        upper = math.ceil((frame.coefficient_error_mean + frame.coefficient_error_se.fillna(0)).max()*1.1/.5)*.5
        upper = max(upper, .5)
        for ax, (key, _) in zip(axes.flat, PANELS):
            panel = frame[frame.scenario.eq(key)]
            panel_upper = max(.1, (panel.coefficient_error_mean + panel.coefficient_error_se.fillna(0)).max()*1.08) if independent_y else upper
            for method, (label, color, marker, ls) in methods.items():
                d = frame[frame.scenario.eq(key) & frame.method.eq(method)].sort_values('n')
                if d.empty:
                    continue
                se = d.coefficient_error_se.fillna(0)
                ax.fill_between(d.n, d.coefficient_error_mean-se, d.coefficient_error_mean+se,
                                color=color, alpha=.10, linewidth=0, zorder=1)
                if tex_labels:
                    label = label.replace('OVR', r'{\methodabbr}')
                ax.plot(d.n, d.coefficient_error_mean, label=label, color=color, marker=marker,
                        linestyle=ls, linewidth=1.35, markersize=3.7, markeredgewidth=.8,
                        markerfacecolor='white' if method == 'DOVR' else color, zorder=3)
            ax.set_xscale('log')
            ax.set_xticks(sample_sizes, [f'{n/1000:g}k' for n in sample_sizes])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_xlim(sample_sizes[0]/1.18, sample_sizes[-1]*1.18)
            ax.set_ylim(0, panel_upper)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]) if independent_y else
                                      FixedLocator([.5*i for i in range(round(upper/.5)+1)]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
            ax.set_title(PLOT_TITLES[key], pad=3, loc='left')
            ax.grid(axis='y', color='#DFE3E7', linewidth=.55, zorder=0)
            ax.set_axisbelow(True)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.535, 1), ncol=len(handles),
                   frameon=False, fontsize=9 if independent_y else 7.8,
                   handlelength=1.7, handletextpad=.4, columnspacing=.9)
        fig.supxlabel(r'Sample size $n$', y=.015 if independent_y else .005, fontsize=10 if independent_y else 8.5)
        fig.supylabel(r'$\|\widehat\beta-\beta_0\|_2$', x=.008 if independent_y else .007,
                      fontsize=10 if independent_y else 8.5)
        if independent_y:
            fig.subplots_adjust(left=.09, right=.985, bottom=.15, top=.855, hspace=.38, wspace=.29)
        else:
            fig.subplots_adjust(left=.105, right=.985, bottom=.15, top=.835, hspace=.34, wspace=.18)
        return fig


def export_figure(frame, sample_sizes, output_dir, *, pgf=False, name='synthetic_main'):
    if name not in ('synthetic_main', 'synthetic_supplementary'):
        raise ValueError('Only the two synthetic figures in the paper are exported')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir/name
    options = dict(include_direct=name == 'synthetic_supplementary', independent_y=True)
    with plt.rc_context(STYLE):
        fig = make_figure(frame, sample_sizes, **options)
        fig.savefig(stem.with_suffix('.pdf'))
        plt.close(fig)
        if pgf:
            # Same Helvetica/LaTeX metrics as the manuscript; pdflatex is optional.
            fig = make_figure(frame, sample_sizes, tex_labels=True, **options)
            fig.savefig(stem.with_suffix('.pgf'), backend='pgf')
            fig.savefig(stem.with_suffix('.pdf'), backend='pgf')
            plt.close(fig)
