"""Export the paper's tables from complete, matched replication grids."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.common.config import paper_config_section
from experiments.common.paper_outputs import TABLES
from experiments.synthetic.paper_plot import PANELS

METHODS=[('SOVR','Smoothed OVR'),('OLS','OLS'),('Environment FE','FE'),
         ('Anchor-g2',r'Anchor ($\gamma=2$)'),('LNN','LNN')]
PREAMBLE=r'\providecommand{\methodabbr}{OVR}'+'\n'

def require_grid(raw, keys, expected, replications):
    """Reject duplicate, missing, or unmatched replication blocks before export."""
    full_keys = keys + ['rep']
    if raw.duplicated(full_keys).any():
        raise ValueError('Duplicate table estimates')
    observed = set(map(tuple, raw[keys].to_numpy()))
    if observed != set(expected):
        raise ValueError('Table scenario/method grid is incomplete or contains extra cells')
    for _, g in raw.groupby(keys):
        if set(g.rep) != set(range(replications)):
            raise ValueError('Table replication IDs do not match the paper protocol')

def number(x, compact=True):
    return f'{x:.3f}'.removeprefix('0') if compact else f'{x:.3f}'

def cell(mean, se, *, bold=False, deterministic=False, compact=True):
    if not np.isfinite(mean):
        return '---'
    values = [number(mean, compact)]
    if np.isfinite(se) and not deterministic:
        values.append(number(se, compact))
    if bold:
        values = [r'\mathbf{'+value+'}' for value in values]
    return '$'+values[0]+(r'\;('+values[1]+')' if len(values) > 1 else '')+'$'

def write_table(output, name, header, rows, alignment):
    if name not in TABLES:
        raise ValueError(f'Table is not in the paper: {name}')
    text = PREAMBLE + r'\begin{tabular}{'+alignment+'}\n'+r'\toprule'+'\n'
    text += ' & '.join(header)+r' \\'+'\n'+r'\midrule'+'\n'
    text += '\n'.join(' & '.join(row)+r' \\' for row in rows)
    text += '\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'
    (output/(name+'.tex')).write_text(text)

def export_ablation(root, output, quick=False):
    from experiments.synthetic.ablation_results import load_replications, paired_step_effects
    config = paper_config_section('synthetic')
    if quick:
        config.update(ablation_sample_sizes=[600],replications=2)
    sizes = tuple(config['ablation_sample_sizes'])
    if len(sizes) != 1:
        raise ValueError('The component-ablation table requires one fixed sample size')
    raw = load_replications(root, sizes, config['replications'])
    expected = {(s, n, v) for s, _ in PANELS for n in sizes
                for v in ('direct', 'regularized', 'early_stopped', 'ensemble', 'smoothed')}
    grid = raw.copy()
    grid['scenario'] = grid.scenario_key
    require_grid(grid, ['scenario', 'n', 'variant_key'], expected, config['replications'])
    effects = paired_step_effects(raw)
    records = []
    for transition, group in effects.groupby('transition', sort=True):
        weight = group.n_pairs
        records.append(dict(transition=transition, n=sizes[0], n_pairs=int(weight.sum()),
                            wins=int(round(np.sum(weight*group.new_wins_fraction))),
                            **{metric: np.exp(np.average(group['log_'+metric+'_ratio_mean'], weights=weight))
                               for metric in ('error', 'moment', 'amplification')}))
    summary = pd.DataFrame(records)
    summary.to_csv(output/'component_ablation.csv', index=False)
    labels = ['Contrast-head regularization', 'Validation and early stopping',
              r'Disjoint split ensemble ($\tau=0$)', r'Uncertainty smoothing ($\tau=1$)']
    rows = []
    for r in summary.to_dict('records'):
        scores = [cell(r[m], np.nan, compact=False) for m in ('error', 'moment', 'amplification')]
        wins = f"${r['wins']}/{r['n_pairs']}$"
        rows.append([labels[r['transition']], scores[0], wins, *scores[1:]])
    write_table(output, 'component_ablation', [
        'Added component', r'$\|\widehat\beta-\beta_0\|_2$', r'\shortstack{Lower\\error}',
        r'$\|\widehat r\|_2$', r'$\dfrac{\|\widehat G_\lambda^{-1}\widehat r\|_2}{\|\widehat r\|_2}$',
    ], rows, 'lcccc')
    path = output/'component_ablation.tex'
    note = (f'% n={sizes[0]}; {int(summary.n_pairs.iloc[0])} paired comparisons per row.\n'
            '% Numeric entries except counts are geometric means of new/previous ratios.\n'
            '% Definitions: paper appendix, Controlled component ablation.\n')
    path.write_text(note + path.read_text())

def export_real(raw, output, reps, quick=False):
    require_grid(raw,['method'],[(m,) for m in ['DOVR',*[m for m,_ in METHODS]]],reps)
    for _, group in raw.groupby('rep'):
        for key in ('data_hash','beta_target','n'):
            if group[key].isna().any() or group[key].nunique()!=1:
                raise ValueError('KuaiRand methods must share training data and reference')
        if not quick and not group.n.eq(160158).all():
            raise ValueError('KuaiRand cohort differs from the reported experiment')
    summary=raw.groupby('method').coefficient_error.agg(mean='mean',se='sem').loc[[m for m,_ in METHODS]]
    if not np.isfinite(summary.to_numpy()).all():
        raise ValueError('Nonfinite coefficient distances')
    summary.to_csv(output/'kuairand.csv')
    best=summary.loc[[m for m,_ in METHODS],'mean'].idxmin()
    values=[cell(summary.loc[m,'mean'],summary.loc[m,'se'],bold=m==best) for m,_ in METHODS]
    write_table(output,'kuairand',[label for _,label in METHODS],[values],'ccccc')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,default=Path('results/paper'))
    parser.add_argument('--output-dir',type=Path,default=Path('figures/paper/tables'))
    parser.add_argument('--studies',nargs='+',choices=['kuairand','ablation'],required=True)
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    root,out=args.results_root,args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    reps=2 if args.quick else 10
    if 'kuairand' in args.studies:
        export_real(pd.read_csv(root/'kuairand/raw_results.csv'),out,reps,args.quick)
    if 'ablation' in args.studies:
        export_ablation(root/'ablation',out,args.quick)


if __name__=='__main__':
    main()
