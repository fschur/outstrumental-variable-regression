"""Reproduce the experiments, figures and tables reported in the paper."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.common.config import paper_config_section
from experiments.common.paper_outputs import validate_exports

ROOT = Path(__file__).resolve().parents[1]


def commands(args):
    """Construct commands without fitting models or touching result files."""
    config=paper_config_section('synthetic')
    studies=set(args.studies or ['synthetic','ablation','tau'] + ([] if args.quick else ['kuairand']))
    quick=['--quick'] if args.quick else []
    sizes=[600] if args.quick else config['sample_sizes']
    reps=2 if args.quick else config['replications']
    result,figures=args.results_root,args.output_dir
    tasks=[]
    def add(module,*options):
        tasks.append([sys.executable,'-m',module,*map(str,options)])
    if 'synthetic' in studies:
        if not args.export_only:
            add('experiments.synthetic.run','--output-dir',result/'synthetic','--workers',args.workers,*quick)
        for name in ('synthetic_main', 'synthetic_supplementary'):
            add('experiments.synthetic.plot','--summary',result/'synthetic/summary.csv','--output-dir',figures,
                '--sample-sizes',*sizes,'--expected-reps',reps,'--name',name,*(['--pgf'] if args.pgf else []))
    if 'ablation' in studies and not args.export_only:
        add('experiments.synthetic.reproduce_ablation','--results-root',result/'ablation',
            '--workers',args.workers,
            '--sample-sizes',*config['ablation_sample_sizes'],
            '--resume',*quick)
    if 'tau' in studies:
        if not args.export_only:
            add('experiments.synthetic.run_tau_sweep','--output-dir',result/'tau','--workers',args.workers,'--resume',
                *(['--sample-sizes','600','--reps','2','--contrast-epochs','2','--residual-epochs','2','--hidden','8']
                  if args.quick else []))
        add('experiments.synthetic.export_tau_sweep','--summary',result/'tau/summary.csv',
            '--output',figures/'tau_sweep.pdf',*quick)
    if 'kuairand' in studies and not args.export_only:
        add('experiments.kuairand.run','--data-dir',args.data_dir,'--output-dir',result/'kuairand',
            *(['--download'] if args.download else []),*quick)
    tables=sorted(studies & {'kuairand','ablation'})
    if tables:
        add('experiments.export_tables','--results-root',result,'--output-dir',figures/'tables',
            '--studies',*tables,*quick)
    return tasks


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--studies',nargs='+',choices=['synthetic','ablation','tau','kuairand'])
    parser.add_argument('--results-root',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--data-dir',type=Path,default=Path('data/kuairand'))
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--download',action='store_true',help='Download KuaiRand if needed')
    parser.add_argument('--quick',action='store_true',help='Reduced-budget pipeline check, not reported results')
    parser.add_argument('--export-only',action='store_true',help='Export saved estimates without fitting models')
    parser.add_argument('--pgf',action='store_true',help='Also export synthetic plots as PGF; requires pdflatex')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    if args.workers<1:
        parser.error('workers must be positive')
    mode='quick' if args.quick else 'paper'
    args.results_root=(args.results_root or Path('results')/mode).resolve()
    args.output_dir=(args.output_dir or Path('figures')/mode).resolve()
    args.data_dir=args.data_dir.resolve()
    return args


def main():
    args=parse_args()
    if not args.dry_run:
        validate_exports(args.output_dir, pgf=args.pgf)
    for command in commands(args):
        print('+ '+shlex.join(command),flush=True)
        if not args.dry_run:
            subprocess.run(command,cwd=ROOT,check=True)
    if not args.dry_run:
        validate_exports(args.output_dir, pgf=args.pgf)


if __name__=='__main__':
    main()
