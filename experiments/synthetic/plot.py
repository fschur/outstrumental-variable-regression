"""Export the main and supplementary synthetic figures from matched summaries."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import pandas as pd

from .paper_plot import validate_summary, export_figure


def _parse_sample_sizes(values):
    sizes = tuple(int(t) for value in values for t in value.split(','))
    if not sizes or min(sizes) <= 0 or len(set(sizes)) != len(sizes):
        raise ValueError('Sample sizes must be distinct positive integers')
    return sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-sizes', nargs='+', required=True)
    parser.add_argument('--expected-reps', type=int, default=10)
    parser.add_argument('--pgf', action='store_true')
    parser.add_argument('--name', choices=['synthetic_main', 'synthetic_supplementary'], default='synthetic_main')
    args = parser.parse_args()
    sizes = _parse_sample_sizes(args.sample_sizes)
    data = pd.read_csv(args.summary)
    data = data[data.n.isin(sizes)].copy()
    validate_summary(data, sizes, args.expected_reps)
    export_figure(data, sizes, args.output_dir, pgf=args.pgf, name=args.name)


if __name__ == '__main__':
    main()
