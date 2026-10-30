"""Checks for the public reproduction entry point and scored exports."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from experiments.reproduce import commands, parse_args
from experiments.common.config import paper_config_section
from experiments.common.paper_outputs import expected_exports, validate_exports
from experiments.export_tables import export_ablation, export_real
from experiments.synthetic.paper_plot import PANELS
from experiments.synthetic.run import default_sample_sizes
from experiments.synthetic import reproduce_ablation
from experiments.synthetic.run_ablation import VARIANTS


class ReleaseTests(unittest.TestCase):
    def test_sample_grids_match_the_figures_and_ablation_table(self):
        config = paper_config_section('synthetic')
        self.assertEqual(config['sample_sizes'], [500, 1000, 2000, 4000, 10000])
        self.assertEqual(config['ablation_sample_sizes'], [10000])
        self.assertEqual(default_sample_sizes(), config['sample_sizes'])
        plan = commands(parse_args(['--studies', 'synthetic', 'ablation']))
        plots = [c for c in plan if c[2] == 'experiments.synthetic.plot']
        self.assertEqual(len(plots), 2)
        for c in plots:
            start, stop = c.index('--sample-sizes')+1, c.index('--expected-reps')
            self.assertEqual(list(map(int, c[start:stop])), config['sample_sizes'])
            self.assertNotIn('--shared-y', c)
        self.assertEqual([c[c.index('--name')+1] for c in plots],
                         ['synthetic_main', 'synthetic_supplementary'])
        ablation = next(c for c in plan if c[2] == 'experiments.synthetic.reproduce_ablation')
        self.assertEqual(ablation[ablation.index('--sample-sizes')+1], '10000')

    def test_ablation_cli_defaults_to_ten_thousand(self):
        with tempfile.TemporaryDirectory() as root:
            argv = ['ablation', '--results-root', root]
            with patch('sys.argv', argv), patch.object(reproduce_ablation, '_run') as run:
                reproduce_ablation.main()
        self.assertEqual(run.call_count, 1)
        training = run.call_args.args[0]
        self.assertEqual(training[training.index('--n')+1], '10000')

    def test_fixed_n_table_uses_sixty_paired_ratios(self):
        # Known paired changes: error halves, moment quarters, scaling doubles.
        rows = []
        for scenario, _ in PANELS:
            for rep in range(10):
                for variant in VARIANTS:
                    error = (rep+1) * .5**variant.step
                    scaling = 2.**variant.step
                    rows.append(dict(scenario_key=scenario, rep=rep, n=10000,
                        variant_key=variant.key, success=True, split_plan_hash='same',
                        fold_hash='same', contrast_artifact_hash='same',
                        coefficient_error=error, ridge_adjusted_moment_norm=error/scaling,
                        effective_amplification=scaling, minimum_residual_eigenvalue=.5,
                        gamma=.5, contrast_rmse=.1))
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            results, out = root/'n10000', root/'tables'
            results.mkdir()
            out.mkdir()
            raw = pd.DataFrame(rows)
            raw.to_csv(results/'raw_results.csv', index=False)
            export_ablation(root, out)
            table = pd.read_csv(out/'component_ablation.csv')
            np.testing.assert_array_equal(table.n, [10000]*4)
            np.testing.assert_array_equal(table.n_pairs, [60]*4)
            np.testing.assert_array_equal(table.wins, [60]*4)
            np.testing.assert_allclose(table.error, .5)
            np.testing.assert_allclose(table.moment, .25)
            np.testing.assert_allclose(table.amplification, 2.)
            text = (out/'component_ablation.tex').read_text()
            self.assertIn(r'\widehat G_\lambda^{-1}', text)
            self.assertIn('% n=10000; 60 paired comparisons per row.', text)
            raw.loc[0, 'n'] = 20000
            raw.to_csv(results/'raw_results.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'sample sizes'):
                export_ablation(root, out)

    def test_quick_default_runs_all_simulations_without_download(self):
        with tempfile.TemporaryDirectory() as root:
            plan=commands(parse_args(['--quick','--results-root',root]))
        joined='\n'.join(' '.join(c) for c in plan)
        for module in ('experiments.synthetic.run','experiments.synthetic.run_tau_sweep',
                       'experiments.synthetic.reproduce_ablation','experiments.export_tables'):
            self.assertIn(module,joined)
        self.assertNotIn('experiments.kuairand.run',joined)
        self.assertNotIn('--download',joined)

    def test_exports_never_fit(self):
        plan=commands(parse_args(['--export-only']))
        self.assertFalse(any(c[2] in ('experiments.synthetic.run','experiments.kuairand.run',
                                     'experiments.synthetic.run_tau_sweep',
                                     'experiments.synthetic.reproduce_ablation') for c in plan))
        tables=next(c for c in plan if c[2]=='experiments.export_tables')
        self.assertIn('ablation',tables)

    def test_export_inventory_matches_the_paper(self):
        expected = {'synthetic_main.pdf', 'synthetic_supplementary.pdf', 'tau_sweep.pdf',
                    'tables/kuairand.tex',
                    'tables/component_ablation.tex'}
        self.assertEqual(expected_exports(), expected)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in expected:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root/'tables/kuairand.csv').touch()
            self.assertEqual(validate_exports(root), expected)
            extra = root/'extra_figure.pdf'
            extra.touch()
            with self.assertRaisesRegex(ValueError, 'Outputs not in the paper'):
                validate_exports(root)
            extra.unlink()
            (root/'synthetic_main.pgf').touch()
            with self.assertRaises(ValueError):
                validate_exports(root)
            self.assertEqual(validate_exports(root, pgf=True), expected | {'synthetic_main.pgf'})

    def test_kuairand_table_has_only_the_five_reported_methods(self):
        methods = ['DOVR', 'SOVR', 'OLS', 'Environment FE', 'Anchor-g2', 'LNN']
        rows = [dict(method=m, rep=rep, data_hash='data', beta_target='[1.0]',
                     n=160158, coefficient_error=(index+1+rep)/100)
                for rep in range(10) for index, m in enumerate(methods)]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            export_real(pd.DataFrame(rows), out, reps=10)
            table = (out/'kuairand.tex').read_text()
            self.assertIn(r'\begin{tabular}{ccccc}', table)
            self.assertNotIn('DOVR', table)
            summary = pd.read_csv(out/'kuairand.csv')
            self.assertEqual(summary.method.tolist(), methods[1:])



if __name__=='__main__':
    unittest.main()
