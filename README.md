# Outstrumental Variable Regression

Code accompanying the paper. This package contains the estimators and experiments reported in the main text and appendix.

## Installation

For ordinary use, install from this directory with `python -m pip install .`.
The package accepts compatible dependency versions. To reproduce the reported
experiments with their exact versions, use Python 3.9.6 and:

```sh
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests
```

The four numerical dependencies for reproduction are pinned in `requirements.txt`. Computation uses CPU PyTorch. Matplotlib exports PDF figures; the optional `--pgf` flag also exports the two synthetic figures in LaTeX format and requires `pdflatex`.

## Estimators

```python
import numpy as np
from ovr import DirectOVR, SmoothedOVR

rng = np.random.default_rng(0)
E = rng.integers(0, 3, size=600)
X = rng.normal(size=(600, 2))
Y = X @ np.array([1., -1.]) + (E + 1) * np.sum(X**2, axis=1) + rng.normal(size=600)

fit = SmoothedOVR(seed=0).fit(X, Y, E)
print(fit.coef_)
```

`X` has shape `(n, d)`, `Y` has shape `(n,)`, and categorical `E` uses consecutive integer labels starting at zero. Pass `groups=group_ids` to keep each group together during sample splitting and cross-fitting.

Smoothed OVR fits two contrast learners on independent subsamples. On the separate slope sample, it cross-fits noisy-input regressions of both `X` and `Y` on the contrast and environment. Held-out predictions average eight perturbations: four independent Gaussian vectors and their negatives. The coefficient regression uses the pooled residuals. Defaults are smoothing scale one, three residualization folds, and a half-sample contrast split. The `seed` argument controls splitting and network initialization without changing the caller's NumPy or PyTorch random state. Thread counts are controlled by the caller; experiment runners use one PyTorch thread per worker.

## Reproduction

Run a small end-to-end check first:

```sh
python -m experiments.reproduce --quick --workers 2
```

This checks all six synthetic settings, the component ablation, the smoothing-scale sweep, and their exports. It uses reduced sample sizes and training budgets. Its outputs in `results/quick` and `figures/quick` are **not the reported results**. It does not download data.

Reproduce the complete paper experiments:

```sh
python -m experiments.reproduce --download --workers 8
```

This downloads the public KuaiRand-Pure release from its official Zenodo archive, verifies its checksum, and runs the full configured grids. The full neural-network experiments are substantially more expensive than the quick check. Use fewer workers if memory is limited. Results go to `results/paper`; figures and LaTeX tables go to `figures/paper`.

To inspect commands, rerun exports, or select studies:

```sh
python -m experiments.reproduce --dry-run
python -m experiments.reproduce --export-only
python -m experiments.reproduce --studies synthetic ablation tau --workers 8
python -m experiments.reproduce --studies kuairand --data-dir data/kuairand
```

The synthetic runner checkpoints individual replications and rejects a resume if the code or configuration changed. Use a new `--results-root` for changed settings. The other runners also record their configurations. Exports check replication grids.

To check the KuaiRand pipeline at a reduced budget after obtaining the data:

```sh
python -m experiments.reproduce --studies kuairand --quick --data-dir data/kuairand
```

## Paper coverage

| Experiment | Settings |
| --- | --- |
| Main synthetic comparison | Six settings; sample sizes 500, 1,000, 2,000, 4,000, 10,000; ten replications |
| Supplementary synthetic comparison | Same observations and sample sizes as the main comparison, with Direct OVR included; ten replications |
| Component ablation | Five nested variants at 10,000 observations; six settings and ten paired replications, giving 60 comparisons per component |
| Smoothing-scale sweep | Scales 0, 0.25, 0.5, 1, 2, 4 at sample sizes 1,000 and 10,000; ten replications |
| KuaiRand | Ten training/reference splits; six video attributes |

The rendered outputs are limited to the following figures and tables:

| Output | Manuscript label |
| --- | --- |
| `synthetic_main.pdf` | `fig:synthetic-core` |
| `synthetic_supplementary.pdf` | `fig:synthetic-large` |
| `tau_sweep.pdf` | `fig:tau-sweep` |
| `tables/kuairand.tex` | `tab:real-data` |
| `tables/component_ablation.tex` | `tab:component-ablation` |

The component ablation produces its table only. The DAGs and conceptual method illustration are manuscript artwork, not experimental outputs. CSV and JSON files retain estimates, summaries, and checks needed to reproduce the reported numbers.

`experiments/common/paper_outputs.py` lists the permitted outputs. Reproduction rejects other rendered files in its output directory; use a fresh directory when updating an earlier export. Reduced-budget checks generate only the corresponding subset of these outputs, in a separate directory.

The comparison methods are OLS, environment fixed effects, Anchor regression with strength two, and a linear-plus-neural-network model (LNN). Direct OVR appears in supplementary comparisons. LNN selects its regularization and initialization using prediction validation, then refits on the full training sample. True coefficients and reference coefficients are used only for scoring.

The main and supplementary figures are exported as `synthetic_main` and `synthetic_supplementary`. They share the grid in `synthetic.sample_sizes`; the ablation uses `synthetic.ablation_sample_sizes`.

For KuaiRand, training/reference splits are made within user-policy cells. Training contains interactions from both policies; the held-out random-insertion interactions supply one global six-dimensional reference regression. Users and videos can occur in both halves. Internal estimator splits keep users disjoint. The reference measures a linear projection under random video assignment; it need not equal the causal effect when other confounding remains.

`configs/paper.json` records the grids and seeds. No datasets or fitted results are bundled. KuaiRand retains its own data license, supplied with the official archive.
