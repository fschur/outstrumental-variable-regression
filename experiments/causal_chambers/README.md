# Light-tunnel experiment

Code for the light-tunnel experiment of the paper. We use the light tunnel (Mk2) of the
Causal Chambers (Gamella et al., 2025). The environment changes how strongly a hidden
confounder affects the outcome, and randomized measurements give a reference for the
causal effect.

| Paper | Chamber variable | Values |
|---|---|---|
| Treatment `X` | `led_3_ir`, brightness setting of the infrared LED next to the third light sensor | 8, ..., 24 |
| Outcome `Y` | `ir_3`, infrared reading of the third light sensor | |
| Hidden confounder `H` | `red`, brightness setting of the red LEDs of the light source | 32, 192 (three-level: also 112; grid runs: 32, 72, 112, 152, 192) |
| Environment `E` | `pol_2`, angle of the second polarizer (the first is fixed at 0°) | 0°, 60° (binary runs also 20°, 40°) |

The methods only see `X`, `Y` and `E` of the confounded measurements.

## Scripts

| Script | Purpose |
|---|---|
| `collect.py` | Chamber API client; prepares the schedules of the binary assignment and the grid runs |
| `prepare_alternatives.py` | Prepares the schedules of the three-level and additive assignments |
| `campaign.py` | Submits prepared runs to the chamber in parts, downloads and validates them |
| `campaign_analysis.py` | Main analysis of one assignment (Table 4, the sample-size figure, the appendix table) |
| `analyze_ovr.py` | Helper functions used by the analyses |
| `kivva_baseline.py` | Our implementation of the fourth-order moment estimator of Kivva et al. (2025, Thm. 3.4) |
| `angle_analysis.py` | Comparison of angles in the binary runs |
| `moment_check.py` | Moment estimator on all three assignments |
| `make_paper_figures.py` | The three light-tunnel figures |
| `watch_campaign.py` | Runs `campaign_analysis.py` on runs as they finish during collection |
| `make_slim_measurements.py` | Writes `measurements/` from the full chamber downloads |

## Environment

The experiment used Python 3.12.6 with the package versions in `requirements.txt`, which
differ from the versions in the repository's main `requirements.txt`. Run all commands
from the repository root.

```sh
python3.12 -m venv .venv-chamber
source .venv-chamber/bin/activate
python -m pip install -r experiments/causal_chambers/requirements.txt
python -m pip install --no-deps -e .
```

## Data

`measurements/` contains all measurements used in the paper: 662,000 rows from the
campaign of 10–12 November 2026 and 1,800 rows from an earlier grid run.

```
measurements/final_campaign_20261110/
  {binary,ternary,uniform}_rep01 ... _rep10/   the 30 repetitions
  grid_before/, grid_middle/, grid_after/      the three grid runs
measurements/campaign_calibration_20261106/    the earlier grid run that gives the
                                               midpoint of the three-level assignment
```

Each run has `measurements.csv`, `manifest.json` (design, seed, fixed settings, chamber
job IDs) and, for the three-level and additive runs, `assignment_law.csv`. The files
keep the 12 columns that the analyses use: `block_id`, `flag` (row number), `kind`
(`confounded` or `reference`, the randomized measurements), `replicate_id`, `E`, `X`,
`Y`, `H`, and the raw chamber variables `red`, `led_3_ir`, `ir_3` and `pol_2`. The full
chamber downloads have about 106 columns, with all other sensor readings and settings;
`make_slim_measurements.py` derives these files from them, keeping every value
unchanged. The analyses give identical results on both.

The analysis checks each `measurements.csv` against the hash in its `manifest.json`.
In the binary runs, `E` = 0, 1, 2, 3 means 0°, 60°, 20°, 40°; in the grid runs, 0°, 20°, 40°, 60°.

## Reproducing the results (no chamber access needed)

Results go to `experiments/causal_chambers/data/`, which is ignored by git.

```sh
C=experiments/causal_chambers/measurements/final_campaign_20261110
D=experiments/causal_chambers/data
export PYTHONPATH=src

# Main analysis, one call per assignment
for design in binary ternary uniform; do
  python -m experiments.causal_chambers.campaign_analysis \
    --data $C/${design}_rep*/measurements.csv --design $design \
    --calibration-data $C/grid_before/measurements.csv \
    --n-per-env 500 1000 2500 5000 --seeds 0 1 2 \
    --output-dir $D/final_campaign_20261110_analysis/$design
done

# Comparison of angles (binary runs)
python -m experiments.causal_chambers.angle_analysis \
  --data $C/binary_rep*/measurements.csv --output-dir $D/final_campaign_20261110_angle_analysis

# Moment estimator on all assignments
python -m experiments.causal_chambers.moment_check --data-root $C \
  --analysis-dir $D/final_campaign_20261110_analysis --out $D/final_campaign_20261110_moment_check

# Figures
python experiments/causal_chambers/make_paper_figures.py --out figures/chamber
```

The fits use three seeds per repetition and nested subsets of whole blocks; all sample
splits keep the measurements of a block together. Errors are in percent of each
repetition's own reference.

## Collecting new data (needs access to a chamber)

Credentials go in `experiments/causal_chambers/.credentials`, which is ignored by git.
The schedules of the reported campaign were prepared with these commands and seeds:

```sh
C=experiments/causal_chambers/data/final_campaign_20261110
for i in $(seq 1 10); do r=$(printf %02d $i)
  python experiments/causal_chambers/collect.py prepare $C/binary_rep$r --stage smoke \
    --blocks 100 --train-per-block 50 --reference-per-block 20 --center 16 --scale 8 \
    --angles 0 60 20 40 --replicate-id final_binary_rep$r --seed $((2026111010 + i))
  python experiments/causal_chambers/prepare_alternatives.py $C/ternary_rep$r --design ternary \
    --blocks 100 --train-per-block 50 --reference-per-block 20 --midpoint -0.00709077 \
    --replicate-id final_ternary_rep$r --seed $((2026111020 + i))
  python experiments/causal_chambers/prepare_alternatives.py $C/uniform_rep$r --design uniform \
    --blocks 100 --train-per-block 50 --reference-per-block 20 --midpoint 0 \
    --replicate-id final_uniform_rep$r --seed $((2026111040 + i))
done
for k in 1 2 3; do run=$(echo before middle after | cut -d' ' -f$k)
  python experiments/causal_chambers/collect.py prepare $C/grid_$run --stage calibration \
    --blocks 100 --levels $(seq 8 24) --red 32 72 112 152 192 --angles 0 20 40 60 --seed $((2026111060 + k))
done
```

The midpoint `-0.00709077` of the three-level assignment is f(112) at 0°, estimated
from a grid run of 6 November 2026 with the same normalization as in the paper.

`campaign.py split RUN --blocks-per-part K --tag-prefix P` divides a prepared run into
chamber jobs of whole blocks (we used 25 blocks for the grid and binary runs and 50
for the others). `campaign.py run QUEUE --stop-utc T` then submits the runs listed in
`QUEUE`, a JSON list of run directories, keeping at most `max_in_flight` jobs on the
chamber (`control.json` next to the queue file, read on every pass). It can be stopped
and restarted with the same command; `campaign.py status QUEUE` shows the progress.
When all parts of a run are downloaded and validated, they are merged into
`measurements.csv`. We ran the 30 repetitions in ten rounds, each with one repetition
per assignment in random order, with grid runs before, after half and after all of them.
