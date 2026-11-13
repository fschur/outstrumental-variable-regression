"""Write the measurements in measurements/ from the full chamber downloads.

The full measurements.csv files have about 106 columns: all sensor readings and
settings of the light tunnel. The analyses use only the columns in KEEP. This
script copies each run's measurements.csv with only these columns, keeping the
text of every value unchanged, together with its manifest.json (with the hash of
the new file; the original hash is kept as full_measurements_sha256) and, where
present, assignment_law.csv.

Usage (from the repository root):
  python experiments/causal_chambers/make_slim_measurements.py \
    --source experiments/causal_chambers/data --out experiments/causal_chambers/measurements
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

KEEP = ["block_id", "flag", "kind", "replicate_id", "E", "X", "Y", "H",
        "red", "led_3_ir", "ir_3", "pol_2"]
RUNS = ["final_campaign_20261110/*", "campaign_calibration_20261106"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def slim(source, target):
    with open(source, newline="") as stream_in, open(target, "w", newline="") as stream_out:
        reader = csv.reader(stream_in)
        header = next(reader)
        columns = [c for c in KEEP if c in header]
        index = [header.index(c) for c in columns]
        writer = csv.writer(stream_out, lineterminator="\n")
        writer.writerow(columns)
        for row in reader:
            writer.writerow([row[i] for i in index])
    return header, columns


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = sorted(p.parent for pattern in RUNS for p in args.source.glob(f"{pattern}/measurements.csv"))
    for run in runs:
        manifest = json.loads((run / "manifest.json").read_text())
        if manifest.get("status") != "DONE" or manifest.get("measurements_sha256") != sha256(run / "measurements.csv"):
            raise ValueError(f"Not a completed, unchanged run: {run}")
        target = args.out / run.relative_to(args.source)
        target.mkdir(parents=True, exist_ok=True)
        header, columns = slim(run / "measurements.csv", target / "measurements.csv")
        manifest.update(full_measurements_sha256=manifest["measurements_sha256"],
                        full_measurements_columns=len(header), columns=columns,
                        measurements_sha256=sha256(target / "measurements.csv"))
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if (run / "assignment_law.csv").exists():
            shutil.copy2(run / "assignment_law.csv", target / "assignment_law.csv")
        print(f"{run.relative_to(args.source)}: {len(header)} -> {len(columns)} columns")


if __name__ == "__main__":
    main()
