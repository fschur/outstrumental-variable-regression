"""Run the paired fits used in the component-ablation table."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from experiments.common.config import DEFAULT_PAPER_CONFIG, paper_config_section


REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_sample_sizes(values: list[str]) -> tuple[int, ...]:
    tokens = [token for value in values for token in value.split(",")]
    sizes = tuple(int(token.strip()) for token in tokens if token.strip())
    if not sizes or any(size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("sample sizes must be distinct positive integers")
    return sizes


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    config = paper_config_section("synthetic", DEFAULT_PAPER_CONFIG)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / "results" / "ovr_component_ablation",
    )
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        default=[str(value) for value in config["ablation_sample_sizes"]],
    )
    parser.add_argument("--reps", type=int, default=int(config["replications"]))
    parser.add_argument("--workers", type=int, default=int(config["workers"]))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.results_root = args.results_root.expanduser().resolve()
    try:
        sample_sizes = _parse_sample_sizes(args.sample_sizes)
    except ValueError as exc:
        parser.error(str(exc))
    if args.quick:
        sample_sizes = (600,)
        args.reps = 2
    if args.reps <= 0 or args.workers <= 0:
        parser.error("reps and workers must be positive")

    for n in sample_sizes:
        command = [
            sys.executable,
            "-m",
            "experiments.synthetic.run_ablation",
            "--n",
            str(n),
            "--reps",
            str(args.reps),
            "--workers",
            str(args.workers),
            "--main-seed",
            str(config["main_seed"]),
            "--shared-seed",
            str(config["shared_decoder_seed"]),
            "--contrast-epochs",
            str(2 if args.quick else config["contrast_epochs"]),
            "--residual-epochs",
            str(2 if args.quick else config["residual_epochs"]),
            "--output-dir",
            str(args.results_root / f"n{n}"),
        ]
        if args.quick:
            command += ["--hidden", "8"]
        if args.resume:
            command.append("--resume")
        _run(command)


if __name__ == "__main__":
    main()
