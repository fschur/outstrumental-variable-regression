"""Load the single configuration used for the paper's reported runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER_CONFIG = REPO_ROOT / "configs" / "paper.json"

REQUIRED_KEYS = {
    "synthetic": {
        "sample_sizes",
        "ablation_sample_sizes",
        "replications",
        "workers",
        "main_seed",
        "shared_decoder_seed",
        "contrast_epochs",
        "residual_epochs",
        "methods",
        "main_scenarios",
        "shared_scenarios",
    },
    "kuairand": {
        "seed",
        "replications",
        "per_user_environment_cap",
        "lnn_epochs",
    },

}


def load_paper_config(path: Path | str = DEFAULT_PAPER_CONFIG) -> dict[str, Any]:
    """Return a validated paper configuration from ``path``."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"paper config must contain a JSON object: {config_path}")
    return config


def paper_config_section(
    name: str,
    path: Path | str = DEFAULT_PAPER_CONFIG,
) -> dict[str, Any]:
    """Return one object-valued section of the paper configuration."""
    config = load_paper_config(path)
    section = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"paper config section {name!r} must be a JSON object")
    required = REQUIRED_KEYS.get(name, set())
    missing = sorted(required - set(section))
    if missing:
        raise ValueError(
            f"paper config section {name!r} is missing: {', '.join(missing)}"
        )
    return section
