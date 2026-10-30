"""The experimental figures and tables present in the manuscript."""

FIGURES = {
    "synthetic_main": "fig:synthetic-core",
    "synthetic_supplementary": "fig:synthetic-large",
    "tau_sweep": "fig:tau-sweep",
}

TABLES = {
    "kuairand": "tab:real-data",
    "component_ablation": "tab:component-ablation",
}


def expected_exports(*, pgf=False):
    """Full reproduction output paths, excluding CSV/JSON experiment records."""
    paths = {name + ".pdf" for name in FIGURES}
    paths.update("tables/" + name + ".tex" for name in TABLES)
    if pgf:
        paths.update(name + ".pgf" for name in ("synthetic_main", "synthetic_supplementary"))
    return paths


def validate_exports(directory, *, pgf=False):
    """Reject rendered figures/tables outside the manuscript's output list."""
    from pathlib import Path
    directory = Path(directory)
    rendered = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
                if p.is_file() and p.suffix.lower() in {'.pdf', '.png', '.svg', '.pgf', '.tex'}}
    unexpected = rendered - expected_exports(pgf=pgf)
    if unexpected:
        raise ValueError('Outputs not in the paper: ' + ', '.join(sorted(unexpected))
                         + '. Use a fresh output directory or remove the obsolete files.')
    return rendered
