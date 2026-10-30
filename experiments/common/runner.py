"""Checkpoint independent replications and aggregate matched estimates."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import pandas as pd

from .results import clean_json


def run_tasks(tasks, worker, output, config, workers=1):
    """Reuse a checkpoint only when its configuration and implementation match."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "replications"
    checkpoints.mkdir(exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    code = hashlib.sha256()
    for directory in (root / "src", root / "experiments"):
        for path in sorted(directory.rglob("*.py")):
            code.update(path.relative_to(root).as_posix().encode())
            code.update(path.read_bytes())
    manifest = clean_json(dict(config=config, code_sha256=code.hexdigest(), tasks=tasks))
    path = output / "config.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Configuration or code changed; use a new results directory")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    pending = [t for t in tasks if not (checkpoints / (t['name'] + '.json')).exists()]
    def save(task, result):
        path = checkpoints / (task['name'] + '.json')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(clean_json(result), sort_keys=True) + '\n')
        temporary.replace(path)
        print('Completed ' + task['name'], flush=True)
    if workers == 1:
        for task in pending:
            save(task, worker(task))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            for task, result in zip(pending, pool.map(worker, pending)):
                save(task, result)
    results = [json.loads((checkpoints / (t['name'] + '.json')).read_text()) for t in tasks]
    return results


def write_estimates(rows, output):
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(output) / 'raw_results.csv', index=False)
    frame.groupby(['dataset', 'scenario', 'n', 'method']).coefficient_error.agg(
        coefficient_error_mean='mean', coefficient_error_se='sem', n_reps='count'
    ).reset_index().to_csv(Path(output) / 'summary.csv', index=False)
    return frame
