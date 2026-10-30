"""Shared hashes, coefficient scoring and report formatting."""

import hashlib
import json

import numpy as np




def array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()



def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value



def estimate_row(method, coefficient, target, diagnostics=None, **metadata):
    diagnostics = diagnostics or {}
    row = dict(metadata, method=method, success=coefficient is not None and bool(np.isfinite(coefficient).all()),
               coefficient_error=float(np.linalg.norm(coefficient-target)) if coefficient is not None else np.nan,
               beta_hat=json.dumps(clean_json(coefficient)), beta_target=json.dumps(clean_json(target)),
               diagnostics=json.dumps(clean_json(diagnostics), sort_keys=True))
    for key, value in diagnostics.items():
        if np.isscalar(value) or value is None:
            row[key] = value
    return row



def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"]*len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])
