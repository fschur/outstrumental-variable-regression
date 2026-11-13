"""Prepare, submit, and retrieve small reproducible Light Tunnel experiments.

Preparation is offline. Only the explicit ``submit`` command starts hardware.
Credentials stay in .credentials; manifests never contain credentials or URLs.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import signal

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CHAMBER = "lt-test-b0ni"
FIXED = {
    "green": 0, "blue": 0, "pol_1": 0,
    "led_1_ir": 0, "led_1_uv": 0, "led_2_ir": 0,
    "led_2_uv": 0, "led_3_uv": 0,
    "t_ir_1": 3, "t_vis_1": 3, "t_ir_2": 3, "t_vis_2": 3,
    "t_ir_3": 3, "t_vis_3": 3,
    "diode_ir_1": 2, "diode_ir_2": 2, "diode_ir_3": 2,
    "diode_vis_1": 1, "diode_vis_2": 1, "diode_vis_3": 1,
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_manifest(directory, manifest):
    path = directory / "manifest.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def prepare(args):
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        raise ValueError("A manifest already exists; use a new output directory.")
    rng = np.random.default_rng(args.seed)
    rows = []
    red = args.red
    angles = args.angles
    if args.stage == "calibration":
        for block in range(args.blocks):
            for e in rng.permutation(len(angles)):
                cells = [(x, h) for x in args.levels for h in range(len(red))]
                for j in rng.permutation(len(cells)):
                    x, h = cells[j]
                    rows.append(dict(kind="calibration", block_id=block,
                                     E=int(e), H=h, X=x,
                                     red=red[h], pol_2=angles[e]))
    else:
        if len(red) != 2 or len(angles) < 2:
            raise ValueError("The binary smoke design requires two source levels and at least two angles.")
        if args.center < args.scale or args.center + args.scale > 4095:
            raise ValueError("The chosen center/scale requires clipping.")
        # E indexes args.angles. Randomized reference rows are taken only at
        # the reference angles (default: all). With two angles the random
        # draws are identical to the original two-angle implementation.
        reference_angles = angles if args.reference_angles is None else args.reference_angles
        if not set(reference_angles).issubset(angles) or len(set(reference_angles)) < 2:
            raise ValueError("Reference angles must be at least two of the angles.")
        environments = dict(confounded=list(range(len(angles))),
                            reference=[angles.index(a) for a in reference_angles])
        for block in range(args.blocks):
            # Different physical rows for training and randomized reference.
            # Every block contains all environments and both data kinds.
            for kind in rng.permutation(["confounded", "reference"]):
                count = args.train_per_block if kind == "confounded" else args.reference_per_block
                for j in rng.permutation(len(environments[kind])):
                    e = environments[kind][j]
                    latent = rng.binomial(1, .5, size=count)
                    r = rng.beta(np.where(latent == 0, 2, 5),
                                 np.where(latent == 0, 5, 2))
                    k = np.rint(args.scale * r).astype(int)
                    s = np.where(rng.random(count) < .8, 1, -1)
                    x = args.center + s * k
                    h = latent if kind == "confounded" else rng.binomial(1, .5, size=count)
                    for xi, hi in zip(x, h):
                        rows.append(dict(kind=kind, block_id=block, E=int(e),
                                         H=int(hi), X=int(xi), red=red[hi],
                                         pol_2=angles[e]))
    schedule = pd.DataFrame(rows)
    schedule.insert(0, "flag", np.arange(1, len(schedule) + 1))
    schedule["led_3_ir"] = schedule.X
    schedule.to_csv(directory / "schedule.csv", index=False)
    manifest = dict(stage=args.stage, created_utc=timestamp(), chamber_id=CHAMBER,
                    config="standard", seed=args.seed, n_measurements=len(schedule),
                    n_blocks=args.blocks, fixed_controls=FIXED, wait_ms=50,
                    red_levels=red, angles=angles,
                    schedule_sha256=hashlib.sha256((directory / "schedule.csv").read_bytes()).hexdigest(),
                    status="PREPARED")
    if args.stage == "calibration":
        manifest.update(treatment_levels=args.levels,
                        pilot_targets={"relative_slope_difference": .05,
                                       "relative_curvature": .02,
                                       "sensor_ceiling": 65535,
                                       "interpretation": "Exploratory tolerances; inspect uncertainty and all cells, not just null-test p-values."})
    else:
        manifest.update(center=args.center, scale=args.scale, sign_probability=.8,
                        h_prior=.5, beta_parameters=[[2, 5], [5, 2]],
                        train_per_environment=args.blocks * args.train_per_block,
                        reference_per_environment=args.blocks * args.reference_per_block,
                        fitting="Existing OVR defaults; fixed seeds 0,1,2; acquisition-block splits; no tuning against reference.")
        if args.reference_angles is not None:
            manifest.update(reference_angles=args.reference_angles)
        if args.replicate_id is not None:
            manifest.update(design="binary", replicate_id=args.replicate_id)
    save_manifest(directory, manifest)
    print(json.dumps(manifest, indent=2))


def api_client():
    with redirect_stdout(io.StringIO()):
        from causalchamber.lab.api import API
    return API(credentials_file=str(HERE / ".credentials"))


def submit(directory):
    with redirect_stdout(io.StringIO()):
        from causalchamber.lab.lab import Protocol
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "PREPARED":
        raise ValueError("This run has already been submitted or attempted; inspect it before retrying.")
    actual_hash = hashlib.sha256((directory / "schedule.csv").read_bytes()).hexdigest()
    if actual_hash != manifest["schedule_sha256"]:
        raise ValueError("Schedule changed after preparation.")
    schedule = pd.read_csv(directory / "schedule.csv")
    protocol = Protocol(manifest["chamber_id"], manifest["config"], api_client())
    for name, value in manifest["fixed_controls"].items():
        protocol.set(name, value)
    protocol.from_df(schedule[["flag", "red", "pol_2", "led_3_ir"]],
                     wait=manifest["wait_ms"])
    (directory / "instructions.txt").write_text("\n".join(protocol.instructions) + "\n")
    manifest.update(status="SUBMISSION_ATTEMPTED", submitted_utc=timestamp())
    save_manifest(directory, manifest)
    experiment_id = protocol.submit(tag=f"ovr_{directory.name}")
    manifest.update(status="SUBMITTED", experiment_id=experiment_id)
    save_manifest(directory, manifest)
    print(json.dumps({"experiment_id": experiment_id, "n_measurements": len(schedule)}), flush=True)


def fetch(directory):
    with redirect_stdout(io.StringIO()):
        from causalchamber.lab.lab import ExperimentDataset
    manifest = json.loads((directory / "manifest.json").read_text())
    if hashlib.sha256((directory / "schedule.csv").read_bytes()).hexdigest() != manifest["schedule_sha256"]:
        raise ValueError("Schedule changed after preparation.")
    eid = manifest["experiment_id"]
    result = api_client().make_request("GET", f"experiments/{eid}").json()
    status = result["status"]
    manifest.update(status=status, last_checked_utc=timestamp())
    # Do not persist signed download URLs or authentication information.
    for key in ("submitted_on", "submission_time", "start_time", "end_time", "runtime", "n_observations", "current_instruction", "total_instructions"):
        if key in result:
            manifest[key] = result[key]
    save_manifest(directory, manifest)
    print(json.dumps({"experiment_id": eid, "status": status,
                      "current_instruction": result.get("current_instruction"),
                      "total_instructions": result.get("total_instructions")}), flush=True)
    if str(status).upper() != "DONE":
        return
    if (directory / "measurements.csv").exists():
        if hashlib.sha256((directory / "measurements.csv").read_bytes()).hexdigest() != manifest["measurements_sha256"]:
            raise ValueError("Previously downloaded measurements changed.")
        print("Measurements already downloaded and validated.")
        return
    raw_dir = directory / "raw"
    raw_dir.mkdir(exist_ok=True)
    dataset = ExperimentDataset(eid, result["download_url"], result["checksum"], raw_dir, verbose=False)
    frame = dataset.dataframe
    frame.to_csv(directory / "raw_measurements.csv", index=False)
    schedule = pd.read_csv(directory / "schedule.csv")
    if len(frame) != len(schedule) or not frame.flag.is_unique:
        raise ValueError("Unexpected measurement count or duplicate flags; raw data saved for inspection.")
    frame["flag"] = frame.flag.astype(int)
    expected = schedule.set_index("flag")
    frame = frame.set_index("flag").loc[expected.index]
    for column in ("red", "pol_2", "led_3_ir"):
        if not np.allclose(frame[column], expected[column], atol=1e-5, rtol=0):
            raise ValueError(f"Measured controls disagree with schedule: {column}")
    for column in manifest["fixed_controls"]:
        if column in frame and not np.allclose(frame[column], manifest["fixed_controls"][column], atol=1e-5, rtol=0):
            raise ValueError(f"Fixed control drift: {column}")
    frame = frame.copy()
    for column in expected.columns:
        if column not in frame:
            frame[column] = expected[column]
    frame["Y"] = frame.ir_3
    frame.reset_index().to_csv(directory / "measurements.csv", index=False)
    manifest.update(measurements_sha256=hashlib.sha256((directory / "measurements.csv").read_bytes()).hexdigest(),
                    downloaded_utc=timestamp(), validated_rows=len(frame))
    save_manifest(directory, manifest)
    print(json.dumps({"validated_rows": len(frame), "ir_3_range": [float(frame.ir_3.min()), float(frame.ir_3.max())],
                      "elapsed_measurement_seconds": float(frame.timestamp.max() - frame.timestamp.min())}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("directory", type=Path)
    prep.add_argument("--stage", choices=["calibration", "smoke"], default="calibration")
    prep.add_argument("--seed", type=int, default=20261106)
    prep.add_argument("--blocks", type=int, default=8)
    prep.add_argument("--levels", type=int, nargs="+", default=[0, 4, 8, 12, 16, 20, 24])
    prep.add_argument("--red", type=int, nargs="+", default=[32, 192])
    prep.add_argument("--angles", type=float, nargs="+", default=[0, 60])
    prep.add_argument("--center", type=int, default=16)
    prep.add_argument("--scale", type=int, default=8)
    prep.add_argument("--train-per-block", type=int, default=32)
    prep.add_argument("--reference-per-block", type=int, default=16)
    prep.add_argument("--reference-angles", type=float, nargs="+",
                      help="Angles with randomized reference rows (default: all angles)")
    prep.add_argument("--replicate-id", help="Record design and replicate ID in the manifest")
    for command in ("submit", "fetch"):
        p = sub.add_parser(command)
        p.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    else:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("API operation timed out")))
        signal.alarm(55)
        try:
            (submit if args.action == "submit" else fetch)(args.directory.resolve())
        finally:
            signal.alarm(0)


if __name__ == "__main__":
    main()
