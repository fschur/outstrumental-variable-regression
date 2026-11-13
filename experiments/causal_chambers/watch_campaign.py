"""Watch completed local campaign files and run offline analyses incrementally.

This process never imports the collection client or calls a hardware/API tool.
Only DONE manifests with matching joined-measurement hashes are analyzed.
Per-fit checkpoints remain inside the requested analysis output directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def utc_now():
    return datetime.now(timezone.utc)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def event(event_log, name, **fields):
    record = dict(event=name, utc=utc_now().isoformat(), **fields)
    text = json.dumps(record, sort_keys=True)
    with event_log.open("a") as stream:
        stream.write(text+"\n")
    print(text, flush=True)


def completed_inputs(data_root):
    result = {name: [] for name in ("binary", "ternary", "uniform")}
    for directory in sorted(data_root.glob("*_rep*")):
        manifest_path, csv = directory/"manifest.json", directory/"measurements.csv"
        if not manifest_path.exists() or not csv.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue  # A collector update may be in progress; read on the next pass.
        if str(manifest.get("status", "")).upper() != "DONE" or "measurements_sha256" not in manifest:
            continue
        design = manifest.get("design", "binary")
        if design not in result:
            continue
        actual = digest(csv)
        if actual != manifest["measurements_sha256"]:
            raise ValueError(f"Completed measurement hash mismatch: {csv}")
        result[design].append(dict(path=str(csv.resolve()), sha256=actual,
                                   replicate_id=manifest.get("replicate_id", directory.name)))
    return result


def frozen_code_hashes():
    files = list((ROOT/"src/ovr").glob("*.py"))
    files += [ROOT/name for name in (
        "configs/paper.json", "experiments/common/baselines.py",
        "experiments/common/linear_neural.py", "experiments/common/linear_neural_benchmark.py",
        "experiments/causal_chambers/analyze_ovr.py", "experiments/causal_chambers/campaign_analysis.py",
        "experiments/causal_chambers/kivva_baseline.py")]
    return {str(path.relative_to(ROOT)): digest(path) for path in files if path.exists()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT/"experiments/causal_chambers/data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path)
    parser.add_argument("--stop-utc", required=True)
    parser.add_argument("--expected-runs", type=int, default=13)
    parser.add_argument("--poll-seconds", type=float, default=45)
    parser.add_argument("--n-per-env", type=int, nargs="+", default=[500, 1000, 2500, 5000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--once", action="store_true", help="Analyze currently complete files once, then exit")
    args = parser.parse_args(argv)
    if not 1 <= args.poll_seconds <= 60:
        parser.error("poll-seconds must be between 1 and 60")
    stop = datetime.fromisoformat(args.stop_utc.replace("Z", "+00:00"))
    if stop.tzinfo is None:
        parser.error("stop-utc must have an explicit UTC offset")
    stop = stop.astimezone(timezone.utc)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output/"logs"
    logs.mkdir(exist_ok=True)
    log = logs/"watch_events.jsonl"
    config = dict(data_root=str(args.data_root.resolve()), output_dir=str(output),
                  calibration_data=str(args.calibration_data.resolve()) if args.calibration_data else None,
                  sizes=args.n_per_env, seeds=args.seeds, code_sha256=frozen_code_hashes())
    config_path = output/"watch_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Watcher configuration/code changed; use a new analysis output directory")
    if not config_path.exists():
        with config_path.open("x") as stream:
            json.dump(config, stream, indent=2, sort_keys=True)
            stream.write("\n")
    with (output/".analysis.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        event(log, "offline_watcher_started", stop_utc=stop.isoformat())
        previous = {}
        state_path = output/"watch_state.json"
        if state_path.exists():
            previous = json.loads(state_path.read_text())
        while utc_now() < stop:
            if frozen_code_hashes() != config["code_sha256"]:
                raise RuntimeError("Frozen estimator/analysis files changed while watcher was running")
            completed = completed_inputs(args.data_root)
            calibration = args.calibration_data if args.calibration_data and args.calibration_data.exists() else None
            if calibration and (calibration.parent/"manifest.json").exists():
                calibration_manifest = json.loads((calibration.parent/"manifest.json").read_text())
                if str(calibration_manifest.get("status", "")).upper() != "DONE" or "measurements_sha256" not in calibration_manifest:
                    calibration = None
                elif digest(calibration) != calibration_manifest["measurements_sha256"]:
                    raise ValueError("Completed calibration hash mismatch")
            calibration_hash = digest(calibration) if calibration else None
            for design, inputs in completed.items():
                if not inputs:
                    continue
                fingerprint = hashlib.sha256(json.dumps(dict(inputs=inputs, calibration=calibration_hash),
                                                        sort_keys=True).encode()).hexdigest()
                if previous.get(design, {}).get("fingerprint") == fingerprint:
                    continue
                timestamp = utc_now().strftime("%Y%m%dT%H%M%S_%fZ")
                run_log = logs/f"{design}_{timestamp}.log"
                command = [sys.executable, "-m", "experiments.causal_chambers.campaign_analysis",
                           "--data", *[item["path"] for item in inputs],
                           "--design", design, "--n-per-env", *map(str, args.n_per_env),
                           "--seeds", *map(str, args.seeds), "--output-dir", str(output/design)]
                if calibration:
                    command += ["--calibration-data", str(calibration)]
                env = os.environ.copy()
                env["PYTHONPATH"] = str(ROOT/"src")+os.pathsep+env.get("PYTHONPATH", "")
                env.setdefault("MPLCONFIGDIR", "/tmp/ovr-campaign-matplotlib")
                event(log, "analysis_started", design=design, physical_replicates=[item["replicate_id"] for item in inputs],
                      calibration_available=bool(calibration), log=str(run_log))
                try:
                    with run_log.open("x") as stream:
                        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                                 stderr=subprocess.STDOUT, timeout=900, check=False)
                    returncode = process.returncode
                    error = None
                except subprocess.TimeoutExpired:
                    returncode, error = -1, "Offline analysis exceeded 15 minutes; process was stopped"
                previous[design] = dict(fingerprint=fingerprint, returncode=returncode,
                                        log=str(run_log), error=error, updated_utc=utc_now().isoformat())
                temporary = state_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(previous, indent=2, sort_keys=True)+"\n")
                temporary.replace(state_path)
                event(log, "analysis_completed" if returncode == 0 else "analysis_failed", design=design,
                      returncode=returncode, error=error, log=str(run_log))
            total = sum(map(len, completed.values()))
            successful = all(previous.get(d, {}).get("returncode") == 0 for d, inputs in completed.items() if inputs)
            if args.once or (total >= args.expected_runs and successful and (calibration is not None or args.calibration_data is None)):
                event(log, "offline_watcher_finished", completed_physical_runs=total)
                return
            time.sleep(min(args.poll_seconds, max(0, (stop-utc_now()).total_seconds())))
        event(log, "offline_watcher_deadline", stop_utc=stop.isoformat())


if __name__ == "__main__":
    main()
