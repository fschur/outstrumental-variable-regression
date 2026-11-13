"""Hardware coordinator for the November 10 campaign; runs are submitted in parts.

Each prepared run (schedule.csv, manifest.json from collect.py or
prepare_alternatives.py) is divided into parts of whole blocks. Each part is
one chamber job, no larger than the jobs of the November 6-7 campaign. When
all parts of a run are downloaded and validated, they are merged into the
usual measurements.csv and the manifest is marked DONE, so campaign_analysis.py
and watch_campaign.py read the run exactly like a single-job run.

Subcommands
  split RUN --blocks-per-part K --tag-prefix P   offline; divide a PREPARED run
  run QUEUE --stop-utc T                         the only process that submits
  status QUEUE                                   offline progress summary
  retry-part RUN PART                            new tag for a FAILED part
  clear-stop QUEUE --note TEXT                   lift a stop after inspection

The coordinator keeps at most max_in_flight parts (control.json next to the
queue file, read on every pass) on the chamber and submits in queue order.
All state is in each run's manifest.json and parts.json, written atomically,
so the coordinator can be stopped at any time and restarted with the same
queue. Network failures are logged and retried; they never stop the process.

A submission whose outcome is uncertain (status SUBMISSION_ATTEMPTED, e.g.
when the connection dropped during the request) is resolved through its unique
tag in the account's experiment list: an experiment with that tag is adopted;
the part is submitted again only if, at least RECHECK_SECONDS after the
attempt, no experiment with that tag exists. Every submission is preceded by
the same tag check. This assumes that an accepted job appears in the list
within RECHECK_SECONDS. If it appears later, two experiments carry the same
tag; the coordinator checks for this whenever it reads the list (before each
submission, every 10 minutes and before it exits) and then stops new submissions.

Stops caused by failures (two failed parts in a row, a validation error, a
repeated tag) are saved in coordinator_state.json next to the queue and
survive restarts; only clear-stop lifts them, while no coordinator runs.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import io
import json
from pathlib import Path
import signal
import time

import numpy as np
import pandas as pd

from collect import api_client, save_manifest, timestamp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UNSUCCESSFUL = {"FAILED", "CANCELED", "CANCELLED", "ERROR", "ABORTED"}
RECHECK_SECONDS = 600
REJECTION_HOLD_SECONDS = 900
API_SECONDS = 55
HEARTBEAT_SECONDS = 1800
TAG_CHECK_SECONDS = 600
DOWNLOAD_SECONDS = 600


class ValidationError(Exception):
    """Downloaded data disagree with the schedule; never retried automatically."""


def utc_now():
    return datetime.now(timezone.utc)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def with_timeout(seconds, function, *args):
    def handler(*_):
        raise TimeoutError(f"API operation exceeded {seconds} s")
    previous = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        return function(*args)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def lab_errors():
    with redirect_stdout(io.StringIO()):
        from causalchamber.lab.exceptions import LabError, UserError
    return LabError, UserError


class Log:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event, **fields):
        text = json.dumps(dict(event=event, utc=utc_now().isoformat(), **fields), sort_keys=True)
        with self.path.open("a") as stream:
            stream.write(text + "\n")
        print(text, flush=True)


# ---------------------------------------------------------------- run state

def resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_schedule(directory, manifest):
    if sha256(directory / "schedule.csv") != manifest["schedule_sha256"]:
        raise ValidationError(f"Schedule changed after preparation: {directory}")
    return pd.read_csv(directory / "schedule.csv")


def load_run(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    parts_path = directory / "parts.json"
    if not parts_path.exists():
        raise ValueError(f"Run has not been split into parts: {directory}")
    state = json.loads(parts_path.read_text())
    if state["schedule_sha256"] != manifest["schedule_sha256"]:
        raise ValidationError(f"parts.json belongs to another schedule: {directory}")
    return manifest, state


def save_parts(directory, state):
    write_json(directory / "parts.json", state)


def part_rows(schedule, part):
    rows = schedule.loc[schedule.block_id.between(part["first_block"], part["last_block"])]
    if len(rows) != part["n_rows"] or rows.flag.min() != part["first_flag"] or rows.flag.max() != part["last_flag"]:
        raise ValidationError(f"Part {part['part']} no longer matches the schedule")
    return rows


def split(args):
    directory = args.directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "PREPARED":
        raise ValueError("Only a PREPARED, never submitted run can be split.")
    if (directory / "parts.json").exists():
        raise ValueError("parts.json already exists.")
    schedule = load_schedule(directory, manifest)
    blocks = schedule.block_id.to_numpy()
    if np.any(np.diff(blocks) < 0) or not np.array_equal(np.unique(blocks), np.arange(manifest["n_blocks"])):
        raise ValueError("Schedule rows must be ordered by consecutive block IDs.")
    if not np.array_equal(schedule.flag.to_numpy(), np.arange(1, len(schedule) + 1)):
        raise ValueError("Flags must be 1..n in schedule order.")
    n_blocks, size = manifest["n_blocks"], args.blocks_per_part
    parts = []
    for k, first in enumerate(range(0, n_blocks, size), start=1):
        last = min(first + size, n_blocks) - 1
        rows = schedule.loc[schedule.block_id.between(first, last)]
        parts.append(dict(part=k, first_block=first, last_block=last,
                          first_flag=int(rows.flag.min()), last_flag=int(rows.flag.max()),
                          n_rows=len(rows), n_instructions=len(manifest["fixed_controls"]) + 6 * len(rows),
                          tag=f"{args.tag_prefix}_{directory.name}_p{k}", status="PREPARED"))
    state = dict(run=directory.name, schedule_sha256=manifest["schedule_sha256"],
                 blocks_per_part=size, created_utc=timestamp(), parts=parts)
    save_parts(directory, state)
    manifest.update(status="IN_PARTS", n_parts=len(parts), parts_file="parts.json")
    save_manifest(directory, manifest)
    print(json.dumps(dict(run=directory.name, parts=len(parts),
                          rows=[p["n_rows"] for p in parts],
                          instructions=[p["n_instructions"] for p in parts])))


# ---------------------------------------------------------- chamber access

def list_experiments():
    return api_client().make_request("GET", "experiments").json()["experiments"]


def experiments_with_tag(tag):
    return [e for e in list_experiments() if e.get("tag") == tag]


def repeated_tags(experiments, items):
    """Our tags (current and earlier attempts) that more than one experiment carries.

    Canceled experiments do not count: canceling the extra job resolves a repeat.
    """
    ours = set()
    for _, _, _, part in items:
        ours.add(part["tag"])
        ours.update(attempt["tag"] for attempt in part.get("previous_attempts", []))
    counts = {}
    for experiment in experiments:
        if str(experiment.get("status", "")).upper() in {"CANCELED", "CANCELLED"}:
            continue
        if experiment.get("tag") in ours:
            counts[experiment["tag"]] = counts.get(experiment["tag"], 0) + 1
    return sorted(tag for tag, n in counts.items() if n > 1)


def submit_part(directory, manifest, state, part):
    with redirect_stdout(io.StringIO()):
        from causalchamber.lab.lab import Protocol
    rows = part_rows(load_schedule(directory, manifest), part)
    protocol = Protocol(manifest["chamber_id"], manifest["config"], api_client())
    for name, value in manifest["fixed_controls"].items():
        protocol.set(name, value)
    protocol.from_df(rows[["flag", "red", "pol_2", "led_3_ir"]], wait=manifest["wait_ms"])
    if len(protocol.instructions) != part["n_instructions"]:
        raise ValidationError(f"Unexpected instruction count for {part['tag']}")
    part.update(status="SUBMISSION_ATTEMPTED", attempted_utc=timestamp(),
                attempts=part.get("attempts", 0) + 1,
                instructions_sha256=hashlib.sha256("\n".join(protocol.instructions).encode()).hexdigest())
    save_parts(directory, state)
    experiment_id = protocol.submit(tag=part["tag"])
    part.update(status="SUBMITTED", experiment_id=experiment_id, submitted_utc=timestamp(),
                chamber_status="SUBMITTED")
    save_parts(directory, state)
    return experiment_id


def validate(frame, expected, fixed_controls):
    """The checks of collect.fetch, for one part or a whole run."""
    if len(frame) != len(expected) or not frame.flag.is_unique:
        raise ValidationError("Unexpected measurement count or duplicate flags")
    frame = frame.copy()
    frame["flag"] = frame.flag.astype(int)
    expected = expected.set_index("flag")
    if set(frame.flag) != set(expected.index):
        raise ValidationError("Measured flags differ from the schedule")
    frame = frame.set_index("flag").loc[expected.index]
    for column in ("red", "pol_2", "led_3_ir"):
        if not np.allclose(frame[column], expected[column], atol=1e-5, rtol=0):
            raise ValidationError(f"Measured controls disagree with schedule: {column}")
    for column in fixed_controls:
        if column in frame and not np.allclose(frame[column], fixed_controls[column], atol=1e-5, rtol=0):
            raise ValidationError(f"Fixed control drift: {column}")
    return frame, expected


def download_part(directory, manifest, state, part, result):
    with redirect_stdout(io.StringIO()):
        from causalchamber.lab.lab import ExperimentDataset
    raw = directory / "raw"
    target = raw / f"p{part['part']}"
    if target.exists():
        target.rename(raw / f"p{part['part']}_incomplete_{utc_now().strftime('%Y%m%dT%H%M%S')}")
    target.mkdir(parents=True)
    dataset = with_timeout(DOWNLOAD_SECONDS, lambda: ExperimentDataset(
        part["experiment_id"], result["download_url"], result["checksum"], target, verbose=False))
    frame = dataset.dataframe
    rows = part_rows(load_schedule(directory, manifest), part)
    validate(frame, rows, manifest["fixed_controls"])
    output = raw / f"part{part['part']}.csv"
    frame.to_csv(output, index=False)
    part.update(status="DOWNLOADED", part_csv=str(output.relative_to(directory)),
                part_csv_sha256=sha256(output), downloaded_utc=timestamp(),
                elapsed_measurement_seconds=float(frame.timestamp.max() - frame.timestamp.min()))
    save_parts(directory, state)


def merge(directory, manifest, state):
    """Join validated parts into raw_measurements.csv and measurements.csv."""
    frames = []
    for part in state["parts"]:
        path = directory / part["part_csv"]
        if sha256(path) != part["part_csv_sha256"]:
            raise ValidationError(f"Part file changed after validation: {path}")
        frames.append(pd.read_csv(path, float_precision="round_trip"))
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(directory / "raw_measurements.csv", index=False)
    schedule = load_schedule(directory, manifest)
    frame, expected = validate(frame, schedule, manifest["fixed_controls"])
    for column in expected.columns:
        if column not in frame:
            frame[column] = expected[column]
    frame["Y"] = frame.ir_3
    frame.reset_index().to_csv(directory / "measurements.csv", index=False)
    manifest.update(status="DONE", measurements_sha256=sha256(directory / "measurements.csv"),
                    raw_measurements_sha256=sha256(directory / "raw_measurements.csv"),
                    parts_sha256=sha256(directory / "parts.json"),
                    experiment_ids=[p["experiment_id"] for p in state["parts"]],
                    downloaded_utc=timestamp(), validated_rows=len(frame))
    save_manifest(directory, manifest)
    return len(frame)


# ------------------------------------------------------------- coordinator

def load_state(path):
    if path.exists():
        return json.loads(path.read_text())
    return dict(blocked=None, failures_in_a_row=0)


def save_state(path, **updates):
    state = load_state(path)
    state.update(updates, updated_utc=timestamp())
    write_json(path, state)


def read_control(path):
    control = dict(max_in_flight=1, pause_submissions=False)
    if path.exists():
        try:
            control.update(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            pass  # An edit may be in progress; keep the defaults for this pass.
    control["max_in_flight"] = max(1, int(control["max_in_flight"]))
    return control


def queue_parts(queue_path):
    """All (directory, manifest, state, part) in queue order, read fresh."""
    items, runs = [], []
    for entry in json.loads(queue_path.read_text()):
        directory = resolve(entry)
        manifest, state = load_run(directory)
        runs.append((directory, manifest, state))
        items.extend((directory, manifest, state, part) for part in state["parts"])
    return runs, items


def run(args):
    LabError, UserError = lab_errors()
    queue_path = args.queue.resolve()
    campaign = queue_path.parent
    control_path = campaign / "control.json"
    log = Log(campaign / "logs" / "coordinator.jsonl")
    stop = datetime.fromisoformat(args.stop_utc.replace("Z", "+00:00"))
    if stop.tzinfo is None:
        raise ValueError("--stop-utc needs an explicit UTC offset")
    lock_path = HERE / "data" / ".hardware.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = campaign / "coordinator_state.json"
        saved = load_state(state_path)
        blocked, failures_in_a_row = saved["blocked"], saved["failures_in_a_row"]
        log("coordinator_started", queue=str(queue_path), stop_utc=stop.isoformat(),
            blocked=blocked, failures_in_a_row=failures_in_a_row)
        hold_until, backoff = utc_now(), args.poll_seconds
        alerted, heartbeat, tag_check = set(), utc_now(), utc_now()

        def block(reason, event, **fields):
            nonlocal blocked
            if blocked == reason:
                return
            blocked = reason
            save_state(state_path, blocked=blocked, failures_in_a_row=failures_in_a_row)
            log(event, **fields)

        def check_tags(experiments, items):
            repeated = repeated_tags(experiments, items)
            if repeated:
                block(f"several experiments carry tags {repeated}; cancel the extra ones",
                      "ALERT_duplicate_tag", tags=repeated)
        while True:
            control = read_control(control_path)
            runs, items = queue_parts(queue_path)
            try:
                # 1. Merge runs whose parts are all validated.
                for directory, manifest, state in runs:
                    if (manifest["status"] != "DONE" and directory.name not in alerted
                            and all(p["status"] == "DOWNLOADED" for p in state["parts"])):
                        try:
                            rows = merge(directory, manifest, state)
                            log("run_completed", run=directory.name, validated_rows=rows)
                        except ValidationError as error:
                            alerted.add(directory.name)
                            block(f"merge of {directory.name}: {error}", "ALERT_validation_failed",
                                  run=directory.name, error=str(error))
                runs, items = queue_parts(queue_path)
                # 2. Poll parts on the chamber, download finished ones.
                for directory, manifest, state, part in items:
                    if part["status"] != "SUBMITTED":
                        continue
                    result = with_timeout(API_SECONDS, lambda: api_client().make_request(
                        "GET", f"experiments/{part['experiment_id']}").json())
                    status = str(result["status"]).upper()
                    changed = status != part.get("chamber_status")
                    part.update(chamber_status=status, last_checked_utc=timestamp(),
                                current_instruction=result.get("current_instruction"),
                                total_instructions=result.get("total_instructions"))
                    for key in ("submitted_on", "start_time", "end_time", "runtime"):
                        if key in result:
                            part[key] = result[key]
                    save_parts(directory, state)
                    if changed:
                        log("part_status", tag=part["tag"], status=status)
                    if status == "DONE":
                        try:
                            download_part(directory, manifest, state, part, result)
                            if failures_in_a_row:
                                failures_in_a_row = 0
                                save_state(state_path, failures_in_a_row=0)
                            log("part_downloaded", tag=part["tag"], rows=part["n_rows"],
                                seconds=round(part["elapsed_measurement_seconds"]))
                        except ValidationError as error:
                            part.update(status="VALIDATION_FAILED", validation_error=str(error))
                            save_parts(directory, state)
                            block(f"{part['tag']}: {error}", "ALERT_validation_failed",
                                  tag=part["tag"], error=str(error))
                    elif status in UNSUCCESSFUL:
                        # One failed job only raises an alert (retry it with retry-part);
                        # two in a row stop all submissions.
                        failures_in_a_row += 1
                        save_state(state_path, failures_in_a_row=failures_in_a_row)
                        part.update(status="FAILED")
                        save_parts(directory, state)
                        log("ALERT_part_failed", tag=part["tag"], status=status, failures_in_a_row=failures_in_a_row)
                        if failures_in_a_row >= 2:
                            block(f"{failures_in_a_row} parts in a row failed, last {part['tag']} ({status})",
                                  "ALERT_submissions_stopped", failures_in_a_row=failures_in_a_row)
                # 3. Resolve uncertain submissions through their tags.
                for directory, manifest, state, part in items:
                    if part["status"] != "SUBMISSION_ATTEMPTED":
                        continue
                    matches = with_timeout(API_SECONDS, experiments_with_tag, part["tag"])
                    age = (utc_now() - datetime.fromisoformat(part["attempted_utc"])).total_seconds()
                    if len(matches) == 1:
                        part.update(status="SUBMITTED", experiment_id=matches[0]["experiment_id"],
                                    chamber_status="SUBMITTED", recovered_from_tag_utc=timestamp())
                        save_parts(directory, state)
                        log("submission_recovered", tag=part["tag"], experiment_id=part["experiment_id"])
                    elif len(matches) > 1:
                        block(f"several experiments carry tags {[part['tag']]}; cancel the extra ones",
                              "ALERT_duplicate_tag", tags=[part["tag"]])
                    elif age >= RECHECK_SECONDS:
                        part.update(status="PREPARED", absent_after_attempt_utc=timestamp())
                        save_parts(directory, state)
                        log("submission_absent", tag=part["tag"], seconds_since_attempt=round(age))
                # 4. Submit the next part in queue order.
                runs, items = queue_parts(queue_path)
                in_flight = sum(p["status"] in {"SUBMITTED", "SUBMISSION_ATTEMPTED"} for _, _, _, p in items)
                nxt = next((item for item in items if item[3]["status"] in {"PREPARED", "SUBMISSION_ATTEMPTED"}), None)
                may_submit = (nxt is not None and nxt[3]["status"] == "PREPARED" and blocked is None
                              and not control["pause_submissions"] and utc_now() < stop
                              and utc_now() >= hold_until and in_flight < control["max_in_flight"])
                if may_submit:
                    tag_check = utc_now()
                    experiments = with_timeout(API_SECONDS, list_experiments)
                    check_tags(experiments, items)
                    may_submit = blocked is None
                elif (utc_now() - tag_check).total_seconds() >= TAG_CHECK_SECONDS:
                    tag_check = utc_now()
                    check_tags(with_timeout(API_SECONDS, list_experiments), items)
                if may_submit:
                    directory, manifest, state, part = nxt
                    existing = [e for e in experiments if e.get("tag") == part["tag"]]
                    if existing:
                        part.update(status="SUBMISSION_ATTEMPTED",
                                    attempted_utc=part.get("attempted_utc", timestamp()))
                        save_parts(directory, state)
                        log("tag_already_exists", tag=part["tag"], n=len(existing))
                    else:
                        try:
                            experiment_id = with_timeout(API_SECONDS, submit_part, directory, manifest, state, part)
                            log("part_submitted", tag=part["tag"], experiment_id=experiment_id,
                                rows=part["n_rows"], in_flight=in_flight + 1)
                        except UserError as error:
                            # The server answered with a rejection: no experiment exists.
                            part.update(status="PREPARED", last_rejection=str(error)[:500],
                                        last_rejection_utc=timestamp())
                            save_parts(directory, state)
                            hold_until = utc_now() + timedelta(seconds=REJECTION_HOLD_SECONDS)
                            log("ALERT_submission_rejected", tag=part["tag"], error=str(error)[:500])
                        except ValidationError:
                            raise
                        except Exception as error:
                            log("submission_uncertain", tag=part["tag"], error=f"{type(error).__name__}: {str(error)[:300]}")
                backoff = args.poll_seconds
            except ValidationError as error:
                block(str(error), "ALERT_validation_failed", error=str(error))
            except (LabError, UserError, TimeoutError, OSError, KeyError, ValueError) as error:
                # Includes connection failures (LabError code 1). Retry later.
                log("api_unavailable", error=f"{type(error).__name__}: {str(error)[:300]}",
                    retry_seconds=round(backoff))
                time.sleep(backoff)
                backoff = min(300, backoff * 2)
                continue
            runs, items = queue_parts(queue_path)
            statuses = [p["status"] for _, _, _, p in items]
            if (utc_now() - heartbeat).total_seconds() >= HEARTBEAT_SECONDS:
                heartbeat = utc_now()
                log("heartbeat", parts_downloaded=statuses.count("DOWNLOADED"), parts_total=len(statuses),
                    on_chamber=[f"{p['tag']} {p.get('chamber_status')} {p.get('current_instruction')}/{p.get('total_instructions')}"
                                for _, _, _, p in items if p["status"] == "SUBMITTED"],
                    max_in_flight=control["max_in_flight"], blocked=blocked)
            active = sum(s in {"SUBMITTED", "SUBMISSION_ATTEMPTED"} for s in statuses)
            done = all(m["status"] == "DONE" for _, m, _ in runs)
            remaining = statuses.count("PREPARED")
            unmerged = any(m["status"] != "DONE" and d.name not in alerted
                           and all(p["status"] == "DOWNLOADED" for p in s["parts"]) for d, m, s in runs)
            if done or (active == 0 and not unmerged and (blocked is not None or utc_now() >= stop or remaining == 0)):
                try:
                    check_tags(with_timeout(API_SECONDS, list_experiments), items)
                except Exception as error:
                    log("final_tag_check_failed", error=f"{type(error).__name__}: {str(error)[:300]}")
                if done and blocked is None:
                    log("queue_complete", runs=len(runs), parts=len(items))
                    return 0
                reason = blocked or ("deadline reached" if remaining else "only failed parts remain")
                log("coordinator_stopped", reason=reason, remaining_parts=remaining,
                    failed_parts=[p["tag"] for _, _, _, p in items if p["status"] in {"FAILED", "VALIDATION_FAILED"}])
                return 0 if reason == "deadline reached" else 2
            time.sleep(args.poll_seconds)


# ------------------------------------------------------------------ status

def status(args):
    saved = load_state(args.queue.resolve().parent / "coordinator_state.json")
    if saved.get("blocked"):
        print(f"SUBMISSIONS STOPPED: {saved['blocked']}")
    _, items = queue_parts(args.queue.resolve())
    by_run = {}
    for directory, manifest, state, part in items:
        by_run.setdefault(directory.name, (manifest, []))[1].append(part)
    rows_total = sum(p["n_rows"] for _, _, _, p in items)
    rows_done = sum(p["n_rows"] for _, _, _, p in items if p["status"] == "DOWNLOADED")
    seconds = [p["elapsed_measurement_seconds"] / p["n_rows"] for _, _, _, p in items if "elapsed_measurement_seconds" in p]
    per_row = float(np.mean(seconds)) if seconds else .2253
    running = []
    for name, (manifest, parts) in by_run.items():
        marks = "".join({"DOWNLOADED": "#", "SUBMITTED": ">", "SUBMISSION_ATTEMPTED": "?",
                         "PREPARED": ".", "FAILED": "X", "VALIDATION_FAILED": "V"}.get(p["status"], "!") for p in parts)
        print(f"{name:16s} {manifest['status']:8s} {marks}")
        running += [p for p in parts if p["status"] == "SUBMITTED"]
    for p in running:
        print(f"  {p['tag']}: {p.get('chamber_status')} {p.get('current_instruction')}/{p.get('total_instructions')}"
              f" (checked {p.get('last_checked_utc', '-')[:19]})")
    remaining = sum(p["n_rows"] for _, _, _, p in items if p["status"] != "DOWNLOADED")
    print(f"rows {rows_done}/{rows_total}; {per_row:.4f} s/row; about {remaining * per_row / 3600:.1f} h of chamber time left")
    print("legend: # downloaded, > on chamber, ? uncertain submission, . waiting, X failed, V validation failed")


def retry_part(args):
    directory = args.directory.resolve()
    manifest, state = load_run(directory)
    part = state["parts"][args.part - 1]
    if part["status"] not in {"FAILED", "VALIDATION_FAILED"}:
        raise ValueError("Only FAILED or VALIDATION_FAILED parts can be retried.")
    history = part.setdefault("previous_attempts", [])
    history.append({k: v for k, v in part.items() if k != "previous_attempts"})
    keep = {k: part[k] for k in ("part", "first_block", "last_block", "first_flag", "last_flag", "n_rows", "n_instructions")}
    base = part["tag"].split("_retry")[0]
    part.clear()
    part.update(keep, tag=f"{base}_retry{len(history)}", status="PREPARED", previous_attempts=history)
    save_parts(directory, state)
    print(json.dumps(dict(run=directory.name, part=args.part, new_tag=part["tag"])))


def clear_stop(args):
    with (HERE / "data" / ".hardware.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A coordinator is running; stop it before clearing its stop.")
        path = args.queue.resolve().parent / "coordinator_state.json"
        saved = load_state(path)
        history = saved.get("cleared", [])
        history.append(dict(blocked=saved.get("blocked"), failures_in_a_row=saved.get("failures_in_a_row"),
                            cleared_utc=timestamp(), note=args.note))
        save_state(path, blocked=None, failures_in_a_row=0, cleared=history)
        print(json.dumps(history[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("split")
    p.add_argument("directory", type=Path)
    p.add_argument("--blocks-per-part", type=int, required=True)
    p.add_argument("--tag-prefix", required=True)
    p = sub.add_parser("run")
    p.add_argument("queue", type=Path)
    p.add_argument("--stop-utc", required=True)
    p.add_argument("--poll-seconds", type=float, default=45)
    p = sub.add_parser("status")
    p.add_argument("queue", type=Path)
    p = sub.add_parser("retry-part")
    p.add_argument("directory", type=Path)
    p.add_argument("part", type=int)
    p = sub.add_parser("clear-stop")
    p.add_argument("queue", type=Path)
    p.add_argument("--note", required=True, help="What was inspected and why submissions may resume")
    args = parser.parse_args()
    if args.action == "split":
        split(args)
    elif args.action == "run":
        raise SystemExit(run(args))
    elif args.action == "status":
        status(args)
    elif args.action == "clear-stop":
        clear_stop(args)
    else:
        retry_part(args)


if __name__ == "__main__":
    main()
