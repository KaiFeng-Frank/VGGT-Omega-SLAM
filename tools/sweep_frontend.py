#!/usr/bin/env python3
"""Design, or explicitly run, a six-configuration frontend development sweep.

The default only writes commands/design files: it never imports Torch, queries
CUDA, acquires a GPU lock, or launches inference. --run serially launches the
current run_slam.py; each child claims the normal shared GPU lock itself.
VOS_* settings and caller thread limits are inherited. A caller may wrap this
tool in a systemd scope with MemoryMax=12G.

Development:
  python tools/sweep_frontend.py --data-root data/tum_rgbd --out runs/frontend_dev
  # Inspect design.json, then repeat with --run when the GPU queue permits.

Validation (no parameter reselection):
  python tools/sweep_frontend.py --data-root data/tum_rgbd --sequence desk2 \
    --out runs/frontend_validation --frozen-selection runs/frontend_dev/frozen_selection.json

An existing s16/d50 baseline can be reused with --baseline-run and optionally
--baseline-cache. A cache alone cannot establish a completed trajectory/ATE.
Only the baseline and frozen selected setting are scheduled in validation.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = "vggt-omega-slam-frontend-sweep-v1"
SELECTION_SCHEMA = "vggt-omega-slam-frontend-selection-v1"
BASELINE_ID = "s16_d50"
GRID = tuple({"id": f"s{n}_d{d}", "submap_size": n, "min_disparity": d}
             for n in (16, 32) for d in (25, 50, 75))
TERMINAL = {"complete", "failed", "hardware_infeasible", "terminated", "incomplete_preserved"}
CORE_DEFAULTS = {"backend_policy": "legacy", "overlap": 1, "seed": 0, "max_loops": 1,
                 "max_frames": 0, "conf_threshold": 25.0, "lc_thres": .95,
                 "scale_confidence_percentile": None, "scale_confidence_scope": "submap",
                 "cross_noise_multiplier": 1.0, "loop_noise_multiplier": 1.0,
                 "edge_huber_delta": None}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def absolute(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sequence_directory(data_root, sequence):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", sequence):
        raise ValueError("--sequence must be a sequence name, not a path")
    name = sequence if sequence.startswith("rgbd_dataset_") else "rgbd_dataset_freiburg1_" + sequence
    return Path(data_root).resolve() / name


def inherited_environment(threads, source=None):
    """Preserve VOS_* and explicit caller limits; fill missing thread settings."""
    result = dict(os.environ if source is None else source)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        result.setdefault(key, str(threads))
    return result


def default_threads():
    limits = [4]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        try:
            limits.append(max(1, int(os.environ.get(name, "4"))))
        except ValueError:
            pass
    return min(limits)


def command(config, sequence_path, backend, output, cache, threads=4):
    argv = [sys.executable, str(ROOT / "run_slam.py"), str(Path(sequence_path).resolve()),
            "--backend", backend, "--output", str(Path(output).resolve()),
            "--submap-size", str(config["submap_size"]),
            "--min-disparity", str(config["min_disparity"]),
            "--overlap", "1", "--seed", "0", "--max-loops", "1", "--max-frames", "0",
            "--conf-threshold", "25", "--backend-policy", "legacy", "--lc-thres", "0.95",
            "--prediction-cache", str(Path(cache).resolve()), "--max-cloud-points", "100000",
            "--threads", str(threads), "--gpu-need-mb", "9000" if config["submap_size"] == 16 else "9500"]
    return argv


def runtime_signature(backend):
    # env.py contains only lightweight path/config definitions; no prepare(),
    # model import, GPU call or checkpoint scan is performed here.
    from vos import env
    signature = {"runner_sha256": sha256(ROOT / "run_slam.py"),
                 "checkpoint_sha256_expected": env.OMEGA_SHA256 if backend == "omega" else env.VGGT_SHA256}
    if backend == "omega":
        config = env.SIM_CONFIG
        values = json.loads(config.read_text())
        signature["similarity"] = {"layer": int(values["layer"]), "tau": float(values["tau"]),
                                   "config_file": str(config.resolve()), "config_sha256": sha256(config)}
    return signature


def load_frozen(path, backend):
    frozen = json.loads(Path(path).read_text())
    if frozen.get("schema") != SELECTION_SCHEMA or frozen.get("status") != "frozen":
        raise ValueError("--frozen-selection must be this tool's frozen development selection")
    if frozen.get("backend") != backend:
        raise ValueError("Frozen selection backend does not match --backend")
    selected = frozen.get("selected", {})
    if not any(all(selected.get(k) == c[k] for k in ("id", "submap_size", "min_disparity")) for c in GRID):
        raise ValueError("Frozen selected configuration is outside the predefined grid")
    return frozen


def make_design(args):
    sequence_path = sequence_directory(args.data_root, args.sequence)
    signature = runtime_signature(args.backend)
    frozen = load_frozen(args.frozen_selection, args.backend) if args.frozen_selection else None
    if frozen and frozen.get("runtime_signature", {}).get("checkpoint_sha256_expected") != signature["checkpoint_sha256_expected"]:
        raise ValueError("Validation checkpoint identity differs from frozen development")
    if frozen and frozen.get("runtime_signature", {}).get("similarity") != signature.get("similarity"):
        raise ValueError("Validation similarity calibration differs from frozen development")
    scheduled = {BASELINE_ID, frozen["selected"]["id"]} if frozen else {c["id"] for c in GRID}
    configs = []
    for config in GRID:
        output = args.out / "runs" / config["id"]
        cache = args.out / "predictions" / config["id"]
        argv = command(config, sequence_path, args.backend, output, cache, args.threads)
        configs.append({**config, "scheduled": config["id"] in scheduled,
                        "output": str(output), "prediction_cache": str(cache),
                        "argv": argv, "command": shlex.join(argv)})
    design = {"schema": SCHEMA, "phase": "validation" if frozen else "development",
              "sequence": args.sequence, "sequence_path": str(sequence_path), "backend": args.backend,
              "runtime_signature": signature, "threads": args.threads,
              "thread_environment": {k: v for k, v in inherited_environment(args.threads).items()
                                     if k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
              "inherited_vos_environment_keys": sorted(k for k in os.environ if k.startswith("VOS_")),
              "baseline_id": BASELINE_ID,
              "baseline_run": str(args.baseline_run) if args.baseline_run else None,
              "baseline_cache": str(args.baseline_cache) if args.baseline_cache else None,
              "source_frozen_selection": str(args.frozen_selection) if args.frozen_selection else None,
              "source_frozen_selection_sha256": sha256(args.frozen_selection) if args.frozen_selection else None,
              "validation_selected_id": frozen["selected"]["id"] if frozen else None,
              "configs": configs,
              "selection_rule": "Minimum finite Sim(3) ATE on this named development sequence only; exact ties prefer baseline, then fixed grid order. Only completed feasible runs are scored. Validation never selects.",
              "resume_rule": "Completed outputs require matching inference/evaluation args and full prediction-cache frame lists. Failed/incomplete directories are preserved and never rerun in place.",
              "allowed_baseline_execution_differences": ["output/cache directory", "point-cloud export cap", "CPU threads", "GPU reservation request", "checkpoint hash-verification toggle"],
              "limitations": ["Development-sequence optimization is not evidence of generalization.",
                              "OOM/resource failures are hardware infeasibility, never accuracy scores.",
                              "Expected checkpoint SHA is recorded; only run_slam --verify-checkpoints records an independently measured full checkpoint SHA.",
                              "Disparity changes keyframe timestamps. The prespecified own-trajectory ATE selection does not control evaluation time support; common-time evaluation must be reported separately.",
                              "Caller is responsible for any systemd MemoryMax scope; each child acquires the shared GPU lock."]}
    design["design_sha256"] = hashlib.sha256(json.dumps(design, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return design


def validate_metadata(metadata, config, design):
    """Reject a completed run from a different inference or evaluation condition."""
    if metadata.get("status") != "complete":
        raise ValueError("run is not complete")
    args = metadata.get("args", {})
    expected = {**CORE_DEFAULTS, "backend": design["backend"],
                "submap_size": config["submap_size"], "min_disparity": config["min_disparity"]}
    for key, value in expected.items():
        actual = args.get(key, CORE_DEFAULTS.get(key))
        if actual != value:
            raise ValueError(f"completed run args mismatch: {key}={actual!r}, expected {value!r}")
    given = args.get("input")
    if given is None or absolute(given) != Path(design["sequence_path"]).resolve():
        raise ValueError("completed run input sequence differs")
    groundtruth = args.get("groundtruth") or metadata.get("input", {}).get("groundtruth")
    if groundtruth and absolute(groundtruth) != Path(design["sequence_path"]) / "groundtruth.txt":
        raise ValueError("completed run ground-truth path differs")
    signature = design["runtime_signature"]
    checkpoint = metadata.get("checkpoints", {}).get(design["backend"], {})
    if checkpoint.get("sha256_expected") != signature["checkpoint_sha256_expected"]:
        raise ValueError("completed run checkpoint identity differs or is missing")
    if checkpoint.get("sha256") and checkpoint["sha256"] != signature["checkpoint_sha256_expected"]:
        raise ValueError("completed run measured checkpoint SHA differs")
    if design["backend"] == "omega":
        actual = metadata.get("similarity", {})
        target = signature["similarity"]
        if actual.get("layer") != target["layer"] or actual.get("tau") != target["tau"]:
            raise ValueError("completed run effective attention calibration differs")
    ate = metadata.get("ate", {})
    score = ate.get("ate_rmse_m")
    if ate.get("status") != "ok" or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0:
        raise ValueError("completed run has no finite valid Sim(3) ATE")
    return float(score)


def validate_cache(cache, metadata):
    """Check cache completeness and exact per-update frame identities on CPU."""
    import numpy as np
    cache = Path(cache)
    if not cache.is_dir():
        raise ValueError("prediction-cache directory is missing")
    paths = sorted(cache.glob("*.npz"))
    if any(not re.fullmatch(r"\d{5}\.npz", p.name) for p in paths):
        raise ValueError("prediction-cache contains nonfinal/temporary NPZ files")
    updates = metadata.get("updates", [])
    if not updates or len(paths) != len(updates):
        raise ValueError("prediction-cache count differs from completed updates")
    for index, (path, update) in enumerate(zip(paths, updates)):
        if path.name != f"{index:05d}.npz":
            raise ValueError("prediction-cache update numbering is not contiguous")
        with np.load(path, allow_pickle=False) as archive:
            record = json.loads(str(archive["__metadata__"]))
        if record.get("schema") != "vos-prediction-cache-v1" or record.get("frame_names") != update["frame_names"]:
            raise ValueError("prediction-cache frame identities differ from the online run")
        expected_id = 0 if index == 0 else updates[index - 1].get("graph_nodes")
        if expected_id is not None and record.get("submap_id") != expected_id:
            raise ValueError("prediction-cache graph-node numbering differs from the online run")
        if len(record.get("values", {}).get("detected_loops", [])) != update.get("loops_accepted_in_update", 0):
            raise ValueError("prediction-cache accepted-loop count differs from the online run")
    return {"status": "verified", "n_updates": len(paths)}


def classify_failure(text, returncode=None):
    message = text.lower()
    if any(word in message for word in ("out of memory", "outofmemoryerror", "oom-kill", "oom_kill", "memoryerror")):
        return "hardware_infeasible"
    if returncode is not None and returncode < 0:
        return "terminated"  # SIGKILL alone does not prove OOM.
    return "failed"


def inspect_run(run_path, config, design, *, cache=None, reused=False):
    path = Path(run_path)
    record = {"id": config["id"], "submap_size": config["submap_size"],
              "min_disparity": config["min_disparity"], "run_dir": str(path),
              "prediction_cache": str(cache) if cache else None, "reused_baseline": reused,
              "status": "planned", "ate_rmse_m": None, "eligible_for_accuracy_selection": False}
    if not path.exists():
        return record
    metadata_file = path / "metadata.json"
    if not metadata_file.is_file():
        return {**record, "status": "incomplete_preserved", "reason": "existing run directory has no metadata.json"}
    metadata = json.loads(metadata_file.read_text())
    if metadata.get("status") != "complete":
        text = metadata_file.read_text()
        log = path / "run.log"
        if log.is_file():
            text += log.read_text(errors="replace")
        return {**record, "status": classify_failure(text), "reason": "existing noncomplete output preserved",
                "recorded_status": metadata.get("status"), "metadata_sha256": sha256(metadata_file)}
    try:
        score = validate_metadata(metadata, config, design)
        if not (path / "poses.tum").is_file():
            raise ValueError("completed output is missing poses.tum")
        cache = cache or metadata.get("args", {}).get("prediction_cache")
        if cache is not None:
            cache = absolute(cache)
            recorded_cache = metadata.get("args", {}).get("prediction_cache")
            if recorded_cache and absolute(recorded_cache) != cache:
                raise ValueError("explicit baseline cache differs from the completed run's recorded cache")
            validation = validate_cache(cache, metadata)
        elif reused:
            validation = {"status": "not_available", "note": "Completed external baseline supplied without a prediction cache"}
        else:
            raise ValueError("completed sweep run has no prediction-cache location")
    except (ValueError, KeyError, OSError) as exc:
        return {**record, "status": "incompatible_existing", "reason": str(exc),
                "metadata_sha256": sha256(metadata_file)}
    stamps = [float(line.split()[0]) for line in (path / "poses.tum").read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    if not stamps or not all(math.isfinite(t) for t in stamps):
        return {**record, "status": "incompatible_existing", "reason": "trajectory timestamps are empty/nonfinite"}
    return {**record, "status": "complete", "ate_rmse_m": score, "eligible_for_accuracy_selection": True,
            "prediction_cache": str(cache) if cache else None, "cache_validation": validation,
            "metadata_sha256": sha256(metadata_file), "poses_sha256": sha256(path / "poses.tum"),
            "n_associated": metadata["ate"].get("n_associated"), "runner_sha256": metadata.get("runner_sha256"),
            "n_keyframes": metadata.get("summary", {}).get("keyframes", len(set(stamps))),
            "n_pose_rows": len(stamps), "n_unique_pose_timestamps": len(set(stamps)),
            "first_timestamp": stamps[0], "last_timestamp": stamps[-1]}


def choose_configuration(records):
    """Pure development rule; infeasible/failed/incomplete runs never get scores."""
    if len(records) != len(GRID) or {r["id"] for r in records} != {c["id"] for c in GRID}:
        raise ValueError("selection requires all six predefined grid records")
    if any(r["status"] not in TERMINAL for r in records):
        raise ValueError("selection requires terminal records for every configuration")
    baseline = next(r for r in records if r["id"] == BASELINE_ID)
    if baseline["status"] != "complete" or not baseline.get("eligible_for_accuracy_selection"):
        raise ValueError("a consistent completed s16/d50 baseline is required before selection")
    scored = [r for r in records if r["status"] == "complete" and r.get("eligible_for_accuracy_selection")
              and not isinstance(r.get("ate_rmse_m"), bool)
              and isinstance(r.get("ate_rmse_m"), (int, float)) and math.isfinite(r["ate_rmse_m"]) and r["ate_rmse_m"] >= 0]
    order = {c["id"]: i for i, c in enumerate(GRID)}
    selected = min(scored, key=lambda r: (r["ate_rmse_m"], r["id"] != BASELINE_ID, order[r["id"]]))
    return selected, baseline


def frozen_selection(design, records):
    if design["phase"] != "development":
        raise ValueError("Validation cannot select or rewrite development parameters")
    selected, baseline = choose_configuration(records)
    # Exclude launch-only details so a legal completed-output resume does not
    # change or rewrite the immutable selection.
    stable_keys = ("id", "submap_size", "min_disparity", "run_dir", "prediction_cache", "status",
                   "ate_rmse_m", "eligible_for_accuracy_selection", "reused_baseline", "metadata_sha256",
                   "poses_sha256", "n_associated", "n_keyframes", "n_pose_rows", "n_unique_pose_timestamps",
                   "first_timestamp", "last_timestamp")
    def stable(record):
        return {k: record[k] for k in stable_keys if k in record}
    return {"schema": SELECTION_SCHEMA, "status": "frozen",
            "development_sequence": design["sequence"], "backend": design["backend"],
            "design_sha256": design["design_sha256"], "runtime_signature": design["runtime_signature"],
            "selection_rule": design["selection_rule"], "selected": stable(selected), "baseline": stable(baseline),
            "selected_improves_baseline": selected["ate_rmse_m"] < baseline["ate_rmse_m"],
            "all_candidates": [stable(r) for r in records], "disclosure": "Parameters selected on the named development sequence only. Resource failures are unscored. Held-out sequences must use --frozen-selection and never reselect. Own-trajectory ATE uses configuration-specific keyframe timestamps."}


def execute_sweep(args):
    args.out.mkdir(parents=True, exist_ok=True)
    design = make_design(args)
    design_file = args.out / "design.json"
    if design_file.exists():
        old = json.loads(design_file.read_text())
        if old.get("design_sha256") != design["design_sha256"]:
            raise ValueError("Existing design differs; use identical arguments/environment or a fresh --out")
    else:
        write_json(design_file, design)
    results_file = args.out / "results.json"
    previous = json.loads(results_file.read_text()) if results_file.exists() else {}
    previous_records = {r["id"]: r for r in previous.get("records", [])}
    records = []
    for config in design["configs"]:
        if not config["scheduled"]:
            records.append({"id": config["id"], "status": "not_scheduled_frozen_validation", "ate_rmse_m": None})
            continue
        external = config["id"] == BASELINE_ID and args.baseline_run is not None
        record = inspect_run(args.baseline_run if external else config["output"], config, design,
                             cache=args.baseline_cache if external else config["prediction_cache"], reused=external)
        old_record = previous_records.get(config["id"], {})
        log = args.out / "logs" / (config["id"] + ".log")
        if record["status"] == "planned" and log.exists():
            status = classify_failure(log.read_text(errors="replace"), old_record.get("returncode"))
            record.update(status=status, launch_log=str(log), reason="Prior failed launch preserved; no retry in place")
        if record["status"] == "complete" and old_record.get("status") == "complete":
            for key in ("metadata_sha256", "poses_sha256"):
                if old_record.get(key) and old_record[key] != record[key]:
                    record.update(status="incompatible_existing", eligible_for_accuracy_selection=False,
                                  reason=f"Previously completed artifact changed: {key}")
        records.append(record)
    state = {"schema": SCHEMA, "phase": design["phase"], "design_sha256": design["design_sha256"],
             "sequence": args.sequence, "backend": args.backend, "executed": False, "records": records,
             "commands": {c["id"]: c["command"] for c in design["configs"]}, "status": "designed"}
    if any(r["status"] == "incompatible_existing" for r in records):
        state["status"] = "incompatible_existing_outputs"
        write_json(args.out / "results.json", state)
        raise ValueError("Existing completed output is incompatible; preserved without launching jobs. See results.json")
    if args.baseline_run is not None and next(r for r in records if r["id"] == BASELINE_ID)["status"] != "complete":
        write_json(args.out / "results.json", state)
        raise ValueError("--baseline-run must be a consistent completed baseline")
    write_json(args.out / "results.json", state)
    if not args.run:
        print(json.dumps({"status": "design_only", "design": str(design_file), "scheduled_commands": [c["command"] for c in design["configs"] if c["scheduled"]]}, indent=2))
        return state
    sequence_path = Path(design["sequence_path"])
    if not (sequence_path / "rgb").is_dir() or not (sequence_path / "groundtruth.txt").is_file():
        raise ValueError("--run requires the sequence rgb/ directory and groundtruth.txt")
    child_env = inherited_environment(args.threads)
    state.update(executed=True, status="running")
    for index, (config, record) in enumerate(zip(design["configs"], records)):
        if not config["scheduled"] or record["status"] != "planned":
            continue  # Completed outputs resume; failed/incomplete outputs are never overwritten.
        cache = Path(config["prediction_cache"])
        if cache.exists() and any(cache.iterdir()):
            records[index] = {**record, "status": "incomplete_preserved", "reason": "Nonempty prediction cache has no consistent completed run; preserved"}
            write_json(args.out / "results.json", state)
            continue
        log = args.out / "logs" / (config["id"] + ".log")
        log.parent.mkdir(exist_ok=True)
        if log.exists():
            records[index] = {**record, "status": "incomplete_preserved", "reason": "Existing launch log preserved; use a fresh output directory to retry"}
            write_json(args.out / "results.json", state)
            continue
        with log.open("x") as stream:
            state["active_id"] = config["id"]
            write_json(args.out / "results.json", state)
            try:
                proc = subprocess.run(config["argv"], cwd=ROOT, env=child_env, stdout=stream, stderr=subprocess.STDOUT, check=False)
                returncode = proc.returncode
            except OSError as exc:
                stream.write(f"Launch failed: {type(exc).__name__}: {exc}\n")
                returncode = 127
        state.pop("active_id", None)
        record = inspect_run(config["output"], config, design, cache=config["prediction_cache"])
        record.update(returncode=returncode, command=config["command"], launch_log=str(log))
        if returncode != 0 or record["status"] == "planned":
            record.update(status=classify_failure(log.read_text(errors="replace"), returncode),
                          eligible_for_accuracy_selection=False, ate_rmse_m=None)
        records[index] = record
        write_json(args.out / "results.json", state)
    state["status"] = "complete" if all(r["status"] in TERMINAL or not c["scheduled"] for r, c in zip(records, design["configs"])) else "incomplete"
    if design["phase"] == "development" and state["status"] == "complete":
        try:
            selection = frozen_selection(design, records)
            path = args.out / "frozen_selection.json"
            if path.exists() and json.loads(path.read_text()) != selection:
                raise ValueError("Existing frozen selection differs; refusing to reselect or overwrite it")
            if not path.exists():
                write_json(path, selection)
            state["frozen_selection"] = str(path)
        except ValueError as exc:
            state.update(status="complete_without_selection", selection_error=str(exc))
    elif design["phase"] == "validation":
        state["frozen_selection_source"] = str(args.frozen_selection)
        state["selection_performed"] = False
    write_json(args.out / "results.json", state)
    print(json.dumps({"status": state["status"], "results": str(args.out / "results.json"),
                      "phase": design["phase"], "frozen_selection": state.get("frozen_selection")}, indent=2))
    return state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequence", default="desk")
    parser.add_argument("--backend", choices=("omega", "vggt"), default="omega")
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--baseline-cache", type=Path)
    parser.add_argument("--frozen-selection", type=Path, help="Validation: only baseline and the frozen selected setting; never reselect")
    parser.add_argument("--threads", type=int, default=default_threads(), choices=range(1, 5))
    parser.add_argument("--run", action="store_true", help="Explicitly execute serial GPU child runs; default only writes design/commands")
    args = parser.parse_args(argv)
    if args.baseline_cache and not args.baseline_run:
        parser.error("--baseline-cache requires the matching completed --baseline-run; a cache alone has no trajectory ATE")
    for name in ("data_root", "out", "baseline_run", "baseline_cache", "frozen_selection"):
        if getattr(args, name) is not None:
            setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    return args


def main(argv=None):
    try:
        state = execute_sweep(parse_args(argv))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Frontend sweep: {exc}", file=sys.stderr)
        return 2
    return 0 if state["status"] in ("designed", "complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
