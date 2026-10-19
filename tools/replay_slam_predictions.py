#!/usr/bin/env python3
"""Replay lossless front-end caches through the SLAM backend on CPU only.

Accepted loops, images, cameras, depths and confidences are frozen by the
cache. No model, retrieval, loop reclassification, GPU query or CUDA lock is
used. Before a nonlegacy policy with --reference-poses runs, the same caches
are replayed through the original backend and checked against all reference
TUM rows, preserving duplicate overlap timestamps and quaternion signs.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

# Set before importing Torch, Open3D, upstream modules, or environment helpers.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from vos import env
from vos.backend_policy import BackendPolicy, POLICY_NAMES, make_solver_class
from vos.prediction_cache import load_predictions


def prediction_paths(directory):
    return sorted(path for path in Path(directory).glob("*.npz") if not path.name.endswith(".tmp.npz"))


def _read_tum_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append([float(v) for v in line.split()])
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 8 or not np.isfinite(array).all():
        raise ValueError(f"Expected finite TUM trajectory rows with eight columns: {path}")
    return array


def compare_reference(reference, replay, *, atol=0.):
    """Compare the complete ordered files, without deduplication/alignment."""
    expected, actual = _read_tum_rows(reference), _read_tum_rows(replay)
    result = {"reference": str(reference), "n_reference_rows": len(expected), "n_replay_rows": len(actual),
              "absolute_tolerance": atol, "relative_tolerance": 0., "exact_numeric_match": False,
              "timestamps_equal": False, "passed": False}
    if expected.shape != actual.shape:
        result["reason"] = "trajectory row counts differ"
        return result
    result["timestamps_equal"] = bool(np.array_equal(expected[:, 0], actual[:, 0]))
    result["max_abs_xyz_difference"] = float(np.max(np.abs(expected[:, 1:4] - actual[:, 1:4])))
    result["max_abs_quaternion_difference"] = float(np.max(np.abs(expected[:, 4:] - actual[:, 4:])))
    result["exact_numeric_match"] = bool(np.array_equal(expected, actual))
    result["passed"] = result["timestamps_equal"] and bool(np.allclose(expected[:, 1:], actual[:, 1:], atol=atol, rtol=0.))
    if not result["passed"]:
        result["reason"] = "ordered trajectory values differ; no coordinate alignment or quaternion-sign correction applied"
    return result


def _runtime(threads):
    # Replay needs source geometry, not SALAD or the feed-forward models.
    for name in ("slam", "vggt"):
        spec = env.SOURCES[name]
        if not (spec["path"] / spec["marker"]).is_file():
            raise RuntimeError(f"Missing {name} sources; run tools/setup_sources.py or set {spec['variable']}")
    env.prepare()
    provenance = env.verify_upstream_snapshot()
    if not provenance["unmodified"]:
        raise RuntimeError(f"Upstream SLAM provenance mismatch: {provenance['mismatches']}")
    env.install_headless_viewer()
    # The upstream Solver module imports this name even when __init__ is not
    # called. A failing stub makes accidental retrieval construction explicit
    # and avoids importing SALAD/Lightning solely for backend replay.
    retrieval = types.ModuleType("vggt_slam.loop_closure")
    class DisabledRetrieval:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("CPU prediction replay must not instantiate image retrieval")
    retrieval.ImageRetrieval = DisabledRetrieval
    sys.modules["vggt_slam.loop_closure"] = retrieval
    try:
        import torch
        import gtsam
        for name in ("SL4", "PriorFactorSL4", "BetweenFactorSL4"):
            if not hasattr(gtsam, name):
                raise RuntimeError(f"GTSAM is missing {name}; use a build with SL(4) bindings")
        from vggt_slam.solver import Solver
        from vggt_slam.map import GraphMap
        from vggt_slam.graph import PoseGraph
        from vggt_slam.submap import Submap
        from vggt_slam.scale_solver import estimate_scale_pairwise
        from vggt_slam.viewer import Viewer
    except ImportError as exc:
        raise RuntimeError(f"Missing CPU replay dependency: {exc}; see docs/INSTALL.md") from exc
    torch.set_num_threads(threads)
    return SimpleRuntime(torch, gtsam, Solver, GraphMap, PoseGraph, Submap, estimate_scale_pairwise, Viewer, provenance)


class SimpleRuntime:
    def __init__(self, torch, gtsam, Solver, GraphMap, PoseGraph, Submap, estimate_scale, Viewer, provenance):
        self.torch, self.gtsam, self.Solver = torch, gtsam, Solver
        self.GraphMap, self.PoseGraph, self.Submap = GraphMap, PoseGraph, Submap
        self.estimate_scale, self.Viewer, self.provenance = estimate_scale, Viewer, provenance


def _replay(paths, output, policy, runtime, cloud_conf_percentile, groundtruth=None):
    output.mkdir(parents=True, exist_ok=True)
    solver_class = make_solver_class(runtime.Solver, policy, scale_estimator=runtime.estimate_scale,
                                     gtsam_module=runtime.gtsam)
    solver = solver_class.__new__(solver_class)  # no SALAD, Torch hub, model loading, or CUDA
    solver.map, solver.graph = runtime.GraphMap(), runtime.PoseGraph()
    solver.init_conf_threshold = cloud_conf_percentile
    solver.vis_voxel_size, solver.vis_imgs = None, False
    solver.viewer, solver.current_working_submap = runtime.Viewer(), None
    updates = []
    started = time.monotonic()
    for index, path in enumerate(paths):
        predictions, metadata = load_predictions(path, restore_torch=True)
        names, submap_id = metadata["frame_names"], int(metadata["submap_id"])
        images = predictions["images"]
        if not isinstance(images, np.ndarray):
            raise TypeError(f"Cached solver images must be NumPy arrays, without reprocessing: {path}")
        if images.ndim != 4 or images.shape[0] != len(names) or images.shape[1] != 3 or not names:
            raise ValueError(f"Invalid image/frame-list dimensions in {path}")
        # Match run_predictions' actual regular-node IDs; auxiliary loop
        # submaps occupy graph IDs between ordinary windows.
        expected_id = 0 if solver.map.get_largest_key() is None else (
            solver.map.get_largest_key() + solver.map.get_latest_submap().get_last_non_loop_frame_index() + 1)
        if submap_id != expected_id:
            raise ValueError(f"Out-of-order/incompatible cache {path}: regular submap {submap_id}, expected {expected_id}")
        submap = runtime.Submap(submap_id)
        submap.add_all_frames(runtime.torch.from_numpy(images))
        submap.set_frame_ids(names)
        submap.set_img_names(names)
        submap.set_last_non_loop_frame_index(len(names) - 1)
        solver.current_working_submap = submap
        accepted = predictions.get("detected_loops", [])
        for loop in accepted:
            if loop.query_submap_id != submap_id:
                raise ValueError(f"Loop query ID disagrees with regular submap ID in {path}")
            solver.graph.increment_loop_closure()
        # No intrinsics/rotation/dtype corrections here: this is the same
        # frozen backend input as the online run, including accepted loops.
        before = time.monotonic()
        solver.add_points(predictions)
        solver.graph.optimize()
        updates.append({"update": index, "cache": path.name, "submap_id": submap_id,
                        "frame_names": names, "n_frames": len(names),
                        "n_submaps": int(solver.map.get_num_submaps()), "graph_nodes": int(solver.graph.values.size()),
                        "loops_accepted_in_update": len(accepted), "loop_closures_total": int(solver.graph.get_num_loops()),
                        "backend_seconds": time.monotonic() - before})
        print(json.dumps(updates[-1]), flush=True)
        del predictions, images
    poses = output / "poses.tum"
    solver.map.write_poses_to_file(str(poses), solver.graph, kitti_format=False)
    diagnostics = getattr(solver, "backend_diagnostics", [])
    (output / "backend_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    result = {"schema": "vggt_omega_slam.cpu_replay.v1", "status": "complete", "policy": policy.to_dict(),
              "cloud_conf_percentile": cloud_conf_percentile, "gpu_used": False, "models_loaded": False,
              "retrieval_executed": False, "accepted_loops_frozen": True, "updates": updates,
              "n_updates": len(updates), "n_submaps": int(solver.map.get_num_submaps()),
              "graph_nodes": int(solver.graph.values.size()), "loop_closures_total": int(solver.graph.get_num_loops()),
              "n_trajectory_rows": len(_read_tum_rows(poses)), "n_backend_diagnostics": len(diagnostics),
              "n_scale_diagnostics": sum("n_scale_pixels" in row for row in diagnostics),
              "wall_seconds": time.monotonic() - started, "poses": str(poses),
              "provenance_check": runtime.provenance}
    if groundtruth is not None:
        from eval_ate import evaluate
        result["ate"] = evaluate(groundtruth, poses)
    (output / "replay.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def _check_reference_counts(reference_poses, replay):
    metadata_path = reference_poses.parent / "metadata.json"
    if not metadata_path.is_file():
        return {"checked": False, "reason": "No online metadata.json next to reference trajectory"}
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("status") != "complete":
        raise RuntimeError("The online reference run is not complete")
    original = metadata.get("updates", [])
    current = replay["updates"]
    if len(original) != len(current):
        raise RuntimeError("Online/replay update counts differ")
    for previous, now in zip(original, current):
        for key in ("frame_names", "n_frames", "n_submaps", "graph_nodes", "loops_accepted_in_update"):
            if key in previous and previous[key] != now[key]:
                raise RuntimeError(f"Online/replay update {now['update']} differs for {key}")
    return {"checked": True, "updates_checked": len(current), "passed": True}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, required=True, help="lossless caches written by run_slam --prediction-cache")
    parser.add_argument("--output", type=Path, required=True, help="fresh replay output directory")
    parser.add_argument("--groundtruth", type=Path)
    parser.add_argument("--reference-poses", type=Path, help="online poses.tum; verifies legacy before policy evaluation")
    parser.add_argument("--reference-atol", type=float, default=0., help="numeric reference tolerance; default 0 is exact")
    parser.add_argument("--conf-threshold", type=float, default=25., help="original cloud-filter percentile, separate from scale selection")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--policy", choices=POLICY_NAMES, default=None)
    parser.add_argument("--policy-config", type=Path, help="JSON object of BackendPolicy fields")
    parser.add_argument("--scale-confidence-percentile", type=float)
    parser.add_argument("--scale-confidence-scope", choices=("submap", "frame"))
    parser.add_argument("--cross-submap-noise-multiplier", type=float)
    parser.add_argument("--loop-noise-multiplier", type=float)
    parser.add_argument("--huber-delta", type=float)
    parser.add_argument("--min-scale-points", type=int)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must be a fresh directory")
    if not args.cache_dir.is_dir() or not prediction_paths(args.cache_dir):
        parser.error("--cache-dir must contain lossless prediction .npz files")
    if not np.isfinite(args.reference_atol) or args.reference_atol < 0:
        parser.error("--reference-atol must be finite and nonnegative")
    if args.threads < 1 or not 0 <= args.conf_threshold <= 100:
        parser.error("invalid CPU thread count or cloud confidence percentile")
    config = json.loads(args.policy_config.read_text()) if args.policy_config else {}
    if args.policy is not None:
        config["name"] = args.policy
    for key in ("scale_confidence_percentile", "scale_confidence_scope", "cross_submap_noise_multiplier",
                "loop_noise_multiplier", "huber_delta", "min_scale_points"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    try:
        args.policy = BackendPolicy(**config)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    paths = prediction_paths(args.cache_dir)
    args.output.mkdir(parents=True)
    result = {"status": "starting", "policy": args.policy.to_dict(), "cache_dir": str(args.cache_dir),
              "gpu_used": False, "cache_count": len(paths)}
    try:
        with (args.output / "run.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            runtime = _runtime(args.threads)
            legacy = None
            if args.reference_poses is not None and args.policy.name != "legacy":
                legacy = _replay(paths, args.output / "legacy_verification", BackendPolicy(), runtime,
                                 args.conf_threshold, args.groundtruth)
            if args.policy.name == "legacy":
                legacy = _replay(paths, args.output, args.policy, runtime, args.conf_threshold, args.groundtruth)
            if args.reference_poses is not None:
                reference_check = compare_reference(args.reference_poses, Path(legacy["poses"]), atol=args.reference_atol)
                reference_check["online_update_counts"] = _check_reference_counts(args.reference_poses, legacy)
                result["legacy_reference_verification"] = reference_check
                if not reference_check["passed"]:
                    raise RuntimeError("Legacy CPU replay differs from online reference; policy replay was not run")
            if args.policy.name == "legacy":
                result.update(legacy)
            else:
                result.update(_replay(paths, args.output, args.policy, runtime, args.conf_threshold, args.groundtruth))
            result["legacy_equivalence_checked"] = args.reference_poses is not None
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        print(result["error"], file=sys.stderr)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    brief = {key: result.get(key) for key in ("status", "policy", "gpu_used", "graph_nodes", "n_trajectory_rows", "error")}
    print(json.dumps(brief), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
