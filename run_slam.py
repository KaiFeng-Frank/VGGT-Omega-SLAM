#!/usr/bin/env python3
"""Headless VGGT-SLAM 2.0 runner with a switchable feed-forward front-end (VGGT or VGGT-Omega).

The default --backend-policy legacy uses the original upstream pipeline
(MIT-SPARK/VGGT-SLAM 35327ac, imported from a verified read-only snapshot):
LK-flow keyframe selection, submaps of --submap-size new keyframes plus one
overlap frame, SALAD retrieval, and an SL(4) factor graph. An opt-in Solver
subclass changes overlap confidence selection and factor noise without editing
the upstream files. The control loop retains the final partial batch and adds

  * a non-rendering Viewer stub (no viser server),
  * offline torch.hub for SALAD/DINOv2 (local checkpoints),
  * instrumentation by wrapping (model call, ImageRetrieval/GraphMap retrieval calls) - no upstream
    file is edited,
  * optional lossless prediction caching for CPU backend experiments,
  * for --backend omega: `vggt_slam.solver.load_and_preprocess_images` and
    `vggt_slam.solver.pose_encoding_to_extri_intri` are replaced by the VGGT-Omega loader/decoder
    and the model is `OmegaForSLAM` (see vos/omega_backend.py).

Input is an image folder (upstream main.py's --image_folder) or a TUM RGB-D sequence directory (its
rgb/ is used; original file names are kept because upstream parses the stamp from the file name).

Outputs in --output (must not exist): poses.tum, metadata.json, keyframes.csv, cloud.ply, run.log.
Translations and the cloud are in model units (monocular, no metric scale).
"""
# Control-loop structure adapted from MIT-SPARK/VGGT-SLAM main.py (BSD 2-Clause, (c) 2025 MIT-SPARK).
# Headless rendering and offline checkpoint loading do not modify upstream source.
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vos import env  # noqa: E402

env.prepare()  # configurable imports, optional GTSAM path; no GPU access

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

SCHEMA = "vggt_omega_slam.run.v1"


class Tee:
    """Duplicate stdout/stderr into run.log."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.stream.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def isatty(self):
        return False


def parse_args(argv=None):
    from vos.backend_policy import POLICY_NAMES

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, nargs="?", help="image folder, or TUM sequence dir (its rgb/ is used)")
    p.add_argument("--backend", choices=["vggt", "omega"], default="omega")
    p.add_argument("--output", type=Path, help="fresh output directory (must not exist)")
    p.add_argument("--submap-size", type=int, default=16, help="new keyframes per submap (upstream --submap_size)")
    p.add_argument("--overlap", type=int, default=1, help="upstream supports only 1 (--overlapping_window_size)")
    p.add_argument("--min-disparity", type=float, default=50.0)
    p.add_argument("--conf-threshold", type=float, default=25.0)
    p.add_argument("--backend-policy", choices=POLICY_NAMES, default="legacy")
    p.add_argument("--scale-confidence-percentile", type=float, help="overlap-scale confidence percentile; separate from cloud filtering")
    p.add_argument("--scale-confidence-scope", choices=("submap", "frame"), default="submap")
    p.add_argument("--cross-noise-multiplier", type=float, default=1.0, help="cross-submap factor standard-deviation multiplier")
    p.add_argument("--loop-noise-multiplier", type=float, default=1.0, help="loop factor standard-deviation multiplier")
    p.add_argument("--edge-huber-delta", type=float, help="optional Huber threshold in whitened residual units")
    p.add_argument("--lc-thres", type=float, default=0.95, help="SALAD distance threshold, upstream --lc_thres")
    p.add_argument("--max-loops", type=int, default=1, choices=[0, 1])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0, help="use only the first N images (0 = all)")
    p.add_argument("--similarity-config", type=Path, default=env.SIM_CONFIG,
                   help="omega: JSON with 'layer' and 'tau' (default config/omega_similarity.json if present)")
    p.add_argument("--omega-layer", type=int, default=None, help="omega: override similarity block (inter-frame index)")
    p.add_argument("--omega-tau", type=float, default=None, help="omega: override raw-ratio threshold tau")
    p.add_argument("--groundtruth", type=Path, default=None, help="TUM groundtruth.txt (auto-found next to rgb/)")
    p.add_argument("--max-cloud-points", type=int, default=2_000_000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cuda-memory-limit-gb", type=float, default=10.0)
    p.add_argument("--gpu-need-mb", type=int, default=None, help="default 9000 (omega) / 7500 (vggt)")
    p.add_argument("--verify-checkpoints", action="store_true", help="recompute sha256 of all checkpoints (slow)")
    p.add_argument("--prediction-cache", type=Path, help="save lossless model predictions for CPU backend replay")
    p.add_argument("--preflight-only", action="store_true", help="CPU import/SL4 check; no model, no GPU")
    args = p.parse_args(argv)
    if not args.preflight_only:
        if args.input is None or args.output is None:
            p.error("input and --output are required unless --preflight-only")
        if args.output.exists():
            p.error(f"{args.output} exists; use a fresh output directory")
    if args.overlap != 1:
        p.error("upstream supports only --overlap 1")
    if args.submap_size < 1 or args.max_frames < 0 or args.threads < 1 or args.max_cloud_points < 1:
        p.error("invalid numeric argument")
    if not 0 <= args.conf_threshold <= 100 or args.min_disparity < 0:
        p.error("confidence percentile must be in [0,100] and disparity must be nonnegative")
    try:
        backend_policy_from_args(args)
    except ValueError as exc:
        p.error(str(exc))
    return args


def backend_policy_from_args(args):
    from vos.backend_policy import BackendPolicy
    return BackendPolicy(name=args.backend_policy,
                         scale_confidence_percentile=args.scale_confidence_percentile,
                         scale_confidence_scope=args.scale_confidence_scope,
                         cross_submap_noise_multiplier=args.cross_noise_multiplier,
                         loop_noise_multiplier=args.loop_noise_multiplier,
                         huber_delta=args.edge_huber_delta)


# ---------------------------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------------------------
def resolve_input(path: Path):
    path = Path(path)
    if (path / "rgb").is_dir():
        gt = path / "groundtruth.txt"
        return path / "rgb", "tum", (gt if gt.is_file() else None)
    gt = path.parent / "groundtruth.txt"
    return path, "folder", (gt if path.name == "rgb" and gt.is_file() else None)


def list_images(folder, utils):
    """upstream main.py: glob, drop depth/txt/db names, sort by embedded number, downsample 1."""
    names = [f for f in glob.glob(os.path.join(folder, "*")) if os.path.isfile(f)
             and "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower()
             and "db" not in os.path.basename(f).lower()]
    names = utils.sort_images_by_number(names)
    return utils.downsample_images(names, 1)


# ---------------------------------------------------------------------------------------------
# instrumentation (wrappers; upstream files untouched)
# ---------------------------------------------------------------------------------------------
class RecordingModel:
    """Callable wrapper around the front-end: synchronised timing and similarity values per call."""

    def __init__(self, model, torch, backend):
        self.model, self.torch, self.backend = model, torch, backend
        self.calls = []

    def __call__(self, images, *args, **kwargs):
        from vos.gpu import ensure_exclusive
        ensure_exclusive()
        torch = self.torch
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = self.model(images, *args, **kwargs)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        similarity = bool(kwargs.get("compute_similarity", args[0] if args else False))
        record = {"input_shape": list(images.shape), "n_frames": int(images.shape[-4]), "seconds": seconds,
                  "compute_similarity": similarity, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())}
        if similarity:
            scaled = float(out["image_match_ratio"])
            raw = out.get("image_match_ratio_raw")
            record.update(image_match_ratio_raw=float(raw) if raw is not None else scaled,
                          image_match_ratio_scaled=scaled)
        self.calls.append(record)
        return out

    def __getattr__(self, name):
        return getattr(self.model, name)


class RetrievalRecorder:
    """Wrap GraphMap.retrieve_best_score_frame and ImageRetrieval.find_loop_closures on one Solver."""

    def __init__(self, solver):
        self.queries, self.matches, self.threshold = [], [], None
        original_best = solver.map.retrieve_best_score_frame
        original_find = solver.image_retrieval.find_loop_closures
        original_descriptors = solver.image_retrieval.get_batch_descriptors

        def descriptors(images):
            from vos.gpu import ensure_exclusive
            ensure_exclusive()
            return original_descriptors(images)

        def best(query_vector, current_submap_id, ignore_last_submap=True):
            result = original_best(query_vector, current_submap_id, ignore_last_submap=ignore_last_submap)
            self.queries.append({"salad_distance": float(result[0]), "submap_id": int(result[1]),
                                 "frame_index": int(result[2])})
            return result

        def find(map_, submap, max_similarity_thres=0.80, max_loop_closures=0):
            self.threshold = float(max_similarity_thres)
            matches = original_find(map_, submap, max_similarity_thres=max_similarity_thres,
                                    max_loop_closures=max_loop_closures)
            self.matches = list(matches)
            return matches

        solver.map.retrieve_best_score_frame = best
        solver.image_retrieval.find_loop_closures = find
        solver.image_retrieval.get_batch_descriptors = descriptors

    def reset(self):
        self.queries, self.matches, self.threshold = [], [], None


# ---------------------------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------------------------
def sample_cloud(solver, max_points):
    """Deterministic global stride over the native confidence-filtered points (internal map untouched).

    Samples exported points without changing the upstream map or confidence mask.
    """
    submaps = list(solver.map.get_submaps())
    counts = [int(np.count_nonzero(s.conf_masks > s.conf_threshold)) for s in submaps]
    total = sum(counts)
    stride = max(1, math.ceil(total / max_points))
    points_out, colors_out, offset, invalid = [], [], 0, 0
    for submap, count in zip(submaps, counts):
        points = submap.get_points_in_world_frame(solver.graph)
        colors = submap.get_points_colors()
        if len(points) != count or len(colors) != count:
            raise ValueError("confidence-mask point/color count mismatch")
        index = np.arange((-offset) % stride, count, stride, dtype=np.int64)
        offset += count
        points, colors = points[index], colors[index]
        finite = np.isfinite(points).all(axis=1)
        invalid += int(np.count_nonzero(~finite))
        points_out.append(points[finite])
        colors_out.append(colors[finite])
    points = np.concatenate(points_out) if points_out else np.empty((0, 3))
    colors = np.concatenate(colors_out) if colors_out else np.empty((0, 3))
    info = {"max_points": max_points, "native_eligible_points": total, "stride": stride,
            "written_points": int(len(points)), "nonfinite_omitted": invalid, "units": "model_units",
            "sampling": "global stride over native confidence-filtered point order"}
    return points, colors, info


def write_cloud(solver, path, max_points):
    import open3d as o3d

    points, colors, info = sample_cloud(solver, max_points)
    if not len(points):
        raise ValueError("map has no finite points")
    if np.issubdtype(colors.dtype, np.integer) or colors.max() > 1.0:
        colors = colors.astype(np.float64) / 255.0
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise RuntimeError("could not write cloud.ply")
    info["file_bytes"] = Path(path).stat().st_size
    return info


def write_poses(solver, path):
    with contextlib.redirect_stdout(open(os.devnull, "w")):  # upstream prints every frame-id list
        solver.map.write_poses_to_file(str(path), solver.graph, kitti_format=False)


def write_keyframes(path, keyframes, updates):
    in_updates = {}
    for u in updates:
        for name in u["frame_names"]:
            in_updates.setdefault(name, []).append(u["update"])
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["keyframe_index", "image_index", "image_name", "frame_id", "submap_updates"])
        for k, (image_index, name, frame_id) in enumerate(keyframes):
            w.writerow([k, image_index, name, f"{frame_id:.6f}", ";".join(map(str, in_updates.get(name, [])))])


def rss_kib():
    """Current resident set (kiB) split into anonymous and file-backed (mmap'd checkpoint) pages."""
    values = {}
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith(("RssAnon:", "RssFile:")):
                values[line.split(":")[0]] = int(line.split()[1])
    return {"anon": values.get("RssAnon"), "file": values.get("RssFile")}


def versions(torch):
    import cv2
    import gtsam

    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "numpy": np.__version__, "cv2": cv2.__version__,
            "gtsam_path": gtsam.__file__, "executable": sys.executable}


# ---------------------------------------------------------------------------------------------
def run(args, report, out, ctx):
    global cv2
    # Invalid inputs and missing setup fail before GPU locking or CUDA queries.
    if not args.preflight_only:
        image_dir, kind, gt_auto = resolve_input(args.input)
        if not image_dir.is_dir() or len(list(image_dir.iterdir())) < 2:
            raise ValueError(f"need at least 2 images in {image_dir}")
    env.require_sources(args.backend)
    env.install_headless_viewer()  # non-rendering Viewer; must precede importing vggt_slam.solver
    try:
        import cv2
        import torch
        import gtsam

        for name in ("SL4", "PriorFactorSL4", "BetweenFactorSL4"):
            if not hasattr(gtsam, name):
                raise RuntimeError(f"gtsam lacks {name}. Install a GTSAM build with SL(4) Python bindings "
                                   "(gtsam-develop or a suitable source build); set VOS_PYLIB for a custom build.")
        from vggt_slam import slam_utils as utils
        from vggt_slam.solver import Solver
        import vggt_slam.solver as solver_module
    except ImportError as exc:
        raise RuntimeError(f"Missing runtime dependency: {exc}. Install the upstream Python dependencies "
                           "and GTSAM with SL(4); source setup alone does not install these. See docs/INSTALL.md.") from exc

    report["versions"] = versions(torch)
    report["upstream"] = {"slam_commit": env.SLAM_COMMIT, "slam_source": str(env.SLAM_SRC),
                          "vggt_spark_source": str(env.VGGT_SRC), "salad_source": str(env.SALAD_SRC),
                          "dinov2_source": str(env.DINOV2_SRC), "omega_source": str(env.OMEGA_SRC),
                          "provenance_check": env.verify_upstream_snapshot()}
    if not report["upstream"]["provenance_check"]["unmodified"]:
        raise RuntimeError(f"upstream snapshot differs from its provenance: {report['upstream']['provenance_check']}")
    if args.preflight_only:
        try:
            if args.backend == "omega":
                import vggt_omega  # noqa: F401
                from vggt_omega.utils.pose_enc import encoding_to_camera  # noqa: F401
                from vos import omega_backend  # noqa: F401
            else:
                from vos import vggt_backend  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(f"Missing front-end dependency: {exc}. See docs/INSTALL.md.") from exc
        report.update(status="preflight_passed", model_loaded=False, gpu_used=False)
        return 0

    groundtruth = args.groundtruth or gt_auto
    image_names = list_images(image_dir, utils)
    if args.max_frames:
        image_names = image_names[: args.max_frames]
    if len(image_names) < 2:
        raise ValueError(f"need at least 2 images in {image_dir}, found {len(image_names)}")
    report["input"] = {"given": str(args.input), "kind": kind, "image_dir": str(image_dir),
                       "n_images_used": len(image_names), "first": os.path.basename(image_names[0]),
                       "last": os.path.basename(image_names[-1]),
                       "groundtruth": str(groundtruth) if groundtruth else None}

    # ---- GPU: lock first, before any model touches CUDA --------------------------------------
    env.require_checkpoints(args.backend)
    need = args.gpu_need_mb or (9000 if args.backend == "omega" else 7500)
    free = env.claim_gpu(need, f"vggt-omega-slam {args.backend} {args.input.name}")
    report["gpu"] = {"need_mb": need, "free_mb_at_claim": free, "lock": "vos.gpu.claim"}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (upstream has no CPU path)")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    from vos.gpu import allocator_budget_bytes
    limit = allocator_budget_bytes(int(args.cuda_memory_limit_gb * 1024 ** 3), total_memory)
    torch.cuda.set_per_process_memory_fraction(limit / total_memory)
    report["gpu"].update(device=torch.cuda.get_device_name(0), total_bytes=int(total_memory),
                         allocator_limit_bytes=int(limit))

    # ---- checkpoints and models --------------------------------------------------------------
    report["offline_hub_calls"] = env.configure_offline_hub(torch, args.output, env.SALAD_CKPT, env.DINOV2_SRC.resolve())
    ckpts = {"salad": {"path": str(env.SALAD_CKPT), "bytes": env.SALAD_CKPT.stat().st_size,
                       "sha256_expected": env.SALAD_SHA256}}
    if args.verify_checkpoints:
        ckpts["salad"]["sha256"] = env.sha256(env.SALAD_CKPT)
        assert ckpts["salad"]["sha256"] == env.SALAD_SHA256
    load_start = time.perf_counter()
    if args.backend == "omega":
        from vos import omega_backend as ob

        ckpts["omega"] = ob.check_omega_checkpoint(env.OMEGA_CKPT, args.verify_checkpoints)
        ckpts["omega"]["sha256_expected"] = env.OMEGA_SHA256
        omega = ob.load_omega(env.OMEGA_CKPT, "cuda")
        model = ob.OmegaForSLAM(omega, layer=args.omega_layer, tau=args.omega_tau, config_path=args.similarity_config)
        # Replace only the two front-end hooks the solver module imported by name.
        solver_module.load_and_preprocess_images = ob.load_images_omega
        solver_module.pose_encoding_to_extri_intri = ob.decode_pose
        report["similarity"] = model.describe()
        report["front_end"] = {"name": "VGGT-Omega 1B (512 balanced)", "weights_dtype": "float32",
                               "aggregator_autocast": "bfloat16", "loader": "vggt_omega.utils.load_fn balanced/512",
                               "decoder": "vggt_omega.utils.pose_enc.encoding_to_camera"}
    else:
        from vos import vggt_backend as vb

        ckpts["vggt"] = vb.check_vggt_checkpoint(env.VGGT_CKPT, args.verify_checkpoints)
        ckpts["vggt"]["sha256_expected"] = env.VGGT_SHA256
        model = vb.load_vggt(env.VGGT_CKPT, "cuda")
        report["similarity"] = {"layer": vb.similarity_layer(model), "token_offset": 5, "tau": 0.95,
                                "source": "VGGT_SPARK target_layer as constructed by VGGT()",
                                "upstream_test": "image_match_ratio < 0.95"}
        report["front_end"] = {"name": "VGGT-1B via VGGT_SPARK", "weights_dtype": "bfloat16",
                               "loader": "vggt.utils.load_fn crop/518", "decoder": "vggt.utils.pose_enc"}
    report["checkpoints"] = ckpts
    torch.cuda.synchronize()
    report["model_load_seconds"] = time.perf_counter() - load_start
    recording = RecordingModel(model, torch, args.backend)

    from vos.backend_policy import make_solver_class
    backend_policy = backend_policy_from_args(args)
    SolverClass = make_solver_class(Solver, backend_policy, scale_estimator=solver_module.estimate_scale_pairwise)
    solver = SolverClass(init_conf_threshold=args.conf_threshold, lc_thres=args.lc_thres, vis_voxel_size=None,
                         vis_imgs=False)
    report["backend_policy"] = backend_policy.to_dict()
    report["backend_diagnostics"] = getattr(solver, "backend_diagnostics", [])
    retrieval = RetrievalRecorder(solver)
    torch.cuda.synchronize()
    report["model_and_salad_load_seconds"] = time.perf_counter() - load_start
    report["status"] = "running"
    report["parameters_effective"] = {"overlapping_window_size": args.overlap, "downsample_factor": 1,
                                      "use_optical_flow_downsample": True}
    torch.cuda.reset_peak_memory_stats()

    keyframes, updates, loop_log = ctx["keyframes"], ctx["updates"], []
    ctx["solver"] = solver
    report.update(updates=updates, loop_log=loop_log)

    def process(names, update_index):
        """One iteration of upstream main.py's submap block, with timing/peak/loop records."""
        torch.cuda.reset_peak_memory_stats()
        retrieval.reset()
        n_calls, loops_before = len(recording.calls), solver.graph.get_num_loops()
        torch.cuda.synchronize()
        t_begin = time.perf_counter()
        predictions = solver.run_predictions(names, recording, args.max_loops, None, None)
        torch.cuda.synchronize()
        t_pred = time.perf_counter()
        if args.prediction_cache is not None:
            from vos.prediction_cache import save_predictions
            save_predictions(args.prediction_cache / f"{update_index:05d}.npz", predictions, names,
                             solver.current_working_submap.get_id())
        solver.add_points(predictions)
        t_add = time.perf_counter()
        solver.graph.optimize()  # upstream: with backend_time
        report["backend_diagnostics"] = getattr(solver, "backend_diagnostics", [])
        t_opt = time.perf_counter()
        calls = recording.calls[n_calls:]
        main_call = calls[0]
        lc_calls = [c for c in calls[1:] if c["compute_similarity"]]
        accepted = solver.graph.get_num_loops() - loops_before
        record = {"update": update_index, "frame_names": [os.path.basename(n) for n in names],
                  "n_frames": len(names), "input_shape": main_call["input_shape"],
                  "forward_seconds": main_call["seconds"],
                  "lc_forward_seconds": lc_calls[0]["seconds"] if lc_calls else None,
                  "run_predictions_seconds": t_pred - t_begin, "add_points_seconds": t_add - t_pred,
                  "optimize_seconds": t_opt - t_add,
                  "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                  "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                  "cuda_allocated_bytes_after": int(torch.cuda.memory_allocated()),
                  "rss_kib_after": rss_kib(),
                  "n_submaps": int(solver.map.get_num_submaps()), "graph_nodes": int(solver.graph.values.size()),
                  "loop_candidates": len(retrieval.matches), "loops_accepted_in_update": int(accepted)}
        updates.append(record)
        # GraphMap.retrieve_best_score_frame starts from a sentinel distance of 1000 when no earlier
        # (non-adjacent) submap exists; those queries carry no information and are not listed.
        real_queries = [q for q in retrieval.queries if q["salad_distance"] < 1000]
        if retrieval.matches or real_queries:
            entry = {"update": update_index, "salad_threshold": retrieval.threshold,
                     "n_queries": len(retrieval.queries), "n_queries_with_reference": len(real_queries),
                     "min_salad_distance": min((q["salad_distance"] for q in real_queries), default=None),
                     "candidates": []}
            current = solver.current_working_submap
            for k, match in enumerate(retrieval.matches):
                cand = {"salad_distance": float(match.similarity_score), "query_submap_id": int(match.query_submap_id),
                        "query_frame": int(match.query_submap_frame),
                        "query_image": os.path.basename(current.get_img_names_at_index(match.query_submap_frame)),
                        "detected_submap_id": int(match.detected_submap_id),
                        "detected_frame": int(match.detected_submap_frame),
                        "detected_image": os.path.basename(solver.map.get_submap(match.detected_submap_id)
                                                           .get_img_names_at_index(match.detected_submap_frame))}
                if k == 0 and lc_calls:
                    cand.update(raw_ratio=lc_calls[0]["image_match_ratio_raw"],
                                scaled_ratio=lc_calls[0]["image_match_ratio_scaled"])
                    cand["accepted"] = bool(accepted > 0)
                else:
                    cand.update(raw_ratio=None, scaled_ratio=None, accepted=False,
                                note="candidate beyond max_loops or no verification pass")
                entry["candidates"].append(cand)
            entry["queries"] = real_queries  # best earlier frame per query frame of this submap
            loop_log.append(entry)
        print(json.dumps({k: v for k, v in record.items() if k != "frame_names"}), flush=True)
        write_metadata(report, out)

    # ---- upstream main.py control loop -----------------------------------------------------
    image_names_subset = []
    count = 0
    image_count = 0
    total_start = time.time()
    keyframe_time, backend_time = utils.Accumulator(), utils.Accumulator()
    for image_index, image_name in enumerate(image_names):
        with keyframe_time:
            img = cv2.imread(image_name)
            if img is None:
                raise ValueError(f"cannot decode {image_name}")
            enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, False)
            if enough_disparity:
                image_names_subset.append(image_name)
                image_count += 1
                # same number extraction as upstream Submap.set_frame_ids
                keyframes.append((image_index, os.path.basename(image_name),
                                  float(re.search(r"\d+(?:\.\d+)?", os.path.basename(image_name)).group())))
        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlap or image_name == image_names[-1]:
            count += 1
            process(image_names_subset, count - 1)
            image_names_subset = image_names_subset[-args.overlap:]
        if image_index % 50 == 0:
            print(f"[progress] image {image_index + 1}/{len(image_names)}, keyframes {image_count}", flush=True)
    total_time = time.time() - total_start

    # ---- outputs ------------------------------------------------------------------------------
    write_poses(solver, out / "poses.tum")
    write_keyframes(out / "keyframes.csv", keyframes, updates)
    report["cloud"] = write_cloud(solver, out / "cloud.ply", args.max_cloud_points)
    forwards = [u["forward_seconds"] for u in updates]
    report["summary"] = {
        "images_used": len(image_names), "keyframes": image_count, "submaps_processed": count,
        "submaps_in_map": int(solver.map.get_num_submaps()), "loop_closures_accepted": int(solver.graph.get_num_loops()),
        "loop_candidates_total": sum(u["loop_candidates"] for u in updates),
        "total_seconds": total_time, "forward_seconds_total": float(sum(forwards)),
        "forward_seconds_mean_per_submap": float(np.mean(forwards)) if forwards else None,
        "forward_seconds_median_per_submap": float(np.median(forwards)) if forwards else None,
        "forward_seconds_per_keyframe": float(sum(forwards) / image_count) if image_count else None,
        "optimize_seconds_total": float(sum(u["optimize_seconds"] for u in updates)),
        "peak_cuda_allocated_bytes": int(max(u["peak_cuda_allocated_bytes"] for u in updates)),
        "peak_cuda_reserved_bytes": int(max(u["peak_cuda_reserved_bytes"] for u in updates)),
        "upstream_style_timers": {"vggt_timer_total": solver.vggt_timer.total_time,
                                  "loop_closure_timer_total": solver.loop_closure_timer.total_time,
                                  "keyframe_selection_total": keyframe_time.total_time}}
    report["poses_file"] = {"path": "poses.tum", "rows": sum(1 for _ in open(out / "poses.tum")),
                            "format": "stamp(parsed from file name) x y z qx qy qz qw; model units",
                            "writer": "solver.map.write_poses_to_file(kitti_format=False)"}
    if groundtruth is not None:
        try:
            import eval_ate

            ate = eval_ate.evaluate(groundtruth, out / "poses.tum")
            (out / "ate.json").write_text(json.dumps(ate, indent=2))
            report["ate"] = {k: ate.get(k) for k in ("status", "ate_rmse_m", "scale", "n_associated",
                                                      "n_estimate_rows", "n_groundtruth", "ate_rmse_se3_no_scale_m")}
            print("[ate]", json.dumps(report["ate"]), flush=True)
        except Exception as exc:  # evaluation must never fail a finished run
            report["ate"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    report["status"] = "complete"
    return 0


def write_metadata(report, out):
    clean = {k: v for k, v in report.items() if not k.startswith("_")}
    (out / "metadata.json").write_text(json.dumps(clean, indent=2, default=str))


def main(argv=None):
    args = parse_args(argv)
    started = time.monotonic()
    out = args.output
    if out is not None:
        out.mkdir(parents=True)
    report = {"schema": SCHEMA, "status": "starting", "backend": args.backend,
              "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "runner_sha256": env.sha256(__file__), "seed": args.seed, "gui": False}
    log = None
    previous_streams = (sys.stdout, sys.stderr)
    if out is not None:
        log = open(out / "run.log", "w")
        sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    code = 1
    ctx = {"keyframes": [], "updates": [], "solver": None}
    try:
        code = run(args, report, out, ctx)
    except BaseException as exc:  # noqa: BLE001 - record everything, then signal failure
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        traceback.print_exc()
        # Best-effort partial products so a late failure does not lose the finished submaps.
        if out is not None and ctx["solver"] is not None:
            for label, action in (("poses_partial.tum", lambda: write_poses(ctx["solver"], out / "poses_partial.tum")),
                                  ("keyframes.csv", lambda: write_keyframes(out / "keyframes.csv", ctx["keyframes"],
                                                                            ctx["updates"]))):
                try:
                    action()
                    report.setdefault("partial_outputs", []).append(label)
                except Exception as inner:  # noqa: BLE001
                    report.setdefault("partial_output_errors", []).append(f"{label}: {inner}")
        code = 1
    finally:
        if ctx["solver"] is not None:
            report["backend_diagnostics"] = getattr(ctx["solver"], "backend_diagnostics", [])
        report["wall_seconds"] = time.monotonic() - started
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if out is not None:
            write_metadata(report, out)
        brief = {k: report.get(k) for k in ("status", "backend", "error", "summary", "ate", "wall_seconds")}
        print("[result]", json.dumps(brief, default=str), flush=True)
        if log is not None:
            sys.stdout, sys.stderr = previous_streams
            log.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
