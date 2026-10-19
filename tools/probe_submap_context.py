#!/usr/bin/env python3
"""Measure the effect of shifting a 17-keyframe window by eight keyframes.

The same nine shared frames are evaluated in timestamp order in both contexts.
Both backends keep their own baseline preprocessing and checkpoint unchanged.
`design` and `analyze` are CPU-only; only `forward` imports torch/acquires CUDA.
The caller must supply matching baseline caches and RGB images (raw or rectified).

Example:
  python tools/probe_submap_context.py design --baseline-root /path/to/analysis \
      --data-root /path/to/tum_rgbd --out runs/submap_context
  # Use exactly the commands recorded in runs/submap_context/design.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from vos import env, why_utils as wu

SCHEMA = "vggt-omega-slam-submap-context-v1"
WINDOW_SIZE = 17
SHIFT = 8
BACKENDS = ("vggt", "omega")
SEQUENCES = ("desk", "desk2", "room")
RAW_INTRINSICS = [[517.3, 0.0, 318.6], [0.0, 516.5, 255.3], [0.0, 0.0, 1.0]]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def frame_names(values):
    """Canonical names from NPZ strings/bytes; reject ambiguous duplicate labels."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("frame_names must be a one-dimensional string array")
    result = []
    for value in values.tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str):
            raise ValueError("frame_names must contain strings, not numeric/object labels")
        name = value.replace("\\", "/").rsplit("/", 1)[-1]
        if not name or not np.isfinite(wu.stamp_of(name)):
            raise ValueError(f"invalid timestamp filename: {value!r}")
        result.append(name)
    if len(set(result)) != len(result):
        raise ValueError("duplicate frame_names make cache correspondence ambiguous")
    return result


def load_cache(path, expected_names=None):
    """Read named cameras only; legacy `names` and new `frame_names` are supported."""
    with np.load(path, allow_pickle=False) as archive:
        keys = [key for key in ("frame_names", "names") if key in archive]
        if not keys:
            raise ValueError(f"{path}: cache has no frame_names/names")
        names = frame_names(archive[keys[0]])
        if len(keys) == 2 and names != frame_names(archive[keys[1]]):
            raise ValueError(f"{path}: conflicting frame_names and names")
        pose = np.asarray(archive["pose_enc"], dtype=np.float64)
        hw = np.asarray(archive["image_hw"], dtype=np.int64)
        result = {"frame_names": names, "pose_enc": pose, "image_hw": hw.tolist()}
        if "design_sha256" in archive:
            result["design_sha256"] = str(archive["design_sha256"].item())
    if expected_names is not None and names != list(expected_names):
        raise ValueError(f"{path}: cached frame order differs from the fixed inference order")
    if pose.shape != (len(names), 9) or not np.isfinite(pose).all():
        raise ValueError(f"{path}: invalid pose_enc shape/values")
    if hw.shape != (2,) or np.any(hw <= 0):
        raise ValueError(f"{path}: invalid image_hw")
    if np.any(np.linalg.norm(pose[:, 3:7], axis=1) < 1e-12):
        raise ValueError(f"{path}: zero quaternion")
    if np.any((pose[:, 7:] <= 0) | (pose[:, 7:] >= np.pi)):
        raise ValueError(f"{path}: FoV must be in (0, pi)")
    return result


def find_baseline(root, backend, tag):
    """Accept either an analysis directory or its cache directory, never guess tags."""
    root = Path(root)
    candidates = [root / "cache" / backend / f"{tag}.npz", root / backend / f"{tag}.npz"]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise ValueError(f"ambiguous baseline paths for {backend}/{tag}: {existing}")
    return existing[0] if existing else None


def build_pairs(manifest, windows_per_sequence=2):
    """Select boundaries using frame positions only, before examining predictions."""
    if windows_per_sequence < 2:
        raise ValueError("at least two internal boundaries per sequence are required")
    pairs, datasets = [], manifest["datasets"]
    for seq in SEQUENCES:
        partitions = manifest["partitions"][seq]
        reference = [frame_names(names) for names in partitions["vggt_loops1"]]
        if any([frame_names(names) for names in windows] != reference for windows in partitions.values()):
            raise ValueError(f"{seq}: backend/loop partitions have different keyframes")
        keyframes = list(reference[0])
        starts = [0]
        for names in reference[1:]:
            if keyframes[-1] != names[0]:
                raise ValueError(f"{seq}: original windows lack their single shared frame")
            starts.append(len(keyframes) - 1)
            keyframes.extend(names[1:])
        if len(set(keyframes)) != len(keyframes):
            raise ValueError(f"{seq}: repeated keyframes outside the expected one-frame overlap")
        stamps = np.asarray([wu.stamp_of(name) for name in keyframes])
        if np.any(np.diff(stamps) <= 0):
            raise ValueError(f"{seq}: keyframes are not strictly chronological")
        candidates = []
        for update, (start, names) in enumerate(zip(starts, reference)):
            shifted_start = start + SHIFT
            if len(names) != WINDOW_SIZE or shifted_start + WINDOW_SIZE > len(keyframes):
                continue
            shifted = keyframes[shifted_start:shifted_start + WINDOW_SIZE]
            common = [name for name in names if name in set(shifted)]
            if len(common) != WINDOW_SIZE - SHIFT:
                raise ValueError("a shifted window must retain exactly nine original frames")
            candidates.append({
                "id": f"{seq}_boundary_{start + WINDOW_SIZE - 1:04d}", "seq": seq,
                "split": "development" if seq == "desk" else "validation",
                "baseline_tag": f"A_{seq}_{update:02d}", "baseline_update": update,
                "boundary_keyframe_index": start + WINDOW_SIZE - 1,
                "baseline_start": start, "shifted_start": shifted_start,
                "baseline_frame_names": names, "shifted_frame_names": shifted,
                "common_frame_names": common, "anchor_frame_name": common[0],
                "sequence_directory": datasets[seq],
            })
        if len(candidates) < windows_per_sequence:
            raise ValueError(f"{seq}: only {len(candidates)} eligible internal boundaries")
        selected = np.rint(np.linspace(0, len(candidates) - 1, windows_per_sequence)).astype(int)
        pairs.extend(candidates[int(index)] for index in selected)
    return pairs


def calibration(path):
    if path is None:
        return {"K": RAW_INTRINSICS, "image_size_wh": [640, 480],
                "source": "published TUM fr1 raw RGB calibration"}
    record = json.loads(Path(path).read_text())
    matrix = np.asarray(record.get("K_new", record.get("K")), dtype=float)
    size = np.asarray(record.get("image_size_wh", [640, 480]), dtype=int)
    if matrix.shape != (3, 3) or size.shape != (2,) or not np.isfinite(matrix).all():
        raise ValueError("calibration needs K_new or K (3x3) and image_size_wh")
    if np.any(np.diag(matrix)[:2] <= 0) or np.any(size <= 0):
        raise ValueError("calibration focal lengths and image dimensions must be positive")
    return {"K": matrix.tolist(), "image_size_wh": size.tolist(),
            "source": str(Path(path).resolve()), "file_sha256": env.sha256(path)}


def saved_path(out, backend, pair, context):
    return Path(out) / "cache" / backend / f"{pair['id']}_{context}.npz"


def make_design(args):
    manifest = json.loads(args.manifest.read_text())
    pairs = build_pairs(manifest, args.windows_per_sequence)
    design = {
        "schema": SCHEMA, "window_size": WINDOW_SIZE, "shift_keyframes": SHIFT,
        "manifest_sha256": env.sha256(args.manifest), "manifest": str(args.manifest.resolve()),
        "baseline_root": str(args.baseline_root.resolve()), "data_root": str(args.data_root.resolve()),
        "max_gt_diff_s": args.max_gt_diff,
        "calibration": calibration(args.calibration), "pairs": pairs,
        "preprocessing": {"omega": "official balanced image_resolution=512, patch_size=16",
                          "vggt": "pinned VGGT_SPARK default crop, width=518"},
        "selection_rule": "per sequence, uniformly select eligible chronological original boundaries; no metric selection",
        "metric_rule": "decode quaternion/FoV in float64; match names; anchor predictions and GT at the same first shared frame; fit each shared trajectory to the identical GT frames with Sim(3)",
        "limitations": ["exploratory correlated windows, not independent trials",
                        "baseline cache has no RGB-content provenance: caller must supply its matching raw/rectified images",
                        "context window order changes the network reference frame; coordinate gauges are removed before scoring",
                        "each backend keeps its original preprocessing; cross-backend pixel budgets are not equalized"],
    }
    missing_baselines = 0
    for pair in pairs:
        pair["baseline_cache"] = {}
        pair["baseline_image_hw"] = {}
        for backend in BACKENDS:
            path = find_baseline(args.baseline_root, backend, pair["baseline_tag"])
            if path is not None:
                payload = load_cache(path, pair["baseline_frame_names"])
                pair["baseline_cache"][backend] = {"path": str(path.resolve()), "sha256": env.sha256(path)}
                pair["baseline_image_hw"][backend] = payload["image_hw"]
            else:
                missing_baselines += 1
        source_names = list(dict.fromkeys(pair["baseline_frame_names"] + pair["shifted_frame_names"]))
        pair["rgb_sha256"] = {}
        for name in source_names:
            path = args.data_root / pair["sequence_directory"] / "rgb" / name
            if not path.is_file():
                raise FileNotFoundError(path)
            pair["rgb_sha256"][name] = env.sha256(path)
    design["planned_forwards"] = {"shifted": len(pairs) * len(BACKENDS), "uncached_baselines": missing_baselines,
                                   "total": len(pairs) * len(BACKENDS) + missing_baselines}
    canonical = json.dumps(design, sort_keys=True, separators=(",", ":"))
    design["design_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    options = ["--baseline-root", str(args.baseline_root.resolve()), "--data-root", str(args.data_root.resolve()),
               "--out", str(args.out.resolve()), "--manifest", str(args.manifest.resolve()),
               "--windows-per-sequence", str(args.windows_per_sequence), "--max-gt-diff", str(args.max_gt_diff)]
    if args.calibration is not None:
        options += ["--calibration", str(args.calibration.resolve())]
    design["commands"] = {stage: shlex.join([sys.executable, str(Path(__file__).resolve()), stage] + options)
                          for stage in ("forward", "analyze")}
    design_path = args.out / "design.json"
    if design_path.is_file():
        previous = json.loads(design_path.read_text())
        if previous.get("design_sha256") != design["design_sha256"]:
            raise ValueError("existing design differs; use a fresh --out directory")
        return previous
    write_json(design_path, design)
    return design


def read_design(args):
    design = json.loads((args.out / "design.json").read_text())
    if design.get("schema") != SCHEMA:
        raise ValueError("unsupported context design")
    if Path(design["baseline_root"]) != args.baseline_root.resolve() or Path(design["data_root"]) != args.data_root.resolve():
        raise ValueError("CLI baseline/data roots differ from the saved design")
    if args.max_gt_diff != design["max_gt_diff_s"]:
        raise ValueError("GT association tolerance differs from the saved design")
    return design


def cached_prediction(design, out, backend, pair, context):
    names = pair[f"{context}_frame_names"]
    if context == "baseline" and backend in pair["baseline_cache"]:
        record = pair["baseline_cache"][backend]
        path = Path(record["path"])
        if env.sha256(path) != record["sha256"]:
            raise ValueError(f"baseline cache changed since design: {path}")
        return load_cache(path, names)
    path = saved_path(out, backend, pair, context)
    if not path.is_file():
        return None
    result = load_cache(path, names)
    if result.get("design_sha256") != design["design_sha256"]:
        raise ValueError(f"cache belongs to a different design: {path}")
    return result


def forward(args, design):
    todo = {backend: [(pair, context) for pair in design["pairs"] for context in ("baseline", "shifted")
                      if cached_prediction(design, args.out, backend, pair, context) is None]
            for backend in BACKENDS}
    print(json.dumps({backend: len(jobs) for backend, jobs in todo.items()}), "forwards remaining", flush=True)
    if not any(todo.values()):
        return
    for pair in design["pairs"]:
        for name, expected in pair["rgb_sha256"].items():
            path = args.data_root / pair["sequence_directory"] / "rgb" / name
            if env.sha256(path) != expected:
                raise ValueError(f"RGB input changed since design: {path}")
    env.prepare()
    env.claim_gpu(9200, "VGGT-Omega-SLAM submap context probe")  # exactly one process-wide claim
    from vos.gpu import ensure_exclusive, allocator_budget_bytes
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("forward stage requires CUDA")
    torch.set_num_threads(4)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(allocator_budget_bytes(10 * 2**30, total) / total)
    log_path = args.out / "forward.json"
    log = json.loads(log_path.read_text()) if log_path.is_file() else {"entries": []}
    log.update(design_sha256=design["design_sha256"], torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0), source_commits_expected={key: record["commit"] for key, record in env.SOURCES.items()},
               checkpoint_sha256_expected={"vggt": env.VGGT_SHA256, "omega": env.OMEGA_SHA256})
    for backend in BACKENDS:
        if not todo[backend]:
            continue
        ensure_exclusive()
        if backend == "vggt":
            from vos.vggt_backend import load_vggt
            from vggt.utils.load_fn import load_and_preprocess_images as loader
            model = load_vggt()
        else:
            from vos.omega_backend import load_omega, load_images_omega as loader
            model = load_omega()
        torch.cuda.synchronize()
        for pair, context in todo[backend]:
            names = pair[f"{context}_frame_names"]
            paths = [str(args.data_root / pair["sequence_directory"] / "rgb" / name) for name in names]
            images = prediction = None
            entry = {"backend": backend, "pair": pair["id"], "context": context}
            try:
                ensure_exclusive()
                images = loader(paths).to("cuda")
                hw = list(images.shape[-2:])
                expected = pair["baseline_image_hw"].get(backend)
                if expected is None and context == "shifted":
                    expected = cached_prediction(design, args.out, backend, pair, "baseline")["image_hw"]
                if expected is not None and hw != expected:
                    raise ValueError(f"preprocessing changed from baseline {expected} to {hw}")
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    prediction = model(images)
                torch.cuda.synchronize()
                entry.update(status="ok", seconds=time.perf_counter() - start, image_hw=hw,
                             peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
                path = saved_path(args.out, backend, pair, context)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, frame_names=np.asarray(names),
                                    pose_enc=prediction["pose_enc"].float().cpu().numpy()[0], image_hw=np.asarray(hw),
                                    design_sha256=np.asarray(design["design_sha256"]))
                temporary.replace(path)
            except Exception as exc:
                entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                log["entries"].append(entry)
                write_json(log_path, log)
                images = prediction = None
                torch.cuda.empty_cache()
            print(json.dumps(entry), flush=True)
        del model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def shared_cameras(payload, common_names):
    """Decode only the fixed named frame set in fp64, ignoring rounded cached matrices."""
    lookup = {name: index for index, name in enumerate(payload["frame_names"])}
    if len(lookup) != len(payload["frame_names"]):
        raise ValueError("duplicate prediction frame names")
    if any(name not in lookup for name in common_names):
        raise ValueError("prediction cache is missing shared frames")
    pose = payload["pose_enc"][[lookup[name] for name in common_names]]
    rotation_w2c = wu.quat_to_rot(pose[:, 3:7])
    rotation_c2w = rotation_w2c.transpose(0, 2, 1)
    centers = -np.einsum("sij,sj->si", rotation_c2w, pose[:, :3])
    height, width = payload["image_hw"]
    focal = np.column_stack((width / 2 / np.tan(pose[:, 8] / 2), height / 2 / np.tan(pose[:, 7] / 2)))
    return centers, rotation_c2w, focal


def anchor(centers, rotations):
    """Express one trajectory in its first shared camera frame, never its window's first camera."""
    basis = rotations[0].T
    return (centers - centers[0]) @ basis.T, np.einsum("ij,sjk->sik", basis, rotations)


def compare_pair(baseline, shifted, pair, groundtruth, camera, max_diff=0.01):
    names = pair["common_frame_names"]
    if baseline["image_hw"] != shifted["image_hw"]:
        raise ValueError("context comparison must keep preprocessing dimensions fixed")
    stamps = [wu.stamp_of(name) for name in names]
    index, valid, dt = wu.associate_frames(stamps, groundtruth[0], max_diff)
    if not valid.all():
        missing = [name for name, ok in zip(names, valid) if not ok]
        raise ValueError(f"all nine shared frames need GT with the same anchor; missing: {missing}")
    gt_centers, gt_rotations = anchor(groundtruth[1][index], wu.quat_to_rot(groundtruth[2][index]))
    path = wu.path_length(gt_centers)
    if path <= 0:
        raise ValueError("shared GT path has zero length")
    true = np.asarray(camera["K"])[[0, 1], [0, 1]] / np.asarray(camera["image_size_wh"])
    result = {"common_frame_names": names, "anchor_frame_name": names[0], "n_frames": len(names),
              "max_gt_dt_s": float(dt.max()), "gt_path_m": path, "contexts": {}}
    anchored, rotations = {}, {}
    fits = {}
    for context, payload in (("baseline", baseline), ("shifted", shifted)):
        centers, orientation, focal = shared_cameras(payload, names)
        anchored[context], rotations[context] = anchor(centers, orientation)
        fit = wu.sim3_ate(anchored[context], gt_centers)
        if fit["status"] != "ok":
            raise ValueError(f"undefined shared-frame Sim(3) for {context}: {fit['reason']}")
        fits[context] = fit
        height, width = payload["image_hw"]
        bias = focal / np.asarray([width, height]) / true - 1
        result["contexts"][context] = {
            "ate_rmse_m": fit["rmse"], "ate_over_gt_path": fit["rmse"] / path,
            "fx_bias_mean": float(bias[:, 0].mean()), "fy_bias_mean": float(bias[:, 1].mean()),
            "fx_bias_per_frame": bias[:, 0].tolist(), "fy_bias_per_frame": bias[:, 1].tolist(),
            "scale_m_per_unit": fit["scale"],
            "rel_rotation_error_deg": wu.summarize(wu.relative_rotation_errors(rotations[context], gt_rotations)),
        }
    between = wu.sim3_ate(anchored["shifted"], anchored["baseline"])
    if between["status"] != "ok":
        raise ValueError(f"undefined between-context Sim(3): {between['reason']}")
    rmse = between["rmse"] * fits["baseline"]["scale"]
    a, b = result["contexts"]["baseline"], result["contexts"]["shifted"]
    result.update(
        fx_bias_change_pp=100 * (b["fx_bias_mean"] - a["fx_bias_mean"]),
        fy_bias_change_pp=100 * (b["fy_bias_mean"] - a["fy_bias_mean"]),
        ate_change_m=b["ate_rmse_m"] - a["ate_rmse_m"],
        ate_ratio_shifted_over_baseline=b["ate_rmse_m"] / a["ate_rmse_m"] if a["ate_rmse_m"] > 1e-12 else None,
        context_shape_change_rmse_m=rmse, context_shape_change_over_gt_path=rmse / path,
        anchored_orientation_change_deg=wu.summarize(wu.rotation_angle_deg(
            rotations["baseline"].transpose(0, 2, 1) @ rotations["shifted"])),
    )
    return result


def analyze(args, design):
    sequence_directories = {pair["seq"]: pair["sequence_directory"] for pair in design["pairs"]}
    groundtruth = {seq: wu.load_groundtruth(args.data_root / directory / "groundtruth.txt")
                  for seq, directory in sequence_directories.items()}
    rows, missing = [], []
    for backend in BACKENDS:
        for pair in design["pairs"]:
            a = cached_prediction(design, args.out, backend, pair, "baseline")
            b = cached_prediction(design, args.out, backend, pair, "shifted")
            if a is None or b is None:
                missing.append({"backend": backend, "pair": pair["id"]})
                continue
            result = compare_pair(a, b, pair, groundtruth[pair["seq"]], design["calibration"], args.max_gt_diff)
            rows.append({"backend": backend, "pair": pair["id"], "seq": pair["seq"], "split": pair["split"], **result})
    summaries = {}
    for backend in BACKENDS:
        summaries[backend] = {}
        for split in ("development", "validation"):
            subset = [row for row in rows if row["backend"] == backend and row["split"] == split]
            summaries[backend][split] = {
                "n_windows": len(subset),
                "fx_bias_change_pp": wu.summarize([row["fx_bias_change_pp"] for row in subset]),
                "context_shape_change_over_gt_path": wu.summarize([row["context_shape_change_over_gt_path"] for row in subset]),
                "ate_ratio_shifted_over_baseline": wu.summarize([
                    row["ate_ratio_shifted_over_baseline"] for row in subset if row["ate_ratio_shifted_over_baseline"] is not None]),
                "n_shifted_ate_better": sum(row["ate_change_m"] < 0 for row in subset),
            }
    output = {"schema": SCHEMA, "design_sha256": design["design_sha256"],
              "status": "complete" if not missing else "partial", "missing": missing,
              "metric_version": "pose_encoding_float64_common_anchor_sim3_v1", "summary": summaries, "rows": rows,
              "limitations": design["limitations"]}
    write_json(args.out / "results.json", output)
    print(json.dumps({"status": output["status"], "missing": missing, "summary": summaries}, indent=2))
    return 0 if not missing else 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("design", "forward", "analyze"))
    parser.add_argument("--baseline-root", type=Path, required=True, help="original analysis directory or its cache directory")
    parser.add_argument("--data-root", type=Path, required=True, help="matching TUM RGB sequence parent directory")
    parser.add_argument("--out", type=Path, required=True, help="new/resumable experiment directory")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/frame_manifest.json")
    parser.add_argument("--calibration", type=Path, help="rectified calibration JSON with K_new and image_size_wh; default is raw fr1")
    parser.add_argument("--windows-per-sequence", type=int, default=2)
    parser.add_argument("--max-gt-diff", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "design":
        design = make_design(args)
        print(json.dumps({"planned_forwards": design["planned_forwards"], "commands": design["commands"]}, indent=2))
        return 0
    design = read_design(args)
    if args.stage == "forward":
        forward(args, design)
        return 0
    return analyze(args, design)


if __name__ == "__main__":
    raise SystemExit(main())
