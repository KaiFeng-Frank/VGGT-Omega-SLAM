#!/usr/bin/env python3
"""Why does VGGT-Omega give a higher ATE than VGGT inside VGGT-SLAM 2.0 on TUM?  (no SLAM is run here)

Stage `forward` (GPU, one process, lock inside): single forwards only, results cached in
    runs/analysis_why/cache/<backend>/<tag>.npz
  A  every regular submap of the existing runs/compare/<seq>_vggt_loops1 (exact frame lists, 17 consecutive keyframes)
  B  sparse wide-baseline sets: N in {8,16,32} frames uniformly spaced over the whole sequence
     (+ two shifted copies of the grid, variant +1 / -1, to gauge the spread)
  R  repeat of A desk/0 (run-to-run nondeterminism of the forward)
Stage `analyze` (CPU): per-submap accuracy (A), sparse regime (B), overlap-frame coupling (C), results.json,
    results_zh.md and 原因分解.png.

Launch the GPU stage with:
  systemd-run --user --scope --quiet -p MemoryMax=12G env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    <python> tools/analysis_why.py forward
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vos import env  # noqa: E402  (sets sys.dont_write_bytecode)

PROJECT = env.PROJECT
OUT = PROJECT / "runs" / "analysis_why"
COMPARE = PROJECT / "runs" / "compare"
TUM = env.TUM_ROOT
SEQS = {"desk": "rgbd_dataset_freiburg1_desk", "desk2": "rgbd_dataset_freiburg1_desk2",
        "room": "rgbd_dataset_freiburg1_room"}
COMPARE_NAME = {"desk": "fr1_desk", "desk2": "fr1_desk2", "room": "fr1_room"}
BACKENDS = ("vggt", "omega")
SPARSE_N = (8, 16, 32)
SPARSE_VARIANTS = (0, 1, -1)
FULL_SUBMAP = 17
MEMORY_CAP_GIB = 10


# ----------------------------------------------------------------------------------------------
# jobs (CPU only)
# ----------------------------------------------------------------------------------------------
def load_update_lists(seq):
    """Frame-name lists of every regular submap for the 4 compare runs of `seq`."""
    lists = {}
    for backend in BACKENDS:
        for loops in (1, 0):
            path = COMPARE / f"{COMPARE_NAME[seq]}_{backend}_loops{loops}" / "metadata.json"
            if path.is_file():
                meta = json.loads(path.read_text())
                lists[(backend, loops)] = [list(u["frame_names"]) for u in meta["updates"]]
            else:
                manifest = json.loads((PROJECT / "results/frame_manifest.json").read_text())
                lists[(backend, loops)] = manifest["partitions"][seq][f"{backend}_loops{loops}"]
    return lists


def frame_list_report():
    report = {}
    for seq in SEQS:
        lists = load_update_lists(seq)
        ref = lists[("vggt", 1)]
        report[seq] = {"n_submaps": len(ref), "frames_per_submap": [len(x) for x in ref],
                       "identical_to_vggt_loops1": {f"{b}_loops{l}": lists[(b, l)] == ref for (b, l) in lists}}
        # overlap frame structure: last frame of k is the first frame of k+1
        report[seq]["overlap_frame_is_shared"] = all(ref[k][-1] == ref[k + 1][0] for k in range(len(ref) - 1))
    return report


def a_jobs():
    jobs = []
    for seq in SEQS:
        ref = load_update_lists(seq)[("vggt", 1)]
        for k, names in enumerate(ref):
            jobs.append({"tag": f"A_{seq}_{k:02d}", "kind": "A", "seq": seq, "update": k, "names": names})
    return jobs


def gt_associated_rgb(seq):
    """rgb/ file names that have a ground-truth row within 0.01 s (sorted by time)."""
    from vos import why_utils as wu

    gt = wu.load_groundtruth(TUM / SEQS[seq] / "groundtruth.txt")
    names = sorted(p for p in os.listdir(TUM / SEQS[seq] / "rgb") if p.endswith(".png"))
    stamps = [wu.stamp_of(n) for n in names]
    _, ok, _ = wu.associate_frames(stamps, gt[0], 0.01)
    return [n for n, k in zip(names, ok) if k], len(names)


def b_jobs():
    from vos import why_utils as wu

    manifest = PROJECT / "results/frame_manifest.json"
    if manifest.is_file():
        return [j for j in json.loads(manifest.read_text())["jobs"] if j["kind"] == "B"]
    jobs = []
    for seq in SEQS:
        names, _ = gt_associated_rgb(seq)
        for n in SPARSE_N:
            for variant in SPARSE_VARIANTS:
                pos = wu.sparse_positions(len(names), n, variant)
                jobs.append({"tag": f"B_{seq}_N{n}_v{variant:+d}", "kind": "B", "seq": seq, "N": n, "variant": variant,
                             "names": [names[i] for i in pos], "positions": [int(i) for i in pos]})
    return jobs


def repeat_job():
    ref = load_update_lists("desk")[("vggt", 1)][0]
    return {"tag": "R_desk_00_rep", "kind": "R", "seq": "desk", "update": 0, "names": ref}


def all_jobs():
    return a_jobs() + [repeat_job()] + b_jobs()


def cache_path(backend, tag):
    return OUT / "cache" / backend / f"{tag}.npz"


# ----------------------------------------------------------------------------------------------
# GPU stage
# ----------------------------------------------------------------------------------------------
def forward_stage(args):
    import numpy as np

    OUT.mkdir(parents=True, exist_ok=True)
    jobs = all_jobs()
    log_path = OUT / "forward_log.json"
    log = json.loads(log_path.read_text()) if log_path.is_file() else {"entries": {}}
    todo = {b: [j for j in jobs if not cache_path(b, j["tag"]).is_file()] for b in args.backends}
    print({b: len(t) for b, t in todo.items()}, "forwards to run", flush=True)
    if not any(todo.values()):
        return 0

    env.prepare()
    need = 9000 if "omega" in args.backends else 7500
    free = env.claim_gpu(need, "vggt-omega-slam analysis_why forward")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    total = torch.cuda.get_device_properties(0).total_memory
    cap = min(MEMORY_CAP_GIB * 1024 ** 3, total * 0.95)
    torch.cuda.set_per_process_memory_fraction(cap / total)
    log["gpu"] = {"device": torch.cuda.get_device_name(0), "total_bytes": int(total), "cap_bytes": int(cap),
                  "free_mb_at_claim": free, "torch": torch.__version__, "lock": "vos.gpu.claim"}
    torch.set_num_threads(4)

    def save_log():
        tmp = log_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(log, indent=1))
        tmp.replace(log_path)

    for backend in args.backends:
        if not todo[backend]:
            continue
        t0 = time.perf_counter()
        if backend == "vggt":
            from vggt.utils.load_fn import load_and_preprocess_images as loader
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri as decode

            from vos import vggt_backend as vb

            model = vb.load_vggt(env.VGGT_CKPT, "cuda")
        else:
            from vos import omega_backend as ob

            model = ob.OmegaForSLAM(ob.load_omega(env.OMEGA_CKPT, "cuda"))
            loader, decode = ob.load_images_omega, ob.decode_pose
        torch.cuda.synchronize()
        log.setdefault("model_load_seconds", {})[backend] = time.perf_counter() - t0
        print(f"[{backend}] model loaded in {time.perf_counter() - t0:.1f}s, "
              f"allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
        for job in todo[backend]:
            tag = job["tag"]
            entry = {"backend": backend, "tag": tag, "n_frames": len(job["names"])}
            images = pred = extrinsic = intrinsic = None
            try:
                paths = [str(TUM / SEQS[job["seq"]] / "rgb" / n) for n in job["names"]]
                images = loader(paths).to("cuda")
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                with torch.no_grad():
                    pred = model(images)
                    extrinsic, intrinsic = decode(pred["pose_enc"], images.shape[-2:])
                torch.cuda.synchronize()
                seconds = time.perf_counter() - t1
                entry.update(seconds=seconds, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                             input_shape=list(images.shape), pose_enc_dtype=str(pred["pose_enc"].dtype),
                             depth_dtype=str(pred["depth"].dtype))

                def np_(x):  # exactly Solver.run_predictions: .float().cpu().numpy().squeeze(0)
                    return x.float().cpu().numpy().squeeze(0)

                payload = {"names": np.array(job["names"]), "pose_enc": np_(pred["pose_enc"]),
                           "extrinsic": np_(extrinsic), "intrinsic": np_(intrinsic),
                           "image_hw": np.array(images.shape[-2:]), "seconds": seconds}
                if job["kind"] in ("A", "R"):
                    depth, conf = np_(pred["depth"]), np_(pred["depth_conf"])  # [S,H,W,1], [S,H,W]
                    payload.update(depth_first=depth[0, ..., 0], conf_first=conf[0], depth_last=depth[-1, ..., 0],
                                   conf_last=conf[-1],
                                   conf_thr=np.float64(np.percentile(conf, 25) + 1e-6),  # Submap.add_all_points
                                   depth_median_all=np.float64(np.median(depth)))
                path = cache_path(backend, tag)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.stem + ".tmp.npz")
                np.savez_compressed(tmp, **payload)
                tmp.replace(path)
                entry["status"] = "ok"
                print(f"[{backend}] {tag}: {entry['n_frames']} frames {seconds:.2f}s "
                      f"peak {entry['peak_allocated_gib']:.2f} GiB", flush=True)
            except torch.cuda.OutOfMemoryError:
                entry.update(status="oom", error=traceback.format_exc(limit=3))
                print(f"[{backend}] {tag}: OOM (cap not raised)\n{entry['error']}", flush=True)
            except Exception:
                entry.update(status="failed", error=traceback.format_exc(limit=6))
                print(f"[{backend}] {tag}: FAILED\n{entry['error']}", flush=True)
            finally:
                images = pred = extrinsic = intrinsic = None
                torch.cuda.empty_cache()
            log["entries"][f"{backend}/{tag}"] = entry
            save_log()
        del model
        torch.cuda.empty_cache()
        print(f"[{backend}] done; allocated after release {torch.cuda.memory_allocated() / 2**20:.0f} MiB", flush=True)
    save_log()
    return 0


# ----------------------------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    fwd = sub.add_parser("forward", help="GPU: single forwards, cached")
    fwd.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS)
    ana = sub.add_parser("analyze", help="CPU: metrics, results.json, results_zh.md, figure")
    ana.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "forward":
        return forward_stage(args)
    return analyze_stage(args)


# ----------------------------------------------------------------------------------------------
# CPU stage
# ----------------------------------------------------------------------------------------------
DEFINITIONS = {
    "association": "each predicted frame -> nearest GT stamp within 0.01 s (one pair per frame)",
    "ate": "Sim(3) Umeyama (with scale) of predicted camera centres onto GT, RMSE in metres; "
           "ate_over_gt_path = ATE / (sum of consecutive GT distances inside the submap / frame set)",
    "rel_rot_err_deg": "angle between predicted and GT relative rotation R_i^T R_i+1 of consecutive frames, mean over pairs",
    "rel_dir_err_deg": "angle between predicted and GT unit translation direction R_i^T (C_i+1 - C_i) in camera i, mean over pairs",
    "fx": "fx of the decoded 3x3 intrinsics in pixels of the preprocessed image; frac = fx / image width "
          "(TUM fr1 calibration: 517.3/640 = 0.808); std is the population std over the frames of the submap",
    "e2e": "end-to-end relative pose first -> last associated frame of the set vs GT: rotation angle error (deg) and "
           "translation-direction error (deg, in the first camera frame); the quantity a one-frame-overlap chain hands "
           "from submap k to k+1 (a heading error theta displaces what follows by about theta * path length)",
    "full_submap": "submaps with 17 frames (16 new + overlap); shorter last submaps are listed but not in the summaries",
    "C(i)": "fx_rel_diff = |fx_k - fx_k+1| / mean of the two, fx of the overlap frame f in submap k (last) and k+1 (first)",
    "C(ii)": "depth_rel_diff_median = median over pixels (conf_k > p25_k and conf_k+1 > p25_k+1) of "
             "|D_k*c_k - D_k+1*c_k+1| / mean(D_k*c_k, D_k+1*c_k+1); c = Sim(3) scale of the per-submap fit (A); "
             "p25 = 25th percentile of the whole submap's depth_conf (+1e-6) as Submap.add_all_points",
    "C(iii)": "scale_ratio_literal = (median D_k(f) / median D_k+1(f)) / (T_k / T_k+1), same pixel set as (ii), "
              "T = mean distance between consecutive predicted camera centres of the submap (own units); GT free. "
              "For a perfectly consistent pair it equals G_k+1 / G_k (ratio of true mean step lengths), so "
              "scale_ratio_gt_step_corrected = literal * G_k / G_k+1 is also given (1 = consistent); "
              "abs_log = |ln ratio|",
    "C(iv)": "s_up = estimate_scale_pairwise exactly as Solver.add_edge (P_temp = inv(K4_k) K4_k+1, median of point-norm "
             "ratios on the good mask incl. its fallbacks); s_true = c_k+1 / c_k; s_up_err = s_up / s_true - 1; "
             "s_depth_err uses the ratio of median depths instead of s_up",
    "junction_aggregate": "only junctions whose two submaps both have a Sim(3) scale (>= 3 non-collinear frames)",
}
A_METRICS = ("ate_rmse_m", "ate_over_gt_path", "scale_m_per_unit", "rel_rot_err_deg_mean", "rel_dir_err_deg_mean",
             "abs_rot_err_deg_mean", "e2e_rot_err_deg", "e2e_dir_err_deg", "fx_px_std", "fx_frac_std", "fx_cv", "fx_frac_mean", "fy_frac_std",
             "gt_path_length_m", "fx_frac_over_tum_calibration")
C_METRICS = ("fx_rel_diff", "depth_rel_diff_median", "scale_ratio_literal_abs_log",
             "scale_ratio_gt_step_corrected_abs_log", "s_up_err_abs", "s_depth_err_abs")


def _clean(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def paired_stats(a_vals, b_vals):
    """a = VGGT, b = Omega, paired by submap/junction. Larger = worse for every metric used here."""
    import numpy as np
    from scipy import stats

    a, b = np.asarray(a_vals, float), np.asarray(b_vals, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    out = {"n_pairs": int(len(a))}
    if len(a) == 0:
        return out
    out.update(n_omega_larger=int((b > a).sum()), n_vggt_larger=int((a > b).sum()),
               median_vggt=float(np.median(a)), median_omega=float(np.median(b)),
               mean_vggt=float(a.mean()), mean_omega=float(b.mean()),
               median_ratio_omega_over_vggt=float(np.median(b / a)) if (a > 0).all() else None,
               ratio_of_means_omega_over_vggt=float(b.mean() / a.mean()) if a.mean() > 0 else None)
    if len(a) >= 5 and np.any(a != b):
        try:
            out["wilcoxon_two_sided_p"] = float(stats.wilcoxon(b, a).pvalue)
        except ValueError as exc:
            out["wilcoxon_error"] = str(exc)
    return out


def analyze_stage(args):
    import datetime

    import numpy as np

    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU stage: never touch the GPU
    env.prepare()
    from vggt.utils.geometry import depth_to_cam_coords_points
    from vggt_slam.scale_solver import estimate_scale_pairwise

    from vos import compare_utils as cu
    from vos import why_utils as wu

    OUT.mkdir(parents=True, exist_ok=True)
    gts = {seq: wu.load_groundtruth(TUM / SEQS[seq] / "groundtruth.txt") for seq in SEQS}
    results = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
               "definitions": DEFINITIONS, "frame_lists": frame_list_report()}
    missing = []

    def get(backend, tag):
        path = cache_path(backend, tag)
        if not path.is_file():
            missing.append(f"{backend}/{tag}")
            return None
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}

    # sanity: cam_points mirrors upstream depth_to_cam_coords_points
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 3.0, (24, 32)).astype(np.float32)
    k_test = np.array([[40.0, 0, 16], [0, 41.0, 12], [0, 0, 1]])
    assert np.array_equal(wu.cam_points(depth, k_test), depth_to_cam_coords_points(depth, k_test))

    # ---- forward log -------------------------------------------------------------------------
    log = json.loads((OUT / "forward_log.json").read_text()) if (OUT / "forward_log.json").is_file() else {"entries": {}}
    fl = {}
    for key, e in log["entries"].items():
        b = e["backend"]
        d = fl.setdefault(b, {"ok": 0, "oom": 0, "failed": 0, "full17_seconds": [], "peak_gib_by_frames": {}})
        d[e["status"]] += 1
        if e["status"] == "ok":
            if e["n_frames"] == 17:
                d["full17_seconds"].append(e["seconds"])
            d["peak_gib_by_frames"][str(e["n_frames"])] = max(d["peak_gib_by_frames"].get(str(e["n_frames"]), 0),
                                                              e["peak_allocated_gib"])
    results["forward_log"] = {"gpu": log.get("gpu"), "model_load_seconds": log.get("model_load_seconds"),
                              "per_backend": fl, "errors": {k: e.get("error") for k, e in log["entries"].items()
                                                            if e["status"] != "ok"}}

    # ---- A: per-submap accuracy ----------------------------------------------------------------
    A = {seq: {b: [] for b in BACKENDS} for seq in SEQS}
    ov = {}  # (backend, seq, update) -> overlap-frame arrays
    for job in a_jobs():
        seq = job["seq"]
        stamps = [wu.stamp_of(n) for n in job["names"]]
        for b in BACKENDS:
            z = get(b, job["tag"])
            if z is None:
                continue
            m = wu.trajectory_metrics(z["extrinsic"], z["intrinsic"], tuple(int(x) for x in z["image_hw"]), stamps,
                                      gts[seq])
            m.update(update=job["update"], full=len(job["names"]) == FULL_SUBMAP, first_frame=job["names"][0],
                     last_frame=job["names"][-1], forward_seconds=float(z["seconds"]))
            A[seq][b].append(m)
            ov[(b, seq, job["update"])] = {
                "first": {"depth": z["depth_first"], "conf": z["conf_first"], "intrinsic": z["intrinsic"][0],
                          "conf_thr": float(z["conf_thr"])},
                "last": {"depth": z["depth_last"], "conf": z["conf_last"], "intrinsic": z["intrinsic"][-1],
                         "conf_thr": float(z["conf_thr"])},
                "extrinsic_first": z["extrinsic"][0]}
    results["A"] = {"per_submap": A}

    def full(seq, b):
        return [m for m in A[seq][b] if m["full"] and m["sim3_status"] == "ok"]

    a_summary = {}
    for b in BACKENDS:
        a_summary[b] = {}
        for seq in list(SEQS) + ["pooled"]:
            items = [m for s in SEQS for m in full(s, b)] if seq == "pooled" else full(seq, b)
            a_summary[b][seq] = {"n_full_submaps": len(items),
                                 **{k: wu.summarize([m[k] for m in items if m.get(k) is not None]) for k in A_METRICS}}
    results["A"]["summary_full_submaps"] = a_summary
    paired = {}
    for seq in list(SEQS) + ["pooled"]:
        seqs = list(SEQS) if seq == "pooled" else [seq]
        rows = []
        for s in seqs:
            v = {m["update"]: m for m in full(s, "vggt")}
            o = {m["update"]: m for m in full(s, "omega")}
            rows += [(v[k], o[k]) for k in sorted(set(v) & set(o))]
        paired[seq] = {k: paired_stats([r[0][k] for r in rows], [r[1][k] for r in rows])
                       for k in ("ate_rmse_m", "ate_over_gt_path", "rel_rot_err_deg_mean", "rel_dir_err_deg_mean",
                                 "abs_rot_err_deg_mean", "e2e_rot_err_deg", "e2e_dir_err_deg", "fx_frac_std", "fx_cv",
                                 "fx_frac_over_tum_calibration")}
    results["A"]["paired_omega_vs_vggt_full_submaps"] = paired

    # ---- existing SLAM results (context) -------------------------------------------------------
    slam = {}
    for seq in SEQS:
        slam[seq] = {}
        for b in BACKENDS:
            for loops in (1, 0):
                f = COMPARE / f"{COMPARE_NAME[seq]}_{b}_loops{loops}" / "ate.json"
                if f.is_file():
                    r = json.loads(f.read_text())
                    slam[seq][f"{b}_loops{loops}"] = {"ate_rmse_m": r.get("ate_rmse_m"), "scale": r.get("scale")}
    results["existing_slam_ate"] = slam
    # descriptive: how much of the sequence-level ratio the per-submap (local) ratio can account for
    ctx = {}
    for seq in SEQS:
        loc = {b: float(np.sqrt(np.mean([m["ate_rmse_m"] ** 2 for m in full(seq, b)]))) for b in BACKENDS}
        entry = {"n_full_submaps": len(full(seq, "vggt")), "local_rms_ate_m": loc,
                 "local_ratio_omega_over_vggt": loc["omega"] / loc["vggt"]}
        for loops in (1, 0):
            v, o = slam[seq][f"vggt_loops{loops}"]["ate_rmse_m"], slam[seq][f"omega_loops{loops}"]["ate_rmse_m"]
            ratio = o / v
            entry[f"slam_loops{loops}"] = {"vggt_m": v, "omega_m": o, "ratio_omega_over_vggt": ratio,
                                          "amplification_vggt": v / loc["vggt"], "amplification_omega": o / loc["omega"],
                                          "log_share_of_gap_from_local": (float(np.log(entry["local_ratio_omega_over_vggt"]) / np.log(ratio))
                                                                          if abs(np.log(ratio)) > 0.05 else None)}
        ctx[seq] = entry
    results["context_local_vs_slam"] = ctx

    # ---- reproduction + nondeterminism checks --------------------------------------------------
    repro = {}
    for seq in SEQS:
        for b in BACKENDS:
            z = get(b, f"A_{seq}_00")
            if z is None:
                continue
            stamps, xyz, _ = cu.read_poses(COMPARE / f"{COMPARE_NAME[seq]}_{b}_loops0" / "poses.tum")
            names = [str(n) for n in z["names"]]
            n = len(names)
            mine, _ = wu.centers_and_rotations(z["extrinsic"])
            theirs = xyz[:n]
            stamps_match = bool(np.allclose(stamps[:n], [wu.stamp_of(x) for x in names], atol=1e-5))
            diff = np.linalg.norm(mine - theirs, axis=1)
            rot, tr, c = cu.umeyama(mine, theirs)
            res = np.linalg.norm(cu.apply_sim3(mine, rot, tr, c) - theirs, axis=1)
            plen = wu.path_length(theirs)
            repro[f"{seq}_{b}"] = {"stamps_match": stamps_match, "n_frames": n, "slam_path_length_units": plen,
                                   "rmse_unaligned_units": float(np.sqrt((diff ** 2).mean())),
                                   "rmse_unaligned_over_path": float(np.sqrt((diff ** 2).mean()) / plen),
                                   "rmse_sim3_aligned_units": float(np.sqrt((res ** 2).mean())), "sim3_scale": float(c)}
    results["reproduction_vs_slam_loops0_first_submap"] = repro
    nd = {}
    for b in BACKENDS:
        z0, z1 = get(b, "A_desk_00"), get(b, "R_desk_00_rep")
        if z0 is None or z1 is None:
            continue
        c0, _ = wu.centers_and_rotations(z0["extrinsic"])
        c1, _ = wu.centers_and_rotations(z1["extrinsic"])
        stamps = [wu.stamp_of(x) for x in z0["names"]]
        m0 = wu.trajectory_metrics(z0["extrinsic"], z0["intrinsic"], tuple(int(x) for x in z0["image_hw"]), stamps, gts["desk"])
        m1 = wu.trajectory_metrics(z1["extrinsic"], z1["intrinsic"], tuple(int(x) for x in z1["image_hw"]), stamps, gts["desk"])
        nd[b] = {"bitwise_identical_pose_enc": bool(np.array_equal(z0["pose_enc"], z1["pose_enc"])),
                 "bitwise_identical_depth_first": bool(np.array_equal(z0["depth_first"], z1["depth_first"])),
                 "max_abs_center_diff_units": float(np.abs(c0 - c1).max()),
                 "max_abs_pose_enc_diff": float(np.abs(z0["pose_enc"] - z1["pose_enc"]).max()),
                 "max_rel_depth_diff_first": float((np.abs(z0["depth_first"] - z1["depth_first"]) / z0["depth_first"]).max()),
                 "ate_run1_m": m0["ate_rmse_m"], "ate_run2_m": m1["ate_rmse_m"],
                 "fx_std_run1_px": m0["fx_px_std"], "fx_std_run2_px": m1["fx_px_std"]}
    results["nondeterminism_desk_submap0"] = nd

    # ---- B: sparse wide-baseline regime --------------------------------------------------------
    B = {seq: {b: {} for b in BACKENDS} for seq in SEQS}
    for job in b_jobs():
        stamps = [wu.stamp_of(n) for n in job["names"]]
        for b in BACKENDS:
            z = get(b, job["tag"])
            if z is None:
                B[job["seq"]][b].setdefault(str(job["N"]), {})[str(job["variant"])] = {"status": "missing"}
                continue
            m = wu.trajectory_metrics(z["extrinsic"], z["intrinsic"], tuple(int(x) for x in z["image_hw"]), stamps,
                                      gts[job["seq"]])
            m["positions"] = job["positions"]
            m["mean_spacing_frames"] = float(np.diff(job["positions"]).mean())
            B[job["seq"]][b].setdefault(str(job["N"]), {})[str(job["variant"])] = m
    b_summary = {}
    for seq in SEQS:
        b_summary[seq] = {}
        for n in SPARSE_N:
            row = {}
            for b in BACKENDS:
                runs = [B[seq][b][str(n)].get(str(v)) for v in SPARSE_VARIANTS]
                runs = [r for r in runs if r and r.get("sim3_status") == "ok"]
                primary = B[seq][b][str(n)].get("0")
                row[b] = {"primary": {k: primary.get(k) for k in ("ate_rmse_m", "ate_over_gt_path", "rel_rot_err_deg_mean",
                                                                  "rel_dir_err_deg_mean", "abs_rot_err_deg_mean",
                                                                  "scale_m_per_unit", "fx_frac_mean", "fx_frac_std")}
                          if primary and primary.get("sim3_status") == "ok" else None,
                          "over_variants": {k: wu.summarize([r[k] for r in runs if r.get(k) is not None])
                                            for k in ("ate_rmse_m", "ate_over_gt_path", "rel_rot_err_deg_mean",
                                                      "rel_dir_err_deg_mean", "abs_rot_err_deg_mean")}}
            b_summary[seq][str(n)] = row
    # pooled over sequences and variants: paired by (seq, N, variant)
    pairs = {k: ([], []) for k in ("ate_rmse_m", "ate_over_gt_path", "rel_rot_err_deg_mean", "rel_dir_err_deg_mean")}
    pairs_by_n = {str(n): {k: ([], []) for k in pairs} for n in SPARSE_N}
    for seq in SEQS:
        for n in SPARSE_N:
            for v in SPARSE_VARIANTS:
                rv, ro = B[seq]["vggt"][str(n)].get(str(v)), B[seq]["omega"][str(n)].get(str(v))
                if rv and ro and rv.get("sim3_status") == "ok" and ro.get("sim3_status") == "ok":
                    for k in pairs:
                        pairs[k][0].append(rv[k]); pairs[k][1].append(ro[k])
                        pairs_by_n[str(n)][k][0].append(rv[k]); pairs_by_n[str(n)][k][1].append(ro[k])
    results["B"] = {"per_set": B, "summary": b_summary,
                    "paired_pooled": {k: paired_stats(*v) for k, v in pairs.items()},
                    "paired_pooled_by_N": {n: {k: paired_stats(*v) for k, v in d.items()} for n, d in pairs_by_n.items()}}

    # ---- C: overlap-frame coupling -------------------------------------------------------------
    C = {seq: {b: [] for b in BACKENDS} for seq in SEQS}
    for seq in SEQS:
        for b in BACKENDS:
            subs = A[seq][b]
            for k in range(len(subs) - 1):
                p, c = subs[k], subs[k + 1]
                assert p["last_frame"] == c["first_frame"], (seq, b, k)  # the shared overlap frame
                op, oc = ov.get((b, seq, k)), ov.get((b, seq, k + 1))
                if op is None or oc is None:
                    continue
                jm = wu.junction_metrics(op["last"], oc["first"], p["scale_m_per_unit"], c["scale_m_per_unit"],
                                         estimate_scale_pairwise, p["pred_step_mean_units"], c["pred_step_mean_units"],
                                         p["gt_step_mean_m"], c["gt_step_mean_m"])
                sim3_both = p["scale_method"] == "sim3" and c["scale_method"] == "sim3"
                jm.update(k=k, overlap_frame=p["last_frame"], n_frames_k=p["n_frames"], n_frames_k1=c["n_frames"],
                          scale_method_k=p["scale_method"], scale_method_k1=c["scale_method"], in_aggregate=sim3_both,
                          extrinsic_first_is_identity=bool(np.allclose(oc["extrinsic_first"][:, :3], np.eye(3), atol=1e-3)
                                                           and np.allclose(oc["extrinsic_first"][:, 3], 0, atol=1e-3)))
                if "s_up_err" in jm:
                    jm["s_up_err_abs"] = abs(jm["s_up_err"])
                if "s_depth_err" in jm:
                    jm["s_depth_err_abs"] = abs(jm["s_depth_err"])
                C[seq][b].append(jm)
    c_summary = {b: {} for b in BACKENDS}
    for b in BACKENDS:
        for seq in list(SEQS) + ["pooled"]:
            items = [j for s in SEQS for j in C[s][b] if j["in_aggregate"]] if seq == "pooled" else \
                [j for j in C[seq][b] if j["in_aggregate"]]
            c_summary[b][seq] = {"n_junctions": len(items), **{k: wu.summarize([j[k] for j in items if k in j])
                                                                for k in C_METRICS}}
            errs = [j["s_up_err"] for j in items if "s_up_err" in j]
            if errs and seq != "pooled":
                cum = np.cumsum(np.log1p(errs))
                c_summary[b][seq]["s_up_cumulative_log_drift_final"] = float(cum[-1])
                c_summary[b][seq]["s_up_cumulative_log_drift_max_abs"] = float(np.abs(cum).max())
                c_summary[b][seq]["s_up_err_signed_mean"] = float(np.mean(errs))
    c_paired = {}
    for seq in list(SEQS) + ["pooled"]:
        seqs = list(SEQS) if seq == "pooled" else [seq]
        rows = []
        for s in seqs:
            v = {j["k"]: j for j in C[s]["vggt"] if j["in_aggregate"]}
            o = {j["k"]: j for j in C[s]["omega"] if j["in_aggregate"]}
            rows += [(v[k], o[k]) for k in sorted(set(v) & set(o))]
        c_paired[seq] = {k: paired_stats([r[0].get(k, np.nan) for r in rows], [r[1].get(k, np.nan) for r in rows])
                         for k in C_METRICS}
    results["C"] = {"per_junction": C, "summary": c_summary, "paired_omega_vs_vggt": c_paired}

    results["missing_cache_entries"] = sorted(set(missing))
    (OUT / "results.json").write_text(json.dumps(_clean(results), indent=1, ensure_ascii=False))
    print(f"wrote {OUT / 'results.json'}; missing cache entries: {len(set(missing))}")
    if not args.no_figure:
        write_figure(results, OUT / "原因分解.png")
    write_markdown(results, OUT / "results_zh.md")
    return 0




COLORS = {"vggt": "#2a78d6", "omega": "#eb6834"}  # categorical slots 1 and 2 of the validated reference palette
LABELS = {"vggt": "VGGT", "omega": "Omega"}
SEQ_ZH = {"desk": "desk", "desk2": "desk2", "room": "room", "pooled": "三序列合计"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"


def _fmt(x, nd=3, scale=1.0):
    return "–" if x is None else f"{x * scale:.{nd}f}"


def write_markdown(results, path):
    A, B, C = results["A"], results["B"], results["C"]
    L = ["# 原因分解:VGGT-Omega 在 VGGT-SLAM 2.0 里 ATE 更高的来源(单次前向,未运行 SLAM)", "",
         f"生成时间 {results['generated']}。所有数字见 `results.json`;口径见其 `definitions`。", ""]

    L += ["## 现有 SLAM 结果(子图 16,回环开/关,ATE RMSE 米)", "",
          "| 序列 | VGGT 开 | VGGT 关 | Omega 开 | Omega 关 |", "|---|---|---|---|---|"]
    for seq, d in results["existing_slam_ate"].items():
        L.append(f"| {seq} | " + " | ".join(_fmt((d.get(k) or {}).get("ate_rmse_m"), 4) for k in
                                          ("vggt_loops1", "vggt_loops0", "omega_loops1", "omega_loops0")) + " |")

    L += ["", "## 局部精度与整条 SLAM 的关系(描述性,SLAM 为单次运行)", "",
          "局部 = 17 帧子图单次前向 ATE 的均方根;放大倍数 = 整条 SLAM 的 ATE ÷ 局部均方根(链式拼接、回环、尺度/旋转累积带来的放大);"
          "对数占比 = ln(局部比) ÷ ln(SLAM 比),>1 表示局部差距比整条差距还大,<0 表示方向相反,SLAM 比≈1 时不定义。", "",
          "| 序列 | 局部 VGGT (cm) | 局部 Omega (cm) | 局部比 | 回环开 SLAM 比 | 放大 VGGT/Omega | 对数占比 | "
          "回环关 SLAM 比 | 放大 VGGT/Omega | 对数占比 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for seq, e in results["context_local_vs_slam"].items():
        a, b_ = e["slam_loops1"], e["slam_loops0"]
        L.append(f"| {seq} | {_fmt(e['local_rms_ate_m']['vggt'], 2, 100)} | {_fmt(e['local_rms_ate_m']['omega'], 2, 100)} | "
                 f"{_fmt(e['local_ratio_omega_over_vggt'], 2)} | {_fmt(a['ratio_omega_over_vggt'], 2)} | "
                 f"{_fmt(a['amplification_vggt'], 1)} / {_fmt(a['amplification_omega'], 1)} | {_fmt(a['log_share_of_gap_from_local'], 2)} | "
                 f"{_fmt(b_['ratio_omega_over_vggt'], 2)} | {_fmt(b_['amplification_vggt'], 1)} / {_fmt(b_['amplification_omega'], 1)} | "
                 f"{_fmt(b_['log_share_of_gap_from_local'], 2)} |")
    L += ["", "## A. 每个子图在 SLAM 工况下的单次前向精度(只统计 17 帧的完整子图)", "",
          "| 序列 | 后端 | 子图数 | ATE 中位 (cm) | ATE/路径长 中位 (%) | 相对旋转误差 (°) | 相对平移方向误差 (°) | "
          "fx 子图内标准差 (px) | fx 标准差/图宽 (%) | fx/图宽 均值 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for seq in list(SEQS) + ["pooled"]:
        for b in BACKENDS:
            d = A["summary_full_submaps"][b][seq]
            L.append(f"| {SEQ_ZH[seq]} | {LABELS[b]} | {d['n_full_submaps']} | {_fmt(d['ate_rmse_m']['median'], 2, 100)} | "
                     f"{_fmt(d['ate_over_gt_path']['median'], 2, 100)} | {_fmt(d['rel_rot_err_deg_mean']['median'], 2)} | "
                     f"{_fmt(d['rel_dir_err_deg_mean']['median'], 1)} | {_fmt(d['fx_px_std']['median'], 1)} | "
                     f"{_fmt(d['fx_frac_std']['median'], 2, 100)} | {_fmt(d['fx_frac_mean']['median'], 3)} |")
    L += ["", "同一子图两后端配对(Omega 更差 = Omega 的值更大):", "",
          "| 范围 | 指标 | 配对数 | Omega 更差的个数 | 中位 VGGT | 中位 Omega | 中位比 Omega/VGGT | Wilcoxon p |", "|---|---|---|---|---|---|---|---|"]
    names = {"ate_rmse_m": "ATE (m)", "ate_over_gt_path": "ATE/路径长", "rel_rot_err_deg_mean": "相对旋转误差 (°)",
             "rel_dir_err_deg_mean": "相对平移方向误差 (°)", "abs_rot_err_deg_mean": "对齐后绝对旋转误差 (°)",
             "e2e_rot_err_deg": "首→末帧相对旋转误差 (°)", "e2e_dir_err_deg": "首→末帧平移方向误差 (°)",
             "fx_frac_over_tum_calibration": "fx/图宽 ÷ TUM 标定值",
             "fx_frac_std": "fx 标准差/图宽", "fx_cv": "fx 变异系数"}
    for seq in list(SEQS) + ["pooled"]:
        for k, nm in names.items():
            d = A["paired_omega_vs_vggt_full_submaps"][seq][k]
            if not d.get("n_pairs"):
                continue
            L.append(f"| {SEQ_ZH[seq]} | {nm} | {d['n_pairs']} | {d['n_omega_larger']} | {_fmt(d['median_vggt'], 4)} | "
                     f"{_fmt(d['median_omega'], 4)} | {_fmt(d.get('median_ratio_omega_over_vggt'), 2)} | "
                     f"{_fmt(d.get('wilcoxon_two_sided_p'), 3)} |")

    L += ["", "## B. 稀疏宽基线(N 帧在整条序列上均匀取,两后端同一批帧)", "",
          "主结果为标准网格;括号内为三种网格(标准、整体平移 ±1/3 间隔)的均值。ATE 单位米,Sim(3) 对齐含尺度。", "",
          "| 序列 | N | 帧间隔(帧) | VGGT ATE | Omega ATE | VGGT 相对旋转误差° | Omega 相对旋转误差° | "
          "VGGT 平移方向误差° | Omega 平移方向误差° |", "|---|---|---|---|---|---|---|---|---|"]
    for seq in SEQS:
        for n in SPARSE_N:
            row = B["summary"][seq][str(n)]
            prim = {b: row[b]["primary"] for b in BACKENDS}
            ov_ = {b: row[b]["over_variants"] for b in BACKENDS}
            gap = B["per_set"][seq]["vggt"][str(n)]["0"].get("mean_spacing_frames")

            def cell(b, k, nd, sc=1.0):
                p = prim[b][k] if prim[b] else None
                m = ov_[b][k]["mean"]
                return f"{_fmt(p, nd, sc)} ({_fmt(m, nd, sc)})"

            L.append(f"| {seq} | {n} | {_fmt(gap, 0)} | {cell('vggt', 'ate_rmse_m', 3)} | {cell('omega', 'ate_rmse_m', 3)} | "
                     f"{cell('vggt', 'rel_rot_err_deg_mean', 2)} | {cell('omega', 'rel_rot_err_deg_mean', 2)} | "
                     f"{cell('vggt', 'rel_dir_err_deg_mean', 1)} | {cell('omega', 'rel_dir_err_deg_mean', 1)} |")
    L += ["", "配对汇总(跨序列与网格):", "", "| N | 指标 | 配对数 | Omega 更差的个数 | 中位比 Omega/VGGT | 比的均值 | Wilcoxon p |", "|---|---|---|---|---|---|---|"]
    for n in SPARSE_N:
        for k, nm in (("ate_rmse_m", "ATE (m)"), ("ate_over_gt_path", "ATE/路径长"), ("rel_rot_err_deg_mean", "相对旋转误差 (°)")):
            d = B["paired_pooled_by_N"][str(n)][k]
            if d.get("n_pairs"):
                L.append(f"| {n} | {nm} | {d['n_pairs']} | {d['n_omega_larger']} | {_fmt(d.get('median_ratio_omega_over_vggt'), 2)} | "
                         f"{_fmt(d.get('ratio_of_means_omega_over_vggt'), 2)} | {_fmt(d.get('wilcoxon_two_sided_p'), 3)} |")

    L += ["", "## C. 重叠帧耦合(相邻子图 k,k+1 共用的那一帧 f)", "",
          "只含两侧子图都有 Sim(3) 尺度的交界。数值为中位数。(i) fx 相对差;(ii) 度量深度不一致(深度乘各自 Sim(3) 尺度,逐像素);"
          "(iii) 无真值的尺度比 |ln|(见 definitions,含真实步长比的混杂);(iv) 上游 `estimate_scale_pairwise` 给出的尺度相对 "
          "GT 隐含尺度的误差。", "",
          "| 序列 | 后端 | 交界数 | (i) fx 相对差 % | (ii) 深度不一致 % | (iii) |ln 比| | (iii) 按真实步长修正 | (iv) |尺度误差| % | 深度中位比的尺度误差 % |",
          "|---|---|---|---|---|---|---|---|---|"]
    for seq in list(SEQS) + ["pooled"]:
        for b in BACKENDS:
            d = C["summary"][b][seq]
            L.append(f"| {SEQ_ZH[seq]} | {LABELS[b]} | {d['n_junctions']} | {_fmt(d['fx_rel_diff']['median'], 2, 100)} | "
                     f"{_fmt(d['depth_rel_diff_median']['median'], 2, 100)} | {_fmt(d['scale_ratio_literal_abs_log']['median'], 3)} | "
                     f"{_fmt(d['scale_ratio_gt_step_corrected_abs_log']['median'], 3)} | {_fmt(d['s_up_err_abs']['median'], 2, 100)} | "
                     f"{_fmt(d['s_depth_err_abs']['median'], 2, 100)} |")
    L += ["", "同一交界两后端配对(合计):", "", "| 指标 | 配对数 | Omega 更差的个数 | 中位 VGGT | 中位 Omega | Wilcoxon p |", "|---|---|---|---|---|---|"]
    cn = {"fx_rel_diff": "(i) fx 相对差", "depth_rel_diff_median": "(ii) 深度不一致", "scale_ratio_literal_abs_log": "(iii) |ln 比|",
          "scale_ratio_gt_step_corrected_abs_log": "(iii') 修正后 |ln 比|", "s_up_err_abs": "(iv) |上游尺度误差|",
          "s_depth_err_abs": "|深度中位比尺度误差|"}
    for k, nm in cn.items():
        d = C["paired_omega_vs_vggt"]["pooled"][k]
        if d.get("n_pairs"):
            L.append(f"| {nm} | {d['n_pairs']} | {d['n_omega_larger']} | {_fmt(d['median_vggt'], 4)} | {_fmt(d['median_omega'], 4)} | "
                     f"{_fmt(d.get('wilcoxon_two_sided_p'), 3)} |")
    L += ["", "链上尺度累计漂移(上游尺度误差的对数累加,每序列):", "", "| 序列 | 后端 | 末端累计 ln | 最大 |累计| | 平均有符号误差 % |", "|---|---|---|---|---|"]
    for seq in SEQS:
        for b in BACKENDS:
            d = C["summary"][b][seq]
            L.append(f"| {seq} | {LABELS[b]} | {_fmt(d.get('s_up_cumulative_log_drift_final'), 3)} | "
                     f"{_fmt(d.get('s_up_cumulative_log_drift_max_abs'), 3)} | {_fmt(d.get('s_up_err_signed_mean'), 2, 100)} |")

    L += ["", "## 自检", ""]
    rep = results["reproduction_vs_slam_loops0_first_submap"]
    if rep:
        L.append("单次前向复现 SLAM 第一个子图(回环关的 poses.tum 前 17 帧):Sim(3) 对齐后相机中心 RMSE 最大 "
                 f"{max(v['rmse_sim3_aligned_units'] for v in rep.values()):.1e}(模型单位),即同一前向。")
    nd = results["nondeterminism_desk_submap0"]
    for b, v in nd.items():
        L.append(f"- {LABELS[b]} 同一子图重复前向:位姿编码逐位相同={v['bitwise_identical_pose_enc']},深度逐位相同="
                 f"{v['bitwise_identical_depth_first']},相机中心最大差 {v['max_abs_center_diff_units']:.2e}。")
    fl = results["forward_log"]["per_backend"]
    for b, v in fl.items():
        L.append(f"- {LABELS[b]} 前向 {v['ok']} 次成功,{v['oom']} 次 OOM,{v['failed']} 次失败;17 帧前向中位 "
                 f"{(sorted(v['full17_seconds'])[len(v['full17_seconds']) // 2] if v['full17_seconds'] else float('nan')):.2f} s;"
                 f"峰值显存 GiB(按帧数):{v['peak_gib_by_frames']}。")
    if results["missing_cache_entries"]:
        L.append(f"- 缺失的前向缓存:{results['missing_cache_entries']}")
    verdict = OUT / "verdict_zh.md"
    if verdict.is_file():
        L += ["", verdict.read_text().rstrip(), ""]
    path.write_text("\n".join(L) + "\n")


def _style_axes(ax, ygrid=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=10, length=3)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)


def _paired_dots(ax, xs, vals_v, vals_o, log=True, width=0.16):
    """Dots of both backends for the same item (submap / junction), joined by a thin grey line."""
    import numpy as np

    for x, a, b in zip(xs, vals_v, vals_o):
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            continue
        ax.plot([x - width, x + width], [a, b], color="#b9b8b2", lw=0.8, zorder=1)
    for b_, off, vals in (("vggt", -width, vals_v), ("omega", width, vals_o)):
        v = np.array([np.nan if y is None else y for y in vals], float)
        ax.scatter(np.asarray(xs, float) + off, v, s=26, color=COLORS[b_], edgecolor=SURFACE, linewidth=0.8, zorder=3)


def _median_marks(ax, x, vals, color, half=0.11):
    import numpy as np

    v = np.array([y for y in vals if y is not None and np.isfinite(y)], float)
    if len(v):
        ax.hlines(np.median(v), x - half, x + half, color=color, lw=3, zorder=4)
    return float(np.median(v)) if len(v) else None


def write_figure(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import font_manager as fm
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullFormatter

    from vos.why_utils import TUM_FR1_FX_FRACTION

    have = {f.name for f in fm.fontManager.ttflist}
    family = next((n for n in ("Noto Sans CJK JP", "Droid Sans Fallback") if n in have), "DejaVu Sans")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"],
                         "axes.unicode_minus": False, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    A, B, C = results["A"], results["B"], results["C"]
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    fig = plt.figure(figsize=(20, 12), dpi=100)
    gs = GridSpec(2, 2, figure=fig, left=0.055, right=0.985, top=0.80, bottom=0.07, hspace=0.50, wspace=0.14)
    seqs = list(SEQS)
    row1_head, row2_head = 0.865, 0.425

    fig.text(0.055, 0.962, "VGGT 与 VGGT-Omega 在 VGGT-SLAM 2.0 上的 ATE 差距:前端精度 vs 后端耦合(只做单次前向,未运行 SLAM)",
             fontsize=18, color=INK, fontweight="bold", va="center")
    s_ate = results["existing_slam_ate"]
    txt = "已有整条 SLAM 的 ATE(米,子图 16,回环开):" + ";".join(
        f"{s} VGGT {s_ate[s]['vggt_loops1']['ate_rmse_m']:.3f} / Omega {s_ate[s]['omega_loops1']['ate_rmse_m']:.3f}" for s in seqs)
    fig.text(0.055, 0.925, txt, fontsize=12.5, color=INK2, va="center")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[b], markeredgecolor=SURFACE, markersize=10,
                      label=LABELS[b]) for b in BACKENDS]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.985, 0.962), ncol=2, frameon=False, fontsize=13)

    # (a) per-submap normalised ATE ---------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    _style_axes(ax)
    for gi, seq in enumerate(seqs):
        v = {m["update"]: m for m in A["per_submap"][seq]["vggt"] if m["full"] and m["sim3_status"] == "ok"}
        o = {m["update"]: m for m in A["per_submap"][seq]["omega"] if m["full"] and m["sim3_status"] == "ok"}
        ks = sorted(set(v) & set(o))
        va = [v[k]["ate_over_gt_path"] * 100 for k in ks]
        oa = [o[k]["ate_over_gt_path"] * 100 for k in ks]
        _paired_dots(ax, np.full(len(ks), float(gi)), va, oa, width=0.17)
        mv, mo = _median_marks(ax, gi - 0.17, va, COLORS["vggt"]), _median_marks(ax, gi + 0.17, oa, COLORS["omega"])
        worse = sum(1 for a, b in zip(va, oa) if b > a)
        ax.text(gi, 1.02, f"Omega 更差 {worse}/{len(ks)}\n中位 {mv:.2f}% → {mo:.2f}%", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=10.5, color=INK)
    ax.set_yscale("log")
    ax.set_yticks([0.6, 1, 2, 3, 4], ["0.6%", "1%", "2%", "3%", "4%"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(range(len(seqs)), seqs, fontsize=12)
    ax.set_xlim(-0.55, len(seqs) - 0.45)
    ax.set_ylabel("ATE ÷ 子图真值路径长(对数轴)", fontsize=11, color=INK2)
    fig.text(0.055, row1_head, "(a) 每个 17 帧子图单次前向的归一化 ATE(每点一个子图,线连同一子图,粗横线为中位)", fontsize=13,
             color=INK, ha="left", va="bottom")

    # (b) sparse regime ---------------------------------------------------------------------------
    sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 1], wspace=0.30)
    for si, seq in enumerate(seqs):
        ax = fig.add_subplot(sub[0, si])
        _style_axes(ax)
        xs = np.arange(len(SPARSE_N))
        for b in BACKENDS:
            prim = [B["summary"][seq][str(n)][b]["primary"] for n in SPARSE_N]
            ys = [p["ate_rmse_m"] if p else np.nan for p in prim]
            for var in SPARSE_VARIANTS[1:]:
                yv = [(B["per_set"][seq][b][str(n)].get(str(var)) or {}).get("ate_rmse_m") for n in SPARSE_N]
                ax.scatter(xs + (-0.06 if b == "vggt" else 0.06), [np.nan if y is None else y for y in yv], s=12,
                           color=COLORS[b], alpha=0.45, edgecolor="none", zorder=2)
            ax.plot(xs, ys, color=COLORS[b], lw=2, marker="o", ms=7, mec=SURFACE, mew=1.2, zorder=3)
            ax.text(xs[-1] + 0.12, ys[-1], LABELS[b], color=INK, fontsize=10, va="center")
        ax.set_xticks(xs, [str(n) for n in SPARSE_N])
        ax.set_xlim(-0.3, len(SPARSE_N) + 0.45)
        ax.set_xlabel("帧数 N", fontsize=10.5, color=INK2)
        ax.set_title(seq, fontsize=12, color=INK, loc="left")
        ax.set_ylim(bottom=0)
        if si == 0:
            ax.set_ylabel("ATE (m, Sim(3) 对齐)", fontsize=11, color=INK2)
    fig.text(0.53, row1_head, "(b) 稀疏宽基线:N 帧在整条序列上均匀取(线=标准网格,浅点=网格整体平移 ±1/3 间隔)", fontsize=13,
             color=INK, ha="left", va="bottom")

    # (c) overlap-frame inconsistency -------------------------------------------------------------
    panels = [("fx_rel_diff", "(i) 焦距相对差", 100, "%"), ("depth_rel_diff_median", "(ii) 度量深度不一致", 100, "%"),
              ("scale_ratio_literal_abs_log", "(iii) 尺度比 |ln|", 1, ""),
              ("scale_ratio_gt_step_corrected_abs_log", "(iii') 按真实步长修正", 1, ""),
              ("s_up_err_abs", "(iv) 上游尺度误差", 100, "%")]
    sub = GridSpecFromSubplotSpec(1, len(panels), subplot_spec=gs[1, 0], wspace=0.62)
    for pi, (key, title, sc, unit) in enumerate(panels):
        ax = fig.add_subplot(sub[0, pi])
        _style_axes(ax)
        vv, oo = [], []
        for seq in seqs:
            v = {j["k"]: j for j in C["per_junction"][seq]["vggt"] if j["in_aggregate"]}
            o = {j["k"]: j for j in C["per_junction"][seq]["omega"] if j["in_aggregate"]}
            for k in sorted(set(v) & set(o)):
                if key in v[k] and key in o[k]:
                    vv.append(v[k][key] * sc)
                    oo.append(o[k][key] * sc)
        _paired_dots(ax, [0.5] * len(vv), vv, oo, width=0.5)
        m1, m2 = _median_marks(ax, 0.0, vv, COLORS["vggt"], 0.3), _median_marks(ax, 1.0, oo, COLORS["omega"], 0.3)
        worse = sum(1 for a, b in zip(vv, oo) if b > a)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([0, 1], ["VGGT", "Omega"], fontsize=10)
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(f"{title}\n中位 {m1:.2f}→{m2:.2f}{unit}\nOmega 更差 {worse}/{len(vv)}", fontsize=9.5, color=INK, loc="left")
    fig.text(0.055, row2_head, "(c) 重叠帧 f 在相邻子图 k 与 k+1 中的预测差异(每点一个交界,三序列共 19 个;线连同一交界,粗横线为中位)",
             fontsize=13, color=INK, ha="left", va="bottom")

    # (d) focal spread ----------------------------------------------------------------------------
    sub = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.28, width_ratios=[1.2, 1])
    for col, (key, scale_, ylabel, fmt_) in enumerate((("fx_frac_std", 100, "子图内 fx 标准差 ÷ 图宽 (%)", "{:.2f}"),
                                                       ("fx_frac_mean", 1, "子图均值 fx ÷ 图宽", "{:.3f}"))):
        ax = fig.add_subplot(sub[0, col])
        _style_axes(ax)
        for gi, seq in enumerate(seqs):
            v = {m["update"]: m for m in A["per_submap"][seq]["vggt"] if m["full"]}
            o = {m["update"]: m for m in A["per_submap"][seq]["omega"] if m["full"]}
            ks = sorted(set(v) & set(o))
            va, oa = [v[k][key] * scale_ for k in ks], [o[k][key] * scale_ for k in ks]
            _paired_dots(ax, np.full(len(ks), float(gi)), va, oa, width=0.17)
            a1, a2 = _median_marks(ax, gi - 0.17, va, COLORS["vggt"]), _median_marks(ax, gi + 0.17, oa, COLORS["omega"])
            ax.text(gi, 1.02, (fmt_ + "→" + fmt_).format(a1, a2), transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=9.5, color=INK)
        ax.set_xticks(range(len(seqs)), seqs, fontsize=11)
        ax.set_xlim(-0.55, len(seqs) - 0.45)
        ax.set_ylabel(ylabel, fontsize=11, color=INK2)
        if col == 0:
            ax.set_ylim(bottom=0)
        else:
            ax.axhline(TUM_FR1_FX_FRACTION, color=INK2, lw=1, ls=(0, (4, 3)))
            ax.text(-0.5, TUM_FR1_FX_FRACTION, " TUM 标定 0.808", color=INK2, fontsize=9.5, va="bottom", ha="left")
    fig.text(0.53, row2_head, "(d) 焦距:子图内离散度(左)与均值相对 TUM 标定值(右);点上方数字=中位 VGGT→Omega", fontsize=13, color=INK, ha="left", va="bottom")

    fig.text(0.055, 0.015, "(iii) 无真值:(重叠帧深度中位数之比) ÷ (两子图相机平移步长均值之比),一致时等于真实步长比,故含混杂;"
             "(iii') 乘以真值步长比修正,1 表示一致。(iv) 上游 estimate_scale_pairwise 的尺度相对真值隐含尺度的误差。",
             fontsize=10.5, color=INK2, ha="left", va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
