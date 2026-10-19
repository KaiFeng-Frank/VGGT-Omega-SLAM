"""CPU stages of tools/analysis_undistort.py (analyze, slam, report). No GPU, no torch."""
from __future__ import annotations

import datetime
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vos import env  # noqa: E402  (sets sys.dont_write_bytecode)

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU stages never touch the GPU

import numpy as np  # noqa: E402

import analysis_why as aw  # noqa: E402
import analysis_undistort as au  # noqa: E402  (redirects aw.TUM / aw.OUT, holds the decision rule)

PROJECT = env.PROJECT
OUT, OUT_ROOT, SLAM_DIR, UND, RAW_ANALYSIS = au.OUT, au.OUT_ROOT, au.SLAM_DIR, au.UND, au.RAW_ANALYSIS
RAW_GT = env.TUM_ROOT  # original TUM folders (read only)
SEQS, BACKENDS = au.SEQS, au.BACKENDS
SEQ_LIST = list(SEQS)
COND = ("raw", "und")
SIZES = (16, 32)
PAPER_S32 = {"desk": 0.025, "desk2": 0.029, "room": 0.063}  # Maggio & Carlone, VGGT-SLAM 2.0, Table I (uncalibrated block)
A_KEYS = ("ate_rmse_m", "ate_over_gt_path", "rel_rot_err_deg_mean", "rel_dir_err_deg_mean", "abs_rot_err_deg_mean",
          "e2e_rot_err_deg", "e2e_dir_err_deg", "fx_frac_mean", "fy_frac_mean", "fx_frac_std", "scale_m_per_unit")
PAIR_KEYS = ("ate_rmse_m", "ate_over_gt_path", "rel_rot_err_deg_mean", "rel_dir_err_deg_mean", "abs_rot_err_deg_mean")


def _clean(obj):
    return aw._clean(obj)


def calibration():
    c = json.loads((UND / SEQS["desk"] / "calibration.json").read_text())
    for seq in SEQ_LIST:  # identical for all sequences, asserted
        other = json.loads((UND / SEQS[seq] / "calibration.json").read_text())
        assert other["K_new"] == c["K_new"] and other["K_old"] == c["K_old"], seq
    w, h = c["image_size_wh"]
    k_old, k_new = np.array(c["K_old"]), np.array(c["K_new"])
    return {"width": w, "height": h, "K_old": c["K_old"], "K_new": c["K_new"],
            "fx_old_frac": float(k_old[0, 0] / w), "fx_new_frac": float(k_new[0, 0] / w),
            "fy_old_frac": float(k_old[1, 1] / h), "fy_new_frac": float(k_new[1, 1] / h)}


PROJ_KEYS = ("ate_rmse_m", "ate_over_gt_path", "rel_rot_err_deg_mean", "rel_dir_err_deg_mean", "abs_rot_err_deg_mean",
             "e2e_rot_err_deg", "e2e_dir_err_deg")


def project_so3(extrinsic):
    """Nearest rotation (SVD polar factor) of the 3x3 block of camera-from-world [S,3,4]; translation untouched.

    VGGT's decoded extrinsics are bf16-quantised and not orthonormal (||R R^T - I|| up to ~0.01, det 0.993..1.005),
    Omega's are orthonormal to float precision.
    """
    ext = np.asarray(extrinsic, np.float64).copy()
    u, _, vt = np.linalg.svd(ext[:, :, :3])
    rot = u @ vt
    flip = np.linalg.det(rot) < 0
    if flip.any():
        u[flip, :, 2] *= -1
        rot = u @ vt
    ext[:, :, :3] = rot
    return ext


def metric_row(z, stamps, gt):
    from vos import why_utils as wu
    from vos.compare_utils import quat_to_rot

    m = wu.trajectory_metrics(z["extrinsic"], z["intrinsic"], tuple(int(x) for x in z["image_hw"]), stamps, gt)
    for k in ("rel_rot_err_deg_list", "rel_dir_err_deg_list"):
        m.pop(k, None)
    # same metrics with the predicted rotations projected to SO(3) first (robustness; see results_zh.md, "旋转矩阵非正交")
    mp = wu.trajectory_metrics(project_so3(z["extrinsic"]), z["intrinsic"], tuple(int(x) for x in z["image_hw"]), stamps, gt)
    m["proj"] = {k: mp.get(k) for k in PROJ_KEYS}
    rot = np.asarray(z["extrinsic"], np.float64)[:, :, :3]
    m["rot_orthonormality_err_max"] = float(max(np.linalg.norm(r @ r.T - np.eye(3)) for r in rot))
    m["rot_det_min"], m["rot_det_max"] = float(np.linalg.det(rot).min()), float(np.linalg.det(rot).max())
    # pairs whose trace-based angle is clipped to exactly 0 deg because trace(R_rel_pred^T R_rel_gt) > 3
    idx, ok, _ = wu.associate_frames(stamps, gt[0], 0.01)
    _, r_pred = wu.centers_and_rotations(z["extrinsic"])
    r_gt = quat_to_rot(gt[2][idx])
    d = np.transpose(np.transpose(r_pred[:-1], (0, 2, 1)) @ r_pred[1:], (0, 2, 1)) @ (np.transpose(r_gt[:-1], (0, 2, 1)) @ r_gt[1:])
    tr = np.trace(d, axis1=1, axis2=2)
    pair_ok = ok[:-1] & ok[1:]
    m["n_rel_rot_pairs"] = int(pair_ok.sum())
    m["n_rel_rot_pairs_clipped_to_zero"] = int(((tr > 3.0) & pair_ok).sum())
    return m


def add_focal_bias(m, cal):
    for axis, key in (("x", "fx_frac_mean"), ("y", "fy_frac_mean")):
        m[f"f{axis}_bias_vs_old"] = m[key] / cal[f"f{axis}_old_frac"] - 1.0
        m[f"f{axis}_bias_vs_new"] = m[key] / cal[f"f{axis}_new_frac"] - 1.0
    return m


def summarize_rows(rows, cal):
    """Summary of a list of per-submap / per-set metric rows (all with sim3 ok)."""
    from vos import why_utils as wu

    out = {"n": len(rows), **{k: wu.summarize([r[k] for r in rows if r.get(k) is not None]) for k in A_KEYS}}
    out["proj"] = {k: wu.summarize([r["proj"][k] for r in rows if r["proj"].get(k) is not None]) for k in PROJ_KEYS}
    out["rot_orthonormality_err_max"] = max((r["rot_orthonormality_err_max"] for r in rows), default=None)
    out["n_rel_rot_pairs"] = int(sum(r["n_rel_rot_pairs"] for r in rows))
    out["n_rel_rot_pairs_clipped_to_zero"] = int(sum(r["n_rel_rot_pairs_clipped_to_zero"] for r in rows))
    if rows:
        fx = float(np.mean([r["fx_frac_mean"] for r in rows]))
        fy = float(np.mean([r["fy_frac_mean"] for r in rows]))
        out["focal"] = {
            "fx_frac_mean_over_sets": fx, "fy_frac_mean_over_sets": fy,
            "fx_bias_vs_new_signed": fx / cal["fx_new_frac"] - 1.0, "fx_bias_vs_old_signed": fx / cal["fx_old_frac"] - 1.0,
            "fy_bias_vs_new_signed": fy / cal["fy_new_frac"] - 1.0, "fy_bias_vs_old_signed": fy / cal["fy_old_frac"] - 1.0,
            "fx_bias_vs_new_per_set": wu.summarize([r["fx_bias_vs_new"] for r in rows]),
            "fx_bias_vs_old_per_set": wu.summarize([r["fx_bias_vs_old"] for r in rows]),
            "fx_px_std_mean_over_sets": float(np.mean([r["fx_px_std"] for r in rows]))}
    return out


# ----------------------------------------------------------------------------------------------
def analyze_stage(args):
    from vos import why_utils as wu

    cal = calibration()
    gts = {seq: wu.load_groundtruth(Path(UND / SEQS[seq] / "groundtruth.txt")) for seq in SEQ_LIST}
    # the copied ground truth is byte-identical to the original (checked in undistort_report.json); assert once more here
    for seq in SEQ_LIST:
        assert (UND / SEQS[seq] / "groundtruth.txt").read_bytes() == (RAW_GT / SEQS[seq] / "groundtruth.txt").read_bytes()
    roots = {"raw": RAW_ANALYSIS, "und": OUT}
    results = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "calibration": cal,
               "decision_rule": au.RULE}
    missing = []

    def get(cond, backend, tag):
        path = roots[cond] / "cache" / backend / f"{tag}.npz"
        if not path.is_file():
            missing.append(f"{cond}/{backend}/{tag}")
            return None
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files}

    # ---- A: per-submap, exact raw frame lists ----------------------------------------------------
    A = {c: {s: {b: [] for b in BACKENDS} for s in SEQ_LIST} for c in COND}
    for cond in COND:
        for job in aw.a_jobs():
            stamps = [wu.stamp_of(n) for n in job["names"]]
            for b in BACKENDS:
                z = get(cond, b, job["tag"])
                if z is None:
                    continue
                assert [str(x) for x in z["names"]] == list(job["names"])
                m = add_focal_bias(metric_row(z, stamps, gts[job["seq"]]), cal)
                m.update(update=job["update"], full=len(job["names"]) == aw.FULL_SUBMAP, first_frame=job["names"][0],
                         last_frame=job["names"][-1], forward_seconds=float(z["seconds"]))
                A[cond][job["seq"]][b].append(m)

    def full(cond, seq, b):
        return [m for m in A[cond][seq][b] if m["full"] and m["sim3_status"] == "ok"]

    summary, paired, ratios = {}, {}, {}
    for cond in COND:
        summary[cond], paired[cond], ratios[cond] = {}, {}, {}
        for b in BACKENDS:
            summary[cond][b] = {}
            for seq in SEQ_LIST + ["pooled"]:
                rows = [m for s in SEQ_LIST for m in full(cond, s, b)] if seq == "pooled" else full(cond, seq, b)
                summary[cond][b][seq] = summarize_rows(rows, cal)
        for seq in SEQ_LIST + ["pooled"]:
            pairs = []
            for s in (SEQ_LIST if seq == "pooled" else [seq]):
                v = {m["update"]: m for m in full(cond, s, "vggt")}
                o = {m["update"]: m for m in full(cond, s, "omega")}
                pairs += [(s, k, v[k], o[k]) for k in sorted(set(v) & set(o))]
            paired[cond][seq] = {k: aw.paired_stats([p[2][k] for p in pairs], [p[3][k] for p in pairs]) for k in PAIR_KEYS}
            paired[cond][seq]["proj"] = {k: aw.paired_stats([p[2]["proj"][k] for p in pairs], [p[3]["proj"][k] for p in pairs])
                                         for k in PROJ_KEYS}
            ratios[cond][seq] = [{"seq": p[0], "update": p[1], "vggt": p[2]["ate_over_gt_path"], "omega": p[3]["ate_over_gt_path"],
                                  "ratio": p[3]["ate_over_gt_path"] / p[2]["ate_over_gt_path"]} for p in pairs]
    results["A"] = {"per_submap": A, "summary_full_submaps": summary, "paired_omega_vs_vggt": paired, "paired_ratio_lists": ratios}

    # same-submap change caused by undistortion, per backend (raw -> undistorted)
    change = {}
    for b in BACKENDS:
        change[b] = {}
        for seq in SEQ_LIST + ["pooled"]:
            rows = []
            for s in (SEQ_LIST if seq == "pooled" else [seq]):
                r = {m["update"]: m for m in full("raw", s, b)}
                u = {m["update"]: m for m in full("und", s, b)}
                rows += [(r[k], u[k]) for k in sorted(set(r) & set(u))]
            entry = {"n_pairs": len(rows)}
            for k in PAIR_KEYS:
                a = np.array([x[0][k] for x in rows], float)
                c = np.array([x[1][k] for x in rows], float)
                st = aw.paired_stats(a, c)  # a = raw, c = undistorted; "omega" fields of paired_stats mean "undistorted"
                entry[k] = {"median_raw": st.get("median_vggt"), "median_und": st.get("median_omega"),
                            "median_ratio_und_over_raw": st.get("median_ratio_omega_over_vggt"),
                            "n_und_larger": st.get("n_omega_larger"), "n_raw_larger": st.get("n_vggt_larger"),
                            "wilcoxon_two_sided_p": st.get("wilcoxon_two_sided_p")}
            entry["proj"] = {}
            for k in PROJ_KEYS:
                a = np.array([x[0]["proj"][k] for x in rows], float)
                c = np.array([x[1]["proj"][k] for x in rows], float)
                st = aw.paired_stats(a, c)
                entry["proj"][k] = {"median_raw": st.get("median_vggt"), "median_und": st.get("median_omega"),
                                    "median_ratio_und_over_raw": st.get("median_ratio_omega_over_vggt"),
                                    "n_und_larger": st.get("n_omega_larger"), "wilcoxon_two_sided_p": st.get("wilcoxon_two_sided_p")}
            change[b][seq] = entry
    results["A"]["change_raw_to_und"] = change

    # ---- R: determinism of the undistorted forward ----------------------------------------------
    nd = {}
    for b in BACKENDS:
        z0, z1 = get("und", b, "A_desk_00"), get("und", b, "R_desk_00_rep")
        if z0 is not None and z1 is not None:
            nd[b] = {"bitwise_identical_pose_enc": bool(np.array_equal(z0["pose_enc"], z1["pose_enc"])),
                     "bitwise_identical_depth_first": bool(np.array_equal(z0["depth_first"], z1["depth_first"])),
                     "max_abs_pose_enc_diff": float(np.abs(z0["pose_enc"] - z1["pose_enc"]).max())}
    results["A"]["determinism_und_desk_submap0"] = nd

    # ---- B: sparse wide-baseline sets ------------------------------------------------------------
    B = {c: {s: {b: {} for b in BACKENDS} for s in SEQ_LIST} for c in COND}
    for cond in COND:
        for job in aw.b_jobs():
            stamps = [wu.stamp_of(n) for n in job["names"]]
            for b in BACKENDS:
                z = get(cond, b, job["tag"])
                if z is None:
                    B[cond][job["seq"]][b].setdefault(str(job["N"]), {})[str(job["variant"])] = {"status": "missing"}
                    continue
                m = add_focal_bias(metric_row(z, stamps, gts[job["seq"]]), cal)
                m["positions"] = job["positions"]
                B[cond][job["seq"]][b].setdefault(str(job["N"]), {})[str(job["variant"])] = m
    b_summary = {}
    for cond in COND:
        b_summary[cond] = {"per_N": {}, "by_seq_N": {}}
        for n in aw.SPARSE_N:
            entry = {}
            for b in BACKENDS:
                rows = [r for s in SEQ_LIST for v in aw.SPARSE_VARIANTS
                        for r in [B[cond][s][b][str(n)].get(str(v))] if r and r.get("sim3_status") == "ok"]
                entry[b] = summarize_rows(rows, cal)
            pairs = {k: ([], []) for k in PAIR_KEYS}
            for s in SEQ_LIST:
                for v in aw.SPARSE_VARIANTS:
                    rv, ro = B[cond][s]["vggt"][str(n)].get(str(v)), B[cond][s]["omega"][str(n)].get(str(v))
                    if rv and ro and rv.get("sim3_status") == "ok" and ro.get("sim3_status") == "ok":
                        for k in pairs:
                            pairs[k][0].append(rv[k])
                            pairs[k][1].append(ro[k])
            entry["paired"] = {k: aw.paired_stats(*pairs[k]) for k in PAIR_KEYS}
            b_summary[cond]["per_N"][str(n)] = entry
        for s in SEQ_LIST:  # primary (standard grid) values per sequence
            b_summary[cond]["by_seq_N"][s] = {
                str(n): {b: ({k: B[cond][s][b][str(n)]["0"].get(k) for k in ("ate_rmse_m", "ate_over_gt_path", "rel_rot_err_deg_mean",
                                                                           "rel_dir_err_deg_mean", "fx_frac_mean", "fx_bias_vs_new",
                                                                           "fx_bias_vs_old")}
                             if B[cond][s][b][str(n)]["0"].get("sim3_status") == "ok" else None) for b in BACKENDS}
                for n in aw.SPARSE_N}
        # pooled over all sparse sets
        allp = {k: ([], []) for k in PAIR_KEYS}
        for n in aw.SPARSE_N:
            for s in SEQ_LIST:
                for v in aw.SPARSE_VARIANTS:
                    rv, ro = B[cond][s]["vggt"][str(n)].get(str(v)), B[cond][s]["omega"][str(n)].get(str(v))
                    if rv and ro and rv.get("sim3_status") == "ok" and ro.get("sim3_status") == "ok":
                        for k in allp:
                            allp[k][0].append(rv[k])
                            allp[k][1].append(ro[k])
        b_summary[cond]["paired_pooled"] = {k: aw.paired_stats(*allp[k]) for k in PAIR_KEYS}
        b_summary[cond]["focal_pooled"] = {
            b: summarize_rows([r for n in aw.SPARSE_N for s in SEQ_LIST for v in aw.SPARSE_VARIANTS
                               for r in [B[cond][s][b][str(n)].get(str(v))] if r and r.get("sim3_status") == "ok"], cal)["focal"]
            for b in BACKENDS}
    results["B"] = {"per_set": B, "summary": b_summary}

    # ---- self-check: the raw side recomputed here equals runs/analysis_why/results.json -----------
    old = json.loads((RAW_ANALYSIS / "results.json").read_text())
    check = {"max_abs_diff": 0.0, "n_compared": 0}
    for b in BACKENDS:
        for seq in SEQ_LIST + ["pooled"]:
            for k in ("ate_over_gt_path", "ate_rmse_m", "rel_rot_err_deg_mean", "rel_dir_err_deg_mean", "fx_frac_mean"):
                a = old["A"]["summary_full_submaps"][b][seq][k]["median"]
                c = summary["raw"][b][seq][k]["median"]
                check["max_abs_diff"] = max(check["max_abs_diff"], abs(a - c))
                check["n_compared"] += 1
    for seq in SEQ_LIST + ["pooled"]:
        a = old["A"]["paired_omega_vs_vggt_full_submaps"][seq]["ate_over_gt_path"]["median_ratio_omega_over_vggt"]
        c = paired["raw"][seq]["ate_over_gt_path"]["median_ratio_omega_over_vggt"]
        check["max_abs_diff"] = max(check["max_abs_diff"], abs(a - c))
        check["n_compared"] += 1
    results["raw_recompute_check_vs_analysis_why"] = check
    assert check["max_abs_diff"] < 1e-9, check

    # ---- decision --------------------------------------------------------------------------------
    pooled = summary["und"]["omega"]["pooled"]["focal"]
    bias = abs(pooled["fx_bias_vs_new_signed"])
    r_desk = paired["und"]["desk"]["ate_over_gt_path"]["median_ratio_omega_over_vggt"]
    r_desk2 = paired["und"]["desk2"]["ate_over_gt_path"]["median_ratio_omega_over_vggt"]
    results["decision"] = {
        "verdict": au.decide(bias, r_desk, r_desk2),
        "omega_focal_bias_und_vs_new_K_pooled_abs": bias,
        "omega_focal_bias_und_vs_new_K_signed_by_sequence": {s: summary["und"]["omega"][s]["focal"]["fx_bias_vs_new_signed"] for s in SEQ_LIST},
        "omega_focal_bias_raw_vs_old_K_pooled_signed": summary["raw"]["omega"]["pooled"]["focal"]["fx_bias_vs_old_signed"],
        "vggt_focal_bias_und_vs_new_K_pooled_signed": summary["und"]["vggt"]["pooled"]["focal"]["fx_bias_vs_new_signed"],
        "vggt_focal_bias_raw_vs_old_K_pooled_signed": summary["raw"]["vggt"]["pooled"]["focal"]["fx_bias_vs_old_signed"],
        "median_paired_ratio_und": {s: paired["und"][s]["ate_over_gt_path"]["median_ratio_omega_over_vggt"] for s in SEQ_LIST + ["pooled"]},
        "median_paired_ratio_raw": {s: paired["raw"][s]["ate_over_gt_path"]["median_ratio_omega_over_vggt"] for s in SEQ_LIST + ["pooled"]},
        "n_pairs_und": {s: paired["und"][s]["ate_over_gt_path"]["n_pairs"] for s in SEQ_LIST + ["pooled"]},
        "ratio_desk": r_desk, "ratio_desk2": r_desk2,
        "robustness_rotations_projected_to_SO3": {
            "note": "same rule with ATE/path recomputed from camera centres built from SO(3)-projected rotations (VGGT's decoded rotations are not orthonormal); focal unchanged",
            "median_paired_ratio_und": {s: paired["und"][s]["proj"]["ate_over_gt_path"]["median_ratio_omega_over_vggt"] for s in SEQ_LIST + ["pooled"]},
            "median_paired_ratio_raw": {s: paired["raw"][s]["proj"]["ate_over_gt_path"]["median_ratio_omega_over_vggt"] for s in SEQ_LIST + ["pooled"]},
            "verdict": au.decide(bias, paired["und"]["desk"]["proj"]["ate_over_gt_path"]["median_ratio_omega_over_vggt"],
                                 paired["und"]["desk2"]["proj"]["ate_over_gt_path"]["median_ratio_omega_over_vggt"])},
        "thresholds": {"explains_bias_lt": au.BIAS_EXPLAINS, "not_bias_gt": au.BIAS_NOT, "explains_ratio_le": au.RATIO_EXPLAINS,
                       "not_ratio_ge": au.RATIO_NOT}}
    results["missing_cache_entries"] = sorted(set(missing))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(_clean(results), indent=1, ensure_ascii=False))
    print(f"wrote {OUT / 'analysis.json'}; missing cache entries: {len(set(missing))}")
    d = results["decision"]
    print("DECISION", d["verdict"], "bias", round(bias, 4), "ratios desk/desk2/room", round(r_desk, 3), round(r_desk2, 3),
          round(paired["und"]["room"]["ate_over_gt_path"]["median_ratio_omega_over_vggt"], 3))
    return 0


# ----------------------------------------------------------------------------------------------
# SLAM runs
# ----------------------------------------------------------------------------------------------
def raw_run_dir(seq, backend, size):
    base = PROJECT / "runs" / ("compare" if size == 16 else "submap32")
    return base / f"fr1_{seq}_{backend}_loops1"


def und_run_dir(seq, backend, size):
    return SLAM_DIR / f"{seq}_{backend}_s{size}"


def read_run(path):
    path = Path(path)
    out = {"dir": str(path), "exists": path.is_dir()}
    meta_p, ate_p, kf_p = path / "metadata.json", path / "ate.json", path / "keyframes.csv"
    if not meta_p.is_file():
        out["status"] = "missing"
        return out
    meta = json.loads(meta_p.read_text())
    out["status"] = meta.get("status")
    if meta.get("status") != "complete":
        out["error"] = meta.get("error")
        out["traceback_tail"] = (meta.get("traceback") or "")[-1200:]
        out["oom"] = "OutOfMemory" in json.dumps(meta, default=str)
        return out
    ate = json.loads(ate_p.read_text()) if ate_p.is_file() else {}
    kf = [line.split(",")[2] for line in kf_p.read_text().splitlines()[1:]]
    updates = meta["updates"]
    accepted, candidates = [], []
    for entry in meta.get("loop_log", []):
        for c in entry["candidates"]:
            pair = (c["query_image"], c["detected_image"])
            candidates.append({"update": entry["update"], "query_image": pair[0], "detected_image": pair[1],
                               "salad_distance": c["salad_distance"], "raw_ratio": c.get("raw_ratio"), "accepted": bool(c.get("accepted"))})
            if c.get("accepted"):
                accepted.append({"update": entry["update"], "query_image": pair[0], "detected_image": pair[1]})
    s = meta["summary"]
    out.update(ate_rmse_m=ate.get("ate_rmse_m"), ate_scale=ate.get("scale"), n_associated=ate.get("n_associated"),
               keyframes=len(kf), keyframe_names=kf, submaps_processed=s["submaps_processed"],
               frames_per_update=[u["n_frames"] for u in updates], partition=[u["frame_names"] for u in updates],
               loop_closures_accepted=s["loop_closures_accepted"], loop_candidates_total=s["loop_candidates_total"],
               accepted_pairs=accepted, candidate_pairs=candidates, forward_seconds_total=s["forward_seconds_total"],
               peak_cuda_allocated_gib=s["peak_cuda_allocated_bytes"] / 2 ** 30, wall_seconds=meta.get("wall_seconds"),
               n_images=meta["input"]["n_images_used"])
    return out


def _pairset(run):
    return {(p["query_image"], p["detected_image"]) for p in run.get("accepted_pairs", [])}


def _stamp(name):
    return float(str(name).replace(".png", ""))


def match_pairs(pairs_a, pairs_b, tol_s=1.0):
    """One-to-one greedy matching of loop pairs by time: both the query and the detected frame within tol_s seconds."""
    a = [(_stamp(q), _stamp(d)) for q, d in sorted(pairs_a)]
    b = [(_stamp(q), _stamp(d)) for q, d in sorted(pairs_b)]
    cand = sorted((max(abs(x[0] - y[0]), abs(x[1] - y[1])), i, j) for i, x in enumerate(a) for j, y in enumerate(b)
                  if abs(x[0] - y[0]) <= tol_s and abs(x[1] - y[1]) <= tol_s)
    used_a, used_b, n = set(), set(), 0
    for _, i, j in cand:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            n += 1
    return n


def compare_runs(a, b):
    """Keyframes / partition / accepted loop pairs of two completed runs."""
    if a.get("status") != "complete" or b.get("status") != "complete":
        return None
    pa, pb = _pairset(a), _pairset(b)
    ka, kb = a["keyframe_names"], b["keyframe_names"]
    return {"keyframes_identical": ka == kb, "n_keyframes": [len(ka), len(kb)], "n_keyframes_common": len(set(ka) & set(kb)),
            "partition_identical": a["partition"] == b["partition"], "frames_per_update": [a["frames_per_update"], b["frames_per_update"]],
            "accepted_pairs_identical": pa == pb, "n_accepted_pairs": [len(pa), len(pb)], "n_accepted_pairs_common": len(pa & pb),
            "n_accepted_pairs_matched_within_1s": match_pairs(pa, pb, 1.0),
            "accepted_pairs_only_first": sorted(pa - pb), "accepted_pairs_only_second": sorted(pb - pa)}


def slam_stage(args):
    res = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "runs": {}, "checks": {}, "paper_s32": PAPER_S32}
    for seq in SEQ_LIST:
        res["runs"][seq] = {}
        res["checks"][seq] = {}
        for size in SIZES:
            runs = {}
            for b in BACKENDS:
                runs[b] = {"raw": read_run(raw_run_dir(seq, b, size)), "und": read_run(und_run_dir(seq, b, size))}
            res["runs"][seq][str(size)] = {b: {c: {k: v for k, v in r.items() if k not in ("keyframe_names", "partition")}
                                               for c, r in runs[b].items()} for b in BACKENDS}
            res["checks"][seq][str(size)] = {
                "vggt_und_vs_raw": compare_runs(runs["vggt"]["raw"], runs["vggt"]["und"]),
                "omega_und_vs_raw": compare_runs(runs["omega"]["raw"], runs["omega"]["und"]),
                "und_vggt_vs_omega": compare_runs(runs["vggt"]["und"], runs["omega"]["und"]),
                "raw_vggt_vs_omega": compare_runs(runs["vggt"]["raw"], runs["omega"]["raw"])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "slam_summary.json").write_text(json.dumps(_clean(res), indent=1, ensure_ascii=False))
    for seq in SEQ_LIST:
        for size in SIZES:
            row = res["runs"][seq][str(size)]

            def f(b, c):
                r = row[b][c]
                return f"{r['ate_rmse_m']:.4f}" if r.get("status") == "complete" and r.get("ate_rmse_m") is not None else str(r.get("status"))

            print(f"{seq} s{size}: VGGT raw {f('vggt', 'raw')} und {f('vggt', 'und')} | Omega raw {f('omega', 'raw')} und {f('omega', 'und')}")
    print(f"wrote {OUT / 'slam_summary.json'}")
    return 0


def report_stage(args):
    import analysis_undistort_report as rep

    return rep.report(args)
