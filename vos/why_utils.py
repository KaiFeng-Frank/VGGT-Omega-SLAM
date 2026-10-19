"""CPU helpers for the front-end vs backend-coupling decomposition (tools/analysis_why.py).

Pure numpy; no torch, no upstream imports (the upstream scale estimator is injected by the caller).

Conventions
  * extrinsic  [S,3,4]  camera-from-world (OpenCV), as decoded by both backends' pose decoders
  * rotations are projected to SO(3) in float64 before inversion and angular evaluation.
    R_c2w = project(R_w2c)^T; camera centre C = -project(R_w2c)^T t.
  * TUM ground truth rows: stamp tx ty tz qx qy qz qw  (camera -> world, optical-frame convention)
  * association: every predicted frame takes the nearest ground-truth stamp (|dt| <= 0.01 s); a frame
    without a partner is dropped. One pair per frame (the sequence-level ATE in eval_ate.py associates
    per ground-truth stamp instead, which gives about two ground-truth rows per keyframe).
"""
from __future__ import annotations

import numpy as np

from eval_ate import ate_rmse, read_tum_trajectory
from .compare_utils import quat_to_rot, stamp_of  # noqa: F401  (stamp_of re-exported for the tool)

# TUM freiburg1 published RGB calibration (fx = 517.3 px on a 640 px wide image). Reference only.
TUM_FR1_FX_PX = 517.3
TUM_FR1_WIDTH_PX = 640.0
TUM_FR1_FX_FRACTION = TUM_FR1_FX_PX / TUM_FR1_WIDTH_PX
GEOMETRY_METRIC_VERSION = "so3_float64_v1"


# ----------------------------------------------------------------------------------------------
# ground truth, association
# ----------------------------------------------------------------------------------------------
def load_groundtruth(path):
    stamps, xyz, quat, _ = read_tum_trajectory(path)
    order = np.argsort(stamps, kind="stable")
    return stamps[order], xyz[order], quat[order]


def associate_frames(frame_stamps, gt_stamps, max_diff=0.01):
    """Nearest ground-truth row per frame (first on ties). Returns (gt_index[n], ok[n] bool, dt[n])."""
    frame_stamps = np.asarray(frame_stamps, dtype=np.float64)
    diffs = np.abs(np.asarray(gt_stamps, dtype=np.float64)[None, :] - frame_stamps[:, None])
    index = diffs.argmin(axis=1)
    dt = diffs[np.arange(len(index)), index]
    return index, dt <= max_diff, dt


# ----------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------
def project_to_so3(rotation):
    """Closest proper rotation to each [...,3,3] matrix, using a float64 polar factor.

    Decoded bf16 matrices need not be orthogonal. Applying the trace/acos formula
    directly can clip a real error to zero or invent an error for identical poses.
    Project individual rotations before composing relative transforms. This is an
    evaluation correction; it does not change inference or saved SLAM trajectories.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    u, _, vt = np.linalg.svd(rotation)
    sign = np.ones(rotation.shape[:-2] + (3,), dtype=np.float64)
    sign[..., -1] = np.where(np.linalg.det(u @ vt) < 0, -1.0, 1.0)
    return (u * sign[..., None, :]) @ vt


def centers_and_rotations(extrinsic):
    """Centres and c2w rotations from camera-from-world poses, after SO(3) projection.

    Translation columns are preserved. Historical unprojected metrics are not
    numerically identical to these corrected evaluation metrics.
    """
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    r_w2c, t = project_to_so3(extrinsic[:, :, :3]), extrinsic[:, :, 3]
    r_c2w = np.transpose(r_w2c, (0, 2, 1))
    centers = -np.einsum("sij,sj->si", r_c2w, t)
    return centers, r_c2w


def rotation_angle_deg(rot):
    """Geodesic angle in degrees, robust to decoded matrix precision and small angles."""
    rot = project_to_so3(rot)
    cos = (np.trace(rot, axis1=-2, axis2=-1) - 1.0) / 2.0
    skew = np.stack((rot[..., 2, 1] - rot[..., 1, 2],
                     rot[..., 0, 2] - rot[..., 2, 0],
                     rot[..., 1, 0] - rot[..., 0, 1]), axis=-1)
    sin = 0.5 * np.linalg.norm(skew, axis=-1)
    return np.degrees(np.arctan2(sin, np.clip(cos, -1.0, 1.0)))


def relative_rotation_errors(r_pred, r_gt, pair_ok=None):
    """Angle (deg) between predicted and ground-truth relative rotation R_i^T R_{i+1} of consecutive frames.

    Invariant to any global rotation of either trajectory. `pair_ok[i]` False drops pair (i, i+1).
    """
    r_pred, r_gt = project_to_so3(r_pred), project_to_so3(r_gt)
    rel_p = np.transpose(r_pred[:-1], (0, 2, 1)) @ r_pred[1:]
    rel_g = np.transpose(r_gt[:-1], (0, 2, 1)) @ r_gt[1:]
    err = rotation_angle_deg(np.transpose(rel_p, (0, 2, 1)) @ rel_g)
    if pair_ok is not None:
        err = np.where(pair_ok, err, np.nan)
    return err


def translation_direction_errors(c_pred, r_pred, c_gt, r_gt, pair_ok=None, min_step=1e-9):
    """Angle (deg) between the predicted and ground-truth direction of the translation C_{i+1}-C_i,
    both expressed in the camera frame of frame i. Scale- and global-similarity-invariant."""
    r_pred, r_gt = project_to_so3(r_pred), project_to_so3(r_gt)

    def local(c, r):
        step = c[1:] - c[:-1]
        return np.einsum("sji,sj->si", r[:-1], step)  # R_i^T (C_{i+1} - C_i)

    d_p, d_g = local(c_pred, r_pred), local(c_gt, r_gt)
    n_p, n_g = np.linalg.norm(d_p, axis=1), np.linalg.norm(d_g, axis=1)
    cos = np.einsum("si,si->s", d_p, d_g) / np.maximum(n_p * n_g, 1e-300)
    err = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    bad = (n_p < min_step) | (n_g < min_step)
    if pair_ok is not None:
        bad = bad | ~pair_ok
    return np.where(bad, np.nan, err)


def path_length(xyz):
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()) if len(xyz) > 1 else 0.0


def sim3_ate(pred_xyz, gt_xyz):
    """Sim(3) Umeyama (with scale) of predicted camera centres onto ground truth.

    Returns dict(status, rmse, scale, errors, rotation, translation) or status='undefined' + reason when
    fewer than 3 points or a degenerate (collinear / coincident) configuration.
    """
    n = len(pred_xyz)
    if n < 3:
        return {"status": "undefined", "reason": f"only {n} associated frames (need >= 3)"}
    try:
        rmse, errors, (rot, trans, scale) = ate_rmse(gt_xyz, pred_xyz, with_scale=True)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return {"status": "undefined", "reason": f"{type(exc).__name__}: {exc}"}
    if not np.isfinite(rmse):
        return {"status": "undefined", "reason": "non-finite alignment"}
    return {"status": "ok", "rmse": float(rmse), "scale": float(scale), "errors": errors,
            "rotation": rot, "translation": trans}


def baseline_ratio_scale(pred_xyz, gt_xyz):
    """GT metres per predicted unit from the two end baselines of a (very short) submap (2 frames)."""
    p = float(np.linalg.norm(pred_xyz[-1] - pred_xyz[0]))
    g = float(np.linalg.norm(gt_xyz[-1] - gt_xyz[0]))
    return g / p if p > 0 else float("nan")


def _stats(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std()), "min": float(v.min()), "max": float(v.max())}


summarize = _stats


def trajectory_metrics(extrinsic, intrinsic, image_hw, frame_stamps, gt, max_diff=0.01):
    """All per-frame-set accuracy numbers (one submap, or one sparse frame set).

    extrinsic [S,3,4], intrinsic [S,3,3], image_hw (H, W), frame_stamps [S], gt = (stamps, xyz, quat_xyzw).
    """
    gt_stamps, gt_xyz, gt_quat = gt
    s = len(frame_stamps)
    idx, ok, dt = associate_frames(frame_stamps, gt_stamps, max_diff)
    centers, r_pred = centers_and_rotations(extrinsic)
    r_gt_all = quat_to_rot(gt_quat[idx])
    gt_c = gt_xyz[idx]

    height, width = image_hw
    fx, fy = np.asarray(intrinsic)[:, 0, 0].astype(np.float64), np.asarray(intrinsic)[:, 1, 1].astype(np.float64)
    out = {"geometry_metric_version": GEOMETRY_METRIC_VERSION,
           "n_frames": int(s), "n_associated": int(ok.sum()), "max_abs_dt_s": float(dt[ok].max()) if ok.any() else None,
           "image_hw": [int(height), int(width)],
           "fx_px_mean": float(fx.mean()), "fx_px_std": float(fx.std()),
           "fx_frac_mean": float(fx.mean() / width), "fx_frac_std": float(fx.std() / width),
           "fx_cv": float(fx.std() / fx.mean()),
           "fy_px_mean": float(fy.mean()), "fy_px_std": float(fy.std()),
           "fy_frac_mean": float(fy.mean() / height), "fy_frac_std": float(fy.std() / height),
           "fx_frac_over_tum_calibration": float(fx.mean() / width / TUM_FR1_FX_FRACTION)}

    pair_ok = ok[:-1] & ok[1:]  # consecutive frames, both associated
    rot_err = relative_rotation_errors(r_pred, r_gt_all, pair_ok)
    dir_err = translation_direction_errors(centers, r_pred, gt_c, r_gt_all, pair_ok)
    out["rel_rot_err_deg"] = _stats(rot_err)
    out["rel_dir_err_deg"] = _stats(dir_err)
    out["rel_rot_err_deg_list"] = [None if not np.isfinite(x) else float(x) for x in rot_err]
    out["rel_dir_err_deg_list"] = [None if not np.isfinite(x) else float(x) for x in dir_err]
    out["rel_rot_err_deg_mean"] = out["rel_rot_err_deg"]["mean"]
    out["rel_dir_err_deg_mean"] = out["rel_dir_err_deg"]["mean"]

    # end-to-end relative pose first -> last associated frame: this is what a single-overlap chain passes on
    # (the last frame of a submap becomes the first frame of the next one)
    ok_idx = np.flatnonzero(ok)
    if len(ok_idx) >= 2:
        i0, i1 = int(ok_idx[0]), int(ok_idx[-1])
        rel_p, rel_g = r_pred[i0].T @ r_pred[i1], r_gt_all[i0].T @ r_gt_all[i1]
        out["e2e_rot_err_deg"] = float(rotation_angle_deg(rel_p.T @ rel_g))
        out["e2e_gt_rot_deg"] = float(rotation_angle_deg(rel_g))
        d_p, d_g = r_pred[i0].T @ (centers[i1] - centers[i0]), r_gt_all[i0].T @ (gt_c[i1] - gt_c[i0])
        n_p, n_g = np.linalg.norm(d_p), np.linalg.norm(d_g)
        out["e2e_dir_err_deg"] = (float(np.degrees(np.arccos(np.clip(d_p @ d_g / (n_p * n_g), -1.0, 1.0))))
                                  if n_p > 1e-9 and n_g > 1e-9 else None)
        out["e2e_gt_baseline_m"] = float(n_g)
    p_assoc, g_assoc = centers[ok], gt_c[ok]
    out["gt_path_length_m"] = path_length(g_assoc)
    steps_p = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    steps_g = np.linalg.norm(np.diff(gt_c, axis=0), axis=1)
    out["pred_step_mean_units"] = float(steps_p[pair_ok].mean()) if pair_ok.any() else None
    out["gt_step_mean_m"] = float(steps_g[pair_ok].mean()) if pair_ok.any() else None
    out["pred_path_length_units"] = float(steps_p.sum())

    fit = sim3_ate(p_assoc, g_assoc)
    out["sim3_status"] = fit["status"]
    if fit["status"] == "ok":
        out["ate_rmse_m"] = fit["rmse"]
        out["scale_m_per_unit"] = fit["scale"]
        out["ate_over_gt_path"] = fit["rmse"] / out["gt_path_length_m"] if out["gt_path_length_m"] > 0 else None
        # absolute orientation error after the same alignment rotation (also a check of the GT frame convention)
        r_fit = fit["rotation"]
        aligned = np.einsum("ij,sjk->sik", r_fit, r_pred[ok])
        out["abs_rot_err_deg_mean"] = float(rotation_angle_deg(np.transpose(aligned, (0, 2, 1)) @ r_gt_all[ok]).mean())
        out["scale_method"] = "sim3"
    else:
        out["sim3_reason"] = fit["reason"]
        out["ate_rmse_m"] = None
        out["ate_over_gt_path"] = None
        out["scale_method"] = None
        out["scale_m_per_unit"] = None
        if ok.sum() == 2:
            out["scale_m_per_unit"] = baseline_ratio_scale(p_assoc, g_assoc)
            out["scale_method"] = "baseline_ratio"
    return out


# ----------------------------------------------------------------------------------------------
# sparse wide-baseline sets
# ----------------------------------------------------------------------------------------------
def sparse_positions(n_valid, n, variant=0):
    """Indices (into the list of GT-associated frames) of n uniformly spaced frames.

    variant 0: linspace(0, M-1, n) rounded (first and last frame included).
    variant +1 / -1: the same grid shifted by +/- one third of the spacing and clipped to [0, M-1].
    """
    step = (n_valid - 1) / (n - 1)
    grid = np.linspace(0, n_valid - 1, n) + {0: 0.0, 1: step / 3.0, -1: -step / 3.0}[variant]
    pos = np.clip(np.round(grid).astype(int), 0, n_valid - 1)
    if len(np.unique(pos)) != n:
        raise ValueError(f"sparse set of {n} frames from {n_valid} is not unique: {pos}")
    return pos


# ----------------------------------------------------------------------------------------------
# overlap-frame coupling
# ----------------------------------------------------------------------------------------------
def cam_points(depth, intrinsic):
    """Camera-frame points [H,W,3] from a depth map (mirror of vggt.utils.geometry.depth_to_cam_coords_points)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    fu, fv, cu, cv = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    x = (u - cu) * depth / fu
    y = (v - cv) * depth / fv
    return np.stack((x, y, depth), axis=-1).astype(np.float32)  # upstream casts to float32


def _k4(intrinsic):
    k4 = np.eye(4)
    k4[:3, :3] = intrinsic
    return k4


def junction_metrics(prev, cur, scale_prev, scale_cur, estimate_scale, step_prev=None, step_cur=None,
                     gt_step_prev=None, gt_step_cur=None):
    """Consistency of overlap frame f between submap k (`prev`, f = its last frame) and k+1 (`cur`, f = its first).

    prev / cur: dict(depth [H,W], conf [H,W], intrinsic [3,3], conf_thr float) of f in that submap
                (conf_thr = percentile-25 of the whole submap's confidence + 1e-6, as Submap.add_all_points).
    scale_prev / scale_cur: GT metres per predicted unit of each submap (Sim3 scale from the per-submap fit).
    estimate_scale: upstream vggt_slam.scale_solver.estimate_scale_pairwise.
    step_*: mean predicted consecutive-camera-centre distance of each submap (units); gt_step_*: same in metres.

    Quantities (a = pixel set where both confidences exceed their submap's p25 threshold):
      fx_rel_diff        |fx_k - fx_k+1| / mean(fx)                                         (i)
      depth_rel_diff_med median_a |Dk*ck - Dk+1*ck+1| / mean(...)  (metric depth, per pixel) (ii)
      scale_ratio_literal (median_a D_k / median_a D_k+1) / (step_k / step_k+1)             (iii), GT free
      scale_ratio_gt_step_corrected  literal * (gt_step_k / gt_step_k+1)  -> 1 when consistent
      s_up        estimate_scale_pairwise exactly as Solver.add_edge computes it (incl. its fallbacks)
      s_true      scale_cur / scale_prev  (predicted-unit k+1 -> unit k implied by the GT fits)
      s_up_err    s_up / s_true - 1 ; s_depth_err  (median D_k / median D_k+1) / s_true - 1
    """
    out = {}
    fx_p, fx_c = float(prev["intrinsic"][0, 0]), float(cur["intrinsic"][0, 0])
    fy_p, fy_c = float(prev["intrinsic"][1, 1]), float(cur["intrinsic"][1, 1])
    out["fx_k_px"], out["fx_k1_px"] = fx_p, fx_c
    out["fx_rel_diff"] = abs(fx_p - fx_c) / (0.5 * (fx_p + fx_c))
    out["fx_ratio_k1_over_k"] = fx_c / fx_p
    out["fy_rel_diff"] = abs(fy_p - fy_c) / (0.5 * (fy_p + fy_c))

    d_p, d_c = prev["depth"].astype(np.float64), cur["depth"].astype(np.float64)
    c_p, c_c = prev["conf"], cur["conf"]
    mask = (c_p > prev["conf_thr"]) & (c_c > cur["conf_thr"]) & (d_p > 1e-8) & (d_c > 1e-8)
    out["n_pixels_common"] = int(mask.sum())
    dm_p, dm_c = float(np.median(d_p[mask])), float(np.median(d_c[mask]))
    out["depth_median_k"], out["depth_median_k1"] = dm_p, dm_c
    out["depth_ratio_median"] = dm_p / dm_c

    if step_prev and step_cur:
        out["step_ratio_pred"] = step_prev / step_cur
        out["scale_ratio_literal"] = (dm_p / dm_c) / (step_prev / step_cur)
        out["scale_ratio_literal_abs_log"] = abs(float(np.log(out["scale_ratio_literal"])))
        if gt_step_prev and gt_step_cur:
            out["gt_step_ratio"] = gt_step_prev / gt_step_cur
            out["scale_ratio_gt_step_corrected"] = out["scale_ratio_literal"] * gt_step_prev / gt_step_cur
            out["scale_ratio_gt_step_corrected_abs_log"] = abs(float(np.log(out["scale_ratio_gt_step_corrected"])))

    if scale_prev and scale_cur and np.isfinite(scale_prev) and np.isfinite(scale_cur):
        a, b = d_p[mask] * scale_prev, d_c[mask] * scale_cur
        rel = np.abs(a - b) / (0.5 * (a + b))
        out["depth_rel_diff_median"] = float(np.median(rel))
        out["depth_rel_diff_mean"] = float(rel.mean())
        out["depth_log_ratio_median"] = float(np.median(np.log(a / b)))
        out["s_true"] = float(scale_cur / scale_prev)
        out["s_depth_err"] = (dm_p / dm_c) / out["s_true"] - 1.0

    # exact replica of Solver.add_edge's scale estimate (proj_mats are the 4x4 intrinsics in add_all_points)
    thr = prev["conf_thr"]
    good = ((c_p > thr) * (c_c > thr)).reshape(-1)
    fallback = 0
    if good.sum() < 100:
        good = (c_p > thr).reshape(-1)
        fallback = 1
        if good.sum() < 100:
            good = (c_p > 0).reshape(-1)
            fallback = 2
    p_temp = np.linalg.inv(_k4(prev["intrinsic"])) @ _k4(cur["intrinsic"])
    t1 = (p_temp[:3, :3] @ cam_points(d_c, cur["intrinsic"]).reshape(-1, 3)[good].T).T
    t2 = cam_points(d_p, prev["intrinsic"]).reshape(-1, 3)[good]
    s_up = float(estimate_scale(t1, t2)[0])
    out["s_up"], out["s_up_fallback_level"] = s_up, fallback
    if "s_true" in out:
        out["s_up_err"] = s_up / out["s_true"] - 1.0
    return out
