#!/usr/bin/env python3
"""ATE RMSE as `evo_ape tum <groundtruth> <estimate> -as` (evo is not installed here).

Re-implementation of evo's behaviour:
  * association: for every reference (ground-truth) stamp the estimated pose with the nearest stamp
    (first one on ties) is taken if |dt| <= max_diff (default 0.01 s, evo's default); this is evo's
    `matching_time_indices`, so a reference stamp near a duplicated estimated stamp (VGGT-SLAM writes
    each overlap frame twice) is associated with its first copy, and one estimated pose can serve
    two reference stamps;
  * alignment: Umeyama similarity (rotation, translation, scale) of the estimate onto the reference
    over the associated pairs (evo `-a -s`);
  * error: Euclidean norm of the translation residual per pair, RMSE over pairs (metres for TUM).

Usage:  eval_ate.py <groundtruth.txt> <poses.tum> [--json out.json] [--max-diff 0.01]
"""
# The associate and umeyama_alignment helpers adapt evo conventions and code
# (Michael Grupp and contributors), under GPL-3.0-or-later. See
# THIRD_PARTY_NOTICES.md and licenses/EVO-GPL-3.0.txt for attribution and terms.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def read_tum_trajectory(path):
    """Rows `stamp tx ty tz qx qy qz qw`; '#' lines skipped. Returns (stamps[n], xyz[n,3], quat_xyzw[n,4], n_dropped)."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 8:
            raise ValueError(f"{path}: expected 8 columns, got {len(parts)}: {line[:80]!r}")
        rows.append([float(value) for value in parts])
    data = np.asarray(rows, dtype=np.float64).reshape(-1, 8)
    finite = np.isfinite(data).all(axis=1)
    data = data[finite]
    return data[:, 0], data[:, 1:4], data[:, 4:8], int((~finite).sum())


def associate(ref_stamps, est_stamps, max_diff=0.01):
    """evo `matching_time_indices(ref, est, max_diff)`. Returns (ref_idx, est_idx) arrays."""
    ref_idx, est_idx = [], []
    for i, stamp in enumerate(ref_stamps):
        diffs = np.abs(est_stamps - stamp)
        j = int(np.argmin(diffs))
        if diffs[j] <= max_diff:
            ref_idx.append(i)
            est_idx.append(j)
    return np.asarray(ref_idx, dtype=np.int64), np.asarray(est_idx, dtype=np.int64)


def umeyama_alignment(x, y, with_scale=True):
    """Least-squares similarity mapping x onto y (evo's umeyama_alignment). x, y: [3, n].

    Returns (R, t, c) with y ~ c * R @ x + t.
    """
    m, n = x.shape
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, None]) ** 2)
    cov_xy = (y - mean_y[:, None]) @ (x - mean_x[:, None]).T / n
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        raise ValueError("degenerate point configuration (rank of covariance < 2)")
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        s[m - 1, m - 1] = -1
    r = u @ s @ v
    c = (1.0 / sigma_x) * np.trace(np.diag(d) @ s) if with_scale else 1.0
    t = mean_y - c * r @ mean_x
    return r, t, float(c)


def ate_rmse(ref_xyz, est_xyz, with_scale=True):
    """Align est onto ref (Sim(3) if with_scale else SE(3)) and return (rmse, errors, (R, t, c))."""
    r, t, c = umeyama_alignment(est_xyz.T, ref_xyz.T, with_scale)
    aligned = (c * (r @ est_xyz.T)).T + t
    errors = np.linalg.norm(aligned - ref_xyz, axis=1)
    return float(np.sqrt(np.mean(errors ** 2))), errors, (r, t, c)


def evaluate(groundtruth, estimate, max_diff=0.01):
    gt_t, gt_xyz, _, gt_bad = read_tum_trajectory(groundtruth)
    est_t, est_xyz, _, est_bad = read_tum_trajectory(estimate)
    result = {"groundtruth": str(groundtruth), "estimate": str(estimate), "max_time_diff_s": max_diff,
              "n_groundtruth": int(len(gt_t)), "n_estimate_rows": int(len(est_t)),
              "n_estimate_nonfinite_dropped": est_bad, "n_groundtruth_nonfinite_dropped": gt_bad,
              "n_estimate_unique_stamps": int(len(np.unique(est_t))),
              "alignment": "Sim(3) Umeyama with scale (evo_ape tum -as)"}
    if len(gt_t) == 0 or len(est_t) == 0:
        result.update(status="failed", error="empty trajectory", n_associated=0, ate_rmse_m=None)
        return result
    ref_idx, est_idx = associate(gt_t, est_t, max_diff)
    result["n_associated"] = int(len(ref_idx))
    if len(ref_idx) < 3:
        result.update(status="failed", error="fewer than 3 associated poses", ate_rmse_m=None)
        return result
    ref_xyz, matched_xyz = gt_xyz[ref_idx], est_xyz[est_idx]
    rmse, errors, (r, t, c) = ate_rmse(ref_xyz, matched_xyz, with_scale=True)
    rmse_se3, _, _ = ate_rmse(ref_xyz, matched_xyz, with_scale=False)
    dts = np.abs(est_t[est_idx] - gt_t[ref_idx])
    result.update(
        status="ok",
        ate_rmse_m=rmse, ate_mean_m=float(errors.mean()), ate_median_m=float(np.median(errors)),
        ate_std_m=float(errors.std()), ate_min_m=float(errors.min()), ate_max_m=float(errors.max()),
        scale=c, rotation=r.tolist(), translation=t.tolist(),
        ate_rmse_se3_no_scale_m=rmse_se3,
        n_distinct_estimate_poses_used=int(len(np.unique(est_idx))),
        mean_abs_time_diff_s=float(dts.mean()), max_abs_time_diff_s=float(dts.max()),
        path_length_groundtruth_m=float(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1).sum()),
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groundtruth", type=Path)
    parser.add_argument("estimate", type=Path)
    parser.add_argument("--json", type=Path, default=None, help="write the full result here")
    parser.add_argument("--max-diff", type=float, default=0.01, help="association tolerance in seconds")
    args = parser.parse_args(argv)
    result = evaluate(args.groundtruth, args.estimate, args.max_diff)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))
    if result["status"] != "ok":
        print(f"FAILED: {result.get('error')}", file=sys.stderr)
        return 1
    print(f"ATE RMSE (Sim3 aligned) = {result['ate_rmse_m']:.4f} m   scale = {result['scale']:.4f}   "
          f"pairs = {result['n_associated']} (distinct est poses {result['n_distinct_estimate_poses_used']}, "
          f"est rows {result['n_estimate_rows']}, gt rows {result['n_groundtruth']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
