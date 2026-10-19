#!/usr/bin/env python3
"""Compare positional ATE on identical time support, with no extrapolation.

Both sparse predictions are linearly interpolated at the same ground-truth
timestamps in their common interval, then independently aligned to those same
ground-truth positions with Sim(3). This is a supplementary interpolation check;
it does not recover the trajectory between sparse keyframes. It does not replace
the original per-run ATE or change a frozen development selection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_key] = "4"

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_ate import ate_rmse, read_tum_trajectory


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "min_s": None, "q25_s": None, "median_s": None,
                "q75_s": None, "q95_s": None, "max_s": None, "mean_s": None}
    quantiles = np.percentile(values, [0, 25, 50, 75, 95, 100])
    return dict(count=int(len(values)), **dict(zip(
        ("min_s", "q25_s", "median_s", "q75_s", "q95_s", "max_s"), map(float, quantiles))),
        mean_s=float(values.mean()))


def _read_positions(path):
    times, xyz, _, dropped = read_tum_trajectory(path)
    # np.unique sorts stamps and returns the first occurrence in original file
    # order, matching eval_ate's np.argmin choice for duplicate estimate stamps.
    unique, first, counts = np.unique(times, return_index=True, return_counts=True)
    info = {"path": str(path), "n_input_rows": int(len(times) + dropped),
            "n_nonfinite_dropped": dropped, "n_finite_rows": int(len(times)),
            "n_unique_timestamps": int(len(unique)),
            "n_duplicate_rows_dropped": int(len(times) - len(unique)),
            "n_duplicate_timestamps": int(np.count_nonzero(counts > 1)),
            "duplicates_policy": "first occurrence in input file, then sort by timestamp",
            "input_had_decreasing_timestamps": bool(np.any(np.diff(times) < 0)),
            "time_range_s": [float(unique[0]), float(unique[-1])] if len(unique) else None,
            "timestamp_intervals": _distribution(np.diff(unique))}
    return unique, xyz[first], info


def _interpolate(times, xyz, query):
    if len(times) < 2 or not len(query):
        raise ValueError("interpolation needs at least two source timestamps and nonempty queries")
    if query[0] < times[0] or query[-1] > times[-1]:
        raise ValueError("extrapolation is forbidden")
    origin = times[0]
    positions = np.column_stack([np.interp(query - origin, times - origin, xyz[:, axis]) for axis in range(3)])
    right = np.searchsorted(times, query, side="left")
    exact = times[right] == query
    gaps = np.zeros(len(query), dtype=np.float64)
    nonexact = ~exact
    gaps[nonexact] = times[right[nonexact]] - times[right[nonexact] - 1]
    return positions, {"n_exact_samples": int(exact.sum()),
                       "n_interpolated_samples": int(nonexact.sum()),
                       "bracketing_gap_distribution": _distribution(gaps[nonexact]),
                       "max_bracketing_gap_s": float(gaps.max())}


def _score_pair(gt, baseline, candidate):
    result = {"n_timestamps": int(len(gt)), "small_sample": bool(len(gt) < 20),
              "small_sample_definition": "fewer than 20 timestamps",
              "alignment": "independent Sim(3) Umeyama alignment to identical ground-truth positions"}
    if len(gt) < 3:
        return dict(result, status="failed", error="fewer than 3 common timestamps")
    try:
        for name, estimate in (("baseline", baseline), ("candidate", candidate)):
            rmse, errors, (rotation, translation, scale) = ate_rmse(gt, estimate, with_scale=True)
            result[name] = {"ate_rmse_m": rmse, "ate_mean_m": float(errors.mean()),
                            "ate_median_m": float(np.median(errors)), "scale": scale,
                            "rotation": rotation.tolist(), "translation": translation.tolist()}
    except (ValueError, np.linalg.LinAlgError) as exc:
        return dict(result, status="failed", error=str(exc))
    base = result["baseline"]["ate_rmse_m"]
    candidate_rmse = result["candidate"]["ate_rmse_m"]
    result.update(status="ok", candidate_over_baseline_rmse=candidate_rmse / base if base > 1e-12 else None,
                  rmse_difference_m=candidate_rmse - base,
                  ratio_unavailable_reason=None if base > 1e-12 else "baseline RMSE <= 1e-12 m")
    return result


def compare_trajectories(baseline, candidate, groundtruth):
    bt, bx, binfo = _read_positions(baseline)
    ct, cx, cinfo = _read_positions(candidate)
    gt, gx, ginfo = _read_positions(groundtruth)
    result = {"schema": "vggt_omega_slam.common_trajectory_support.v1", "gpu_used": False,
              "baseline": binfo, "candidate": cinfo, "groundtruth": ginfo,
              "limitation": "Piecewise linear interpolation between sparse keyframes is supplementary; "
                            "it cannot establish the actual motion between those frames. No extrapolation is used.",
              "selection": "supplementary validation only; original development ATE selection remains frozen"}
    if min(len(bt), len(ct), len(gt)) < 2:
        return dict(result, status="failed", error="each input needs at least two unique finite timestamps")
    start, end = max(bt[0], ct[0], gt[0]), min(bt[-1], ct[-1], gt[-1])
    result["common_time_range_s"] = [float(start), float(end)] if start <= end else None
    if start > end:
        return dict(result, status="failed", error="no overlapping time interval")
    mask = (gt >= start) & (gt <= end)
    query, truth = gt[mask], gx[mask]
    result["common_groundtruth_support"] = {"query_source": "unique ground-truth timestamps in common interval",
                                            "n_gt_timestamps_excluded": int((~mask).sum())}
    support = result["common_groundtruth_support"]
    if len(query) < 3:
        support.update(status="failed", n_timestamps=int(len(query)), error="fewer than 3 common ground-truth timestamps")
        return dict(result, status="failed", error=support["error"])
    bquery, binterp = _interpolate(bt, bx, query)
    cquery, cinterp = _interpolate(ct, cx, query)
    support.update(_score_pair(truth, bquery, cquery), evaluated_time_range_s=[float(query[0]), float(query[-1])],
                   baseline_interpolation=binterp, candidate_interpolation=cinterp,
                   timestamp_intervals=_distribution(np.diff(query)))
    # A second view uses exact shared *prediction* stamps. Only ground truth
    # is interpolated here; prediction positions are retained without resampling.
    shared, bi, ci = np.intersect1d(bt, ct, assume_unique=True, return_indices=True)
    within_gt = (shared >= gt[0]) & (shared <= gt[-1])
    shared, bi, ci = shared[within_gt], bi[within_gt], ci[within_gt]
    if len(shared):
        shared_gt, gtinterp = _interpolate(gt, gx, shared)
        shared_result = _score_pair(shared_gt, bx[bi], cx[ci])
        shared_result["groundtruth_interpolation"] = gtinterp
    else:
        shared_result = _score_pair(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)))
    shared_result.update(query_source="exact shared prediction timestamps within ground-truth range; ground truth linearly interpolated",
                         time_range_s=[float(shared[0]), float(shared[-1])] if len(shared) else None,
                         timestamp_intervals=_distribution(np.diff(shared)))
    result["exact_shared_prediction_support"] = shared_result
    result["status"] = support["status"]
    if support["status"] != "ok":
        result["error"] = support["error"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--groundtruth", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    result = compare_trajectories(args.baseline, args.candidate, args.groundtruth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    support = result.get("common_groundtruth_support", {})
    print(json.dumps({"status": result["status"], "n_timestamps": support.get("n_timestamps"),
                      "candidate_over_baseline_rmse": support.get("candidate_over_baseline_rmse"),
                      "error": result.get("error")}, allow_nan=False))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
