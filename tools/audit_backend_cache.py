#!/usr/bin/env python3
"""CPU audit of confidence filtering and overlap scales from existing dense caches.

No model, GPU, raw image, or point-cloud export is used. Ground-truth pose scales
are computed after float64 SO(3) projection. Shared-frame depth consistency is a
geometric diagnostic, not absolute depth ground truth.

Example:
  python tools/audit_backend_cache.py --raw-cache runs/analysis_why/cache \
    --und-cache runs/undistort/analysis/cache --tum-root data/tum_rgbd
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import rankdata, spearmanr
from vos import why_utils as wu

FRAME_QUANTILES = (10, 25, 50, 75)
POLICIES = ("all_valid", "legacy_prior_q25", "each_submap_q25") + tuple(
    f"each_frame_q{q}" for q in FRAME_QUANTILES)
SEQUENCES = ("desk", "desk2", "room")


def _number(x):
    return float(x) if np.isfinite(x) else None


def stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max())} if len(a) else {"n": 0}


def valid_pixels(prior_depth, current_depth, prior_conf, current_conf):
    arrays = [np.asarray(a) for a in (prior_depth, current_depth, prior_conf, current_conf)]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("Shared-frame arrays must have identical shapes")
    return np.logical_and.reduce([np.isfinite(a) & (a > 0) for a in arrays])


def confidence_mask(prior_conf, current_conf, prior_threshold, current_threshold,
                    valid, minimum_points=100):
    """Replicate upstream's two fallback levels; report when thresholds are bypassed."""
    prior_conf, current_conf, valid = map(np.asarray, (prior_conf, current_conf, valid))
    mask = (prior_conf > prior_threshold) & (current_conf > current_threshold) & valid
    initial_count = int(mask.sum())
    fallback = 0
    if initial_count < minimum_points:
        mask = (prior_conf > prior_threshold) & valid
        fallback = 1
        if int(mask.sum()) < minimum_points:
            mask = (prior_conf > 0) & valid
            fallback = 2
    return mask, {"initial_count": initial_count, "fallback_level": fallback}


def policy_mask(name, prior, current, valid, minimum_points=100):
    pc, cc = prior["conf"], current["conf"]
    if name == "all_valid":
        return valid.copy(), {"initial_count": int(valid.sum()), "fallback_level": 0,
                              "prior_threshold": None, "current_threshold": None}
    if name == "legacy_prior_q25":
        pt = ct = float(prior["submap_q25"])
    elif name == "each_submap_q25":
        pt, ct = float(prior["submap_q25"]), float(current["submap_q25"])
    elif name.startswith("each_frame_q"):
        q = int(name.removeprefix("each_frame_q"))
        pt, ct = float(np.percentile(pc, q) + 1e-6), float(np.percentile(cc, q) + 1e-6)
    else:
        raise ValueError(f"Unknown policy: {name}")
    mask, extra = confidence_mask(pc, cc, pt, ct, valid, minimum_points)
    return mask, {**extra, "prior_threshold": pt, "current_threshold": ct}


def scale_metrics(prior_depth, current_depth, mask, true_scale):
    """Upstream scale after K_prior^-1 K_current cancels the current rays.

    For positive depth at the same pixel, median(norm(Y)/norm(X)) reduces to
    median(D_prior/D_current). This leaves the estimator unchanged; only masks vary.
    """
    if not np.isfinite(true_scale) or true_scale <= 0:
        raise ValueError("true_scale must be finite and positive")
    ratio = np.asarray(prior_depth, float)[mask] / np.asarray(current_depth, float)[mask]
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if not len(ratio):
        return {"n_pixels": 0, "estimated_scale": None, "abs_log_scale_error": None}
    estimate = float(np.median(ratio))
    residual = np.log(ratio / true_scale)
    symmetric = 2 * np.abs(ratio / true_scale - 1) / (ratio / true_scale + 1)
    return {"n_pixels": len(ratio), "estimated_scale": estimate,
            "true_scale": float(true_scale), "signed_scale_error": estimate / true_scale - 1,
            "abs_log_scale_error": float(abs(np.log(estimate / true_scale))),
            "median_abs_log_pixel_residual": float(np.median(abs(residual))),
            "median_symmetric_metric_depth_error": float(np.median(symmetric)),
            "log_ratio_mad": float(np.median(abs(np.log(ratio) - np.median(np.log(ratio))))) }


def confidence_ranking(prior, current, valid, true_scale, max_samples=20000):
    pc, cc = prior["conf"][valid], current["conf"][valid]
    pd, cd = prior["depth"][valid], current["depth"][valid]
    # Deterministic spatial stride, not random pixels chosen using residuals.
    stride = max(1, int(np.ceil(len(pc) / max_samples)))
    pc, cc, pd, cd = [a[::stride] for a in (pc, cc, pd, cd)]
    residual = np.abs(np.log(pd.astype(float) / cd.astype(float) / true_scale))
    rp, rc = rankdata(pc, method="average") / len(pc), rankdata(cc, method="average") / len(cc)
    score = np.minimum(rp, rc)
    def rho(values):
        return _number(spearmanr(values, residual).statistic) if np.ptp(values) > 0 and np.ptp(residual) > 0 else None
    bins = []
    for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.0)):
        mask = (score > lo) & (score <= hi)
        bins.append({"min_rank_range": [lo, hi], "n": int(mask.sum()),
                     "median_abs_log_pixel_residual": float(np.median(residual[mask])) if mask.any() else None})
    return {"n_sampled_pixels": len(pc), "spatial_stride": stride,
            "prior_conf_vs_error_spearman": rho(pc), "current_conf_vs_error_spearman": rho(cc),
            "min_conf_rank_vs_error_spearman": rho(score), "rank_bins": bins,
            "interpretation": "Negative rho indicates higher confidence predicts lower shared-frame inconsistency; no independent-pixel p-values are reported."}


def pose_scale(z, groundtruth):
    stamps = [wu.stamp_of(n) for n in z["names"]]
    idx, ok, _ = wu.associate_frames(stamps, groundtruth[0], .01)
    centers, _ = wu.centers_and_rotations(z["extrinsic"])
    fit = wu.sim3_ate(centers[ok], groundtruth[1][idx[ok]])
    return fit["scale"] if fit["status"] == "ok" else None


def load_window(path, groundtruth):
    with np.load(path, allow_pickle=False) as z:
        return {"tag": path.stem, "names": z["names"].tolist(), "n_frames": len(z["names"]),
                "pose_scale": pose_scale(z, groundtruth), "intrinsic": z["intrinsic"].astype(float),
                "depth_first": z["depth_first"], "depth_last": z["depth_last"],
                "conf_first": z["conf_first"], "conf_last": z["conf_last"],
                "submap_q25": float(z["conf_thr"])}


def summarize_rows(rows):
    out = {}
    for policy in POLICIES:
        values = [r["policies"][policy] for r in rows]
        metrics = {k: stats([v[k] for v in values if v.get(k) is not None]) for k in (
            "abs_log_scale_error", "signed_scale_error", "median_abs_log_pixel_residual",
            "median_symmetric_metric_depth_error", "retained_fraction")}
        comparisons = [(v, r["policies"]["legacy_prior_q25"]) for r, v in zip(rows, values)]
        metrics["n_scale_error_improves_vs_legacy"] = sum(a["abs_log_scale_error"] < b["abs_log_scale_error"] for a, b in comparisons)
        metrics["median_paired_abs_log_scale_error_change_vs_legacy"] = float(np.median([
            a["abs_log_scale_error"] - b["abs_log_scale_error"] for a, b in comparisons])) if comparisons else None
        metrics["n_fallback"] = sum(v["fallback_level"] > 0 for v in values)
        metrics["cumulative_signed_log_scale_error"] = float(sum(np.log(v["estimated_scale"] / v["true_scale"]) for v in values))
        out[policy] = metrics
    ranking = {k: stats([r["confidence_ranking"][k] for r in rows if r["confidence_ranking"][k] is not None])
               for k in ("prior_conf_vs_error_spearman", "current_conf_vs_error_spearman", "min_conf_rank_vs_error_spearman")}
    return {"n_junctions": len(rows), "policies": out, "confidence_ranking": ranking,
            "submap_threshold_ratio_current_over_prior": stats([r["submap_threshold_ratio_current_over_prior"] for r in rows])}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    design = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
              "development": ["desk"], "validation": ["desk2", "room"],
              "disclosure": "All three sequences and earlier headline results were examined before this audit. Validation is an adaptation split, not an untouched benchmark.",
              "primary_condition": "raw", "primary_geometry": wu.GEOMETRY_METRIC_VERSION,
              "primary_junctions": "Both adjacent windows contain 17 frames; fit-valid short tails are reported separately.",
              "frozen_structural_candidate": "each_submap_q25",
              "exploratory_candidates": [f"each_frame_q{q}" for q in FRAME_QUANTILES],
              "development_selection": "Among exploratory candidates, minimize mean abs-log scale error on raw desk full-window junctions only; apply selected candidate unchanged to desk2/room.",
              "metric": "abs(log(estimated overlap scale / GT-pose-implied scale)); true scale uses separate per-window Sim3 scales after SO3 projection",
              "confidence_metric": "Spearman rank and rank-bin residuals of identical overlap-frame metric depth; descriptive spatially dependent pixels",
              "limitations": ["Only first/last dense maps are cached. Full-submap q25 is exact; other quantiles are per overlap frame, not full-submap quantiles.",
                              "Pose-fit-implied scale contains trajectory error and does not supply absolute depth ground truth.",
                              "No whole-SLAM intervention, image, model weight or GPU operation is included.",
                              "Few dependent junctions, especially desk. Do not select policy using validation outcomes."]}
    (args.output / "backend_cache_design.json").write_text(json.dumps(design, indent=2) + "\n")
    rows, skipped = [], []
    for cond, root in (("raw", args.raw_cache), ("und", args.und_cache)):
        for seq in SEQUENCES:
            gt = wu.load_groundtruth(args.tum_root / ("rgbd_dataset_freiburg1_" + seq) / "groundtruth.txt")
            for backend in ("omega", "vggt"):
                paths = sorted((root / backend).glob(f"A_{seq}_*.npz"))
                if not paths:
                    raise FileNotFoundError(f"No A windows for {cond}/{seq}/{backend}")
                previous = load_window(paths[0], gt)
                for path in paths[1:]:
                    current = load_window(path, gt)
                    if previous["names"][-1] != current["names"][0]:
                        raise ValueError("Consecutive caches do not share their boundary frame")
                    if previous["pose_scale"] is None or current["pose_scale"] is None:
                        skipped.append({"condition": cond, "seq": seq, "backend": backend, "tag": current["tag"], "reason": "Undefined GT-pose Sim3 scale"})
                        previous = current
                        continue
                    prior = {"depth": previous["depth_last"], "conf": previous["conf_last"], "submap_q25": previous["submap_q25"]}
                    cur = {"depth": current["depth_first"], "conf": current["conf_first"], "submap_q25": current["submap_q25"]}
                    valid = valid_pixels(prior["depth"], cur["depth"], prior["conf"], cur["conf"])
                    truth = current["pose_scale"] / previous["pose_scale"]
                    policies = {}
                    for policy in POLICIES:
                        mask, details = policy_mask(policy, prior, cur, valid)
                        policies[policy] = {**scale_metrics(prior["depth"], cur["depth"], mask, truth), **details,
                                            "retained_fraction": float(mask.sum() / valid.sum())}
                    row = {"condition": cond, "seq": seq, "split": "development" if seq == "desk" else "validation", "backend": backend,
                           "prior_tag": previous["tag"], "current_tag": current["tag"], "shared_frame": current["names"][0],
                           "primary_full_windows": previous["n_frames"] == current["n_frames"] == 17,
                           "prior_pose_scale": previous["pose_scale"], "current_pose_scale": current["pose_scale"],
                           "submap_threshold_ratio_current_over_prior": current["submap_q25"] / previous["submap_q25"],
                           "n_valid_pixels": int(valid.sum()), "policies": policies,
                           "confidence_ranking": confidence_ranking(prior, cur, valid, truth)}
                    rows.append(row)
                    previous = current
    full = [r for r in rows if r["primary_full_windows"]]
    dev = [r for r in full if r["condition"] == "raw" and r["seq"] == "desk" and r["backend"] == "omega"]
    candidates = design["exploratory_candidates"]
    scores = {p: float(np.mean([r["policies"][p]["abs_log_scale_error"] for r in dev])) for p in candidates}
    selected = min(candidates, key=lambda p: (scores[p], candidates.index(p)))
    summary = {}
    for cond in ("raw", "und"):
        summary[cond] = {}
        for backend in ("omega", "vggt"):
            summary[cond][backend] = {seq: summarize_rows([r for r in full if r["condition"] == cond and r["backend"] == backend and r["seq"] == seq]) for seq in SEQUENCES}
            for split, seqs in (("validation_pooled", ("desk2", "room")), ("all_sequences_descriptive", SEQUENCES)):
                summary[cond][backend][split] = summarize_rows([r for r in full if r["condition"] == cond and r["backend"] == backend and r["seq"] in seqs])
    result = {"design": design, "development_selected_frame_policy": selected, "development_candidate_scores": scores,
              "full_window_summary": summary, "all_rows_including_fit_valid_tails": rows, "skipped": skipped,
              "counts": {"all_fit_valid_junction_rows": len(rows), "full_window_junction_rows": len(full), "tail_sensitivity_rows": len(rows)-len(full)}}
    (args.output / "backend_cache_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "selected_on_raw_desk_only": selected, "development_scores": scores, "counts": result["counts"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-cache", type=Path, default=ROOT / "runs/analysis_why/cache")
    parser.add_argument("--und-cache", type=Path, default=ROOT / "runs/undistort/analysis/cache")
    parser.add_argument("--tum-root", type=Path, default=ROOT / "data/tum_rgbd")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/adaptation_audit/backend_cache")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
