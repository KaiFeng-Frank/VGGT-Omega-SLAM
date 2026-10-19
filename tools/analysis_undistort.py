#!/usr/bin/env python3
"""Does lens distortion explain VGGT-Omega's focal over-estimate and its worse translation accuracy on TUM fr1?

Re-runs tools/analysis_why.py's single-forward analysis on the UNDISTORTED frames (tools/undistort_tum.py) and compares
with the raw-image analysis (runs/analysis_why, read only).  Nothing of analysis_why.py is rewritten: its job lists,
`forward_stage` (GPU, lock inside) and the per-set metrics (`vos.why_utils.trajectory_metrics`, Sim(3) Umeyama) are reused
with two module globals redirected (image root `TUM` -> data/tum_undistorted, cache/log root `OUT` -> runs/undistort/analysis).
The frame lists stay those of the raw runs (runs/compare/<seq>_vggt_loops1 partitions, read only) and are asserted to be
identical to the frames of the raw forward cache.

Stages
  forward   GPU, one process, both backends: A (every regular submap), R (repeat), B (sparse N=8/16/32 + shifted grids)
  analyze   CPU: per-submap and sparse metrics raw vs undistorted, focal vs new K and old K, paired ratios, decision
  slam      CPU: summarise runs/undistort/slam/* against the raw s16 (runs/compare) and s32 (runs/submap32) runs
  report    CPU: results.json, results_zh.md, 去畸变对比.png

Launch the GPU stage with:
  systemd-run --user --scope --quiet -p MemoryMax=12G env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    <python> tools/analysis_undistort.py forward
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vos import env  # noqa: E402  (sets sys.dont_write_bytecode)

import analysis_why as aw  # noqa: E402  (tools/ is sys.path[0] when this file is run as a script)

PROJECT = env.PROJECT
UND = PROJECT / "data" / "tum_undistorted"
OUT_ROOT = PROJECT / "runs" / "undistort"
OUT = OUT_ROOT / "analysis"
RAW_ANALYSIS = PROJECT / "runs" / "analysis_why"
SLAM_DIR = OUT_ROOT / "slam"
SEQS, BACKENDS, SPARSE_N, SPARSE_VARIANTS = aw.SEQS, aw.BACKENDS, aw.SPARSE_N, aw.SPARSE_VARIANTS
FULL_SUBMAP = aw.FULL_SUBMAP

# redirect the reused module's roots (its functions read these module globals at call time)
aw.TUM = UND
aw.OUT = OUT

# ----------------------------------------------------------------------------------------------
# Pre-declared decision rule, fixed before any undistorted forward was run.  Readings of the wording:
#   * focal bias of a backend = | mean over the full 17-frame submaps of (per-submap mean fx / image width) / (fx_new / 640) - 1 |
#     pooled over the three sequences (all full submaps, 20 of them); per-sequence values are reported next to it.
#   * ratio = median over the paired submaps of (Omega ATE/path) / (VGGT ATE/path), one value per sequence.
#   * "ratio stays >= 1.2 on desk and desk2" = both desk and desk2 have ratio >= 1.2.
# ----------------------------------------------------------------------------------------------
RULE = {
    "EXPLAINS": "undistorted Omega focal bias < 5% AND median paired ratio (Omega ATE/path)/(VGGT ATE/path) <= 1.05 on desk AND on desk2",
    "DOES NOT EXPLAIN": "undistorted Omega focal bias > 8% OR median paired ratio >= 1.2 on desk AND on desk2",
    "PARTIAL": "anything else; room reported separately",
    "focal_bias": "|mean over full submaps of (fx/width) / (fx_new/640) - 1|, pooled over desk, desk2, room",
    "ratio": "median over paired full submaps of (Omega ATE/path) / (VGGT ATE/path), per sequence",
}
BIAS_EXPLAINS, BIAS_NOT, RATIO_EXPLAINS, RATIO_NOT = 0.05, 0.08, 1.05, 1.2


def decide(bias_omega, ratio_desk, ratio_desk2):
    if bias_omega < BIAS_EXPLAINS and ratio_desk <= RATIO_EXPLAINS and ratio_desk2 <= RATIO_EXPLAINS:
        return "EXPLAINS"
    if bias_omega > BIAS_NOT or (ratio_desk >= RATIO_NOT and ratio_desk2 >= RATIO_NOT):
        return "DOES NOT EXPLAIN"
    return "PARTIAL"


def assert_same_windows():
    """Every frame list of the jobs equals the frame list stored in the raw forward cache (same submaps, same sparse sets)."""
    import numpy as np

    n = 0
    for job in aw.all_jobs():
        for b in BACKENDS:
            raw = RAW_ANALYSIS / "cache" / b / f"{job['tag']}.npz"
            if not raw.is_file():
                raise RuntimeError(f"raw cache entry missing: {raw}")
            z = np.load(raw, allow_pickle=False)
            if [str(x) for x in z["names"]] != list(job["names"]):
                raise RuntimeError(f"{b}/{job['tag']}: frame list differs from the raw analysis")
            n += 1
    return n


def forward_stage(args):
    n = assert_same_windows()
    print(f"[check] {n} (backend, job) frame lists identical to the raw analysis cache", flush=True)
    for seq in SEQS:
        if not (UND / SEQS[seq] / "rgb").is_dir():
            raise RuntimeError(f"run tools/undistort_tum.py first ({UND / SEQS[seq]} missing)")
    return aw.forward_stage(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    fwd = sub.add_parser("forward", help="GPU: single forwards on undistorted frames, cached")
    fwd.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS)
    for name in ("analyze", "slam", "report"):
        sub.add_parser(name)
    args = parser.parse_args(argv)
    if args.stage == "forward":
        return forward_stage(args)
    import analysis_undistort_cpu as cpu  # noqa: PLC0415  (CPU stages live in tools/analysis_undistort_cpu.py)

    return getattr(cpu, f"{args.stage}_stage")(args)


if __name__ == "__main__":
    raise SystemExit(main())
