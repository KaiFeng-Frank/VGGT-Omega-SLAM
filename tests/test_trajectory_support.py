"""Known geometric transforms and counterexamples for common-support ATE."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.compare_trajectory_support import _interpolate, compare_trajectories


def _write(path, times, xyz):
    quat = np.tile([0., 0., 0., 1.], (len(times), 1))
    np.savetxt(path, np.column_stack((times, xyz, quat)), fmt="%.15g")


def _curve(times):
    return np.column_stack((times, times ** 2, np.sin(times)))


def test_known_sim3_at_different_sampling_is_exact_for_piecewise_linear_motion(tmp_path):
    knots = np.arange(6., dtype=float)
    points = _curve(knots)
    query = np.linspace(0., 5., 101)
    truth = np.column_stack([np.interp(query, knots, points[:, i]) for i in range(3)])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    _write(tmp_path / "gt", query, truth)
    _write(tmp_path / "base", knots, 2.5 * points @ rotation.T + [4., -3., 2.])
    # Different timestamps retain every motion knot, so interpolating the
    # known Sim(3) commutes with the exact linear motion inside each segment.
    dense = np.arange(0., 5.01, .5)
    dense_xyz = np.column_stack([np.interp(dense, knots, points[:, i]) for i in range(3)])
    _write(tmp_path / "candidate", dense, .4 * dense_xyz - [1., 2., 3.])
    result = compare_trajectories(tmp_path / "base", tmp_path / "candidate", tmp_path / "gt")
    assert result["status"] == "ok"
    support = result["common_groundtruth_support"]
    assert support["n_timestamps"] == 101
    assert support["baseline"]["ate_rmse_m"] < 1e-12
    assert support["candidate"]["ate_rmse_m"] < 1e-12
    assert support["baseline"]["scale"] == pytest.approx(.4)
    assert support["candidate"]["scale"] == pytest.approx(2.5)
    assert support["baseline_interpolation"]["max_bracketing_gap_s"] == 1.
    assert support["candidate_interpolation"]["max_bracketing_gap_s"] == .5
    assert support["candidate_over_baseline_rmse"] is None


def test_duplicates_keep_first_input_row_even_when_unsorted(tmp_path):
    times = np.arange(6., dtype=float)
    xyz = _curve(times)
    _write(tmp_path / "gt", times, xyz)
    _write(tmp_path / "base", np.r_[times, 2.], np.vstack([xyz, [999., 999., 999.]]))
    order = np.array([4, 1, 5, 0, 2, 3])
    _write(tmp_path / "candidate", times[order], xyz[order])
    result = compare_trajectories(tmp_path / "base", tmp_path / "candidate", tmp_path / "gt")
    assert result["common_groundtruth_support"]["baseline"]["ate_rmse_m"] < 1e-12
    assert result["baseline"]["n_duplicate_rows_dropped"] == 1
    assert result["baseline"]["n_duplicate_timestamps"] == 1
    assert result["candidate"]["input_had_decreasing_timestamps"]
    assert result["exact_shared_prediction_support"]["n_timestamps"] == 6
    assert result["exact_shared_prediction_support"]["small_sample"]


def test_intersection_uses_identical_gt_timestamps_without_extrapolation(tmp_path):
    times = np.arange(8., dtype=float)
    xyz = _curve(times)
    _write(tmp_path / "gt", times, xyz)
    _write(tmp_path / "base", times[:6], xyz[:6])
    candidate = xyz[2:].copy()
    candidate[1, 2] += .1
    _write(tmp_path / "candidate", times[2:], candidate)
    result = compare_trajectories(tmp_path / "base", tmp_path / "candidate", tmp_path / "gt")
    support = result["common_groundtruth_support"]
    assert result["common_time_range_s"] == [2., 5.]
    assert support["n_timestamps"] == 4 and support["n_gt_timestamps_excluded"] == 4
    assert support["baseline"]["ate_rmse_m"] < 1e-12
    assert support["candidate"]["ate_rmse_m"] > .01
    with pytest.raises(ValueError, match="extrapolation"):
        _interpolate(times[2:], xyz[2:], np.array([1., 2.]))


def test_no_overlap_returns_failure_without_interpolating(tmp_path):
    times = np.arange(6., dtype=float)
    for name, shift in (("gt", 0.), ("base", 0.), ("candidate", 10.)):
        _write(tmp_path / name, times + shift, _curve(times))
    result = compare_trajectories(tmp_path / "base", tmp_path / "candidate", tmp_path / "gt")
    assert result["status"] == "failed" and result["common_time_range_s"] is None
    assert result["error"] == "no overlapping time interval"


def test_nonfinite_rows_are_reported_and_single_point_support_fails(tmp_path):
    times = np.arange(6., dtype=float)
    _write(tmp_path / "gt", times, _curve(times))
    _write(tmp_path / "base", times[:4], _curve(times[:4]))
    xyz = _curve(times[3:])
    xyz[-1] = np.nan
    _write(tmp_path / "candidate", times[3:], xyz)
    result = compare_trajectories(tmp_path / "base", tmp_path / "candidate", tmp_path / "gt")
    assert result["candidate"]["n_nonfinite_dropped"] == 1
    assert result["status"] == "failed"
    assert result["common_groundtruth_support"]["n_timestamps"] == 1


def test_cli_writes_complete_json_with_finite_ratio(tmp_path):
    times = np.linspace(0., 4., 41)
    xyz = _curve(times)
    _write(tmp_path / "gt", times, xyz)
    perturbation = np.column_stack((np.sin(5 * times), np.cos(7 * times), np.sin(9 * times))) * .01
    _write(tmp_path / "base", times, xyz + perturbation)
    _write(tmp_path / "candidate", times, xyz + 2 * perturbation)
    output = tmp_path / "result.json"
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "tools/compare_trajectory_support.py"),
               "--baseline", str(tmp_path / "base"), "--candidate", str(tmp_path / "candidate"),
               "--groundtruth", str(tmp_path / "gt"), "--out", str(output)]
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(output.read_text())
    assert result["gpu_used"] is False
    assert result["common_groundtruth_support"]["candidate_over_baseline_rmse"] == pytest.approx(2., rel=.01)
