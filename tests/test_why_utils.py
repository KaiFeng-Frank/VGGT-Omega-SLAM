"""CPU tests for vos/why_utils.py (association, alignment, per-frame-set metrics, overlap-frame coupling)."""
import numpy as np
import pytest

from vos import why_utils as wu


def _rot(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _quat_xyzw(rot):
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(rot).as_quat()


def _trajectory(n=12, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.cumsum(rng.normal(scale=0.1, size=(n, 3)), axis=0)
    rots = np.stack([_rot(rng.normal(size=3), 0.1 * i) for i in range(n)])
    return centers, rots


def _extrinsic(centers, rots_c2w):
    r_w2c = np.transpose(rots_c2w, (0, 2, 1))
    t = -np.einsum("sij,sj->si", r_w2c, centers)
    return np.concatenate([r_w2c, t[:, :, None]], axis=-1)


def test_associate_frames_nearest_within_tolerance():
    gt = np.array([0.0, 0.01, 0.02, 0.03])
    idx, ok, dt = wu.associate_frames([0.0104, 0.0299, 0.5], gt, 0.01)
    assert idx.tolist() == [1, 3, 3]
    assert ok.tolist() == [True, True, False]
    assert dt[0] == pytest.approx(0.0004)


def test_centers_and_rotations_roundtrip():
    centers, rots = _trajectory()
    c2, r2 = wu.centers_and_rotations(_extrinsic(centers, rots))
    assert np.allclose(c2, centers) and np.allclose(r2, rots)


def test_rotation_angle_deg():
    assert wu.rotation_angle_deg(_rot([1, 2, 3], np.radians(37.0))) == pytest.approx(37.0)
    assert wu.rotation_angle_deg(np.eye(3)) == pytest.approx(0.0)


def test_relative_rotation_error_invariant_to_global_rotation_and_detects_error():
    _, rots = _trajectory()
    g = _rot([0, 1, 0.3], 0.9)
    assert np.allclose(wu.relative_rotation_errors(g @ rots, rots), 0.0, atol=1e-4)
    bad = rots.copy()
    bad[5] = bad[5] @ _rot([1, 0, 0], np.radians(10.0))  # perturb one frame
    err = wu.relative_rotation_errors(bad, rots)
    assert err[4] == pytest.approx(10.0, abs=1e-4) and err[5] == pytest.approx(10.0, abs=1e-4)
    assert np.allclose(np.delete(err, [4, 5]), 0.0, atol=1e-4)
    masked = wu.relative_rotation_errors(bad, rots, np.array([True] * 4 + [False] + [True] * 6))
    assert np.isnan(masked[4])


def test_translation_direction_error_is_similarity_invariant():
    centers, rots = _trajectory()
    g = _rot([1, 1, 0], 0.7)
    pred_c = 3.0 * centers @ g.T + np.array([1.0, 2.0, 3.0])
    pred_r = g @ rots
    assert np.allclose(wu.translation_direction_errors(pred_c, pred_r, centers, rots), 0.0, atol=1e-4)
    # a displacement of the last camera perpendicular to the last step gives a known angle
    step = centers[-1] - centers[-2]
    perp = np.cross(step, [0.3, 0.2, 0.9])
    perp = perp / np.linalg.norm(perp) * np.linalg.norm(step)
    moved = centers.copy()
    moved[-1] = centers[-2] + perp  # 90 degrees away from the true last step
    err = wu.translation_direction_errors(moved, rots, centers, rots)
    assert err[-1] == pytest.approx(90.0, abs=1e-3)


def test_sim3_ate_known_transform_and_degenerate_cases():
    rng = np.random.default_rng(1)
    gt = rng.normal(size=(10, 3))
    r, t, c = _rot([1, 2, 3], 0.5), np.array([0.3, -1.0, 2.0]), 0.4
    pred = ((gt - t) @ r) / c  # gt = c * R @ pred + t
    fit = wu.sim3_ate(pred, gt)
    assert fit["status"] == "ok" and fit["rmse"] < 1e-9 and fit["scale"] == pytest.approx(c)
    assert wu.sim3_ate(pred[:2], gt[:2])["status"] == "undefined"
    line = np.outer(np.arange(6.0), [1.0, 0.0, 0.0])
    assert wu.sim3_ate(line, line)["status"] == "undefined"
    assert wu.baseline_ratio_scale(np.array([[0, 0, 0], [2.0, 0, 0]]), np.array([[0, 0, 0], [0, 1.0, 0]])) == pytest.approx(0.5)


def test_trajectory_metrics_recovers_similarity_and_reports_focal():
    centers, rots = _trajectory(n=10)
    n = len(centers)
    stamps = 100.0 + 0.1 * np.arange(n)
    gt = (stamps, centers, np.stack([_quat_xyzw(r) for r in rots]))
    g, s, shift = _rot([0, 0, 1], 0.4), 2.5, np.array([0.5, -0.2, 1.0])
    pred_c = s * centers @ g.T + shift
    pred_r = g @ rots
    k = np.tile(np.array([[400.0, 0, 259], [0, 400.0, 196], [0, 0, 1]]), (n, 1, 1))
    m = wu.trajectory_metrics(_extrinsic(pred_c, pred_r), k, (392, 518), stamps + 0.004, gt)
    assert m["n_associated"] == n and m["sim3_status"] == "ok"
    assert m["ate_rmse_m"] < 1e-9 and m["scale_m_per_unit"] == pytest.approx(1 / s)
    assert m["rel_rot_err_deg_mean"] < 1e-3 and m["rel_dir_err_deg_mean"] < 1e-3 and m["abs_rot_err_deg_mean"] < 1e-3
    assert m["e2e_rot_err_deg"] < 1e-3 and m["e2e_dir_err_deg"] < 1e-3 and m["e2e_gt_baseline_m"] > 0
    assert m["fx_px_mean"] == pytest.approx(400.0) and m["fx_px_std"] == pytest.approx(0.0, abs=1e-9)
    assert m["fx_frac_mean"] == pytest.approx(400.0 / 518)
    assert m["gt_path_length_m"] == pytest.approx(wu.path_length(centers))
    assert m["ate_over_gt_path"] == pytest.approx(m["ate_rmse_m"] / m["gt_path_length_m"])
    # frames without a ground-truth partner are dropped
    m2 = wu.trajectory_metrics(_extrinsic(pred_c, pred_r), k, (392, 518), stamps + np.r_[50.0, np.zeros(n - 1)], gt)
    assert m2["n_associated"] == n - 1


def test_two_frame_set_uses_baseline_ratio_scale():
    centers, rots = _trajectory(n=2)
    stamps = np.array([1.0, 1.1])
    gt = (stamps, centers, np.stack([_quat_xyzw(r) for r in rots]))
    k = np.tile(np.eye(3) * 300.0, (2, 1, 1))
    m = wu.trajectory_metrics(_extrinsic(centers * 4.0, rots), k, (100, 100), stamps, gt)
    assert m["sim3_status"] == "undefined" and m["scale_method"] == "baseline_ratio"
    assert m["scale_m_per_unit"] == pytest.approx(0.25)


def test_sparse_positions():
    pos = wu.sparse_positions(613, 8, 0)
    assert pos[0] == 0 and pos[-1] == 612 and len(set(pos.tolist())) == 8
    plus, minus = wu.sparse_positions(613, 8, 1), wu.sparse_positions(613, 8, -1)
    assert (plus >= pos).all() and (minus <= pos).all() and plus[-1] == 612 and minus[0] == 0
    assert len(set(wu.sparse_positions(613, 32, 1).tolist())) == 32


def _median_scale(x, y):
    return float(np.median(np.linalg.norm(y, axis=1) / np.linalg.norm(x, axis=1))), None


def _frame(depth, fx, conf_thr=0.5, h=24, w=32):
    k = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]])
    conf = np.ones((h, w)) * 2.0
    return {"depth": depth, "conf": conf, "intrinsic": k, "conf_thr": conf_thr}


def test_junction_consistent_pair_has_zero_error_even_with_different_focal():
    rng = np.random.default_rng(3)
    d_prev = rng.uniform(1.0, 3.0, (24, 32))
    c_prev, c_cur = 1.5, 3.0  # metres per unit of each submap
    d_cur = d_prev * c_prev / c_cur  # same metric depth seen in the other submap's units
    jm = wu.junction_metrics(_frame(d_prev, 40.0), _frame(d_cur, 44.0), c_prev, c_cur, _median_scale,
                             step_prev=0.3, step_cur=0.15, gt_step_prev=0.45, gt_step_cur=0.45)
    assert jm["fx_rel_diff"] == pytest.approx(4.0 / 42.0)
    assert jm["depth_rel_diff_median"] == pytest.approx(0.0, abs=1e-9)
    assert jm["s_true"] == pytest.approx(2.0) and jm["s_up"] == pytest.approx(2.0, rel=1e-5)
    assert abs(jm["s_up_err"]) < 1e-5 and abs(jm["s_depth_err"]) < 1e-5
    assert jm["scale_ratio_gt_step_corrected"] == pytest.approx(1.0, rel=1e-6)
    assert jm["scale_ratio_literal"] == pytest.approx(1.0, rel=1e-6)  # equal true steps -> literal ratio also 1


def test_junction_detects_scale_error_and_masks_low_confidence():
    rng = np.random.default_rng(4)
    d_prev = rng.uniform(1.0, 3.0, (24, 32))
    d_cur = d_prev * 0.5 * 1.2  # cur depth 20 % too large for c_cur / c_prev = 2
    prev, cur = _frame(d_prev, 40.0), _frame(d_cur, 40.0)
    cur["conf"][:12] = 0.0  # top half excluded
    jm = wu.junction_metrics(prev, cur, 1.0, 2.0, _median_scale)
    assert jm["n_pixels_common"] == 12 * 32
    assert jm["depth_rel_diff_median"] == pytest.approx(abs(1 - 1.2) / 1.1, rel=1e-6)
    assert jm["s_up_err"] == pytest.approx(1 / 1.2 - 1, rel=1e-5)


def test_end_to_end_error_detects_heading_error_of_last_frame():
    centers, rots = _trajectory(n=6)
    stamps = np.arange(6) * 0.1
    gt = (stamps, centers, np.stack([_quat_xyzw(r) for r in rots]))
    bad_r = rots.copy()
    bad_r[-1] = bad_r[-1] @ _rot([0, 1, 0], np.radians(7.0))  # only the last orientation is off
    k = np.tile(np.eye(3) * 300.0, (6, 1, 1))
    m = wu.trajectory_metrics(_extrinsic(centers, bad_r), k, (100, 100), stamps, gt)
    assert m["e2e_rot_err_deg"] == pytest.approx(7.0, abs=1e-4) and m["e2e_dir_err_deg"] < 1e-3
