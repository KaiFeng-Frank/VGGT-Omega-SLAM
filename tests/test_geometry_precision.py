"""Regressions for false angular errors caused by nonorthogonal decoded rotations."""
import numpy as np
import pytest

from vos import why_utils as wu


def _z(degrees):
    theta = np.radians(degrees)
    return np.array([[np.cos(theta), -np.sin(theta), 0.0],
                     [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])


def test_nonorthogonal_relative_rotation_does_not_clip_real_error_to_zero():
    gt = np.stack((np.eye(3), np.eye(3)))
    predicted = 1.002 * np.stack((np.eye(3), _z(0.7)))
    # The old trace formula really does return zero for this small nonzero error.
    rel = predicted[0].T @ predicted[1]
    legacy = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
    assert legacy == 0
    assert wu.relative_rotation_errors(predicted, gt)[0] == pytest.approx(0.7, abs=1e-10)


def test_contracted_identical_rotations_do_not_invent_angular_error():
    predicted = 0.998 * np.stack((_z(20), _z(20)))
    gt = np.stack((_z(20), _z(20)))
    assert wu.relative_rotation_errors(predicted, gt)[0] == pytest.approx(0, abs=1e-10)


def test_projection_uses_proper_rotations_and_preserves_translation_column():
    rotation = _z(31)
    stretch = np.diag([1.003, 0.997, 1.001])
    translation = np.array([0.3, -0.6, 1.2])
    pose = np.column_stack((rotation @ stretch, translation))[None]
    centers, c2w = wu.centers_and_rotations(pose)
    assert np.allclose(c2w[0], rotation.T, atol=1e-12)
    assert np.allclose(centers[0], -rotation.T @ translation, atol=1e-12)
    assert np.array_equal(pose[0, :, 3], translation)
    proper = wu.project_to_so3(np.diag([1.0, 1.0, -1.0]))
    assert np.linalg.det(proper) == pytest.approx(1)
    assert np.allclose(proper.T @ proper, np.eye(3))


def test_rotation_angle_is_accurate_at_small_and_half_turn_angles():
    assert wu.rotation_angle_deg(_z(1e-7)) == pytest.approx(1e-7, rel=1e-6)
    assert wu.rotation_angle_deg(_z(180)) == pytest.approx(180)


def test_trajectory_metrics_marks_corrected_geometry_version():
    centers = np.array([[0, 0, 0], [0.3, 0, 0], [0.3, 0.2, 0], [0.2, 0.1, 0.4]], float)
    stamps = np.arange(4) * 0.1
    distorted_rotation = np.diag([1.002, 0.998, 1.001])
    poses = np.concatenate((np.tile(distorted_rotation, (4, 1, 1)), -centers[:, :, None]), axis=2)
    gt = (stamps, centers, np.tile([0.0, 0.0, 0.0, 1.0], (4, 1)))
    intrinsics = np.tile(np.eye(3), (4, 1, 1))
    metrics = wu.trajectory_metrics(poses, intrinsics, (100, 100), stamps, gt)
    assert metrics["geometry_metric_version"] == "so3_float64_v1"
    assert metrics["ate_rmse_m"] < 1e-12
    assert metrics["rel_rot_err_deg_mean"] < 1e-10
    assert metrics["e2e_rot_err_deg"] < 1e-10
