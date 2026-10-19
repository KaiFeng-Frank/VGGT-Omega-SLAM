"""CPU tests for vos/compare_utils.py (summary and figure helpers)."""
import struct

import numpy as np
import pytest

from vos import compare_utils as cu


def _rot(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def test_read_poses_keeps_first_duplicate(tmp_path):
    f = tmp_path / "p.tum"
    f.write_text("1.0 0 0 0 0 0 0 1\n2.0 1 0 0 0 0 0 1\n2.0 9 9 9 0 0 0 1\n3.0 2 0 0 0 0 0 1\n")
    stamps, xyz, quat = cu.read_poses(f)
    assert stamps.tolist() == [1.0, 2.0, 3.0]
    assert xyz[1].tolist() == [1.0, 0.0, 0.0]
    assert quat.shape == (3, 4)


def test_stamp_of_matches_upstream_regex():
    assert cu.stamp_of("/a/b/frame_0130.jpg") == 130.0
    assert cu.stamp_of("1305031452.791720.png") == pytest.approx(1305031452.79172)


def test_umeyama_recovers_known_sim3():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(40, 3))
    r, t, c = _rot([1, 2, 3], 0.7), np.array([1.0, -2.0, 0.5]), 2.5
    dst = c * src @ r.T + t
    r2, t2, c2 = cu.umeyama(src, dst)
    assert c2 == pytest.approx(c) and np.allclose(r2, r) and np.allclose(t2, t)
    assert np.abs(cu.apply_sim3(src, r2, t2, c2) - dst).max() < 1e-9


def test_gap_fraction_closed_square():
    xyz = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.0, 0.1, 0]])
    g = cu.start_end_gap(xyz)
    assert g["path_length"] == pytest.approx(3.9)
    assert g["gap"] == pytest.approx(0.1)
    assert g["gap_fraction"] == pytest.approx(0.1 / 3.9)


def test_ply_roundtrip_and_subsample(tmp_path):
    n = 50
    xyz = np.random.default_rng(1).normal(size=(n, 3))
    rgb = np.random.default_rng(2).integers(0, 255, size=(n, 3), dtype=np.uint8)
    header = ("ply\nformat binary_little_endian 1.0\ncomment Created by Open3D\nelement vertex %d\n"
              "property double x\nproperty double y\nproperty double z\nproperty uchar red\n"
              "property uchar green\nproperty uchar blue\nend_header\n" % n).encode()
    body = b"".join(struct.pack("<3d3B", *p, *c) for p, c in zip(xyz, rgb))
    f = tmp_path / "c.ply"
    f.write_bytes(header + body)
    x2, c2 = cu.read_ply_xyz_rgb(f)
    assert np.allclose(x2, xyz) and (c2 == rgb).all()
    x3, c3 = cu.read_ply_xyz_rgb(f, max_points=10)
    assert len(x3) == len(c3) == 10
    with pytest.raises(ValueError):
        f.write_bytes(header + body[:-5])
        cu.read_ply_xyz_rgb(f)


def test_accepted_loop_edges_only_accepted():
    stamps = np.array([1.0, 2.0, 40.0, 130.0])
    xyz = np.arange(12, dtype=float).reshape(4, 3)
    log = [{"update": 2, "candidates": [
        {"accepted": False, "query_image": "frame_0130.jpg", "detected_image": "frame_0040.jpg"},
        {"accepted": True, "query_image": "frame_0130.jpg", "detected_image": "frame_0001.jpg", "raw_ratio": 1.0},
        {"accepted": True, "query_image": "frame_0999.jpg", "detected_image": "frame_0001.jpg"}]}]
    edges = cu.accepted_loop_edges(log, stamps, xyz)
    assert len(edges) == 1
    assert edges[0][0].tolist() == xyz[3].tolist() and edges[0][1].tolist() == xyz[0].tolist()


def test_top_down_basis_is_orthonormal_and_up_is_negative_down():
    rng = np.random.default_rng(3)
    centres = rng.normal(size=(30, 3)) * [3, 0.05, 1]  # wide in x and z, flat in y (OpenCV vertical axis)
    down = np.array([0.0, 1.0, 0.0])
    e1, e2, up = cu.top_down_basis(down, centres)
    assert np.allclose([e1 @ e1, e2 @ e2, up @ up], 1.0)
    assert abs(e1 @ e2) < 1e-9 and abs(e1 @ up) < 1e-9
    assert up.tolist() == [0.0, -1.0, 0.0]
    assert abs(e1[0]) > 0.9  # principal axis is x


def test_gravity_down_identity_camera():
    q = np.tile([0.0, 0.0, 0.0, 1.0], (5, 1))
    assert np.allclose(cu.gravity_down_from_cameras(q), [0, 1, 0])
    r = _rot([1, 0, 0], np.pi / 2)  # camera rolled about x by 90 deg: its y axis becomes world z
    q2 = np.array([[np.sin(np.pi / 4), 0, 0, np.cos(np.pi / 4)]])
    assert np.allclose(cu.quat_to_rot(q2)[0], r, atol=1e-9)
