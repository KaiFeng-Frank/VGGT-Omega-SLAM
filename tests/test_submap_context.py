"""CPU regressions for context selection, frame identity, and camera-gauge removal."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("probe_submap_context", Path(__file__).resolve().parents[1] / "tools/probe_submap_context.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def manifest():
    names = [f"{index + 1:.6f}.png" for index in range(65)]
    windows = [names[start:start + 17] for start in (0, 16, 32, 48)]
    return {"datasets": {seq: seq for seq in probe.SEQUENCES}, "partitions": {
        seq: {key: windows for key in ("vggt_loops1", "vggt_loops0", "omega_loops1", "omega_loops0")}
        for seq in probe.SEQUENCES}}


def rotation_z(angle):
    theta = np.radians(angle)
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.]])


def payload(names, centers, world_rotation=None, offset=None, scale=1., focal=500.):
    world_rotation = np.eye(3) if world_rotation is None else world_rotation
    offset = np.zeros(3) if offset is None else offset
    locations = scale * centers @ world_rotation.T + offset
    # Every camera's optical orientation is identity in the canonical trajectory.
    r_w2c = world_rotation.T
    quaternion = np.array([0., 0., np.sin(np.radians(-30) / 2), np.cos(np.radians(-30) / 2)]) if np.allclose(world_rotation, rotation_z(30)) else np.array([0., 0., 0., 1.])
    translation = -locations @ r_w2c.T
    fov = np.array([2 * np.arctan(480 / 2 / focal), 2 * np.arctan(640 / 2 / focal)])
    enc = np.column_stack((translation, np.tile(quaternion, (len(names), 1)), np.tile(fov, (len(names), 1))))
    return {"frame_names": list(names), "pose_enc": enc, "image_hw": [480, 640]}


def geometry():
    names = [f"{index + 1:.6f}.png" for index in range(9)]
    phase = np.linspace(0, 1, 9)
    centers = np.column_stack((phase, phase ** 2, np.sin(phase * 2)))
    gt = (np.arange(1, 10, dtype=float), centers, np.tile([0., 0., 0., 1.], (9, 1)))
    camera = {"K": [[500, 0, 320], [0, 500, 240], [0, 0, 1]], "image_size_wh": [640, 480]}
    return names, centers, gt, camera


def test_boundary_selection_has_same_nine_frames_in_same_order_for_both_backends():
    pairs = probe.build_pairs(manifest())
    assert len(pairs) == 6
    for pair in pairs:
        assert len(pair["baseline_frame_names"]) == len(pair["shifted_frame_names"]) == 17
        assert pair["common_frame_names"] == pair["baseline_frame_names"][8:]
        assert pair["common_frame_names"] == pair["shifted_frame_names"][:9]
        assert pair["anchor_frame_name"] == pair["common_frame_names"][0]
    assert {p["split"] for p in pairs if p["seq"] == "desk"} == {"development"}
    assert {p["split"] for p in pairs if p["seq"] != "desk"} == {"validation"}


def test_different_frontend_frame_partitions_are_rejected():
    record = manifest()
    record["partitions"]["desk"]["omega_loops0"] = [["999.000000.png"]]
    with pytest.raises(ValueError, match="different keyframes"):
        probe.build_pairs(record)


def test_legacy_cache_names_are_checked_against_fixed_inference_order(tmp_path):
    names, centers, _, _ = geometry()
    data = payload(names, centers)
    path = tmp_path / "legacy.npz"
    np.savez(path, names=np.asarray(names, dtype="S"), pose_enc=data["pose_enc"], image_hw=[480, 640])
    assert probe.load_cache(path, names)["frame_names"] == names
    with pytest.raises(ValueError, match="frame order"):
        probe.load_cache(path, names[::-1])
    with pytest.raises(ValueError, match="duplicate"):
        probe.frame_names(["a/1.000000.png", "b/1.000000.png"])


def test_shared_frames_are_matched_by_name_instead_of_array_index():
    names, centers, _, _ = geometry()
    data = payload(names, centers)
    permutation = np.array([8, 2, 0, 4, 6, 1, 5, 7, 3])
    shuffled = {**data, "frame_names": [names[i] for i in permutation], "pose_enc": data["pose_enc"][permutation]}
    c, _, f = probe.shared_cameras(shuffled, names)
    assert np.allclose(c, centers, atol=1e-12)
    assert np.allclose(f, 500.)


def test_changed_world_origin_rotation_and_scale_do_not_become_context_error():
    names, centers, gt, camera = geometry()
    baseline = payload(names, centers)
    shifted = payload(names, centers, rotation_z(30), [2., -3., 1.], scale=2.3)
    result = probe.compare_pair(baseline, shifted, {"common_frame_names": names}, gt, camera)
    assert result["anchor_frame_name"] == names[0]
    assert result["context_shape_change_rmse_m"] < 1e-12
    assert result["anchored_orientation_change_deg"]["max"] < 1e-10
    assert result["contexts"]["baseline"]["ate_rmse_m"] < 1e-12
    assert result["contexts"]["shifted"]["ate_rmse_m"] < 1e-12


def test_different_seventeen_frame_reference_origins_share_the_same_nine_frame_anchor():
    names = [f"{index + 1:.6f}.png" for index in range(25)]
    phase = np.linspace(0, 2, 25)
    centers = np.column_stack((phase, np.sin(phase), phase ** 2))
    baseline = payload(names[:17], centers[:17] - centers[0])
    shifted = payload(names[8:], centers[8:] - centers[8], scale=1.7)
    gt = (np.arange(1, 26, dtype=float), centers, np.tile([0., 0., 0., 1.], (25, 1)))
    _, _, _, camera = geometry()
    result = probe.compare_pair(baseline, shifted, {"common_frame_names": names[8:17]}, gt, camera)
    assert result["anchor_frame_name"] == names[8]
    assert result["context_shape_change_rmse_m"] < 1e-12
    assert result["contexts"]["baseline"]["ate_rmse_m"] < 1e-12
    assert result["contexts"]["shifted"]["ate_rmse_m"] < 1e-12


def test_pose_encoding_is_redecoded_instead_of_reusing_rounded_legacy_matrices():
    names, centers, _, _ = geometry()
    data = {**payload(names, centers), "extrinsic": np.zeros((9, 3, 4)), "intrinsic": np.zeros((9, 3, 3))}
    c, rotations, focal = probe.shared_cameras(data, names)
    assert np.allclose(c, centers)
    assert np.allclose(rotations, np.eye(3))
    assert np.allclose(focal, 500.)


def test_resume_rejects_predictions_from_a_different_design(tmp_path):
    names, centers, _, _ = geometry()
    pair = {"id": "test", "shifted_frame_names": names}
    path = probe.saved_path(tmp_path, "omega", pair, "shifted")
    path.parent.mkdir(parents=True)
    np.savez(path, frame_names=np.asarray(names), pose_enc=payload(names, centers)["pose_enc"],
             image_hw=[480, 640], design_sha256=np.asarray("old design"))
    with pytest.raises(ValueError, match="different design"):
        probe.cached_prediction({"design_sha256": "new design"}, tmp_path, "omega", pair, "shifted")


def test_true_shape_change_and_focal_change_remain_visible_after_gauge_removal():
    names, centers, gt, camera = geometry()
    warped = centers.copy()
    warped[4, 1] += .15
    result = probe.compare_pair(payload(names, centers), payload(names, warped, focal=550.),
                                {"common_frame_names": names}, gt, camera)
    assert result["context_shape_change_rmse_m"] > .01
    assert result["contexts"]["shifted"]["ate_rmse_m"] > .01
    assert result["fx_bias_change_pp"] == pytest.approx(10.)


def test_missing_anchor_gt_is_rejected_instead_of_reanchoring_one_context():
    names, centers, gt, camera = geometry()
    truncated_gt = (gt[0][1:], gt[1][1:], gt[2][1:])
    with pytest.raises(ValueError, match="same anchor"):
        probe.compare_pair(payload(names, centers), payload(names, centers),
                           {"common_frame_names": names}, truncated_gt, camera)


def test_preprocessing_mismatch_is_rejected():
    names, centers, gt, camera = geometry()
    shifted = {**payload(names, centers), "image_hw": [448, 592]}
    with pytest.raises(ValueError, match="preprocessing"):
        probe.compare_pair(payload(names, centers), shifted, {"common_frame_names": names}, gt, camera)


def test_analyze_is_cpu_only_and_writes_missing_cache_status(tmp_path, monkeypatch):
    # No forward/cache call can acquire CUDA while analyzing an incomplete design.
    def fail(*args, **kwargs):
        raise AssertionError("CPU analysis must not claim a GPU")
    monkeypatch.setattr(probe.env, "claim_gpu", fail)
    pairs = probe.build_pairs(manifest())
    for pair in pairs:
        pair["baseline_cache"] = {}
        folder = tmp_path / "data" / pair["sequence_directory"]
        folder.mkdir(parents=True, exist_ok=True)
        rows = [f"{i + 1}.0 {i * .01} {i ** 2 * .001} 0 0 0 0 1" for i in range(65)]
        (folder / "groundtruth.txt").write_text("\n".join(rows))
    design = {"pairs": pairs, "design_sha256": "test", "calibration": probe.calibration(None), "limitations": []}
    args = SimpleNamespace(data_root=tmp_path / "data", out=tmp_path / "out", max_gt_diff=.01)
    assert probe.analyze(args, design) == 2
    assert (args.out / "results.json").is_file()
