from types import SimpleNamespace

import numpy as np
import pytest

from vos.prediction_cache import load_predictions, save_predictions


def test_lossless_arrays_and_loop_identities_survive_cache(tmp_path):
    pred = {"extrinsic": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            "intrinsic": np.eye(3, dtype=np.float64)[None],
            "depth_conf": np.array([1.001, 12.345], dtype=np.float32),
            "detected_loops": [SimpleNamespace(query_submap_id=18, query_submap_frame=2,
                                               detected_submap_id=0, detected_submap_frame=6,
                                               similarity_score=np.float64(.4))],
            "image_match_ratio": None, "frames_lc_names": ["/data/1.0.png", "/data/2.0.png"]}
    path = tmp_path / "00000.npz"
    save_predictions(path, pred, ["/data/1.0.png", "/data/2.0.png"], 18)
    restored, metadata = load_predictions(path)
    for key in ("extrinsic", "intrinsic", "depth_conf"):
        assert restored[key].dtype == pred[key].dtype
        np.testing.assert_array_equal(restored[key], pred[key])
    assert vars(restored["detected_loops"][0]) == vars(pred["detected_loops"][0])
    assert metadata["frame_names"] == restored["frames_lc_names"] == ["1.0.png", "2.0.png"]
    assert metadata["submap_id"] == 18
    with pytest.raises(FileExistsError):
        save_predictions(path, pred, [], 19)


def test_cache_rejects_pickle_requiring_arrays(tmp_path):
    with pytest.raises(TypeError, match="Object array"):
        save_predictions(tmp_path / "bad.npz", {"bad": np.array([object()])}, [], 0)


def test_upstream_namedtuple_loop_serialization(tmp_path):
    from collections import namedtuple
    LoopMatch = namedtuple("LoopMatch", "query_submap_id query_submap_frame detected_submap_id detected_submap_frame similarity_score")
    loop = LoopMatch(17, 3, 0, 5, .4)
    path = tmp_path / "loop.npz"
    save_predictions(path, {"detected_loops": [loop]}, ["1.png"], 17)
    restored, _ = load_predictions(path)
    assert vars(restored["detected_loops"][0]) == loop._asdict()


def test_loop_frames_restore_as_cpu_tensor(tmp_path):
    torch = pytest.importorskip("torch")
    frames = torch.randn(2, 3, 4, 5)
    path = tmp_path / "tensor.npz"
    save_predictions(path, {"frames_lc": frames, "detected_loops": []}, ["1.png", "2.png"], 0)
    restored, _ = load_predictions(path)
    assert restored["frames_lc"].device.type == "cpu"
    assert torch.equal(restored["frames_lc"], frames)


def test_vggt_bfloat16_tokens_and_nested_camera_iterations_are_lossless(tmp_path):
    torch = pytest.importorskip("torch")
    # Include signed zero, infinities and a NaN payload; check bits, not allclose.
    bits = torch.tensor([0, 32768, 16256, 32640, 65408, 32705], dtype=torch.uint16)
    tokens = bits.view(torch.bfloat16).reshape(2, 3).T  # noncontiguous source
    cameras = [torch.arange(18).reshape(2, 9).to(torch.bfloat16), torch.randn(2, 9)]
    path = tmp_path / "vggt.npz"
    save_predictions(path, {"target_tokens": tokens, "pose_enc_list": cameras,
                            "aux": ({"data": np.array([2.0], dtype=np.float16)}, None)}, [], 0)
    restored, _ = load_predictions(path)
    assert restored["target_tokens"].dtype == torch.bfloat16
    assert torch.equal(restored["target_tokens"].view(torch.uint16), tokens.view(torch.uint16))
    assert isinstance(restored["pose_enc_list"], list)
    for got, expected in zip(restored["pose_enc_list"], cameras):
        assert got.dtype == expected.dtype and torch.equal(got, expected)
    assert isinstance(restored["aux"], tuple)
    assert restored["aux"][0]["data"].dtype == np.float16
    assert not any(key.startswith("__nested_array_") for key in restored)
    raw, metadata = load_predictions(path, restore_torch=False)
    assert raw["target_tokens"].dtype == np.uint16
    assert metadata["tensor_storage"]["target_tokens"] == "bfloat16_bits"


def test_old_v1_cache_without_nested_storage_fields_remains_readable(tmp_path):
    import json
    path = tmp_path / "old.npz"
    metadata = {"schema": "vos-prediction-cache-v1", "submap_id": 0,
                "frame_names": [], "tensor_keys": [], "values": {"detected_loops": []}}
    np.savez(path, depth=np.array([1.5], dtype=np.float32), __metadata__=json.dumps(metadata))
    restored, _ = load_predictions(path)
    np.testing.assert_array_equal(restored["depth"], [1.5])
