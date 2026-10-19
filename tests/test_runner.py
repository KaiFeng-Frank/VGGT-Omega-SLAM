"""(e) runner --preflight-only passes on CPU; input handling matches upstream main.py."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from vos import env as vos_env

ROOT = Path(__file__).resolve().parents[1]


def test_preflight_only_passes_and_checks_upstream_provenance(tmp_path):
    try:
        vos_env.require_sources("omega")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    gtsam = pytest.importorskip("gtsam", reason="full preflight needs GTSAM with SL(4)")
    if not all(hasattr(gtsam, name) for name in ("SL4", "PriorFactorSL4", "BetweenFactorSL4")):
        pytest.skip("full preflight needs GTSAM with SL(4) Python bindings")
    out = tmp_path / "pre"
    env = dict(os.environ, OMP_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([sys.executable, str(ROOT / "run_slam.py"), "--preflight-only", "--output", str(out)],
                          capture_output=True, text=True, env=env, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert "preflight_passed" in proc.stdout
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["status"] == "preflight_passed" and meta["gpu_used"] is False and meta["model_loaded"] is False
    assert meta["upstream"]["provenance_check"]["unmodified"] is True
    assert meta["upstream"]["provenance_check"]["files_checked"] == 25


def test_runner_refuses_existing_output(tmp_path):
    proc = subprocess.run([sys.executable, str(ROOT / "run_slam.py"), str(tmp_path), "--output", str(tmp_path)],
                          capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert proc.returncode != 0 and "exists" in proc.stderr


def test_input_resolution_and_listing_follow_upstream(tmp_path):
    import run_slam
    utils = pytest.importorskip("vggt_slam.slam_utils", reason="needs pinned SLAM sources")

    seq = tmp_path / "rgbd_dataset_x"
    (seq / "rgb").mkdir(parents=True)
    (seq / "groundtruth.txt").write_text("# gt\n")
    names = ["1305031452.823674.png", "1305031452.791720.png", "1305031453.100000.png", "rgb_depth.png", "x.txt", "a.db"]
    for n in names:
        (seq / "rgb" / n).write_bytes(b"")
    image_dir, kind, gt = run_slam.resolve_input(seq)
    assert (image_dir, kind, gt) == (seq / "rgb", "tum", seq / "groundtruth.txt")
    listed = [Path(p).name for p in run_slam.list_images(image_dir, utils)]
    assert listed == ["1305031452.791720.png", "1305031452.823674.png", "1305031453.100000.png"]  # numeric order, filters
    folder = tmp_path / "imgs"
    folder.mkdir()
    for i in (10, 2, 1):
        (folder / f"frame_{i:04d}.jpg").write_bytes(b"")
    image_dir, kind, gt = run_slam.resolve_input(folder)
    assert kind == "folder" and gt is None
    assert [Path(p).name for p in run_slam.list_images(image_dir, utils)] == ["frame_0001.jpg", "frame_0002.jpg", "frame_0010.jpg"]


def test_keyframe_stamp_regex_matches_upstream_submap():
    import re

    Submap = pytest.importorskip("vggt_slam.submap", reason="needs pinned SLAM sources and runtime dependencies").Submap

    s = Submap(0)
    names = ["/x/1305031452.791720.png", "/x/frame_0042.jpg"]
    s.set_frame_ids(names)
    ours = [float(re.search(r"\d+(?:\.\d+)?", Path(n).name).group()) for n in names]
    assert s.get_frame_ids() == ours


def test_failure_writes_failed_metadata_with_traceback(tmp_path):
    out = tmp_path / "fail"
    proc = subprocess.run([sys.executable, str(ROOT / "run_slam.py"), str(tmp_path / "no_such_dir"), "--output", str(out)],
                          capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), timeout=300)
    assert proc.returncode == 1
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["status"] == "failed" and "need at least 2 images" in meta["error"]
    assert "Traceback" in meta["traceback"] and "gpu" not in meta


def test_missing_sources_fail_clearly_without_gpu_access(tmp_path):
    out = tmp_path / "missing_sources"
    process_env = dict(os.environ, CUDA_VISIBLE_DEVICES="", VOS_DEPS=str(tmp_path / "absent"),
                       VOS_GPU_LOCK=str(tmp_path / "must_not_exist.lock"))
    for source in vos_env.SOURCES.values():
        process_env[source["variable"]] = str(tmp_path / "absent" / source["variable"])
    proc = subprocess.run([sys.executable, str(ROOT / "run_slam.py"), "--preflight-only", "--output", str(out)],
                          capture_output=True, text=True, env=process_env, timeout=60)
    assert proc.returncode == 1
    meta = json.loads((out / "metadata.json").read_text())
    assert "Missing pinned upstream sources" in meta["error"]
    assert "tools/setup_sources.py" in meta["error"]
    assert "gpu" not in meta and not (tmp_path / "must_not_exist.lock").exists()


def test_late_failure_keeps_partial_outputs(tmp_path, monkeypatch):
    import types

    import run_slam

    def fake_run(args, report, out, ctx):
        written = []
        fake_map = types.SimpleNamespace(
            write_poses_to_file=lambda path, graph, kitti_format=False: Path(path).write_text("1.0 0 0 0 0 0 0 1\n"))
        ctx["solver"] = types.SimpleNamespace(map=fake_map, graph=None)
        ctx["keyframes"].append((0, "frame_0001.jpg", 1.0))
        ctx["updates"].append({"update": 0, "frame_names": ["frame_0001.jpg"]})
        raise RuntimeError("boom after the first submap")

    monkeypatch.setattr(run_slam, "run", fake_run)
    out = tmp_path / "late"
    assert run_slam.main(["x", "--output", str(out)]) == 1
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["status"] == "failed" and meta["error"] == "RuntimeError: boom after the first submap"
    assert sorted(meta["partial_outputs"]) == ["keyframes.csv", "poses_partial.tum"]
    assert (out / "poses_partial.tum").read_text().startswith("1.0")
    assert "0" in (out / "keyframes.csv").read_text().splitlines()[1].split(",")[-1]


def test_retrieval_checks_gpu_before_descriptor_forward(monkeypatch):
    from types import SimpleNamespace
    import run_slam
    from vos import gpu

    calls = []
    retrieval = SimpleNamespace(
        find_loop_closures=lambda *args, **kwargs: [],
        get_batch_descriptors=lambda images: calls.append(images) or "descriptors",
    )
    solver = SimpleNamespace(map=SimpleNamespace(retrieve_best_score_frame=lambda *a, **kw: None),
                             image_retrieval=retrieval)
    run_slam.RetrievalRecorder(solver)
    monkeypatch.setattr(gpu, "ensure_exclusive", lambda: None)
    assert retrieval.get_batch_descriptors("first") == "descriptors"

    def busy():
        raise RuntimeError("Other GPU compute processes appeared")

    monkeypatch.setattr(gpu, "ensure_exclusive", busy)
    with pytest.raises(RuntimeError, match="Other GPU"):
        retrieval.get_batch_descriptors("blocked")
    assert calls == ["first"]
