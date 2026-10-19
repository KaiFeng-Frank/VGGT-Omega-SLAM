"""Path configuration and integrity checks need neither upstream installs nor CUDA."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from vos import env


def test_path_overrides_are_independent_and_import_does_not_lock_gpu(tmp_path):
    process_env = dict(os.environ, VOS_DEPS=str(tmp_path / "deps"), VOS_CHECKPOINTS=str(tmp_path / "weights"),
                       VOS_OMEGA_SRC=str(tmp_path / "omega"), VOS_TUM_ROOT=str(tmp_path / "tum"),
                       VOS_OFFICE_ROOT=str(tmp_path / "office"), VOS_GPU_LOCK=str(tmp_path / "gpu.lock"))
    for key in ("VOS_SLAM_SRC", "VOS_SALAD_CKPT"):
        process_env.pop(key, None)
    script = ("import json; from vos import env; env.prepare(); "
              "print(json.dumps({k:str(getattr(env,k)) for k in "
              "['DEPS','SLAM_SRC','OMEGA_SRC','SALAD_CKPT','TUM_ROOT','OFFICE_ROOT']}))")
    proc = subprocess.run([sys.executable, "-B", "-c", script], cwd=env.PROJECT, env=process_env,
                          capture_output=True, text=True, check=True, timeout=60)
    paths = json.loads(proc.stdout)
    assert paths["SLAM_SRC"] == str(tmp_path / "deps" / "vggt_slam2_upstream_35327ac")
    assert paths["OMEGA_SRC"] == str(tmp_path / "omega")
    assert paths["SALAD_CKPT"] == str(tmp_path / "weights" / "dino_salad.ckpt")
    assert paths["TUM_ROOT"] == str(tmp_path / "tum")
    assert paths["OFFICE_ROOT"] == str(tmp_path / "office")
    assert not (tmp_path / "gpu.lock").exists()


def test_snapshot_verification_detects_modified_file(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    entry = source / "solver.py"
    entry.write_text("original\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"commit": env.SLAM_COMMIT,
                                    "files": [{"path": "solver.py", "sha256": env.sha256(entry)}]}))
    monkeypatch.setattr(env, "SLAM_SRC", source)
    monkeypatch.setattr(env, "SLAM_PROVENANCE", manifest)
    assert env.verify_upstream_snapshot()["unmodified"] is True
    entry.write_text("modified\n")
    assert env.verify_upstream_snapshot()["mismatches"] == ["solver.py"]


def test_missing_source_and_checkpoint_errors_are_actionable(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "SOURCES", {key: {**spec, "path": tmp_path / key} for key, spec in env.SOURCES.items()})
    with pytest.raises(RuntimeError, match="tools/setup_sources.py"):
        env.require_sources()
    monkeypatch.setattr(env, "SALAD_CKPT", tmp_path / "missing.ckpt")
    with pytest.raises(RuntimeError, match="VOS_SALAD_CKPT"):
        env.require_checkpoints()
