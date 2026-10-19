"""Validate lock ordering and timeout behavior without querying a real GPU."""
import fcntl
import time
from types import SimpleNamespace

import pytest

from vos import gpu


def test_disabled_cuda_fails_before_lock_or_memory_query(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(gpu, "_take_lock", lambda *_: pytest.fail("CPU mode acquired GPU lock"))
    monkeypatch.setattr(gpu, "free_mb", lambda: pytest.fail("CPU mode queried GPU memory"))
    with pytest.raises(RuntimeError, match="CUDA is disabled"):
        gpu.claim(100)


def test_claim_takes_lock_before_memory_query_and_reuses_it(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("VOS_GPU_RESERVE_MB", "2048")
    monkeypatch.delenv("VOS_GPU_DEVICE", raising=False)
    monkeypatch.setattr(gpu, "_fd", None)
    events = []
    monkeypatch.setattr(gpu, "_take_lock", lambda *_: events.append("lock") or 123)
    monkeypatch.setattr(gpu, "free_mb", lambda: events.append("memory") or 12000)
    assert gpu.device_selector() == "2"
    assert gpu.claim(9000, "test") == 12000
    assert gpu.claim(9000, "test") == 12000
    assert events == ["lock", "memory", "memory"]


def test_memory_timeout_does_not_proceed_with_insufficient_capacity(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("VOS_GPU_WAIT_S", "0")
    monkeypatch.setenv("VOS_GPU_RESERVE_MB", "2048")
    monkeypatch.setattr(gpu, "_fd", 123)
    monkeypatch.setattr(gpu, "free_mb", lambda: 100)
    with pytest.raises(RuntimeError, match="only 100 MiB"):
        gpu.claim(9000, "test")


def test_contended_process_lock_times_out(tmp_path, monkeypatch):
    path = tmp_path / "gpu.lock"
    monkeypatch.setenv("VOS_GPU_LOCK", str(path))
    with path.open("w+") as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="still held"):
            gpu._take_lock("contender", time.monotonic() - 1)


def test_memory_query_uses_visible_device_and_has_no_torch_fallback(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-example")
    monkeypatch.delenv("VOS_GPU_DEVICE", raising=False)
    calls = []
    def fake_run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="12000\n")
    monkeypatch.setattr(gpu.subprocess, "run", fake_run)
    assert gpu.free_mb() == 12000
    assert calls[0][-2:] == ["-i", "GPU-example"]
    monkeypatch.setattr(gpu.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(RuntimeError, match="nvidia-smi"):
        gpu.free_mb()
