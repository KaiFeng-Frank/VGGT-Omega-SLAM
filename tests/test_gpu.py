"""Validate lock ordering and timeout behavior without querying a real GPU."""
import fcntl
import time
from types import SimpleNamespace

import pytest

from vos import gpu


@pytest.fixture(autouse=True)
def isolate_optional_idle_policy(monkeypatch):
    monkeypatch.delenv("VOS_GPU_REQUIRE_IDLE", raising=False)


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


def test_unlimited_wait_queues_on_lock_without_polling(tmp_path, monkeypatch):
    monkeypatch.setenv("VOS_GPU_LOCK", str(tmp_path / "gpu.lock"))
    calls = []
    def flock(fd, operation):
        calls.append(operation)
        if operation & fcntl.LOCK_NB:
            raise BlockingIOError()
    monkeypatch.setattr(gpu.fcntl, "flock", flock)
    monkeypatch.setattr(gpu.time, "sleep", lambda *_: pytest.fail("unlimited lock wait polled"))
    fd = gpu._take_lock("queued job", None)
    try:
        assert calls == [fcntl.LOCK_EX | fcntl.LOCK_NB, fcntl.LOCK_EX]
        assert "queued job" in (tmp_path / "gpu.lock").read_text()
    finally:
        gpu.os.close(fd)


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


def test_idle_policy_waits_even_when_memory_is_available(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("VOS_GPU_REQUIRE_IDLE", "1")
    monkeypatch.setenv("VOS_GPU_WAIT_S", "0")
    monkeypatch.setattr(gpu, "_fd", 123)
    monkeypatch.setattr(gpu, "free_mb", lambda: 24000)
    monkeypatch.setattr(gpu, "other_compute_pids", lambda: [987])
    with pytest.raises(RuntimeError, match="still active"):
        gpu.claim(9000, "test")
    with pytest.raises(RuntimeError, match="leaving them running"):
        gpu.ensure_exclusive()


def test_compute_process_query_excludes_only_this_process(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=f"{gpu.os.getpid()}\n987\n987\n"))
    assert gpu.other_compute_pids() == [987]


def test_allocator_budget_preserves_reserve_and_external_allocation_margin(monkeypatch):
    monkeypatch.setenv("VOS_GPU_RESERVE_MB", "2048")
    monkeypatch.setattr(gpu, "free_mb", lambda: 11000)
    assert gpu.allocator_budget_bytes(10 * 2**30, 12 * 2**30) == (11000 - 2048 - 256) * 2**20
    assert gpu.allocator_budget_bytes(4 * 2**30, 12 * 2**30) == 4 * 2**30
    monkeypatch.setattr(gpu, "free_mb", lambda: 2000)
    with pytest.raises(RuntimeError, match="No GPU allocator budget"):
        gpu.allocator_budget_bytes(10 * 2**30, 12 * 2**30)
