"""Linux process-wide CUDA lock and free-memory check, independent of Torch.

The lock is held until process exit. Imports and CPU preflight do not query
CUDA. VOS_GPU_LOCK overrides the lock path, VOS_GPU_WAIT_S sets an optional
timeout, VOS_GPU_RESERVE_MB overrides the default 2048 MiB reserve.
VOS_GPU_DEVICE is the nvidia-smi selector corresponding to Torch's logical
CUDA device zero; the first CUDA_VISIBLE_DEVICES entry or 0 is the default.
All cooperating jobs must use the same lock path.
VOS_GPU_REQUIRE_IDLE=1 also refuses concurrent compute processes outside that
lock. It does not stop them or reserve the display's GPU memory.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

POLL_S = 5.0
_fd = None


def lock_path() -> Path:
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return Path(os.environ.get("VOS_GPU_LOCK") or cache / "vggt-omega-slam" / "gpu.lock").expanduser()


def device_selector() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and (not visible.strip() or visible.strip() == "-1"):
        raise RuntimeError("CUDA is disabled by CUDA_VISIBLE_DEVICES; GPU execution cannot start.")
    return os.environ.get("VOS_GPU_DEVICE") or (visible.split(",")[0].strip() if visible else "0")


def free_mb() -> int:
    """Query free memory in MiB without opening a CUDA context."""
    selector = device_selector()
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", selector],
                                capture_output=True, text=True, timeout=20, check=True)
        return int(result.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise RuntimeError(f"Cannot query NVIDIA GPU {selector!r} with nvidia-smi. "
                           "Check the NVIDIA driver and VOS_GPU_DEVICE/CUDA_VISIBLE_DEVICES.") from exc


def other_compute_pids() -> list[int]:
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
                                 "-i", device_selector()], capture_output=True, text=True, timeout=20, check=True)
        return sorted({int(line.strip()) for line in result.stdout.splitlines() if line.strip()} - {os.getpid()})
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError("Cannot verify other GPU compute processes with nvidia-smi") from exc


def ensure_exclusive() -> None:
    """Before a new forward, fail safely if a non-cooperating CUDA job appeared."""
    if os.environ.get("VOS_GPU_REQUIRE_IDLE") == "1":
        others = other_compute_pids()
        if others:
            raise RuntimeError(f"Other GPU compute processes appeared: {others}; leaving them running")


def allocator_budget_bytes(requested_bytes: int, total_bytes: int) -> int:
    """Budget the allocator after context creation, retaining reserve plus headroom."""
    reserve = int(os.environ.get("VOS_GPU_RESERVE_MB", "2048"))
    # cuBLAS/cuDNN and display allocations are outside PyTorch's allocator.
    available = (free_mb() - reserve - 256) * 2**20
    if available <= 0:
        raise RuntimeError("No GPU allocator budget remains after the display reserve")
    return min(int(requested_bytes), int(total_bytes * .95), available)


def _holder(fd) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    return os.read(fd, 400).decode(errors="replace").strip() or "unknown"


def _take_lock(who, deadline):
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    announced = False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not announced:
                print(f"[gpu] {who}: waiting for {path}, held by {_holder(fd)}", file=sys.stderr, flush=True)
                announced = True
            if deadline is None:
                # Queue in the kernel instead of missing short release windows
                # between another project's consecutive jobs. Timed waits keep
                # the interruptible polling path below.
                fcntl.flock(fd, fcntl.LOCK_EX)
                break
            if deadline is not None and time.monotonic() >= deadline:
                os.close(fd)
                raise RuntimeError(f"GPU lock {path} is still held after VOS_GPU_WAIT_S") from None
            time.sleep(POLL_S)
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, f"pid {os.getpid()}: {who}\n".encode())
    return fd


def claim(need_mb: int, who: str = "") -> int:
    """Take the shared lock, then wait for ``need_mb + reserve`` free MiB."""
    global _fd
    device_selector()
    reserve = int(os.environ.get("VOS_GPU_RESERVE_MB", "2048"))
    wait = os.environ.get("VOS_GPU_WAIT_S")
    if need_mb < 0 or reserve < 0 or (wait and float(wait) < 0):
        raise ValueError("GPU memory needs, reserve and timeout must be nonnegative")
    deadline = time.monotonic() + float(wait) if wait else None
    if _fd is None:
        _fd = _take_lock(who, deadline)
    announced = False
    while True:
        others = other_compute_pids() if os.environ.get("VOS_GPU_REQUIRE_IDLE") == "1" else []
        free = free_mb()
        if not others and free >= need_mb + reserve:
            return free
        if not announced:
            print(f"[gpu] {who}: {free} MiB free, other compute PIDs {others}; "
                  f"waiting for {need_mb} + {reserve} MiB", file=sys.stderr, flush=True)
            announced = True
        if deadline is not None and time.monotonic() >= deadline:
            if others:
                raise RuntimeError(f"Other GPU compute processes still active after VOS_GPU_WAIT_S: {others}")
            raise RuntimeError(f"{who}: only {free} MiB free after VOS_GPU_WAIT_S; need {need_mb} + {reserve} MiB")
        time.sleep(POLL_S)


def claim_for(device, need_mb: int, who: str = "") -> None:
    if str(device).startswith("cuda"):
        claim(need_mb, who)
