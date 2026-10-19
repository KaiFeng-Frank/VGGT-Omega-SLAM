"""Configurable paths and offline imports; upstream source trees stay unmodified.

Importing this module or calling ``prepare`` never acquires a GPU lock or opens
a CUDA context. Source downloads and Python/GTSAM installation are separate;
see ``tools/setup_sources.py`` and the setup documentation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

# The upstream trees are read-only inputs: never let Python write .pyc files into them.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

PROJECT = Path(__file__).resolve().parents[1]

def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name) or default).expanduser().resolve()


DEPS = _path("VOS_DEPS", PROJECT / "deps")
PYLIB = _path("VOS_PYLIB", DEPS / "python")  # optional custom GTSAM Python build
SLAM_SRC = _path("VOS_SLAM_SRC", DEPS / "vggt_slam2_upstream_35327ac")
VGGT_SRC = _path("VOS_VGGT_SRC", DEPS / "vggt_spark_upstream_6e6e161")
SALAD_SRC = _path("VOS_SALAD_SRC", DEPS / "salad_upstream_33ca9c0")
DINOV2_SRC = _path("VOS_DINOV2_SRC", DEPS / "dinov2_upstream_7764ea0")
OMEGA_SRC = _path("VOS_OMEGA_SRC", DEPS / "vggt_omega_upstream_48b23c8")
CHECKPOINTS = _path("VOS_CHECKPOINTS", PROJECT / "checkpoints")
SALAD_CKPT = _path("VOS_SALAD_CKPT", CHECKPOINTS / "dino_salad.ckpt")
SALAD_SHA256 = "6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada"
VGGT_CKPT = _path("VOS_VGGT_CKPT", CHECKPOINTS / "vggt_original_1b.pt")
VGGT_SHA256 = "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0"
OMEGA_CKPT = _path("VOS_OMEGA_CKPT", CHECKPOINTS / "vggt_omega_1b_512.pt")
OMEGA_CKPT_BYTES = 4_576_706_117
OMEGA_SHA256 = "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"
SLAM_COMMIT = "35327ac28b7d193df9ccc39ba6346052bb6f1207"

TUM_ROOT = _path("VOS_TUM_ROOT", PROJECT / "data" / "tum_rgbd")
OFFICE_ROOT = _path("VOS_OFFICE_ROOT", PROJECT / "data" / "office_loop")
SIM_CONFIG = _path("VOS_SIM_CONFIG", PROJECT / "config" / "omega_similarity.json")
SLAM_PROVENANCE = _path("VOS_SLAM_PROVENANCE", PROJECT / "config" / "upstream_slam.provenance.json")

# The same pinned revisions used for the reported experiments.
SOURCES = {
    "slam": {"url": "https://github.com/MIT-SPARK/VGGT-SLAM.git", "commit": SLAM_COMMIT,
             "path": SLAM_SRC, "marker": "vggt_slam/solver.py", "variable": "VOS_SLAM_SRC"},
    "vggt": {"url": "https://github.com/MIT-SPARK/VGGT_SPARK.git", "commit": "6e6e16107b88e8e76c751826af10d4295d87ecd2",
             "path": VGGT_SRC, "marker": "vggt/models/vggt.py", "variable": "VOS_VGGT_SRC"},
    "salad": {"url": "https://github.com/Dominic101/salad.git", "commit": "33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d",
              "path": SALAD_SRC, "marker": "hubconf.py", "variable": "VOS_SALAD_SRC"},
    "dinov2": {"url": "https://github.com/facebookresearch/dinov2.git", "commit": "7764ea0f912e53c92e82eb78a2a1631e92725fc8",
               "path": DINOV2_SRC, "marker": "hubconf.py", "variable": "VOS_DINOV2_SRC"},
    "omega": {"url": "https://github.com/facebookresearch/vggt-omega.git", "commit": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
              "path": OMEGA_SRC, "marker": "vggt_omega/models/vggt_omega.py", "variable": "VOS_OMEGA_SRC"},
}


def sha256(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(upstream: bool = True) -> None:
    """Set offline imports, loading OpenCV before torchvision when available.

    Missing runtime packages do not prevent running pure CPU tests or --help.
    The runner's preflight reports missing sources and dependencies.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if PYLIB.is_dir() and str(PYLIB) not in sys.path:
        sys.path.insert(0, str(PYLIB))
    if importlib.util.find_spec("cv2") is not None:
        import cv2  # noqa: F401  (must precede torchvision)

    if upstream:
        for path in (OMEGA_SRC, SALAD_SRC, VGGT_SRC, SLAM_SRC):  # last inserted = first searched
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
    if str(PROJECT) not in sys.path:
        sys.path.append(str(PROJECT))


def require_sources(backend="omega") -> None:
    names = ("slam", "vggt", "salad", "dinov2") + (("omega",) if backend == "omega" else ())
    missing = [f"{name}: {SOURCES[name]['path']} ({SOURCES[name]['variable']})"
               for name in names if not (SOURCES[name]["path"] / SOURCES[name]["marker"]).is_file()]
    if missing:
        raise RuntimeError("Missing pinned upstream sources:\n  " + "\n  ".join(missing)
                           + "\nRun python tools/setup_sources.py, or set the VOS_*_SRC paths. "
                           "Python dependencies and GTSAM with SL(4) must be installed separately.")


def require_checkpoints(backend="omega") -> None:
    paths = [("VOS_SALAD_CKPT", SALAD_CKPT),
             ("VOS_OMEGA_CKPT", OMEGA_CKPT) if backend == "omega" else ("VOS_VGGT_CKPT", VGGT_CKPT)]
    for variable, path in paths:
        if not path.is_file():
            raise RuntimeError(f"Missing checkpoint {path}. Download the approved upstream weights and set {variable}; "
                               "source setup does not download weights.")


def install_headless_viewer() -> None:
    """Replace only the rendering module; no server, window or socket is created.

    Must run before `vggt_slam.solver` is imported.
    """
    module = types.ModuleType("vggt_slam.viewer")

    class Viewer:
        def __init__(self):
            self.server = types.SimpleNamespace(scene=types.SimpleNamespace(add_point_cloud=lambda **_: None))

        def visualize_frames(self, *args, **kwargs):
            return None

        def visualize_obb(self, *args, **kwargs):
            return None

    module.Viewer = Viewer
    sys.modules["vggt_slam.viewer"] = module


def configure_offline_hub(torch, output, salad_checkpoint, dinov2_source):
    """Keep original SALAD load_model; suppress its redundant DINO download.

    The complete SALAD checkpoint strictly supplies the backbone weights, so only the initial, immediately overwritten
    pretrained DINOv2 download is replaced with the identical local architecture initialisation.
    """
    hub = Path(output) / "torch_hub"
    (hub / "checkpoints").mkdir(parents=True)
    (hub / "checkpoints" / "dino_salad.ckpt").symlink_to(Path(salad_checkpoint).resolve())
    torch.hub.set_dir(str(hub))
    original_load = torch.hub.load
    calls = []

    def local_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir != "facebookresearch/dinov2" or model != "dinov2_vitb14":
            raise RuntimeError(f"Offline wrapper blocked unexpected torch.hub request {repo_or_dir}/{model}")
        calls.append({"requested_repository": repo_or_dir, "model": model, "local_source": str(dinov2_source),
                      "initial_pretrained": False,
                      "reason": "complete SALAD checkpoint strictly supplies backbone weights"})
        for key in ("source", "trust_repo", "skip_validation", "force_reload", "pretrained"):
            kwargs.pop(key, None)
        return original_load(str(dinov2_source), model, *args, source="local", pretrained=False, **kwargs)

    def blocked_download(*args, **kwargs):
        raise RuntimeError("Network model download blocked: provide explicit local checkpoints")

    torch.hub.load = local_load
    torch.hub.load_state_dict_from_url = blocked_download
    torch.hub.download_url_to_file = blocked_download
    return calls


def claim_gpu(need_mb: int, who: str):
    """Acquire the standalone process lock before the first CUDA operation."""
    from . import gpu
    return gpu.claim(need_mb, who)


def verify_upstream_snapshot():
    """Verify SLAM against the bundled, path-free checksum manifest."""
    if not SLAM_PROVENANCE.is_file():
        raise RuntimeError(f"Missing SLAM provenance manifest: {SLAM_PROVENANCE}")
    provenance = json.loads(SLAM_PROVENANCE.read_text())
    if provenance["commit"] != SLAM_COMMIT:
        raise RuntimeError(f"SLAM provenance must describe pinned commit {SLAM_COMMIT}")
    mismatches = []
    for entry in provenance["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid provenance path: {relative}")
        path = SLAM_SRC / relative
        if not path.is_file() or sha256(path) != entry["sha256"]:
            mismatches.append(entry["path"])
    return {"commit": provenance["commit"], "files_checked": len(provenance["files"]),
            "mismatches": mismatches, "unmodified": not mismatches}
