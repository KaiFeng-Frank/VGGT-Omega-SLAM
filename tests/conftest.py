"""CPU-only test setup: project + upstream source paths, no CUDA, no GPU lock."""
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # set before any third-party import
os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vos import env  # noqa: E402

env.prepare()  # optional GTSAM path, cv2 first, available upstream sources
