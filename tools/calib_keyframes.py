#!/usr/bin/env python3
"""Calibration step 1 (CPU): the ordered keyframe list K of office_loop, exactly as the SLAM sees it.

Runs the unmodified upstream LK selector (vggt_slam.frame_overlap.FrameTracker.compute_disparity with
min_disparity 50) over all images in upstream order, as main.py / run_slam.py do. Cross-checks the
result against the keyframes of the earlier 150-frame smoke runs (same selector, same prefix).

  OMP_NUM_THREADS=4 <python> tools/calib_keyframes.py
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vos import env  # noqa: E402

env.prepare()

import cv2  # noqa: E402

from vos import calibration as cal  # noqa: E402

IMAGES = env.OFFICE_ROOT
OUT = env.PROJECT / "runs" / "calibration"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, default=IMAGES)
    p.add_argument("--min-disparity", type=float, default=cal.MIN_DISPARITY)
    p.add_argument("--output", type=Path, default=OUT / "keyframes.json")
    args = p.parse_args()
    if args.output.exists():
        sys.exit(f"{args.output} exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from vggt_slam import slam_utils as utils
    from vggt_slam.frame_overlap import FrameTracker

    import run_slam  # listing identical to upstream main.py (same helper the runner uses)

    names = run_slam.list_images(args.images, utils)
    tracker = FrameTracker()
    keyframes = []
    for index, name in enumerate(names):
        img = cv2.imread(name)
        if img is None:
            raise ValueError(f"cannot decode {name}")
        if tracker.compute_disparity(img, args.min_disparity, False):
            keyframes.append({"kf": len(keyframes), "image_index": index, "image": os.path.basename(name),
                              "path": str(Path(name).resolve())})

    check = {}
    for tag in ("smoke_vggt_office150", "smoke_omega_office150"):
        csv_path = env.PROJECT / "runs" / tag / "keyframes.csv"
        if csv_path.is_file():
            with csv_path.open() as f:
                smoke = [row["image_name"] for row in csv.DictReader(f)]
            mine = [k["image"] for k in keyframes[: len(smoke)]]
            check[tag] = {"n_smoke_keyframes": len(smoke), "prefix_identical": mine == smoke}
    result = {"dataset": cal.DATASET, "image_dir": str(args.images), "n_images": len(names),
              "min_disparity": args.min_disparity, "n_keyframes": len(keyframes), "smoke_prefix_check": check,
              "keyframes": keyframes}
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "keyframes"}, indent=1))


if __name__ == "__main__":
    main()
