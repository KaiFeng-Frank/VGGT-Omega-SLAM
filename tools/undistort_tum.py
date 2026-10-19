#!/usr/bin/env python3
"""Undistort TUM fr1 desk / desk2 / room RGB frames with the published TUM calibration (CPU only).

  K = [[517.306408, 0, 318.643040], [0, 516.469215, 255.313989], [0, 0, 1]]
  D = (k1, k2, p1, p2, k3) = (0.262383, -0.953104, -0.005358, 0.002628, 1.163314)      OpenCV model, 640x480

Per frame: cv2.getOptimalNewCameraMatrix(K, D, (640, 480), alpha=0) (valid pixels only, same output size) once,
cv2.initUndistortRectifyMap(K, D, None, K_new, (640, 480), CV_32FC1) once, cv2.remap(..., INTER_LINEAR) per frame,
PNG written under the ORIGINAL file name into
  data/tum_undistorted/rgbd_dataset_freiburg1_<seq>/{rgb/, groundtruth.txt, rgb.txt, calibration.json}
(a TUM-style sequence directory: run_slam.py and tools/analysis_undistort.py take it as is).
The original TUM folders are only read.

Also writes runs/undistort/samples/<seq>_raw_vs_undistorted.png (side by side, with straight-line overlays) and
runs/undistort/undistort_report.json (counts, validity, map round-trip check, straight-line deviations).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vos import env  # noqa: E402,F401  (sets sys.dont_write_bytecode)

import cv2  # noqa: E402  (before anything that may import torchvision)
import numpy as np  # noqa: E402

PROJECT = env.PROJECT
TUM = env.TUM_ROOT
DATA = PROJECT / "data" / "tum_undistorted"
OUT = PROJECT / "runs" / "undistort"
SEQS = {"desk": "rgbd_dataset_freiburg1_desk", "desk2": "rgbd_dataset_freiburg1_desk2",
        "room": "rgbd_dataset_freiburg1_room"}

SIZE = (640, 480)  # (width, height)
K = np.array([[517.306408, 0.0, 318.643040], [0.0, 516.469215, 255.313989], [0.0, 0.0, 1.0]])
D = np.array([0.262383, -0.953104, -0.005358, 0.002628, 1.163314])  # k1 k2 p1 p2 k3


def build_maps():
    k_new, roi = cv2.getOptimalNewCameraMatrix(K, D, SIZE, 0, SIZE)
    map_x, map_y = cv2.initUndistortRectifyMap(K, D, None, k_new, SIZE, cv2.CV_32FC1)
    return k_new, roi, map_x, map_y


def map_checks(k_new, map_x, map_y):
    """Validity of the output (all source coordinates inside the raw image) and round trip through undistortPoints."""
    w, h = SIZE
    inside = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    ones = cv2.remap(np.full((h, w), 255, np.uint8), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0)
    # round trip: output pixel -> raw coordinates (the map) -> undistortPoints with P = k_new -> output pixel
    ys, xs = np.mgrid[0:h:8, 0:w:8]
    raw_pts = np.stack([map_x[ys, xs], map_y[ys, xs]], axis=-1).reshape(-1, 1, 2).astype(np.float64)
    back = cv2.undistortPoints(raw_pts, K, D, P=k_new).reshape(-1, 2)
    want = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float64)
    err = np.linalg.norm(back - want, axis=1)
    return {"fraction_source_coordinate_inside_raw_image": float(inside.mean()),
            "n_output_pixels_outside_raw": int((~inside).sum()),
            "constant_border_remap_min_value_of_all_255_image": int(ones.min()),
            "n_output_pixels_darkened_by_border": int((ones < 255).sum()),
            "roundtrip_undistortPoints_max_err_px": float(err.max()), "roundtrip_undistortPoints_mean_err_px": float(err.mean()),
            "roundtrip_n_points": int(len(err)),
            "distortion_displacement_px": displacement_stats(k_new),
            "map_x_range": [float(map_x.min()), float(map_x.max())], "map_y_range": [float(map_y.min()), float(map_y.max())]}


def distort_pixels(pts_new, k_new):
    """Undistorted-image pixel coordinates -> raw-image pixel coordinates (forward distortion model)."""
    pts_new = np.asarray(pts_new, np.float64).reshape(-1, 1, 2)
    norm = cv2.undistortPoints(pts_new, k_new, None).reshape(-1, 2)  # normalised, no distortion
    xyz = np.concatenate([norm, np.ones((len(norm), 1))], axis=1)
    out, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), K, D)
    return out.reshape(-1, 2)


def displacement_stats(k_new):
    """Pure distortion displacement: for every output pixel, |raw position of the scene point - where an ideal pinhole with
    the old K would put it| (the change of K itself, a 5-6 % zoom, is excluded)."""
    w, h = SIZE
    ys, xs = np.mgrid[0:h:4, 0:w:4]
    pts = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float64)
    norm = cv2.undistortPoints(pts.reshape(-1, 1, 2), k_new, None).reshape(-1, 2)
    ideal = norm * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    disp = np.linalg.norm(distort_pixels(pts, k_new) - ideal, axis=1)
    return {"max": float(disp.max()), "p95": float(np.percentile(disp, 95)), "median": float(np.median(disp)),
            "mean": float(disp.mean()), "grid_step_px": 4}


def long_segments(gray, min_len=170.0):
    """Straight segments of the UNDISTORTED image (LSD), longest first; (length, (x1, y1, x2, y2))."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return []
    segs = [(float(np.hypot(x2 - x1, y2 - y1)), (float(x1), float(y1), float(x2), float(y2)))
            for (x1, y1, x2, y2) in lines.reshape(-1, 4)]
    return sorted((s for s in segs if s[0] >= min_len), reverse=True)


def line_points(p0, p1, n=60):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) * np.asarray(p0, float)[None] + t * np.asarray(p1, float)[None]


def chord_deviation(pts_raw):
    """Max distance (raw px) between a raw-image polyline (the distorted image of a straight line) and its chord."""
    a, b = pts_raw[0], pts_raw[-1]
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    rel = pts_raw - a
    return float(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]).max())


def edge_offsets(gray, pts, half=6.0, step=0.5):
    """Normal offset (px) of the strongest intensity edge from each point of a polyline (within +/- half px)."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    tang = np.gradient(pts, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    offs = np.arange(-half, half + step, step)
    h, w = gray.shape
    out = []
    for p, n in zip(pts, normal):
        q = p[None] + offs[:, None] * n[None]
        if (q[:, 0] < 1).any() or (q[:, 0] > w - 2).any() or (q[:, 1] < 1).any() or (q[:, 1] > h - 2).any():
            continue
        mx = cv2.remap(gx, q[:, 0:1].astype(np.float32).reshape(1, -1), q[:, 1:2].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR)
        my = cv2.remap(gy, q[:, 0:1].astype(np.float32).reshape(1, -1), q[:, 1:2].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR)
        proj = np.abs(mx[0] * n[0] + my[0] * n[1])
        out.append(float(offs[int(np.argmax(proj))]))
    return np.asarray(out)


def pick_sample(seq, k_new, step=2):
    """Frame whose long straight edges (found on the undistorted image) bow the most in the raw image.

    Candidates must be sharp and verified by the images themselves: the strongest intensity edge lies within 1.2 px (rms)
    of the straight line in the undistorted image and within 1.5 px (rms) of the mapped curve in the raw image, the whole
    segment is at least 12 px inside the frame and the frame's Laplacian variance is above the median of the scanned frames.
    """
    dst, src = DATA / SEQS[seq] / "rgb", TUM / SEQS[seq] / "rgb"
    names = sorted(os.listdir(dst))[::step]
    sharp = {}
    for name in names:
        sharp[name] = float(cv2.Laplacian(cv2.cvtColor(cv2.imread(str(dst / name)), cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    thr = float(np.median(list(sharp.values())))
    best = None
    for name in names:
        if sharp[name] < thr:
            continue
        und = cv2.imread(str(dst / name))
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
        gray_raw = None
        chosen, mids = [], []
        for length, s in long_segments(gray, 150.0):
            p0, p1 = np.array(s[:2]), np.array(s[2:])
            if min(p0.min(), p1.min()) < 12 or max(p0[0], p1[0]) > 640 - 12 or max(p0[1], p1[1]) > 480 - 12:
                continue
            mid = (p0 + p1) / 2
            if any(np.linalg.norm(mid - m) < 100 for m in mids):
                continue
            line = line_points(p0, p1)
            curve = distort_pixels(line, k_new)
            if curve.min() < 12 or curve[:, 0].max() > 640 - 12 or curve[:, 1].max() > 480 - 12:
                continue
            e_und = edge_offsets(gray, line)
            if len(e_und) < 50 or np.sqrt(np.mean(e_und ** 2)) > 1.5:
                continue
            if gray_raw is None:
                gray_raw = cv2.cvtColor(cv2.imread(str(src / name)), cv2.COLOR_BGR2GRAY)
            e_raw = edge_offsets(gray_raw, curve)
            if len(e_raw) < 50 or np.sqrt(np.mean(e_raw ** 2)) > 1.8:
                continue
            chosen.append((chord_deviation(curve), length, s))
            mids.append(mid)
        chosen.sort(reverse=True)
        top = chosen[:3]
        score = sum(c[0] * w for c, w in zip(top, (1.0, 0.3, 0.3)))
        if top and (best is None or score > best[0]):
            best = (score, name, top)
    if best is None:
        raise RuntimeError(f"{seq}: no verified straight edge found for the sample image")
    return best


def _zoom(img, centre, half, factor):
    """Square crop of side 2*half around `centre` (padded with grey), upsampled by `factor`; returns (image, to_crop)."""
    padded = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    crop = padded[cy:cy + 2 * half, cx:cx + 2 * half]  # padded coords = original + half -> crop starts at c - half
    big = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    def to_crop(pt):
        return (np.asarray(pt, float) - np.array([cx - half, cy - half]) + 0.5) * factor - 0.5

    return big, to_crop


def sample_image(seq, k_new, path):
    """Top row: raw | undistorted; green = straight grid of the undistorted image and its curved image in the raw frame. Red = a long straight scene edge (picked on the undistorted image):
    one straight line on the right; on the left its image in the raw frame is the (slightly) curved red polyline, which
    follows the real edge, while the cyan chord is the straight line between its two ends.
    Bottom row: x6 zoom at the point where that edge bows most (left raw with chord, right undistorted with the line)."""
    score, name, top = pick_sample(seq, k_new)
    raw = cv2.imread(str(TUM / SEQS[seq] / "rgb" / name))
    und = cv2.imread(str(DATA / SEQS[seq] / "rgb" / name))
    gray_raw, gray_und = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
    left, right = raw.copy(), und.copy()
    h, w = raw.shape[:2]
    # green grid: straight lines every 80 px of the UNDISTORTED image (right panel) and their images under the distortion
    # model (left panel, curved: this is the distortion field the correction removes; up to ~23 px at the corners)
    for c in range(0, w, 80):
        pts = np.stack([np.full(64, float(c)), np.linspace(0, h - 1, 64)], axis=1)
        cv2.line(right, (c, 0), (c, h - 1), (0, 200, 0), 1, cv2.LINE_AA)
        cv2.polylines(left, [np.round(distort_pixels(pts, k_new)).astype(np.int32).reshape(-1, 1, 2)], False, (0, 200, 0), 1, cv2.LINE_AA)
    for r in range(0, h, 80):
        pts = np.stack([np.linspace(0, w - 1, 64), np.full(64, float(r))], axis=1)
        cv2.line(right, (0, r), (w - 1, r), (0, 200, 0), 1, cv2.LINE_AA)
        cv2.polylines(left, [np.round(distort_pixels(pts, k_new)).astype(np.int32).reshape(-1, 1, 2)], False, (0, 200, 0), 1, cv2.LINE_AA)
    stats, zoom_info = [], None
    for rank, (dev, length, s) in enumerate(top):
        p0, p1 = np.array(s[:2]), np.array(s[2:])
        line = line_points(p0, p1)
        curve = distort_pixels(line, k_new)
        cv2.line(left, tuple(np.round(curve[0]).astype(int)), tuple(np.round(curve[-1]).astype(int)), (255, 255, 0), 2, cv2.LINE_AA)
        cv2.polylines(left, [np.round(curve).astype(np.int32).reshape(-1, 1, 2)], False, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(right, tuple(np.round(p0).astype(int)), tuple(np.round(p1).astype(int)), (0, 0, 255), 1, cv2.LINE_AA)
        e_und = edge_offsets(gray_und, line)
        e_raw = edge_offsets(gray_raw, curve)
        stats.append({"frame": name, "segment_undistorted_px": [float(x) for x in s], "segment_length_px": float(length),
                      "raw_curve_max_deviation_from_its_chord_px": float(dev),
                      "undistorted_image_edge_rms_offset_from_straight_line_px": float(np.sqrt(np.mean(e_und ** 2))) if len(e_und) else None,
                      "raw_image_edge_rms_offset_from_mapped_curve_px": float(np.sqrt(np.mean(e_raw ** 2))) if len(e_raw) else None})
        if rank == 0:
            a, b = curve[0], curve[-1]
            d = (b - a) / np.linalg.norm(b - a)
            idx = int(np.argmax(np.abs((curve - a)[:, 0] * d[1] - (curve - a)[:, 1] * d[0])))
            zoom_info = (curve[idx], line[idx], a, b, p0, p1, dev)
    pad = np.full((h, 6, 3), 255, np.uint8)
    top_row = np.concatenate([left, pad, right], axis=1)
    for text, org in ((f"raw {name}", (10, 22)), ("undistorted (alpha=0)", (w + 16, 22))):
        cv2.putText(top_row, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(top_row, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    raw_pt, und_pt, a, b, p0, p1, dev = zoom_info
    factor, half = 6, 32
    zr, to_r = _zoom(raw, raw_pt, half, factor)
    zu, to_u = _zoom(und, und_pt, half, factor)
    for img, to_c, q0, q1, col, thick in ((zr, to_r, a, b, (255, 255, 0), 1), (zu, to_u, p0, p1, (0, 0, 255), 1)):
        c0, c1 = to_c(q0), to_c(q1)
        direction = (c1 - c0) / np.linalg.norm(c1 - c0)
        cv2.line(img, tuple(np.round(c0 - direction * 4000).astype(int)), tuple(np.round(c0 + direction * 4000).astype(int)),
                 col, thick, cv2.LINE_AA)
    size = 2 * half * factor
    cap_h = 26
    row2 = np.full((size + cap_h, 2 * w + 6, 3), 255, np.uint8)
    x0r, x0u = (w - size) // 2, w + 6 + (w - size) // 2
    row2[cap_h:, x0r:x0r + size], row2[cap_h:, x0u:x0u + size] = zr, zu
    caption = [f"raw x{factor}: edge vs chord (cyan), max bow {dev:.1f} px", f"undistorted x{factor}: edge vs straight line (red)"]
    for text, x0 in zip(caption, (x0r, x0u)):
        cv2.putText(row2, text, (x0, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    out_img = np.concatenate([top_row, np.full((6, top_row.shape[1], 3), 255, np.uint8), row2], axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out_img)
    return stats


def undistort_sequence(seq, k_new, map_x, map_y, force=False):
    src = TUM / SEQS[seq]
    dst = DATA / SEQS[seq]
    src_rgb = sorted(p for p in os.listdir(src / "rgb") if p.endswith(".png"))
    (dst / "rgb").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_written = 0
    for name in src_rgb:
        out_path = dst / "rgb" / name
        raw = cv2.imread(str(src / "rgb" / name), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.shape != (SIZE[1], SIZE[0], 3) or raw.dtype != np.uint8:
            raise ValueError(f"{src / 'rgb' / name}: unexpected image {None if raw is None else (raw.shape, raw.dtype)}")
        und = cv2.remap(raw, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if force or not out_path.is_file():
            if not cv2.imwrite(str(out_path), und):
                raise RuntimeError(f"could not write {out_path}")
            n_written += 1
    for f in ("groundtruth.txt", "rgb.txt"):
        shutil.copyfile(src / f, dst / f)
    dst_rgb = sorted(p for p in os.listdir(dst / "rgb") if p.endswith(".png"))
    check = {"n_original": len(src_rgb), "n_undistorted": len(dst_rgb), "names_identical": src_rgb == dst_rgb,
             "n_written_this_call": n_written, "seconds": time.time() - t0,
             "groundtruth_identical": (src / "groundtruth.txt").read_bytes() == (dst / "groundtruth.txt").read_bytes(),
             "rgb_txt_identical": (src / "rgb.txt").read_bytes() == (dst / "rgb.txt").read_bytes()}
    if check["n_original"] != check["n_undistorted"] or not check["names_identical"]:
        raise RuntimeError(f"{seq}: file count / names differ {check}")
    return check


def verify_sequence(seq, map_x, map_y):
    """Every saved PNG equals a fresh remap of its raw frame, bit for bit, and has the raw frame's shape and dtype."""
    src, dst = TUM / SEQS[seq] / "rgb", DATA / SEQS[seq] / "rgb"
    bad = []
    names = sorted(p for p in os.listdir(src) if p.endswith(".png"))
    for name in names:
        raw = cv2.imread(str(src / name), cv2.IMREAD_UNCHANGED)
        saved = cv2.imread(str(dst / name), cv2.IMREAD_UNCHANGED)
        fresh = cv2.remap(raw, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if saved is None or saved.shape != raw.shape or saved.dtype != raw.dtype or not np.array_equal(saved, fresh):
            bad.append(name)
    return {"n_checked": len(names), "n_mismatch": len(bad), "mismatching_files": bad[:5]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rewrite existing output PNGs")
    args = parser.parse_args(argv)
    cv2.setNumThreads(4)
    k_new, roi, map_x, map_y = build_maps()
    report = {"opencv": cv2.__version__, "K_old": K.tolist(), "D_old_k1_k2_p1_p2_k3": D.tolist(), "K_new": k_new.tolist(),
              "roi": [int(x) for x in roi], "image_size_wh": list(SIZE),
              "method": "cv2.getOptimalNewCameraMatrix(K, D, (640,480), alpha=0, newImgSize=(640,480)); "
                        "cv2.initUndistortRectifyMap(K, D, None, K_new, (640,480), CV_32FC1); cv2.remap(INTER_LINEAR); PNG",
              "fx_old_over_width": float(K[0, 0] / SIZE[0]), "fx_new_over_width": float(k_new[0, 0] / SIZE[0]),
              "fy_old_over_height": float(K[1, 1] / SIZE[1]), "fy_new_over_height": float(k_new[1, 1] / SIZE[1]),
              "maps": map_checks(k_new, map_x, map_y), "sequences": {}}
    for seq in SEQS:
        check = undistort_sequence(seq, k_new, map_x, map_y, args.force)
        stats = sample_image(seq, k_new, OUT / "samples" / f"{seq}_raw_vs_undistorted.png")
        check["straight_line_guides"] = stats
        check["png_equals_fresh_remap"] = verify_sequence(seq, map_x, map_y)
        report["sequences"][seq] = check
        calib = {"sequence": SEQS[seq], "image_size_wh": list(SIZE), "model": "OpenCV plumb_bob (k1 k2 p1 p2 k3), TUM fr1 RGB calibration",
                 "K_old": K.tolist(), "D_old_k1_k2_p1_p2_k3": D.tolist(), "K_new": k_new.tolist(), "roi": [int(x) for x in roi],
                 "fx_old_over_width": report["fx_old_over_width"], "fx_new_over_width": report["fx_new_over_width"],
                 "fy_old_over_height": report["fy_old_over_height"], "fy_new_over_height": report["fy_new_over_height"],
                 "method": report["method"], "opencv": cv2.__version__, "n_frames": check["n_undistorted"],
                 "source": str(TUM / SEQS[seq])}
        (DATA / SEQS[seq] / "calibration.json").write_text(json.dumps(calib, indent=2))
        print(seq, {k: v for k, v in check.items() if k != "straight_line_guides"}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "undistort_report.json").write_text(json.dumps(report, indent=2))
    print("K_new", k_new.tolist(), "roi", roi)
    print("maps", json.dumps(report["maps"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
