"""CPU helpers for the VGGT vs VGGT-Omega comparison (summary + figure). No GPU, no upstream imports.

* poses.tum reader that keeps one pose per stamp (VGGT-SLAM writes each overlap frame twice)
* binary PLY reader for the cloud.ply written by run_slam.py (Open3D layout: double xyz + uchar rgb)
* Sim(3) Umeyama alignment between two sets of camera centres (display alignment of Omega -> VGGT frame)
* start-to-end gap proxy for sequences that return to the start
* loop-closure edge resolution (accepted loops in metadata.json -> pose positions)
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np


def read_poses(path):
    """TUM rows -> (stamps[n], xyz[n,3], quat_xyzw[n,4]); first occurrence of a stamp is kept, order preserved."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append([float(v) for v in line.split()])
    data = np.asarray(rows, dtype=np.float64).reshape(-1, 8)
    _, first = np.unique(data[:, 0], return_index=True)
    data = data[np.sort(first)]
    return data[:, 0], data[:, 1:4], data[:, 4:8]


def stamp_of(image_name):
    """Same number extraction as upstream Submap.set_frame_ids."""
    return float(re.search(r"\d+(?:\.\d+)?", Path(image_name).name).group())


def quat_to_rot(q):
    """xyzw quaternions [n,4] -> rotation matrices [n,3,3]."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q.T
    r = np.empty((len(q), 3, 3))
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - z * w)
    r[:, 0, 2] = 2 * (x * z + y * w)
    r[:, 1, 0] = 2 * (x * y + z * w)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - x * w)
    r[:, 2, 0] = 2 * (x * z - y * w)
    r[:, 2, 1] = 2 * (y * z + x * w)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def read_ply_xyz_rgb(path, max_points=None, seed=0):
    """Binary little-endian PLY with properties (double x y z, uchar red green blue). Returns xyz[n,3], rgb[n,3] uint8."""
    with Path(path).open("rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            chunk = f.readline()
            if not chunk:
                raise ValueError("PLY header not terminated")
            header += chunk
        text = header.decode("ascii")
        if "binary_little_endian" not in text:
            raise ValueError("only binary_little_endian PLY is supported")
        n = int(re.search(r"element vertex (\d+)", text).group(1))
        props = re.findall(r"property (\w+) (\w+)", text)
        if props != [("double", "x"), ("double", "y"), ("double", "z"),
                     ("uchar", "red"), ("uchar", "green"), ("uchar", "blue")]:
            raise ValueError(f"unexpected PLY layout {props}")
        dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        data = np.fromfile(f, dtype=dtype, count=n)
    if len(data) != n:
        raise ValueError(f"PLY truncated: {len(data)} of {n} vertices")
    if max_points and n > max_points:
        index = np.sort(np.random.default_rng(seed).choice(n, max_points, replace=False))
        data = data[index]
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    rgb = np.stack([data["r"], data["g"], data["b"]], axis=1)
    return xyz, rgb


def umeyama(src, dst, with_scale=True):
    """Least-squares similarity with dst ~ c * R @ src + t for [n,3] arrays. Returns (R, t, c)."""
    mean_s, mean_d = src.mean(0), dst.mean(0)
    xs, yd = src - mean_s, dst - mean_d
    n = len(src)
    cov = yd.T @ xs / n
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1
    r = u @ s @ vt
    var_s = (xs ** 2).sum() / n
    c = float(np.trace(np.diag(d) @ s) / var_s) if with_scale else 1.0
    t = mean_d - c * r @ mean_s
    return r, t, c


def apply_sim3(points, r, t, c):
    return c * points @ r.T + t


def common_stamps(stamps_a, stamps_b):
    """Indices of the stamps present in both (exact float equality; both come from the same file names)."""
    index_b = {s: i for i, s in enumerate(stamps_b)}
    pairs = [(i, index_b[s]) for i, s in enumerate(stamps_a) if s in index_b]
    return np.array([p[0] for p in pairs], dtype=int), np.array([p[1] for p in pairs], dtype=int)


def path_length(xyz):
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()) if len(xyz) > 1 else 0.0


def start_end_gap(xyz):
    """Distance first -> last camera position, trajectory length and their ratio (model units; scale cancels)."""
    gap = float(np.linalg.norm(xyz[-1] - xyz[0]))
    length = path_length(xyz)
    return {"gap": gap, "path_length": length, "gap_fraction": gap / length if length > 0 else None,
            "extent_bbox_diagonal": float(np.linalg.norm(xyz.max(0) - xyz.min(0)))}


def accepted_loop_edges(loop_log, stamps, xyz):
    """[(p_query, p_detected, info)] for every accepted loop candidate that has both stamps in the pose table."""
    index = {s: i for i, s in enumerate(stamps)}
    edges = []
    for entry in loop_log or []:
        for cand in entry.get("candidates", []):
            if not cand.get("accepted"):
                continue
            a, b = stamp_of(cand["query_image"]), stamp_of(cand["detected_image"])
            if a in index and b in index:
                edges.append((xyz[index[a]], xyz[index[b]],
                              {"update": entry["update"], "query_image": cand["query_image"],
                               "detected_image": cand["detected_image"], "raw_ratio": cand.get("raw_ratio"),
                               "salad_distance": cand.get("salad_distance")}))
    return edges


def gravity_down_from_cameras(quat_xyzw):
    """Mean camera +y axis in world coordinates (OpenCV cameras: +y points down when held upright)."""
    down = quat_to_rot(quat_xyzw)[:, :, 1].mean(axis=0)
    return down / np.linalg.norm(down)


def top_down_basis(down, centres):
    """Orthonormal (e1, e2, up) with `up = -down`; e1 = principal horizontal direction of the camera centres."""
    up = -down / np.linalg.norm(down)
    flat = centres - np.outer(centres @ up, up)
    flat = flat - flat.mean(0)
    _, _, vt = np.linalg.svd(flat, full_matrices=False)
    e1 = vt[0] - (vt[0] @ up) * up
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    return e1, e2, up
