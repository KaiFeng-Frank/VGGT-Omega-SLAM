#!/usr/bin/env python3
"""One-page comparison figure (Simplified Chinese text) from runs/compare + summary.json.

(a) office_loop top-down point maps + trajectories of both backends (accepted loop closures marked)
(b) TUM sequences: Sim(3)-aligned trajectories of both backends vs ground truth, top-down
(c) grouped ATE bars (loops on/off, both backends) with the VGGT-SLAM 2.0 paper numbers as markers
(d) per-submap forward time and peak CUDA memory per backend, plus a compact throughput table

Numbers shown in (c), (d) and the table are read from summary.json (written by compare_summarize.py).
Trajectories and maps are read from the run directories. CPU only.
Usage:  OMP_NUM_THREADS=4 python tools/compare_figure.py [--compare-dir runs/compare] [--out PNG]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import eval_ate  # noqa: E402
from vos import compare_utils as cu  # noqa: E402

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# dataviz reference palette, categorical slots 1-2 (validated for all-pairs use) + neutrals
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_LIGHT, ORANGE_LIGHT = "#a9c9ee", "#f6bba4"
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
COL = {"vggt": BLUE, "omega": ORANGE}
COL_LIGHT = {"vggt": BLUE_LIGHT, "omega": ORANGE_LIGHT}
NAME = {"vggt": "VGGT", "omega": "VGGT-Omega"}
SEQ_LABEL = {"office_loop": "office_loop", "fr1_desk": "desk", "fr1_desk2": "desk2", "fr1_room": "room"}
TUM = ["fr1_desk", "fr1_desk2", "fr1_room"]


def load_run(cdir, name):
    d = Path(cdir) / name
    if not (d / "poses.tum").is_file():
        return None
    meta = json.loads((d / "metadata.json").read_text()) if (d / "metadata.json").is_file() else {}
    if meta.get("status") != "complete":
        return None
    stamps, xyz, quat = cu.read_poses(d / "poses.tum")
    return {"dir": d, "meta": meta, "stamps": stamps, "xyz": xyz, "quat": quat,
            "loops": cu.accepted_loop_edges(meta.get("loop_log"), stamps, xyz)}


def style_axis(ax, grid=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK3)
    ax.tick_params(colors=INK2, labelsize=10.5, length=3)
    if grid:
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def panel_title(ax, text, sub=None):
    ax.set_title(text, loc="left", fontsize=13.5, color=INK, fontweight="bold", pad=18 if sub else 8)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=10.5, color=INK2, va="bottom", ha="left")


# ---------------------------------------------------------------------------------------------
# (a) office_loop top-down maps
# ---------------------------------------------------------------------------------------------
def office_maps(fig, axes, cdir, summary, loops_key=1):
    runs = {b: load_run(cdir, f"office_loop_{b}_loops{loops_key}") for b in ("vggt", "omega")}
    ref = runs["vggt"] or runs["omega"]
    if ref is None:
        for ax in axes:
            ax.text(0.5, 0.5, "office_loop 运行缺失", ha="center", va="center", transform=ax.transAxes, color=INK2)
            ax.set_axis_off()
        return
    down = cu.gravity_down_from_cameras(ref["quat"])
    e1, e2, up = cu.top_down_basis(down, ref["xyz"])
    shared_lims = [None]
    for ax, backend in zip(axes, ("vggt", "omega")):
        run = runs[backend]
        gap = summary["office_loop"].get(f"{backend}_loops{loops_key}")
        if run is None:
            ax.text(0.5, 0.5, f"{NAME[backend]} 运行失败或缺失", ha="center", va="center", transform=ax.transAxes, color=INK2)
            ax.set_axis_off()
            continue
        pts, rgb = cu.read_ply_xyz_rgb(run["dir"] / "cloud.ply", max_points=350_000)
        cams = run["xyz"]
        loops = run["loops"]
        aligned = backend == "omega" and runs["vggt"] is not None
        if aligned:  # display only: Omega frame -> VGGT frame by Sim(3) on the common keyframe centres
            ia, ib = cu.common_stamps(runs["vggt"]["stamps"], run["stamps"])
            r, t, c = cu.umeyama(run["xyz"][ib], runs["vggt"]["xyz"][ia])
            pts, cams = cu.apply_sim3(pts, r, t, c), cu.apply_sim3(cams, r, t, c)
            loops = [(cu.apply_sim3(a[None], r, t, c)[0], cu.apply_sim3(b[None], r, t, c)[0], i) for a, b, i in loops]
        h = pts @ up
        hc = float(np.median(cams @ up))
        lo, hi = np.percentile(h, [2, 98])
        keep = (h <= hc + 0.15 * (hi - lo)) & (h >= lo)  # slice: floor to just above camera height (no ceiling)
        order = np.argsort(h[keep])
        p2 = np.stack([pts[keep] @ e1, pts[keep] @ e2], axis=1)[order]
        ax.scatter(p2[:, 0], p2[:, 1], c=rgb[keep][order] / 255.0, s=0.35, linewidths=0, rasterized=True, zorder=1)
        c2 = np.stack([cams @ e1, cams @ e2], axis=1)
        ax.plot(c2[:, 0], c2[:, 1], color="white", lw=4.2, zorder=3, solid_capstyle="round")
        ax.plot(c2[:, 0], c2[:, 1], color=COL[backend], lw=2.4, zorder=4, solid_capstyle="round")
        ax.scatter(*c2[0], s=90, marker="o", color="#1baf7a", edgecolor="white", lw=1.6, zorder=6)
        ax.scatter(*c2[-1], s=90, marker="s", color=INK, edgecolor="white", lw=1.6, zorder=6)
        for a, b, _ in loops:
            pa, pb = np.array([a @ e1, a @ e2]), np.array([b @ e1, b @ e2])
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=COL[backend], lw=2.0, ls=(0, (4, 2)), zorder=5)
            ax.scatter([pa[0], pb[0]], [pa[1], pb[1]], s=46, marker="D", color="white", edgecolor=COL[backend],
                       lw=1.4, zorder=6)
        if shared_lims[0] is None:  # both panels share the VGGT-frame limits so the boxes are identical
            lim = np.vstack([c2, np.percentile(p2, [1, 99], axis=0)])
            pad = 0.08 * (lim.max(0) - lim.min(0))
            shared_lims[0] = ((lim[:, 0].min() - pad[0], lim[:, 0].max() + pad[0]), (lim[:, 1].min() - pad[1], lim[:, 1].max() + pad[1]))
        ax.set_xlim(*shared_lims[0][0])
        ax.set_ylim(*shared_lims[0][1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        for k, (a_, b_, info) in enumerate(loops):  # call-out for the (short) loop edge
            mid = np.array([(a_ + b_)[0:3] @ e1, (a_ + b_)[0:3] @ e2]) / 2
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            fx, fy = (mid[0] - x0) / (x1 - x0), (mid[1] - y0) / (y1 - y0)
            tx, ty = fx + (0.17 if fx < 0.5 else -0.17), fy + (0.16 if fy < 0.5 else -0.16)
            ax.annotate(f"回环 {info['query_image'].split('.')[0].replace('frame_', '')}→{info['detected_image'].split('.')[0].replace('frame_', '')}",
                        xy=(mid[0], mid[1]), xycoords="data", xytext=(tx, ty), textcoords="axes fraction", fontsize=10.5, color=INK,
                        ha="center", va="center", arrowprops=dict(arrowstyle="-", color=INK, lw=1.2), zorder=11,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=INK3, alpha=0.95), annotation_clip=False)
        n_acc = gap["loop_closures_accepted"] if gap else len(loops)
        txt = f"接受回环 {n_acc} 条    首尾间隙/轨迹长 {gap['gap_fraction'] * 100:.1f}%" if gap else f"接受回环 {n_acc} 条"
        sub = "顶视（模型单位，Sim(3) 对齐到 VGGT 坐标系，仅作显示）" if aligned else "顶视（模型单位，重力方向取相机平均 y 轴）"
        panel_title(ax, f"(a) office_loop · {NAME[backend]}", sub)
        ax.text(0.015, 0.02, txt, transform=ax.transAxes, fontsize=11, color=INK, va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, alpha=0.92), zorder=10)


def office_note(fig, summary, x=0.045, y=0.652):
    """Three text lines under the (a) panels, all numbers from summary.json."""
    off = summary.get("office_loop", {})
    runs = summary["runs"]
    tau = summary["protocol"]["omega_similarity"]["tau"]
    parts, accepted = [], {}
    for backend, thr in (("vggt", 0.95), ("omega", tau)):
        o = off.get(f"{backend}_loops1")
        if not o:
            continue
        acc = [e for e in o["loop_events"] if e["accepted"]]
        accepted[backend] = [(e["query_image"], e["detected_image"]) for e in acc]
        detail = "，".join(f"原始比值 {e['raw_ratio']:.3f}" for e in acc) or "无"
        parts.append(f"{NAME[backend]} {len(acc)}/{o['loop_candidates']}（{detail}，阈值 {thr:.3f}）")
    same = len(accepted) == 2 and accepted["vggt"] == accepted["omega"] and accepted["vggt"]
    r0 = runs.get("office_loop_vggt_loops1") or runs.get("office_loop_omega_loops1")
    line1 = (f"office_loop：{r0['n_images']} 帧 → {r0['keyframes']} 关键帧 → {r0['submaps']} 个子图" if r0 and r0["status"] == "complete" else "office_loop")
    if same:
        line1 += "；两个前端接受的是同一对帧 " + "、".join(f"{a.split('.')[0][-4:]}→{b.split('.')[0][-4:]}" for a, b in accepted["vggt"])
    line2 = "回环 接受/候选：" + "；".join(parts)
    gaps = []
    for loops, label in ((1, "回环开"), (0, "回环关")):
        seg = [f"{NAME[b]} {off[f'{b}_loops{loops}']['gap_fraction'] * 100:.1f}%" for b in ("vggt", "omega") if f"{b}_loops{loops}" in off]
        if seg:
            gaps.append(f"{label} " + " / ".join(seg))
    line3 = "首尾间隙/轨迹长（代理指标，无真值，仅因相机回到起点附近才有意义）：" + "；".join(gaps)
    pair = []
    for backend in ("vggt", "omega"):
        rows = off.get(f"{backend}_loops1", {}).get("accepted_pair_gap", [])
        if rows:
            pair.append(f"{NAME[backend]} {rows[0]['pair_gap_fraction_loops1'] * 100:.1f}% / {rows[0]['pair_gap_fraction_loops0'] * 100:.1f}%")
    lines = [line1, line2, line3]
    if pair:
        lines.append("回环所连两帧的间距/轨迹长（开 / 关）：" + "；".join(pair))
    for k, line in enumerate(lines):
        fig.text(x, y - 0.0195 * k, line, fontsize=11, color=INK2, va="top")


# ---------------------------------------------------------------------------------------------
# (b) TUM trajectories
# ---------------------------------------------------------------------------------------------
def tum_panel(ax, cdir, summary, seq, loops_key=1):
    meta_any = None
    gt_xyz = None
    ate_text = []
    for backend in ("vggt", "omega"):
        run = load_run(cdir, f"{seq}_{backend}_loops{loops_key}")
        if run is None:
            ate_text.append(f"{NAME[backend]} 失败/缺失")
            continue
        meta_any = run["meta"]
        gt_file = run["meta"]["input"]["groundtruth"]
        if gt_xyz is None:
            _, gt_xyz, _, _ = eval_ate.read_tum_trajectory(gt_file)
            ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], color=INK3, lw=1.6, zorder=2, solid_capstyle="round")
            ax.scatter(*gt_xyz[0, :2], s=60, marker="o", color="#1baf7a", edgecolor="white", lw=1.3, zorder=7)
        ate_file = run["dir"] / "ate.json"
        ate = json.loads(ate_file.read_text()) if ate_file.is_file() else eval_ate.evaluate(gt_file, run["dir"] / "poses.tum")
        r, t, c = np.asarray(ate["rotation"]), np.asarray(ate["translation"]), ate["scale"]
        est = cu.apply_sim3(run["xyz"], r, t, c)
        ax.plot(est[:, 0], est[:, 1], color=COL[backend], lw=2.0, zorder=4, solid_capstyle="round")
        ax.scatter(est[:, 0], est[:, 1], s=9, color=COL[backend], zorder=5, linewidths=0)
        # Each backend draws only the loops it accepted, in its own color.
        for a, b, _ in run["loops"]:
            pa, pb = cu.apply_sim3(a[None], r, t, c)[0], cu.apply_sim3(b[None], r, t, c)[0]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=COL[backend], lw=1.4, ls=(0, (3, 2)), zorder=6)
            ax.scatter([pa[0], pb[0]], [pa[1], pb[1]], s=26, marker="D", color="white", edgecolor=COL[backend],
                       lw=1.1, zorder=7)
        row = next(r_ for r_ in summary["ate_table"] if r_["seq"] == seq)
        v = row[f"{backend}_loops{loops_key}"]
        ate_text.append(f"{NAME[backend]} {v:.3f}" if v is not None else f"{NAME[backend]} -")
    ax.set_aspect("equal", adjustable="datalim")
    style_axis(ax)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel("x（米，真值坐标系）", fontsize=10.5, color=INK2)
    ax.set_ylabel("y（米）", fontsize=10.5, color=INK2)
    row = next(r_ for r_ in summary["ate_table"] if r_["seq"] == seq)
    panel_title(ax, f"(b) TUM {SEQ_LABEL[seq]} · 回环开", f"ATE（米）：{'   '.join(ate_text)}   论文 {row['paper_submap32']:.3f}")


# ---------------------------------------------------------------------------------------------
# (c) ATE bars
# ---------------------------------------------------------------------------------------------
def ate_bars(ax, summary):
    table, means = summary["ate_table"], summary["ate_mean_of_three"]
    groups = [(r["label"], r) for r in table] + [("三序列均值", means)]
    series = [("vggt", 1), ("vggt", 0), ("omega", 1), ("omega", 0)]
    width, gap = 0.19, 0.0
    xs = np.arange(len(groups))
    top = 0
    for k, (backend, loops) in enumerate(series):
        offset = (k - 1.5) * (width + gap)
        for gi, (_, row) in enumerate(groups):
            v = row.get(f"{backend}_loops{loops}")
            x = xs[gi] + offset
            if v is None:
                ax.text(x, 0.001, "×", ha="center", va="bottom", fontsize=12, color=INK2)
                continue
            top = max(top, v)
            ax.bar(x, v, width * 0.94, color=COL[backend] if loops else COL_LIGHT[backend],
                   edgecolor=COL[backend], lw=1.2, hatch=None if loops else "///", zorder=3)
            ax.text(x, v + 0.0008, f"{v * 100:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    for gi, (_, row) in enumerate(groups):
        p = row["paper_submap32"]
        ax.plot([xs[gi] - 0.42, xs[gi] + 0.42], [p, p], color=INK, lw=2.0, zorder=5)
        ax.scatter([xs[gi]], [p], marker="D", s=46, color="white", edgecolor=INK, lw=1.6, zorder=6)
        top = max(top, p)
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], fontsize=11.5, color=INK)
    ax.set_ylim(0, top * 1.42)
    ax.set_ylabel("ATE RMSE（米）", fontsize=10.5, color=INK2)
    style_axis(ax, grid=True)
    ax.legend(handles=[Patch(fc=BLUE, ec=BLUE, label="VGGT 回环开"), Patch(fc=BLUE_LIGHT, ec=BLUE, hatch="///", label="VGGT 回环关"),
                       Patch(fc=ORANGE, ec=ORANGE, label="Omega 回环开"), Patch(fc=ORANGE_LIGHT, ec=ORANGE, hatch="///", label="Omega 回环关"),
                       Line2D([0], [0], color=INK, lw=2, marker="D", mfc="white", mec=INK, label="论文 VGGT-SLAM 2.0（子图 32）")],
              loc="upper left", fontsize=9.5, frameon=False, ncol=2, columnspacing=1.2, handlelength=1.6)
    panel_title(ax, "(c) ATE RMSE：回环开/关 × 两个前端", "柱顶数字为厘米；Sim(3) 对齐含尺度；本实验子图 16，论文子图 32；单次运行，种子 0")


# ---------------------------------------------------------------------------------------------
# (d) timing and memory
# ---------------------------------------------------------------------------------------------
def timing_strip(ax, summary):
    seqs = list(SEQ_LABEL)
    runs = summary["runs"]
    for gi, seq in enumerate(seqs):
        for k, backend in enumerate(("vggt", "omega")):
            r = runs.get(f"{seq}_{backend}_loops1")
            x0 = gi + (-0.2 if k == 0 else 0.2)
            if not r or r["status"] != "complete":
                ax.text(x0, 0.05, "×", ha="center", fontsize=12, color=INK2)
                continue
            f = np.asarray(r["forward_per_submap_s"])
            full = np.asarray(r["frames_per_submap"]) == max(r["frames_per_submap"])
            jitter = (np.arange(len(f)) - (len(f) - 1) / 2) * 0.012
            ax.scatter(x0 + jitter[full], f[full], s=34, color=COL[backend], edgecolor="white", lw=0.8, zorder=4)
            if (~full).any():
                ax.scatter(x0 + jitter[~full], f[~full], s=34, facecolor="white", edgecolor=COL[backend], lw=1.4, zorder=4)
            med = r["forward_median_per_submap_s"]
            ax.plot([x0 - 0.15, x0 + 0.15], [med, med], color=INK, lw=2.0, zorder=5)
    ax.set_xticks(range(len(seqs)))
    ax.set_xticklabels([SEQ_LABEL[s] for s in seqs], fontsize=11.5, color=INK)
    ax.set_xlim(-0.6, len(seqs) - 0.4)
    ymax = max([max(r["forward_per_submap_s"]) for r in runs.values() if r["status"] == "complete"] or [1.0])
    ax.set_ylim(0, ymax * 1.38)
    ax.set_ylabel("单个子图前向耗时（秒）", fontsize=10.5, color=INK2)
    style_axis(ax, grid=True)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=BLUE, mec="white", label="VGGT 子图"),
                       Line2D([0], [0], marker="o", ls="", color=ORANGE, mec="white", label="Omega 子图"),
                       Line2D([0], [0], marker="o", ls="", mfc="white", mec=INK2, label="末尾不足 17 帧的批"),
                       Line2D([0], [0], color=INK, lw=2, label="中位数")],
              loc="upper left", fontsize=9.5, frameon=False, ncol=2)
    panel_title(ax, "(d1) 每个子图的前向耗时（回环开）", "17 帧/子图；两后端输入分辨率不同（VGGT 294x518，Omega 384x688；TUM 为 392x518 与 448x592）")


def memory_bars(ax, summary):
    seqs = list(SEQ_LABEL)
    runs = summary["runs"]
    width = 0.34
    for gi, seq in enumerate(seqs):
        for k, backend in enumerate(("vggt", "omega")):
            r = runs.get(f"{seq}_{backend}_loops1")
            x = gi + (k - 0.5) * (width + 0.03)
            if not r or r["status"] != "complete":
                ax.text(x, 0.1, "×", ha="center", fontsize=12, color=INK2)
                continue
            ax.bar(x, r["peak_cuda_allocated_gib"], width, color=COL[backend], zorder=3)
            ax.plot([x - width / 2, x + width / 2], [r["peak_cuda_reserved_gib"]] * 2, color=INK, lw=1.8, zorder=5)
            ax.text(x, r["peak_cuda_reserved_gib"] + 0.12, f"{r['peak_cuda_allocated_gib']:.1f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(seqs)))
    ax.set_xticklabels([SEQ_LABEL[s] for s in seqs], fontsize=11, color=INK)
    ax.set_ylim(0, 11.5)
    ax.set_ylabel("CUDA 显存峰值（GiB）", fontsize=10.5, color=INK2)
    style_axis(ax, grid=True)
    ax.legend(handles=[Patch(fc=BLUE, label="VGGT allocated"), Patch(fc=ORANGE, label="Omega allocated"),
                       Line2D([0], [0], color=INK, lw=2, label="reserved")], loc="upper left", fontsize=9.5, frameon=False, ncol=3)
    panel_title(ax, "(d2) 显存峰值", "含常驻权重：VGGT bf16，Omega fp32")


def table_panel(ax, summary):
    ax.set_axis_off()
    panel_title(ax, "(e) 吞吐与回环（回环开）", "关键帧/秒 = 关键帧数 ÷ 前向总耗时")
    head = ["序列", "后端", "中位数 s", "关键帧/秒", "回环 接受/候选"]
    rows = []
    for seq in SEQ_LABEL:
        for backend in ("vggt", "omega"):
            r = summary["runs"].get(f"{seq}_{backend}_loops1")
            if not r or r["status"] != "complete":
                rows.append([SEQ_LABEL[seq], NAME[backend], "失败", "-", "-"])
            else:
                rows.append([SEQ_LABEL[seq], NAME[backend], f"{r['forward_median_per_submap_s']:.2f}", f"{r['fps_keyframes_forward']:.2f}",
                             f"{r['loop_closures_accepted']}/{r['loop_candidates']}"])
    xs = [0.0, 0.2, 0.43, 0.6, 0.79]
    for x, h in zip(xs, head):
        ax.text(x, 0.93, h, transform=ax.transAxes, fontsize=10.5, color=INK2, fontweight="bold", va="top")
    ax.plot([0, 1], [0.865, 0.865], transform=ax.transAxes, color=INK3, lw=1.0)
    for i, row in enumerate(rows):
        y = 0.83 - i * 0.092
        for x, v in zip(xs, row):
            color = COL["vggt"] if (x == xs[1] and v == "VGGT") else COL["omega"] if (x == xs[1] and v == "VGGT-Omega") else INK
            ax.text(x, y, v, transform=ax.transAxes, fontsize=10.5, color=color, va="top")
        if i % 2 == 1:
            ax.plot([0, 1], [y - 0.05, y - 0.05], transform=ax.transAxes, color=GRID, lw=0.8)


# ---------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compare-dir", type=Path, default=ROOT / "runs" / "compare")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    summary = json.loads((args.compare_dir / "summary.json").read_text())
    out = args.out or args.compare_dir / "对比_VGGT_vs_Omega.png"

    fig = plt.figure(figsize=(22, 13), dpi=100, facecolor="white")
    gs = fig.add_gridspec(3, 12, height_ratios=[1.28, 1.0, 0.92], left=0.045, right=0.985, top=0.875, bottom=0.055,
                          hspace=0.5, wspace=1.6)
    fig.text(0.045, 0.972, "VGGT 与 VGGT-Omega 作为前端接入同一个未修改的 VGGT-SLAM 2.0 后端", fontsize=21, fontweight="bold", color=INK, va="center")
    proto = summary["protocol"]
    fig.text(0.045, 0.943, f"后端：MIT-SPARK 上游 35327ac 原样；子图 {proto['submap_size']}、min_disparity {proto['min_disparity']}、"
             f"conf {proto['conf_threshold']}、lc_thres {proto['lc_thres']}；Omega 回环验证层 {proto['omega_similarity']['layer']}、"
             f"tau {proto['omega_similarity']['tau']:.3f}（仅在 office_loop 上标定，TUM 未参与）",
             fontsize=12.5, color=INK2, va="center")
    handles = [Line2D([0], [0], color=BLUE, lw=3, label="VGGT"), Line2D([0], [0], color=ORANGE, lw=3, label="VGGT-Omega"),
               Line2D([0], [0], color=INK3, lw=2, label="真值轨迹"),
               Line2D([0], [0], color=INK, lw=2, ls=(0, (3, 2)), marker="D", mfc="white", mec=INK,
                      label="已接受的回环边（颜色同前端）"),
               Line2D([0], [0], marker="o", ls="", color="#1baf7a", mec="white", ms=9, label="起点"),
               Line2D([0], [0], marker="s", ls="", color=INK, mec="white", ms=9, label="终点（office_loop）")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.04, 0.913), ncol=6, fontsize=11.5, frameon=False,
               columnspacing=2.2, handlelength=2.2)

    ax_a1, ax_a2 = fig.add_subplot(gs[0, 0:4]), fig.add_subplot(gs[0, 4:8])
    office_maps(fig, (ax_a1, ax_a2), args.compare_dir, summary)
    office_note(fig, summary)
    ate_bars(fig.add_subplot(gs[0, 8:12]), summary)
    for i, seq in enumerate(TUM):
        tum_panel(fig.add_subplot(gs[1, 4 * i:4 * i + 4]), args.compare_dir, summary, seq)
    timing_strip(fig.add_subplot(gs[2, 0:5]), summary)
    memory_bars(fig.add_subplot(gs[2, 5:8]), summary)
    table_panel(fig.add_subplot(gs[2, 8:12]), summary)
    fig.savefig(out, dpi=100, facecolor="white")
    print("wrote", out, plt.imread(out).shape)


if __name__ == "__main__":
    main()
