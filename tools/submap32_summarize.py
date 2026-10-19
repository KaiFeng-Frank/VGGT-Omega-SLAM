#!/usr/bin/env python3
"""Collect the submap-32 runs (runs/submap32/<seq>_<backend>_loops1) next to the existing submap-16 runs
(runs/compare/<seq>_<backend>_loops1 and _loops0) into summary.json, summary_zh.md and 子图32对比.png.

CPU only; reads run directories, never writes into them. ATE is recomputed with eval_ate.evaluate from poses.tum and
compared with the ate.json the runner wrote. Optional --verdict-file is appended to summary_zh.md (hand-written
interpretation, kept in its own file so this script stays purely data-driven).
Usage:  OMP_NUM_THREADS=4 python tools/submap32_summarize.py [--verdict-file runs/submap32/verdict_zh.md]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import eval_ate  # noqa: E402
from compare_summarize import PAPER  # noqa: E402

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# same reference-palette slots as runs/compare figure (categorical 1-2) + neutrals
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_LIGHT, ORANGE_LIGHT = "#a9c9ee", "#f6bba4"
INK, INK2, INK3, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
COL = {"vggt": BLUE, "omega": ORANGE}
COL_LIGHT = {"vggt": BLUE_LIGHT, "omega": ORANGE_LIGHT}
NAME = {"vggt": "VGGT", "omega": "VGGT-Omega"}
SEQS = ["fr1_desk", "fr1_desk2", "fr1_room"]
LABEL = {"fr1_desk": "desk", "fr1_desk2": "desk2", "fr1_room": "room"}
BACKENDS = ["vggt", "omega"]
GIB = 1024 ** 3
CAP_GIB = 10.0


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def stamp_pair(c):
    return (c["query_image"], c["detected_image"])


def run_record(root, seq, backend, loops):
    name = f"{seq}_{backend}_loops{loops}"
    d = Path(root) / name
    rec = {"name": name, "dir": str(d), "seq": seq, "backend": backend, "max_loops": loops}
    meta = load_json(d / "metadata.json")
    if meta is None:
        rec["status"] = "missing"
        return rec
    rec["status"] = meta.get("status")
    if meta.get("status") != "complete":
        rec["error"] = meta.get("error")
        rec["traceback_tail"] = (meta.get("traceback") or "")[-1500:]
        rec["partial_updates"] = len(meta.get("updates", []))
        rec["peak_cuda_allocated_gib_partial"] = max(
            [u["peak_cuda_allocated_bytes"] for u in meta.get("updates", [])] or [0]) / GIB
        return rec
    s, updates, args = meta["summary"], meta["updates"], meta["args"]
    n_full = args["submap_size"] + args["overlap"]
    fwd = [u["forward_seconds"] for u in updates]
    full = [u["forward_seconds"] for u in updates if u["n_frames"] == n_full]
    sub = [u["n_frames"] for u in updates]
    cands = []
    for entry in meta.get("loop_log", []):
        for c in entry["candidates"]:
            cands.append({"update": entry["update"], "query_image": c["query_image"],
                          "detected_image": c["detected_image"], "salad_distance": c["salad_distance"],
                          "raw_ratio": c.get("raw_ratio"), "scaled_ratio": c.get("scaled_ratio"),
                          "accepted": bool(c["accepted"]), "note": c.get("note")})
    tau = (meta.get("similarity") or {}).get("tau") if backend == "omega" else 0.95
    rec.update(
        submap_size=args["submap_size"], n_images=meta["input"]["n_images_used"], keyframes=s["keyframes"],
        submaps=s["submaps_processed"], frames_per_submap=sub,
        loop_candidates=s["loop_candidates_total"], loop_closures_accepted=s["loop_closures_accepted"],
        loop_candidate_list=cands, loop_threshold_raw=tau,
        forward_per_submap_s=fwd, forward_total_s=float(sum(fwd)),
        forward_median_full_submap_s=float(np.median(full)) if full else None,
        forward_max_s=float(max(fwd)), n_full_submaps=len(full),
        s_per_frame_full=float(np.median(full) / n_full) if full else None,
        lc_forward_total_s=float(sum(u["lc_forward_seconds"] or 0.0 for u in updates)),
        add_points_total_s=float(sum(u["add_points_seconds"] for u in updates)),
        optimize_total_s=float(s["optimize_seconds_total"]), run_predictions_total_s=float(
            sum(u["run_predictions_seconds"] for u in updates)),
        main_loop_seconds=float(s["total_seconds"]), wall_seconds=float(meta["wall_seconds"]),
        model_load_seconds=meta.get("model_load_seconds"),
        peak_cuda_allocated_gib=s["peak_cuda_allocated_bytes"] / GIB,
        peak_cuda_reserved_gib=s["peak_cuda_reserved_bytes"] / GIB,
        peak_cuda_allocated_per_submap_gib=[u["peak_cuda_allocated_bytes"] / GIB for u in updates],
        peak_rss_gib=meta.get("peak_rss_kib", 0) / 1024 ** 2,
        gpu=meta.get("gpu"), cuda_memory_cap_gib=args.get("cuda_memory_limit_gb"),
        input_shape_first_submap=updates[0]["input_shape"], seed=meta.get("seed"),
        alloc_conf=None)
    gt = meta["input"].get("groundtruth")
    ate_runner = load_json(d / "ate.json")
    if gt and (d / "poses.tum").is_file():
        ate = eval_ate.evaluate(gt, d / "poses.tum")
        rec["ate"] = {k: ate.get(k) for k in ("status", "ate_rmse_m", "ate_mean_m", "ate_median_m", "ate_max_m", "scale",
                                              "n_associated", "ate_rmse_se3_no_scale_m")}
        rec["ate_matches_runner_json"] = (ate_runner is not None and
                                          abs(ate_runner.get("ate_rmse_m", -1) - ate["ate_rmse_m"]) < 1e-12)
    return rec


def pair_sets(rec):
    cand = {stamp_pair(c) for c in rec.get("loop_candidate_list", [])}
    acc = {stamp_pair(c) for c in rec.get("loop_candidate_list", []) if c["accepted"]}
    return cand, acc


def cross_backend_loops(runs, root_tag, seq):
    a, b = runs.get(f"{seq}_vggt_loops1"), runs.get(f"{seq}_omega_loops1")
    if not a or not b or a["status"] != "complete" or b["status"] != "complete":
        return None
    ca, aa = pair_sets(a)
    cb, ab = pair_sets(b)
    return {"candidates_vggt": len(ca), "candidates_omega": len(cb), "candidates_identical": ca == cb,
            "accepted_vggt": len(aa), "accepted_omega": len(ab), "accepted_identical": aa == ab,
            "accepted_both": sorted(map(list, aa & ab)), "accepted_only_vggt": sorted(map(list, aa - ab)),
            "accepted_only_omega": sorted(map(list, ab - aa)),
            "candidates_only_vggt": sorted(map(list, ca - cb)), "candidates_only_omega": sorted(map(list, cb - ca)),
            "keyframes_identical": a["keyframes"] == b["keyframes"] and a["frames_per_submap"] == b["frames_per_submap"]}


def fmt(v, nd=3, none="-"):
    return none if v is None else f"{v:.{nd}f}"


def build(root32, root16):
    r32 = {f"{s}_{b}_loops1": run_record(root32, s, b, 1) for s in SEQS for b in BACKENDS}
    r16 = {f"{s}_{b}_loops{l}": run_record(root16, s, b, l) for s in SEQS for b in BACKENDS for l in (1, 0)}

    def ate(runs, key):
        r = runs[key]
        return (r.get("ate") or {}).get("ate_rmse_m") if r["status"] == "complete" else None

    table = []
    for s in SEQS:
        row = {"seq": s, "label": LABEL[s], "paper_submap32": PAPER[s]}
        for b in BACKENDS:
            row[f"{b}_16_loops1"] = ate(r16, f"{s}_{b}_loops1")
            row[f"{b}_16_loops0"] = ate(r16, f"{s}_{b}_loops0")
            row[f"{b}_32_loops1"] = ate(r32, f"{s}_{b}_loops1")
            row[f"{b}_32_accepted"] = r32[f"{s}_{b}_loops1"].get("loop_closures_accepted")
            row[f"{b}_16_accepted"] = r16[f"{s}_{b}_loops1"].get("loop_closures_accepted")
            row[f"{b}_32_status"] = r32[f"{s}_{b}_loops1"]["status"]
        for k in (16, 32):
            v, o = row[f"vggt_{k}_loops1"], row[f"omega_{k}_loops1"]
            row[f"ratio_omega_over_vggt_{k}"] = (o / v) if (v and o) else None
        v32 = row["vggt_32_loops1"]
        row["vggt32_over_paper"] = (v32 / PAPER[s]) if v32 else None
        row["vggt32_minus_paper_m"] = (v32 - PAPER[s]) if v32 else None
        row["vggt16_over_paper"] = (row["vggt_16_loops1"] / PAPER[s]) if row["vggt_16_loops1"] else None
        table.append(row)
    means = {"paper_submap32": PAPER["mean_of_the_three"]}
    for b in BACKENDS:
        for k, tag in ((16, "16_loops1"), (32, "32_loops1"), (16, "16_loops0")):
            vals = [r[f"{b}_{tag}"] for r in table]
            means[f"{b}_{tag}"] = float(np.mean(vals)) if all(v is not None for v in vals) else None
    for k in (16, 32):
        v, o = means[f"vggt_{k}_loops1"], means[f"omega_{k}_loops1"]
        means[f"ratio_omega_over_vggt_{k}"] = (o / v) if (v and o) else None
    geo = {}
    for k in (16, 32):
        rs = [r[f"ratio_omega_over_vggt_{k}"] for r in table]
        geo[f"geomean_ratio_{k}"] = float(np.exp(np.mean(np.log(rs)))) if all(rs) else None
    cross32 = {s: cross_backend_loops(r32, "32", s) for s in SEQS}
    cross16 = {s: cross_backend_loops(r16, "16", s) for s in SEQS}
    return {"schema": "vggt_omega_slam.submap32.v1", "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "protocol": {"slam": "VGGT-SLAM 2.0 upstream 35327ac, unmodified", "submap_size_new": 32,
                         "submap_size_existing": 16, "min_disparity": 50, "conf_threshold": 25, "lc_thres": 0.95,
                         "max_loops": 1, "seed": 0, "omega_similarity": "config/omega_similarity.json (layer 15, tau 0.3764), frozen",
                         "ate": "Sim(3) Umeyama with scale, 0.01 s association (evo_ape tum -as equivalent, own code)",
                         "cuda_cap_gib": CAP_GIB, "gpu_claim_mb": {"vggt": 8500, "omega": 9500},
                         "allocator_env": "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for the submap-32 runs only "
                                          "(the submap-16 runs were launched without it, so 'reserved' is not comparable)"},
            "paper_reference": PAPER, "runs_submap32": r32, "runs_submap16": r16, "ate_table": table,
            "ate_mean_of_three": means, "ate_geomean_ratio": geo, "loops_cross_backend_submap32": cross32,
            "loops_cross_backend_submap16": cross16}


# ---------------------------------------------------------------------------------------------
def write_markdown(S, path, verdict=None):
    r32, r16 = S["runs_submap32"], S["runs_submap16"]
    T, M = S["ate_table"], S["ate_mean_of_three"]
    L = ["# 子图 32(论文协议)下 VGGT 与 VGGT-Omega 的 VGGT-SLAM 2.0 对比", "",
         f"生成时间 {S['generated']}。上游 VGGT-SLAM 2.0(35327ac)未修改;子图 32(每个子图 32 个新关键帧加 1 个重叠帧,"
         "即 33 帧一次前向)、min_disparity 50、conf 25、lc_thres 0.95、max_loops 1、种子 0,回环开;Omega 用冻结的 "
         "config/omega_similarity.json(层 15,tau 0.3764)。每个配置只跑一次。", "",
         "## ATE RMSE(米,Sim(3) 对齐含尺度)", "",
         "| 序列 | VGGT 子图16 | VGGT 子图32 | 论文(子图32) | VGGT 32 / 论文 | Omega 子图16 | Omega 子图32 | Omega/VGGT 比 @16 | Omega/VGGT 比 @32 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in T:
        L.append(f"| {r['label']} | {fmt(r['vggt_16_loops1'], 4)} | {fmt(r['vggt_32_loops1'], 4)} | {fmt(r['paper_submap32'], 3)} | "
                 f"{fmt(r['vggt32_over_paper'], 2)} | {fmt(r['omega_16_loops1'], 4)} | {fmt(r['omega_32_loops1'], 4)} | "
                 f"{fmt(r['ratio_omega_over_vggt_16'], 2)} | {fmt(r['ratio_omega_over_vggt_32'], 2)} |")
    L.append(f"| 三序列均值 | {fmt(M['vggt_16_loops1'], 4)} | {fmt(M['vggt_32_loops1'], 4)} | {fmt(M['paper_submap32'], 3)} | "
             f"{fmt(M['vggt_32_loops1'] / M['paper_submap32'] if M['vggt_32_loops1'] else None, 2)} | {fmt(M['omega_16_loops1'], 4)} | "
             f"{fmt(M['omega_32_loops1'], 4)} | {fmt(M['ratio_omega_over_vggt_16'], 2)} | {fmt(M['ratio_omega_over_vggt_32'], 2)} |")
    g = S["ate_geomean_ratio"]
    L += ["", f"三序列 Omega/VGGT 比的几何平均:子图 16 为 {fmt(g['geomean_ratio_16'], 2)},子图 32 为 {fmt(g['geomean_ratio_32'], 2)}。"
          "子图 16 的数字取自 runs/compare(回环开,种子 0,单次);论文数字为 Table I(Uncalibrated 栏 VGGT-SLAM 2.0 一行),"
          "三序列均值是本表自算(论文只给 9 序列平均 0.041)。", "",
          "参考:子图 16 回环关的 ATE(同一协议,仅关回环)", "",
          "| 序列 | VGGT 回环关 | Omega 回环关 |", "|---|---|---|"]
    for r in T:
        L.append(f"| {r['label']} | {fmt(r['vggt_16_loops0'], 4)} | {fmt(r['omega_16_loops0'], 4)} |")

    # ---- loops
    L += ["", "## 子图结构与回环决定", "",
          "每个序列的关键帧选择(LK 光流、min_disparity 50)与后端无关,两个后端的关键帧序列和子图划分逐项相同"
          "(见 summary.json 的 keyframes_identical)。回环候选由 SALAD 在图像上检索,同样与前端无关;前端只决定验证前向的相似度比值。", "",
          "| 序列 | 子图大小 | 关键帧 | 子图帧数 | 后端 | 候选 | 接受 | 候选(查询帧→被检测帧):原始比值 / 阈值,✓接受 ✗拒绝 |", "|---|---|---|---|---|---|---|---|"]
    for s in SEQS:
        for tag, runs in (("32", r32), ("16", r16)):
            for b in BACKENDS:
                r = runs[f"{s}_{b}_loops1"]
                if r["status"] != "complete":
                    L.append(f"| {LABEL[s]} | {tag} | 运行{r['status']} | | {NAME[b]} | | | {r.get('error')} |")
                    continue
                thr = r["loop_threshold_raw"]
                ev = "; ".join(f"{c['query_image'][:-4]}→{c['detected_image'][:-4]}: {fmt(c['raw_ratio'], 3)} / {fmt(thr, 4)}"
                               f"{'✓' if c['accepted'] else '✗'}" if c["raw_ratio"] is not None else
                               f"{c['query_image'][:-4]}→{c['detected_image'][:-4]}: 未验证✗" for c in r["loop_candidate_list"]) or "无候选"
                L.append(f"| {LABEL[s]} | {tag} | {r['keyframes']} | {'+'.join(map(str, r['frames_per_submap']))} | {NAME[b]} | "
                         f"{r['loop_candidates']} | {r['loop_closures_accepted']} | {ev} |")
    L += ["", "两个后端在子图 32 下的回环决定是否相同:", "", "| 序列 | 候选集相同 | 接受集相同 | 两者都接受 | 仅 VGGT 接受 | 仅 Omega 接受 |", "|---|---|---|---|---|---|"]
    for s in SEQS:
        c = S["loops_cross_backend_submap32"][s]
        if c is None:
            L.append(f"| {LABEL[s]} | (有运行失败) | | | | |")
            continue
        fp = lambda lst: "; ".join(f"{a[:-4]}→{b[:-4]}" for a, b in lst) or "无"  # noqa: E731
        L.append(f"| {LABEL[s]} | {'是' if c['candidates_identical'] else '否'} | {'是' if c['accepted_identical'] else '否'} | "
                 f"{fp(c['accepted_both'])} | {fp(c['accepted_only_vggt'])} | {fp(c['accepted_only_omega'])} |")

    # ---- timing / memory
    L += ["", "## 耗时与显存(回环开,子图 32;括号内为同序列子图 16 的值)", "",
          "| 序列 | 后端 | 前向总计 s | 33 帧子图前向中位数 s(17 帧) | 秒/帧(33 帧) | 验证前向合计 s | 优化合计 s | 主循环 s | 整个进程 s | 显存峰值 allocated GiB(16) | reserved GiB | 进程峰值内存 GiB | 输入分辨率 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in SEQS:
        for b in BACKENDS:
            r, q = r32[f"{s}_{b}_loops1"], r16[f"{s}_{b}_loops1"]
            if r["status"] != "complete":
                L.append(f"| {LABEL[s]} | {NAME[b]} | 运行{r['status']}:{r.get('error')} | | | | | | | | | | |")
                continue
            shp = r["input_shape_first_submap"]
            L.append(f"| {LABEL[s]} | {NAME[b]} | {fmt(r['forward_total_s'], 1)} | {fmt(r['forward_median_full_submap_s'], 2)} "
                     f"({fmt(q.get('forward_median_full_submap_s'), 2)}) | {fmt(r['s_per_frame_full'], 3)} | {fmt(r['lc_forward_total_s'], 2)} | "
                     f"{fmt(r['optimize_total_s'], 3)} | {fmt(r['main_loop_seconds'], 1)} | {fmt(r['wall_seconds'], 1)} | "
                     f"{fmt(r['peak_cuda_allocated_gib'], 2)} ({fmt(q.get('peak_cuda_allocated_gib'), 2)}) | "
                     f"{fmt(r['peak_cuda_reserved_gib'], 2)} | {fmt(r['peak_rss_gib'], 1)} | {shp[-2]}x{shp[-1]} |")
    L += ["", f"CUDA 分配器上限 {CAP_GIB:.0f} GiB(torch.cuda.set_per_process_memory_fraction),GPU 锁申请量 VGGT 8500 MB、Omega 9500 MB(另留 2048 MB 给桌面)。"
          "子图 32 的运行带 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,子图 16 的运行没有,所以 reserved 不可比,allocated 可比。"
          "输入分辨率:VGGT 392x518,Omega 448x592(同为 1036 个 patch token/帧)。", ""]

    # ---- notes
    L += ["## 说明与限制", "",
          "- 每个配置只跑一次(种子 0),没有多种子重复;ATE 用自写的 evo -as 等价实现(本机未装 evo)。",
          "- 子图 32 时序列被切成很少的子图(desk 2 个、desk2 4 个、room 6 个)。上游只在与当前子图不相邻的早期子图里找回环候选,"
          "所以少子图的序列几乎没有回环机会;「回环开」在 desk 上与「回环关」等价。子图 16 与 32 之间的差别因此同时混入了"
          "「子图大小」和「可用回环数」两个因素。",
          "- 论文 Table I 用子图 32;其 ATE 评测工具链与本机自写实现、种子、关键帧选择实现细节是否逐位一致无法核实。", ""]
    if verdict:
        L += [verdict.strip(), ""]
    failed = [r for r in list(r32.values()) if r["status"] != "complete"]
    if failed:
        L += ["## 失败或缺失的运行", ""] + [f"- {r['name']}:{r['status']},{r.get('error')}" for r in failed] + [""]
    Path(path).write_text("\n".join(L))


# ---------------------------------------------------------------------------------------------
def style_axis(ax, grid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK3)
    ax.tick_params(colors=INK2, labelsize=10.5, length=3)
    if grid:
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def make_figure(S, path):
    r32, r16, T = S["runs_submap32"], S["runs_submap16"], S["ate_table"]
    fig = plt.figure(figsize=(15, 10.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.5, wspace=0.22, left=0.06, right=0.985, top=0.9, bottom=0.07)
    fig.text(0.015, 0.968, "子图 32(论文协议)与子图 16:VGGT 与 VGGT-Omega 在 VGGT-SLAM 2.0 中的对比", fontsize=15.5,
             color=INK, fontweight="bold", ha="left")
    fig.text(0.015, 0.940, "TUM fr1 desk / desk2 / room,回环开,种子 0,每个配置单次运行;上游 VGGT-SLAM 2.0 未修改", fontsize=11, color=INK2, ha="left")

    # (a) ATE bars ------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    width, gap = 0.19, 0.012
    offsets = {("vggt", 16): -1.5, ("vggt", 32): -0.5, ("omega", 16): 0.5, ("omega", 32): 1.5}
    ymax = 0.0
    for i, row in enumerate(T):
        for (b, k), o in offsets.items():
            v = row[f"{b}_{k}_loops1"]
            x = i + o * (width + gap)
            if v is None:
                ax.text(x, 0.002, "失败", ha="center", va="bottom", fontsize=9.5, color=INK2, rotation=90)
                continue
            ymax = max(ymax, v)
            if k == 16:
                ax.bar(x, v, width, color=COL_LIGHT[b], edgecolor=COL[b], linewidth=1.2, zorder=3)
            else:
                ax.bar(x, v, width, color=COL[b], edgecolor=COL[b], linewidth=1.2, zorder=3)
            p_ = row["paper_submap32"]
            ly = v + 0.0016
            if ly - 0.0010 < p_ < ly + 0.0036:  # keep the value label clear of the paper line
                ly = p_ + 0.0016
            ax.text(x, ly, f"{v:.3f}", ha="center", va="bottom", fontsize=10, color=INK, zorder=5)
        p = row["paper_submap32"]
        ax.plot([i - 2.05 * (width + gap), i + 2.05 * (width + gap)], [p, p], color=INK, lw=1.6, zorder=6, solid_capstyle="butt")
        ax.scatter([i], [p], marker="D", s=46, color="white", edgecolor=INK, lw=1.5, zorder=7)
        ax.text(i + 2.1 * (width + gap), p, f"论文 {p:.3f}", ha="left", va="center", fontsize=9.5, color=INK2, zorder=7)
        ymax = max(ymax, p)
    ticks = []
    for row in T:
        s = row["seq"]
        n = lambda tag, b: row.get(f"{b}_{tag}_accepted")  # noqa: E731
        ticks.append(f"{row['label']}\n接受回环数 子图16:VGGT {n('16', 'vggt')}、Omega {n('16', 'omega')}"
                     f"\n接受回环数 子图32:VGGT {n('32', 'vggt')}、Omega {n('32', 'omega')}")
    ax.set_xticks(range(len(T)))
    ax.set_xticklabels(ticks, fontsize=10.5)
    ax.set_xlim(-0.6, len(T) - 0.35)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_ylabel("ATE RMSE(米,越低越好)", fontsize=11, color=INK2)
    style_axis(ax)
    ax.set_title("(a) 轨迹精度:子图 16 与 32,论文数字作标记", loc="left", fontsize=13, color=INK, fontweight="bold")
    handles = [Patch(facecolor=BLUE_LIGHT, edgecolor=BLUE, label="VGGT 子图16"), Patch(facecolor=BLUE, edgecolor=BLUE, label="VGGT 子图32"),
               Patch(facecolor=ORANGE_LIGHT, edgecolor=ORANGE, label="Omega 子图16"), Patch(facecolor=ORANGE, edgecolor=ORANGE, label="Omega 子图32"),
               Line2D([0], [0], color=INK, lw=1.6, marker="D", markerfacecolor="white", markeredgecolor=INK, ms=6, label="论文 Table I(子图32)")]
    ax.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=10, columnspacing=1.6, handlelength=1.4)

    # (b) per-submap forward time vs frames --------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    for b in BACKENDS:
        xs_all, ys_all = [], []
        for runs, k in ((r16, 16), (r32, 32)):
            for s in SEQS:
                r = runs[f"{s}_{b}_loops1"]
                if r["status"] != "complete":
                    continue
                jit = (SEQS.index(s) - 1) * 0.5
                xs = np.array(r["frames_per_submap"], dtype=float) + jit
                ax.scatter(xs, r["forward_per_submap_s"], s=36, marker="o", color=COL[b] if k == 32 else "white",
                           edgecolor=COL[b], lw=1.4, zorder=4)
                xs_all += r["frames_per_submap"]
                ys_all += r["forward_per_submap_s"]
        full = {}
        for n_, y_ in zip(xs_all, ys_all):
            full.setdefault(n_, []).append(y_)
        ns = sorted(n_ for n_ in full if len(full[n_]) >= 3)
        ax.plot(ns, [np.median(full[n_]) for n_ in ns], color=COL[b], lw=1.8, zorder=3, label=f"{NAME[b]} 各帧数中位数")
    ax.set_xlabel("每个子图的帧数(含 1 个重叠帧)", fontsize=11, color=INK2)
    ax.set_ylabel("单次前向耗时(秒,GPU 同步)", fontsize=11, color=INK2)
    ax.set_xlim(0, 36)
    ax.set_ylim(0, None)
    style_axis(ax)
    ax.set_title("(b) 每个子图的前向耗时", loc="left", fontsize=13, color=INK, fontweight="bold")
    h = [Line2D([0], [0], color=COL["vggt"], lw=1.8, label="VGGT 中位数连线"), Line2D([0], [0], color=COL["omega"], lw=1.8, label="Omega 中位数连线"),
         Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=INK2, ms=7, label="空心:子图16 的运行"),
         Line2D([0], [0], marker="o", color="none", markerfacecolor=INK2, markeredgecolor=INK2, ms=7, label="实心:子图32 的运行")]
    ax.legend(handles=h, loc="upper left", frameon=False, fontsize=10)

    # (c) peak CUDA allocated ----------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    w = 0.19
    for i, s in enumerate(SEQS):
        for (b, k), o in offsets.items():
            r = (r32 if k == 32 else r16)[f"{s}_{b}_loops1"]
            x = i + o * (w + 0.012)
            if r["status"] != "complete":
                ax.text(x, 0.1, "失败", ha="center", va="bottom", fontsize=9.5, color=INK2, rotation=90)
                continue
            v = r["peak_cuda_allocated_gib"]
            ax.bar(x, v, w, color=(COL_LIGHT[b] if k == 16 else COL[b]), edgecolor=COL[b], linewidth=1.2, zorder=3)
            ax.text(x, v + 0.12, f"{v:.1f}", ha="center", va="bottom", fontsize=9.5, color=INK, zorder=5)
    ax.axhline(CAP_GIB, color=INK, lw=1.4, ls=(0, (5, 3)), zorder=6)
    ax.text(len(SEQS) - 0.62, CAP_GIB + 0.15, f"分配器上限 {CAP_GIB:.0f} GiB", ha="right", va="bottom", fontsize=10, color=INK2)
    ax.set_xticks(range(len(SEQS)))
    ax.set_xticklabels([LABEL[s] for s in SEQS], fontsize=11)
    ax.set_xlim(-0.6, len(SEQS) - 0.4)
    ax.set_ylim(0, CAP_GIB * 1.18)
    ax.set_ylabel("CUDA 峰值 allocated(GiB)", fontsize=11, color=INK2)
    style_axis(ax)
    ax.set_title("(c) 显存峰值(含常驻权重、SALAD、已存帧)", loc="left", fontsize=13, color=INK, fontweight="bold")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir32", type=Path, default=ROOT / "runs" / "submap32")
    p.add_argument("--dir16", type=Path, default=ROOT / "runs" / "compare")
    p.add_argument("--verdict-file", type=Path, default=None)
    args = p.parse_args()
    S = build(args.dir32, args.dir16)
    (args.dir32 / "summary.json").write_text(json.dumps(S, indent=2, default=float, ensure_ascii=False))
    verdict = args.verdict_file.read_text() if args.verdict_file and args.verdict_file.is_file() else None
    write_markdown(S, args.dir32 / "summary_zh.md", verdict)
    make_figure(S, args.dir32 / "子图32对比.png")
    print("wrote summary.json, summary_zh.md, 子图32对比.png in", args.dir32)
    for r in S["ate_table"]:
        print(r["label"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("seq", "label")})


if __name__ == "__main__":
    main()
