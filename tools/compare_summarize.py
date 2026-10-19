#!/usr/bin/env python3
"""Collect the comparison runs (runs/compare/<seq>_<backend>_loops<0|1>) into summary.json and summary_zh.md.

CPU only. Every number the figure uses is written to summary.json by this script; the figure script reads it.
Usage:  OMP_NUM_THREADS=4 python tools/compare_summarize.py [--compare-dir runs/compare]
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import eval_ate  # noqa: E402
from vos import compare_utils as cu  # noqa: E402

SEQS = {"office_loop": {"label": "office_loop", "tum": None},
        "fr1_desk": {"label": "desk", "tum": "rgbd_dataset_freiburg1_desk"},
        "fr1_desk2": {"label": "desk2", "tum": "rgbd_dataset_freiburg1_desk2"},
        "fr1_room": {"label": "room", "tum": "rgbd_dataset_freiburg1_room"}}
TUM_SEQS = ["fr1_desk", "fr1_desk2", "fr1_room"]
BACKENDS = ["vggt", "omega"]
GIB = 1024 ** 3

# VGGT-SLAM 2.0 paper (Maggio & Carlone 2026), Table I, PDF page 7 of 10, block "Uncalibrated", row
# "VGGT-SLAM 2.0": 360 .050, desk .025, desk2 .029, floor .102, plant .026, room .063, rpy .026, teddy .038,
# xyz .014, avg .041. Read from the text layer and checked against a rendering of the page. The paper uses
# submaps of 32 frames for this table ("For fairness, we use submaps of size 32 frames", Sec. V-B).
PAPER = {"source": "Maggio & Carlone, VGGT-SLAM 2.0, Table I (ATE RMSE, m, TUM RGB-D), PDF page 7",
         "row": "VGGT-SLAM 2.0 (Uncalibrated block)", "submap_size": 32,
         "fr1_desk": 0.025, "fr1_desk2": 0.029, "fr1_room": 0.063,
         "avg_all_9_sequences": 0.041,
         "all_9_sequences": {"360": 0.050, "desk": 0.025, "desk2": 0.029, "floor": 0.102, "plant": 0.026,
                             "room": 0.063, "rpy": 0.026, "teddy": 0.038, "xyz": 0.014},
         "mean_of_the_three": float(np.mean([0.025, 0.029, 0.063])),
         "note": "The paper's average (0.041) is over all 9 TUM sequences; only 3 are run here, so the 3-sequence "
                 "mean is given separately. Columns are 360 desk desk2 floor plant room rpy teddy xyz avg; the "
                 "room value is 0.063 (the 0.026 in the build report was the neighbouring plant/rpy column)."}


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def run_record(compare_dir, seq, backend, loops):
    name = f"{seq}_{backend}_loops{loops}"
    d = compare_dir / name
    rec = {"name": name, "seq": seq, "backend": backend, "max_loops": loops, "dir": str(d)}
    meta = load_json(d / "metadata.json")
    if meta is None:
        rec.update(status="missing")
        return rec
    rec["status"] = meta.get("status")
    if meta.get("status") != "complete":
        rec["error"] = meta.get("error")
        rec["traceback_tail"] = (meta.get("traceback") or "")[-800:]
        rec["partial_updates"] = len(meta.get("updates", []))
        return rec
    s, updates = meta["summary"], meta["updates"]
    forwards = [u["forward_seconds"] for u in updates]
    nframes = [u["n_frames"] for u in updates]
    full = [f for f, n in zip(forwards, nframes) if n == meta["args"]["submap_size"] + meta["args"]["overlap"]]
    sim = meta.get("similarity", {})
    rec.update(
        n_images=meta["input"]["n_images_used"], keyframes=s["keyframes"], submaps=s["submaps_processed"],
        input_shape_first_submap=updates[0]["input_shape"],
        loop_candidates=s["loop_candidates_total"], loop_closures_accepted=s["loop_closures_accepted"],
        loop_events=[{"update": e["update"], "salad_distance": c["salad_distance"], "raw_ratio": c.get("raw_ratio"),
                      "scaled_ratio": c.get("scaled_ratio"), "accepted": c["accepted"],
                      "query_image": c["query_image"], "detected_image": c["detected_image"]}
                     for e in meta.get("loop_log", []) for c in e["candidates"]],
        similarity={k: sim.get(k) for k in ("layer", "tau", "token_offset", "source")},
        forward_per_submap_s=forwards, frames_per_submap=nframes,
        forward_total_s=float(sum(forwards)),
        forward_median_per_submap_s=float(np.median(forwards)),
        forward_median_per_full_submap_s=float(np.median(full)) if full else None,
        forward_first_submap_s=forwards[0],
        lc_forward_total_s=float(sum(u["lc_forward_seconds"] or 0.0 for u in updates)),
        optimize_total_s=s["optimize_seconds_total"],
        fps_keyframes_forward=s["keyframes"] / float(sum(forwards)),
        fps_keyframes_whole_loop=s["keyframes"] / s["total_seconds"],
        main_loop_seconds=s["total_seconds"], keyframe_selection_seconds=s["upstream_style_timers"]["keyframe_selection_total"],
        model_load_seconds=meta.get("model_load_seconds"), wall_seconds=meta.get("wall_seconds"),
        peak_cuda_allocated_gib=s["peak_cuda_allocated_bytes"] / GIB,
        peak_cuda_reserved_gib=s["peak_cuda_reserved_bytes"] / GIB,
        peak_cuda_allocated_per_submap_gib=[u["peak_cuda_allocated_bytes"] / GIB for u in updates],
        peak_rss_gib=meta.get("peak_rss_kib", 0) / 1024 ** 2,
        cloud_points_written=meta.get("cloud", {}).get("written_points"),
        runner_sha256=meta.get("runner_sha256"), seed=meta.get("seed"))
    gt = meta["input"].get("groundtruth")
    if gt:
        ate = load_json(d / "ate.json") or eval_ate.evaluate(gt, d / "poses.tum")
        rec["ate"] = {k: ate.get(k) for k in ("status", "ate_rmse_m", "ate_mean_m", "ate_median_m", "ate_max_m", "scale",
                                              "n_associated", "n_estimate_unique_stamps", "ate_rmse_se3_no_scale_m",
                                              "path_length_groundtruth_m", "error")}
    else:
        stamps, xyz, _ = cu.read_poses(d / "poses.tum")
        rec["start_end"] = cu.start_end_gap(xyz)
        rec["start_end"]["n_poses"] = int(len(stamps))
        rec["start_end"]["label"] = "proxy: only meaningful because the camera returns near its start"
    return rec


def omega_vs_vggt(compare_dir, seq, loops):
    a, b = compare_dir / f"{seq}_vggt_loops{loops}" / "poses.tum", compare_dir / f"{seq}_omega_loops{loops}" / "poses.tum"
    if not (a.is_file() and b.is_file()):
        return None
    sa, xa, _ = cu.read_poses(a)
    sb, xb, _ = cu.read_poses(b)
    ia, ib = cu.common_stamps(sa, sb)
    if len(ia) < 4:
        return None
    r, t, c = cu.umeyama(xb[ib], xa[ia])
    res = np.linalg.norm(cu.apply_sim3(xb[ib], r, t, c) - xa[ia], axis=1)
    extent = float(np.linalg.norm(xa.max(0) - xa.min(0)))
    return {"n_common_stamps": int(len(ia)), "sim3_rmse_vggt_units": float(np.sqrt((res ** 2).mean())),
            "vggt_bbox_diagonal": extent, "rmse_over_diagonal": float(np.sqrt((res ** 2).mean()) / extent),
            "scale_omega_to_vggt": c}


def pair_gaps(compare_dir, runs, backend):
    """Distance between the two frames of every loop closure accepted in the loops-on run, measured in the loops-on
    and the loops-off poses (same model units inside a run), as a fraction of that run's trajectory length."""
    on, off = runs.get(f"office_loop_{backend}_loops1"), runs.get(f"office_loop_{backend}_loops0")
    if not on or on["status"] != "complete" or not off or off["status"] != "complete":
        return None
    out = []
    for e in (e for e in on["loop_events"] if e["accepted"]):
        row = {"query_image": e["query_image"], "detected_image": e["detected_image"]}
        for tag, name in (("loops1", "office_loop_%s_loops1" % backend), ("loops0", "office_loop_%s_loops0" % backend)):
            stamps, xyz, _ = cu.read_poses(compare_dir / name / "poses.tum")
            index = {st: i for i, st in enumerate(stamps)}
            a, b = index[cu.stamp_of(e["query_image"])], index[cu.stamp_of(e["detected_image"])]
            length = cu.path_length(xyz)
            row[f"pair_distance_{tag}"] = float(np.linalg.norm(xyz[a] - xyz[b]))
            row[f"pair_gap_fraction_{tag}"] = row[f"pair_distance_{tag}"] / length
        out.append(row)
    return out


def fmt(v, nd=3, none="-"):
    return none if v is None else f"{v:.{nd}f}"


def build(compare_dir):
    runs = {}
    for seq in SEQS:
        for backend in BACKENDS:
            for loops in (1, 0):
                if seq == "office_loop" and loops == 0 and not (compare_dir / f"{seq}_{backend}_loops0").exists():
                    continue
                runs[f"{seq}_{backend}_loops{loops}"] = run_record(compare_dir, seq, backend, loops)
    # ATE table
    table = []
    for seq in TUM_SEQS:
        row = {"seq": seq, "label": SEQS[seq]["label"], "paper_submap32": PAPER[seq]}
        for backend in BACKENDS:
            for loops in (1, 0):
                rec = runs[f"{seq}_{backend}_loops{loops}"]
                ate = rec.get("ate") or {}
                row[f"{backend}_loops{loops}"] = ate.get("ate_rmse_m")
                row[f"{backend}_loops{loops}_status"] = rec["status"] if ate.get("status") in (None, "ok") else ate["status"]
        table.append(row)
    means = {"paper_submap32": PAPER["mean_of_the_three"]}
    for backend in BACKENDS:
        for loops in (1, 0):
            vals = [r[f"{backend}_loops{loops}"] for r in table]
            means[f"{backend}_loops{loops}"] = float(np.mean(vals)) if all(v is not None for v in vals) else None
    # vggt harness sanity vs paper
    sanity = []
    for r in table:
        v = r["vggt_loops1"]
        sanity.append({"seq": r["label"], "ours_vggt_loops1_m": v, "paper_m": r["paper_submap32"],
                       "ratio_ours_over_paper": (v / r["paper_submap32"]) if v is not None else None,
                       "difference_m": (v - r["paper_submap32"]) if v is not None else None,
                       "known_differences": "submap 16 (upstream default / eval_tum.sh) vs 32 in the paper's Table I; "
                                            "seed 0; evo not installed (own evo-equivalent ATE); single run"})
    office = {}
    for backend in BACKENDS:
        for loops in (1, 0):
            rec = runs.get(f"office_loop_{backend}_loops{loops}")
            if rec and rec["status"] == "complete":
                office[f"{backend}_loops{loops}"] = {
                    "keyframes": rec["keyframes"], "submaps": rec["submaps"], "loop_closures_accepted": rec["loop_closures_accepted"],
                    "loop_candidates": rec["loop_candidates"], "loop_events": rec["loop_events"], **rec["start_end"]}
    for backend in BACKENDS:
        pg = pair_gaps(compare_dir, runs, backend)
        if pg is not None and f"{backend}_loops1" in office:
            office[f"{backend}_loops1"]["accepted_pair_gap"] = pg
    separation = {}
    for backend in BACKENDS:
        events = [e for seq in SEQS for e in runs[f"{seq}_{backend}_loops1"].get("loop_events", []) if e["raw_ratio"] is not None]
        rej = [e["raw_ratio"] for e in events if not e["accepted"]]
        acc = [e["raw_ratio"] for e in events if e["accepted"]]
        separation[backend] = {"n_verified_candidates": len(events), "max_rejected_raw_ratio": max(rej) if rej else None,
                               "min_accepted_raw_ratio": min(acc) if acc else None,
                               "threshold": 0.95 if backend == "vggt" else (load_json(ROOT / "config" / "omega_similarity.json") or {}).get("tau")}
    cross = {f"{seq}_loops{loops}": omega_vs_vggt(compare_dir, seq, loops) for seq in SEQS for loops in (1, 0)
             if (compare_dir / f"{seq}_omega_loops{loops}").exists()}
    cfg = load_json(ROOT / "config" / "omega_similarity.json") or {}
    return {"schema": "vggt_omega_slam.compare.v1", "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "protocol": {"slam": "VGGT-SLAM 2.0 upstream 35327ac, unmodified", "submap_size": 16, "min_disparity": 50,
                         "conf_threshold": 25, "lc_thres": 0.95, "max_loops": "1 (on) / 0 (off)", "seed": 0,
                         "omega_similarity": {k: cfg.get(k) for k in ("layer", "tau", "dataset")},
                         "vggt_similarity": "layer 21, token_offset 5, threshold 0.95 (upstream)",
                         "ate": "Sim(3) Umeyama with scale, max time diff 0.01 s (evo_ape tum -as equivalent, own code)",
                         "input_resolution": {"vggt": "294x518 (16:9) / 392x518 (4:3)", "omega": "384x688 (16:9) / 448x592 (4:3)"}},
            "paper_reference": PAPER, "runs": runs, "ate_table": table, "ate_mean_of_three": means,
            "harness_sanity_vggt_vs_paper": sanity, "loop_ratio_separation": separation, "office_loop": office, "omega_vs_vggt_trajectory_sim3": cross}


def write_markdown(summary, path):
    runs, L = summary["runs"], []
    L += ["# VGGT 与 VGGT-Omega 在 VGGT-SLAM 2.0 中的对比", "",
          f"生成时间 {summary['generated']}。上游 VGGT-SLAM 2.0（35327ac）未修改；默认协议：子图 16、min_disparity 50、"
          "conf 25、lc_thres 0.95、种子 0。Omega 相似度层与阈值取自 office_loop 上冻结的校准"
          f"（层 {summary['protocol']['omega_similarity']['layer']}，tau {fmt(summary['protocol']['omega_similarity']['tau'], 4)}）。", "",
          "## TUM RGB-D：ATE RMSE（米，Sim(3) 对齐含尺度）", "",
          "| 序列 | VGGT 回环开 | VGGT 回环关 | Omega 回环开 | Omega 回环关 | 论文 VGGT-SLAM 2.0（子图 32） |", "|---|---|---|---|---|---|"]
    for r in summary["ate_table"]:
        L.append(f"| {r['label']} | {fmt(r['vggt_loops1'])} | {fmt(r['vggt_loops0'])} | {fmt(r['omega_loops1'])} | "
                 f"{fmt(r['omega_loops0'])} | {fmt(r['paper_submap32'])} |")
    m = summary["ate_mean_of_three"]
    L.append(f"| 三序列均值 | {fmt(m['vggt_loops1'])} | {fmt(m['vggt_loops0'])} | {fmt(m['omega_loops1'])} | "
             f"{fmt(m['omega_loops0'])} | {fmt(m['paper_submap32'])} |")
    L += ["", "论文出处：Table I，PDF 第 7 页，Uncalibrated 栏 VGGT-SLAM 2.0 一行；论文给出的 9 序列平均为 0.041（本表只含 3 条序列，故另算三序列均值）。"
          "论文该表使用 32 帧子图，本次用上游默认的 16 帧子图，二者不可直接等同。", "",
          "## 回环接受情况（回环开）", "", "| 序列 | 后端 | 关键帧 | 子图 | 回环候选 | 接受 | 候选原始比值 |", "|---|---|---|---|---|---|---|"]
    for seq in SEQS:
        for backend in BACKENDS:
            r = runs.get(f"{seq}_{backend}_loops1")
            if not r or r["status"] != "complete":
                L.append(f"| {SEQS[seq]['label']} | {backend} | 运行失败或缺失（{(r or {}).get('status')}） | | | | |")
                continue
            ev = "; ".join(f"{fmt(e['raw_ratio'], 3)}{'✓' if e['accepted'] else '✗'}" for e in r["loop_events"]) or "-"
            L.append(f"| {SEQS[seq]['label']} | {backend} | {r['keyframes']} | {r['submaps']} | {r['loop_candidates']} | "
                     f"{r['loop_closures_accepted']} | {ev} |")
    L += ["", "## office_loop 首尾间隙（代理指标，无真值）", "",
          "首末相机位置距离占轨迹总长的比例。该指标仅因相机最终回到起点附近才有意义，不是精度指标。", "",
          "| 配置 | 回环接受 | 首尾间隙/轨迹长 |", "|---|---|---|"]
    for key, o in summary["office_loop"].items():
        L.append(f"| {key} | {o['loop_closures_accepted']} | {fmt(o['gap_fraction'], 4)} |")
    L += ["", "回环对间距：回环开运行里被接受的回环所连两帧，在回环开/关两次运行的位姿中的距离，除以各自的轨迹长（同样是代理指标）。", "",
          "| 后端 | 回环对 | 间距/轨迹长 回环开 | 回环关 |", "|---|---|---|---|"]
    for backend in BACKENDS:
        for row in summary["office_loop"].get(f"{backend}_loops1", {}).get("accepted_pair_gap", []):
            L.append(f"| {backend} | {row['query_image']} → {row['detected_image']} | {fmt(row['pair_gap_fraction_loops1'], 4)} | "
                     f"{fmt(row['pair_gap_fraction_loops0'], 4)} |")
    L += ["", "## 耗时与显存（回环开）", "",
          "| 序列 | 后端 | 前向总计 s | 子图前向中位数 s | 满 17 帧子图中位数 s | 关键帧/秒（仅前向） | 关键帧/秒（含选帧的主循环） | 显存峰值 allocated GiB | reserved GiB | 输入分辨率 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for seq in SEQS:
        for backend in BACKENDS:
            r = runs.get(f"{seq}_{backend}_loops1")
            if not r or r["status"] != "complete":
                continue
            shp = r["input_shape_first_submap"]
            L.append(f"| {SEQS[seq]['label']} | {backend} | {fmt(r['forward_total_s'], 1)} | {fmt(r['forward_median_per_submap_s'], 2)} | "
                     f"{fmt(r['forward_median_per_full_submap_s'], 2)} | {fmt(r['fps_keyframes_forward'], 2)} | "
                     f"{fmt(r['fps_keyframes_whole_loop'], 2)} | {fmt(r['peak_cuda_allocated_gib'], 2)} | "
                     f"{fmt(r['peak_cuda_reserved_gib'], 2)} | {shp[-2]}x{shp[-1]} |")
    L += ["", "显存峰值含常驻权重（VGGT 为 bf16，Omega 为 fp32）。两个后端的输入分辨率不同，前向耗时差异同时来自模型与分辨率。", ""]
    sep, san = summary["loop_ratio_separation"], summary["harness_sanity_vggt_vs_paper"]
    L += ["## 说明与限制", "",
          "- 每个配置只跑了一次（种子 0），三条 TUM 序列加一条 office_loop，样本量小，不足以给出统计结论。"
          "VGGT 在 desk 上的 ATE（0.0249015）与上一阶段路径检查运行逐位相同，说明该配置可重复，但没有做多种子对比。",
          "- VGGT 复现与论文 Table I 的比值（本次/论文）：" + "，".join(f"{r['seq']} {fmt(r['ratio_ours_over_paper'], 2)}" for r in san)
          + "。论文用 32 帧子图，本次 16 帧，且 ATE 为自写的 evo 等价实现（本机未装 evo），只能说明量级一致，不能当作逐位复现。",
          "- Omega 的回环阈值 tau 由 office_loop 一次性标定后冻结（TUM 未参与）。回环开运行中已验证候选的原始比值：Omega 被拒最大 "
          f"{fmt(sep['omega']['max_rejected_raw_ratio'])}、被接受最小 {fmt(sep['omega']['min_accepted_raw_ratio'])}、tau {fmt(sep['omega']['threshold'], 4)}；"
          f"VGGT 被拒最大 {fmt(sep['vggt']['max_rejected_raw_ratio'])}、被接受最小 {fmt(sep['vggt']['min_accepted_raw_ratio'])}、阈值 0.95。"
          "Omega 最大被拒值距 tau 很近，结论对 tau 的取值有依赖。",
          "- 两个前端的输入分辨率不同（VGGT 518 宽，Omega 平衡 512 档），回环验证用的相似度层与阈值也各自标定，二者的原始比值不可直接比较。",
          "- office_loop 无真值。首尾间隙/轨迹长在回环开时没有比回环关时更小（" + "；".join(
              f"{b} {fmt(summary['office_loop'][f'{b}_loops1']['gap_fraction'] * 100, 1)}% 对 {fmt(summary['office_loop'][f'{b}_loops0']['gap_fraction'] * 100, 1)}%"
              for b in BACKENDS if f"{b}_loops0" in summary["office_loop"]) + "），"
          "这个指标衡量的是第一个与最后一个位姿，而被接受的回环连的是最后一帧与第 18 帧（未做进一步验证这是否是原因）；"
          "回环所连两帧的间距是更直接的代理，见上表。两者都只是代理指标，不是精度。",
          "- 论文 room 的数值是 0.063（构建阶段报告里的 0.026 读错了列）。", ""]
    failed = [r for r in runs.values() if r["status"] != "complete"]
    if failed:
        L += ["## 失败或缺失的运行", ""] + [f"- {r['name']}：{r['status']}，{r.get('error')}" for r in failed] + [""]
    Path(path).write_text("\n".join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--compare-dir", type=Path, default=ROOT / "runs" / "compare")
    args = p.parse_args()
    summary = build(args.compare_dir)
    (args.compare_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_markdown(summary, args.compare_dir / "summary_zh.md")
    print(f"wrote {args.compare_dir / 'summary.json'} and summary_zh.md")
    for r in summary["ate_table"]:
        print(r["label"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k.endswith(("loops1", "loops0"))})


if __name__ == "__main__":
    main()
