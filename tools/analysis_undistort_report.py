"""results.json, results_zh.md and 去畸变对比.png of the undistortion test (CPU only)."""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vos import env  # noqa: E402,F401

import numpy as np  # noqa: E402

import analysis_undistort_cpu as cpu  # noqa: E402

OUT, OUT_ROOT = cpu.OUT, cpu.OUT_ROOT
SEQS_L = cpu.SEQ_LIST
BACKENDS = cpu.BACKENDS
COLORS = {"vggt": "#2a78d6", "omega": "#eb6834"}  # categorical slots 1 and 2 of the validated reference palette
LABELS = {"vggt": "VGGT", "omega": "Omega"}
INK, INK2, GRID, SURFACE, RAWGREY = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb", "#b9b8b2"
VERDICT_ZH = {"EXPLAINS": "能解释", "DOES NOT EXPLAIN": "不能解释", "PARTIAL": "部分解释"}


def tint(hex_color, amount=0.55):
    """Blend a colour towards the surface (raw-image bars/markers use the lighter tint of their backend colour)."""
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], float)
    s = np.array([int(SURFACE[i:i + 2], 16) for i in (1, 3, 5)], float)
    m = (c * (1 - amount) + s * amount).round().astype(int)
    return "#%02x%02x%02x" % tuple(m)


def pct(x, nd=1, signed=False):
    return "–" if x is None else (f"{x * 100:+.{nd}f}%" if signed else f"{x * 100:.{nd}f}%")


def num(x, nd=3, scale=1.0):
    return "–" if x is None else f"{x * scale:.{nd}f}"


# ----------------------------------------------------------------------------------------------
def build_results(an, sl, und_report):
    d = an["decision"]
    res = {
        "schema": "vggt_omega_slam.undistort.v1", "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "question": "Does lens distortion explain VGGT-Omega's focal over-estimate and worse translation accuracy on TUM fr1?",
        "protocol": {
            "undistortion": und_report["method"], "images": "data/tum_undistorted/rgbd_dataset_freiburg1_<seq>/rgb (original file names)",
            "per_submap": "exact frame lists of runs/analysis_why (17-frame windows of runs/compare/<seq>_vggt_loops1), one forward per "
                          "submap and backend, same metrics as tools/analysis_why.py (Sim(3) Umeyama ATE, ATE/GT path, relative rotation, "
                          "translation direction, fx/width)",
            "sparse": "N=8/16/32 frames uniformly spaced over the sequence, standard grid and two shifted grids (variants 0, +1, -1)",
            "slam": "run_slam.py (unmodified VGGT-SLAM 2.0), submap 16 and 32, loops on, seed 0, min-disparity 50, conf 25, lc-thres 0.95, "
                    "Omega frozen config/omega_similarity.json; CUDA cap 10 GiB",
            "ate": "Sim(3) Umeyama with scale, 0.01 s association (evo_ape tum -as equivalent, eval_ate.py)",
            "focal_reference": {"old_K_fx_over_width": an["calibration"]["fx_old_frac"], "new_K_fx_over_width": an["calibration"]["fx_new_frac"],
                                "old_K_fy_over_height": an["calibration"]["fy_old_frac"], "new_K_fy_over_height": an["calibration"]["fy_new_frac"]}},
        "decision_rule": an["decision_rule"], "decision": d,
        "undistortion_report": {k: und_report[k] for k in ("opencv", "K_old", "D_old_k1_k2_p1_p2_k3", "K_new", "roi", "image_size_wh",
                                                          "fx_old_over_width", "fx_new_over_width", "fy_old_over_height", "fy_new_over_height",
                                                          "maps")},
        "undistortion_files": {s: {k: v for k, v in und_report["sequences"][s].items() if k != "straight_line_guides"} for s in SEQS_L},
        "straight_line_guides": {s: und_report["sequences"][s]["straight_line_guides"] for s in SEQS_L},
        "A": {"summary_full_submaps": an["A"]["summary_full_submaps"], "paired_omega_vs_vggt": an["A"]["paired_omega_vs_vggt"],
              "paired_ratio_lists": an["A"]["paired_ratio_lists"], "change_raw_to_und": an["A"]["change_raw_to_und"],
              "determinism_und_desk_submap0": an["A"]["determinism_und_desk_submap0"], "per_submap": an["A"]["per_submap"]},
        "B": {"summary": an["B"]["summary"], "per_set": an["B"]["per_set"]},
        "slam": sl, "raw_recompute_check_vs_analysis_why": an["raw_recompute_check_vs_analysis_why"],
        "missing_cache_entries": an["missing_cache_entries"]}
    return res


# ----------------------------------------------------------------------------------------------
def slam_table(sl):
    """rows[seq][size][backend] = (raw_ate, und_ate, und_status)"""
    rows = {}
    for seq in SEQS_L:
        rows[seq] = {}
        for size in cpu.SIZES:
            rows[seq][size] = {}
            for b in BACKENDS:
                r = sl["runs"][seq][str(size)][b]
                rows[seq][size][b] = {c: (r[c].get("ate_rmse_m") if r[c].get("status") == "complete" else None) for c in cpu.COND}
                rows[seq][size][b]["und_status"] = r["und"].get("status")
    return rows


def write_markdown(res, path):
    an_cal = res["protocol"]["focal_reference"]
    d = res["decision"]
    A, B, sl = res["A"], res["B"], res["slam"]
    S = A["summary_full_submaps"]
    P = A["paired_omega_vs_vggt"]
    L = ["# 去畸变对照:镜头畸变能否解释 VGGT-Omega 的焦距偏大与平移变差(TUM fr1 desk / desk2 / room)", "",
         f"生成时间 {res['generated']}。所有数字见 `results.json`;逐子图、逐稀疏集的原始数值在其 `A.per_submap`、`B.per_set`。", ""]
    L += ["## 判定(规则在看到去畸变数字之前已写死)", "",
          f"**判定:{VERDICT_ZH[d['verdict']]}({d['verdict']})**", "",
          "规则:能解释 = 去畸变后 Omega 焦距偏差 < 5% 且 desk、desk2 的逐子图 ATE/路径长比(Omega÷VGGT)中位数都 ≤ 1.05;"
          "不能解释 = Omega 焦距偏差 > 8% 或 desk、desk2 的比值都 ≥ 1.2;其余为部分解释。", "",
          "| 判据 | 去畸变前 | 去畸变后 | 规则阈值 |", "|---|---|---|---|",
          f"| Omega 焦距偏差 abs(mean fx/新 K fx − 1)(三序列 {d['n_pairs_und']['pooled']} 个完整子图合计;去畸变前对旧 K) | "
          f"{pct(abs(d['omega_focal_bias_raw_vs_old_K_pooled_signed']))} | **{pct(d['omega_focal_bias_und_vs_new_K_pooled_abs'])}** | < 5% 解释;> 8% 不解释 |",
          f"| desk 配对比中位数(Omega÷VGGT 的 ATE/路径长,{d['n_pairs_und']['desk']} 个子图) | {num(d['median_paired_ratio_raw']['desk'], 2)} | "
          f"**{num(d['ratio_desk'], 2)}** | ≤ 1.05 解释;≥ 1.2 不解释 |",
          f"| desk2 配对比中位数({d['n_pairs_und']['desk2']} 个子图) | {num(d['median_paired_ratio_raw']['desk2'], 2)} | **{num(d['ratio_desk2'], 2)}** | 同上 |",
          f"| room 配对比中位数({d['n_pairs_und']['room']} 个子图,单列,不入规则) | {num(d['median_paired_ratio_raw']['room'], 2)} | "
          f"{num(d['median_paired_ratio_und']['room'], 2)} | – |",
          f"| 三序列合计配对比中位数({d['n_pairs_und']['pooled']} 个子图,不入规则) | {num(d['median_paired_ratio_raw']['pooled'], 2)} | "
          f"{num(d['median_paired_ratio_und']['pooled'], 2)} | – |", "",
          "读法约定:焦距偏差 = 各完整子图(17 帧)内 fx 均值÷图宽,再对子图取均值,与参考值之比减 1,三序列合计;配对比 = 同一子图两后端 ATE/路径长之比,取中位数;"
          "「desk 与 desk2 的比值都 ≥ 1.2」按两者各自都 ≥ 1.2 理解。desk 只有 3 个完整子图。", ""]

    L += ["## 去畸变做了什么", "",
          f"TUM fr1 官方标定(OpenCV 模型):fx={res['undistortion_report']['K_old'][0][0]:.3f}, fy={res['undistortion_report']['K_old'][1][1]:.3f}, "
          f"cx={res['undistortion_report']['K_old'][0][2]:.3f}, cy={res['undistortion_report']['K_old'][1][2]:.3f};"
          f"k1,k2,p1,p2,k3 = {', '.join(f'{x:.6f}' for x in res['undistortion_report']['D_old_k1_k2_p1_p2_k3'])}。"
          f"`getOptimalNewCameraMatrix(alpha=0)`(只留有效像素,输出仍 640×480)得到新 K:fx={res['undistortion_report']['K_new'][0][0]:.3f}, "
          f"fy={res['undistortion_report']['K_new'][1][1]:.3f}, cx={res['undistortion_report']['K_new'][0][2]:.3f}, "
          f"cy={res['undistortion_report']['K_new'][1][2]:.3f}(ROI 为整幅,无黑边)。"
          f"fx/图宽:旧 K {an_cal['old_K_fx_over_width']:.4f} → 新 K {an_cal['new_K_fx_over_width']:.4f}"
          f"(去畸变本身把真值焦距放大 {pct(an_cal['new_K_fx_over_width'] / an_cal['old_K_fx_over_width'] - 1)});"
          f"fy/图高:旧 {an_cal['old_K_fy_over_height']:.4f} → 新 {an_cal['new_K_fy_over_height']:.4f}。", ""]
    disp = res["undistortion_report"]["maps"]["distortion_displacement_px"]
    mp = res["undistortion_report"]["maps"]
    L += [f"纯畸变位移(不含新旧 K 的缩放):中位 {disp['median']:.1f} px,95 分位 {disp['p95']:.1f} px,最大 {disp['max']:.1f} px(图角)。"
          f"输出每个像素的源坐标都在原图内({mp['n_output_pixels_outside_raw']} 个越界),全白图 remap 无黑边像素;"
          f"网格点经 undistortPoints 往返最大误差 {mp['roundtrip_undistortPoints_max_err_px']:.3f} px。", "",
          "| 序列 | 原图张数 | 去畸变张数 | 文件名一致 | 与 remap 逐位一致 | 真值/rgb.txt 一致 |", "|---|---|---|---|---|---|"]
    for s in SEQS_L:
        f = res["undistortion_files"][s]
        L.append(f"| {s} | {f['n_original']} | {f['n_undistorted']} | {f['names_identical']} | "
                 f"{f['png_equals_fresh_remap']['n_checked'] - f['png_equals_fresh_remap']['n_mismatch']}/{f['png_equals_fresh_remap']['n_checked']} | "
                 f"{f['groundtruth_identical'] and f['rgb_txt_identical']} |")
    L += ["", "肉眼核对见 `samples/<序列>_raw_vs_undistorted.png`(左:原图及去畸变直线网格在原图中的弯曲像;右:去畸变图。红线=在去畸变图上找到的长直边;"
          "下排放大 6 倍)。长直边在去畸变图中沿直线(强边偏离直线 rms 见下),在原图中沿弯曲像;单条边的弯曲量不大(沿径向的边几乎不弯),整体畸变量见上面的位移。", "",
          "| 序列 | 帧 | 线段长 px | 原图中该线的最大弯曲 px | 去畸变图强边偏离直线 rms px | 原图强边偏离映射曲线 rms px |", "|---|---|---|---|---|---|"]
    for s in SEQS_L:
        for g in res["straight_line_guides"][s]:
            L.append(f"| {s} | {g['frame']} | {g['segment_length_px']:.0f} | {g['raw_curve_max_deviation_from_its_chord_px']:.2f} | "
                     f"{num(g['undistorted_image_edge_rms_offset_from_straight_line_px'], 2)} | {num(g['raw_image_edge_rms_offset_from_mapped_curve_px'], 2)} |")

    L += ["", "## 1. 焦距(单次前向,完整 17 帧子图)", "",
          "fx/图宽 为子图内各帧 fx 的均值÷预处理后图宽(VGGT 518、Omega 592 宽,比值与缩放无关)。偏差 = 对子图取均值后的 fx/图宽 ÷ 参考 − 1。"
          "原图的主参考是旧 K(0.8083),去畸变图的主参考是新 K(0.8543);另一个参考只作对照。", "",
          "| 序列 | 后端 | 图像 | 子图数 | fx/图宽 均值 | 对旧 K 偏差 | 对新 K 偏差 | fy/图高 均值 | fy 对旧 K | fy 对新 K |", "|---|---|---|---|---|---|---|---|---|---|"]
    for seq in SEQS_L + ["pooled"]:
        for b in BACKENDS:
            for cond, nm in (("raw", "原图"), ("und", "去畸变")):
                x = S[cond][b][seq]
                f = x["focal"]
                L.append(f"| {'合计' if seq == 'pooled' else seq} | {cpu_label(b)} | {nm} | {x['n']} | {f['fx_frac_mean_over_sets']:.4f} | "
                         f"{'**' if cond == 'raw' else ''}{pct(f['fx_bias_vs_old_signed'], 1, True)}{'**' if cond == 'raw' else ''} | "
                         f"{'**' if cond == 'und' else ''}{pct(f['fx_bias_vs_new_signed'], 1, True)}{'**' if cond == 'und' else ''} | "
                         f"{f['fy_frac_mean_over_sets']:.4f} | {pct(f['fy_bias_vs_old_signed'], 1, True)} | {pct(f['fy_bias_vs_new_signed'], 1, True)} |")
    L += ["", "(加粗 = 该图像条件下的主参考。)", ""]

    L += ["## 2. 每个 17 帧子图的精度(同一批窗口,原图 vs 去畸变)", "",
          "| 序列 | 后端 | 图像 | 子图数 | ATE 中位 (cm) | ATE/路径长 中位 (%) | 相对旋转误差 (°,原定义) | 相对旋转误差 (°,旋转投影到 SO(3) 后) | 相对平移方向误差 (°) | 对齐后绝对旋转误差 (°) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for seq in SEQS_L + ["pooled"]:
        for b in BACKENDS:
            for cond, nm in (("raw", "原图"), ("und", "去畸变")):
                x = S[cond][b][seq]
                L.append(f"| {'合计' if seq == 'pooled' else seq} | {cpu_label(b)} | {nm} | {x['n']} | {num(x['ate_rmse_m']['median'], 2, 100)} | "
                         f"{num(x['ate_over_gt_path']['median'], 2, 100)} | {num(x['rel_rot_err_deg_mean']['median'], 2)} | "
                         f"{num(x['proj']['rel_rot_err_deg_mean']['median'], 2)} | "
                         f"{num(x['rel_dir_err_deg_mean']['median'], 1)} | {num(x['abs_rot_err_deg_mean']['median'], 1)} |")
    L += ["", "同一子图两后端配对(比 = Omega÷VGGT,>1 表示 Omega 更差):", "",
          "| 序列 | 指标 | 图像 | 配对数 | Omega 更差的个数 | 中位 VGGT | 中位 Omega | 中位比 | Wilcoxon p |", "|---|---|---|---|---|---|---|---|---|"]
    names = {"ate_over_gt_path": "ATE/路径长", "ate_rmse_m": "ATE (m)", "rel_rot_err_deg_mean": "相对旋转误差 (°)",
             "rel_dir_err_deg_mean": "相对平移方向误差 (°)", "abs_rot_err_deg_mean": "对齐后绝对旋转误差 (°)"}
    for seq in SEQS_L + ["pooled"]:
        for k, nm in names.items():
            for cond, cn in (("raw", "原图"), ("und", "去畸变")):
                p = P[cond][seq][k]
                if not p.get("n_pairs"):
                    continue
                L.append(f"| {'合计' if seq == 'pooled' else seq} | {nm} | {cn} | {p['n_pairs']} | {p['n_omega_larger']} | {num(p['median_vggt'], 4)} | "
                         f"{num(p['median_omega'], 4)} | {num(p.get('median_ratio_omega_over_vggt'), 2)} | {num(p.get('wilcoxon_two_sided_p'), 3)} |")

    L += ["", "### 去畸变对每个后端本身的影响(同一子图,去畸变 ÷ 原图;<1 表示变好)", "",
          "| 序列 | 后端 | 配对数 | ATE/路径长 中位 原图 → 去畸变 | 中位比 | 去畸变更差的个数 | Wilcoxon p | 相对旋转误差 中位 原图 → 去畸变 | 平移方向误差 中位 原图 → 去畸变 |", "|---|---|---|---|---|---|---|---|---|"]
    ch = A["change_raw_to_und"]
    for b in BACKENDS:
        for seq in SEQS_L + ["pooled"]:
            e = ch[b][seq]
            a, r, t = e["ate_over_gt_path"], e["rel_rot_err_deg_mean"], e["rel_dir_err_deg_mean"]
            L.append(f"| {'合计' if seq == 'pooled' else seq} | {cpu_label(b)} | {e['n_pairs']} | {pct(a['median_raw'], 2)} → {pct(a['median_und'], 2)} | "
                     f"{num(a['median_ratio_und_over_raw'], 2)} | {a['n_und_larger']}/{e['n_pairs']} | {num(a['wilcoxon_two_sided_p'], 3)} | "
                     f"{num(r['median_raw'], 2)} → {num(r['median_und'], 2)} | {num(t['median_raw'], 1)} → {num(t['median_und'], 1)} |")

    cr, cu = S["raw"]["vggt"]["pooled"], S["und"]["vggt"]["pooled"]
    Pp = A["paired_omega_vs_vggt"]
    L += ["", "### 旋转矩阵非正交:对 `analysis_why` 相对旋转结论的一处订正", "",
          "VGGT 解码出的外参旋转是 bf16 量化的,不是正交矩阵;`relative_rotation_errors` 用迹求角,迹 > 3 时被截成恰好 0°,迹 < 3 时又被放大成虚假角度"
          "(迹偏差 0.005 在小角度下相当于约 4°)。Omega 的旋转正交到浮点精度,不受影响。把旋转先投影到最近的 SO(3)(SVD 极分解)再算同一批指标:", "",
          "| 项 | VGGT 原图 | VGGT 去畸变 | Omega 原图 | Omega 去畸变 |", "|---|---|---|---|---|",
          f"| 旋转正交性误差 max‖RRᵀ−I‖ | {num(S['raw']['vggt']['pooled']['rot_orthonormality_err_max'], 4)} | {num(S['und']['vggt']['pooled']['rot_orthonormality_err_max'], 4)} | "
          f"{S['raw']['omega']['pooled']['rot_orthonormality_err_max']:.1e} | {S['und']['omega']['pooled']['rot_orthonormality_err_max']:.1e} |",
          f"| 被截成 0° 的相邻帧对 | {cr['n_rel_rot_pairs_clipped_to_zero']}/{cr['n_rel_rot_pairs']} | {cu['n_rel_rot_pairs_clipped_to_zero']}/{cu['n_rel_rot_pairs']} | "
          f"{S['raw']['omega']['pooled']['n_rel_rot_pairs_clipped_to_zero']}/{S['raw']['omega']['pooled']['n_rel_rot_pairs']} | "
          f"{S['und']['omega']['pooled']['n_rel_rot_pairs_clipped_to_zero']}/{S['und']['omega']['pooled']['n_rel_rot_pairs']} |",
          f"| 相对旋转误差中位 原定义 (°) | {num(cr['rel_rot_err_deg_mean']['median'], 2)} | {num(cu['rel_rot_err_deg_mean']['median'], 2)} | "
          f"{num(S['raw']['omega']['pooled']['rel_rot_err_deg_mean']['median'], 2)} | {num(S['und']['omega']['pooled']['rel_rot_err_deg_mean']['median'], 2)} |",
          f"| 相对旋转误差中位 投影后 (°) | {num(cr['proj']['rel_rot_err_deg_mean']['median'], 2)} | {num(cu['proj']['rel_rot_err_deg_mean']['median'], 2)} | "
          f"{num(S['raw']['omega']['pooled']['proj']['rel_rot_err_deg_mean']['median'], 2)} | {num(S['und']['omega']['pooled']['proj']['rel_rot_err_deg_mean']['median'], 2)} |",
          f"| ATE/路径长 中位 原定义 → 投影后 (%) | {num(cr['ate_over_gt_path']['median'], 2, 100)} → {num(cr['proj']['ate_over_gt_path']['median'], 2, 100)} | "
          f"{num(cu['ate_over_gt_path']['median'], 2, 100)} → {num(cu['proj']['ate_over_gt_path']['median'], 2, 100)} | "
          f"{num(S['raw']['omega']['pooled']['ate_over_gt_path']['median'], 2, 100)} → {num(S['raw']['omega']['pooled']['proj']['ate_over_gt_path']['median'], 2, 100)} | "
          f"{num(S['und']['omega']['pooled']['ate_over_gt_path']['median'], 2, 100)} → {num(S['und']['omega']['pooled']['proj']['ate_over_gt_path']['median'], 2, 100)} |", "",
          f"同一子图配对的相对旋转误差比(Omega÷VGGT 中位,三序列合计):原图 原定义 {num(Pp['raw']['pooled']['rel_rot_err_deg_mean'].get('median_ratio_omega_over_vggt'), 2)}"
          f"(Omega 更差 {Pp['raw']['pooled']['rel_rot_err_deg_mean']['n_omega_larger']}/{Pp['raw']['pooled']['rel_rot_err_deg_mean']['n_pairs']})→ 投影后 "
          f"{num(Pp['raw']['pooled']['proj']['rel_rot_err_deg_mean'].get('median_ratio_omega_over_vggt'), 2)}(Omega 更差 {Pp['raw']['pooled']['proj']['rel_rot_err_deg_mean']['n_omega_larger']}/{Pp['raw']['pooled']['proj']['rel_rot_err_deg_mean']['n_pairs']});"
          f"去畸变 原定义 {num(Pp['und']['pooled']['rel_rot_err_deg_mean'].get('median_ratio_omega_over_vggt'), 2)} → 投影后 "
          f"{num(Pp['und']['pooled']['proj']['rel_rot_err_deg_mean'].get('median_ratio_omega_over_vggt'), 2)}。"
          "即 `analysis_why` 里「相对旋转 Omega 更好(0.73° 对 1.25°)」主要来自 VGGT 旋转矩阵非正交的度量伪影;旋转正交化后两后端的逐帧相对旋转误差基本相同。"
          "ATE 几乎不受影响(中心由 −Rᵀt 得到,正交化只改变约 1% 量级)。判定规则只用 ATE/路径长和焦距,不受此影响;规则在正交化后的 ATE/路径长上重算,结果为 "
          f"{res['decision']['robustness_rotations_projected_to_SO3']['verdict']}(desk {num(res['decision']['robustness_rotations_projected_to_SO3']['median_paired_ratio_und']['desk'], 2)},"
          f" desk2 {num(res['decision']['robustness_rotations_projected_to_SO3']['median_paired_ratio_und']['desk2'], 2)})。", ""]
    k_ratio = an_cal["new_K_fx_over_width"] / an_cal["old_K_fx_over_width"]
    L += ["### 焦距是否跟着真值焦距走", "",
          f"去畸变把真值焦距 fx/图宽 放大了 {pct(k_ratio - 1)}(新 K ÷ 旧 K = {k_ratio:.4f})。预测焦距的变化(三序列合计,去畸变 ÷ 原图):"
          + ";".join(f"{cpu_label(b)} {S['und'][b]['pooled']['focal']['fx_frac_mean_over_sets'] / S['raw'][b]['pooled']['focal']['fx_frac_mean_over_sets']:.4f}" for b in BACKENDS)
          + "。两个后端的预测焦距都随真值焦距放大(比值与新旧 K 之比同量级),各自保留原有的乘性偏差。", ""]
    L += ["", "## 3. 稀疏宽基线(N 帧在整条序列上均匀取;标准网格 + 两个平移网格)", "",
          "| N | 图像 | 配对数 | ATE 中位比 Omega/VGGT | Omega 更差的个数 | ATE/路径长 中位比 | 相对旋转误差 中位比 | VGGT fx 偏差(对该图的主参考) | Omega fx 偏差 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n in cpu.aw.SPARSE_N:
        for cond, cn, key in (("raw", "原图", "fx_bias_vs_old_signed"), ("und", "去畸变", "fx_bias_vs_new_signed")):
            sm = B["summary"][cond]["per_N"][str(n)]
            pa, pr, pp = sm["paired"]["ate_rmse_m"], sm["paired"]["ate_over_gt_path"], sm["paired"]["rel_rot_err_deg_mean"]
            L.append(f"| {n} | {cn} | {pa['n_pairs']} | {num(pa.get('median_ratio_omega_over_vggt'), 2)} | {pa['n_omega_larger']}/{pa['n_pairs']} | "
                     f"{num(pr.get('median_ratio_omega_over_vggt'), 2)} | {num(pp.get('median_ratio_omega_over_vggt'), 2)} | "
                     f"{pct(sm['vggt']['focal'][key], 1, True)} | {pct(sm['omega']['focal'][key], 1, True)} |")
    L += ["", "(「–」= VGGT 有被截成恰为 0° 的相邻帧对,按组配对的比值无定义,见上文「旋转矩阵非正交」。)"]
    L += ["", "各序列标准网格的 ATE(m):", "", "| 序列 | N | VGGT 原图 | VGGT 去畸变 | Omega 原图 | Omega 去畸变 |", "|---|---|---|---|---|---|"]
    for s in SEQS_L:
        for n in cpu.aw.SPARSE_N:
            cells = []
            for b in BACKENDS:
                for cond in cpu.COND:
                    v = B["summary"][cond]["by_seq_N"][s][str(n)][b]
                    cells.append(num(v["ate_rmse_m"], 3) if v else "–")
            L.append(f"| {s} | {n} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
    pp = B["summary"]
    L += ["", "跨 N、序列、网格合并后的配对(Omega÷VGGT 中位比):原图 ATE "
          f"{num(pp['raw']['paired_pooled']['ate_rmse_m'].get('median_ratio_omega_over_vggt'), 2)}(Omega 更差 {pp['raw']['paired_pooled']['ate_rmse_m']['n_omega_larger']}/{pp['raw']['paired_pooled']['ate_rmse_m']['n_pairs']}),"
          f"去畸变 ATE {num(pp['und']['paired_pooled']['ate_rmse_m'].get('median_ratio_omega_over_vggt'), 2)}(Omega 更差 {pp['und']['paired_pooled']['ate_rmse_m']['n_omega_larger']}/{pp['und']['paired_pooled']['ate_rmse_m']['n_pairs']})。", ""]

    # ---- SLAM ----
    L += ["## 4. 整条 SLAM(上游 VGGT-SLAM 2.0,回环开,seed 0)", "",
          "ATE RMSE(m,Sim(3) 对齐含尺度)。论文值为其子图 32 协议(Table I,Uncalibrated block);子图 16 没有论文值。", "",
          "| 序列 | 子图 | VGGT 原图 | VGGT 去畸变 | Omega 原图 | Omega 去畸变 | Omega÷VGGT 原图 | Omega÷VGGT 去畸变 | 论文 |", "|---|---|---|---|---|---|---|---|---|"]
    tab = slam_table(sl)
    for s in SEQS_L:
        for size in cpu.SIZES:
            t = tab[s][size]
            vr, vu, orr, ou = t["vggt"]["raw"], t["vggt"]["und"], t["omega"]["raw"], t["omega"]["und"]
            ratio_r = (orr / vr) if (orr and vr) else None
            ratio_u = (ou / vu) if (ou and vu) else None
            stat = "" if (t["vggt"]["und_status"] == "complete" and t["omega"]["und_status"] == "complete") else \
                f"(去畸变 VGGT {t['vggt']['und_status']} / Omega {t['omega']['und_status']})"
            L.append(f"| {s} | {size} | {num(vr, 4)} | {num(vu, 4)} | {num(orr, 4)} | {num(ou, 4)} | {num(ratio_r, 2)} | {num(ratio_u, 2)} | "
                     f"{num(cpu.PAPER_S32[s], 3) if size == 32 else '–'} {stat} |")
    L += ["", "关键帧选择 / 子图划分 / 被接受的回环对,去畸变 vs 原图,以及两后端之间:", "",
          "| 序列 | 子图 | 后端 | 关键帧数 原→去畸变 | 共有关键帧 | 关键帧相同 | 划分相同 | 接受的回环对数 原→去畸变 | 回环对逐帧相同 | 按时间(±1 s)配上的回环对 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SEQS_L:
        for size in cpu.SIZES:
            for b in BACKENDS:
                c = sl["checks"][s][str(size)][f"{b}_und_vs_raw"]
                if c is None:
                    L.append(f"| {s} | {size} | {cpu_label(b)} | – | – | – | – | – | – | – |")
                    continue
                L.append(f"| {s} | {size} | {cpu_label(b)} | {c['n_keyframes'][0]}→{c['n_keyframes'][1]} | {c['n_keyframes_common']} | {c['keyframes_identical']} | {c['partition_identical']} | "
                         f"{c['n_accepted_pairs'][0]}→{c['n_accepted_pairs'][1]} | {c['accepted_pairs_identical']} | {c['n_accepted_pairs_matched_within_1s']} |")
    L += ["", "两后端之间(同为去畸变图 / 同为原图):", "",
          "| 序列 | 子图 | 图像 | 关键帧相同 | 划分相同 | VGGT 接受的回环对数 | Omega 接受的回环对数 | 回环对相同 | 共同回环对 |", "|---|---|---|---|---|---|---|---|---|"]
    for s in SEQS_L:
        for size in cpu.SIZES:
            for key, cn in (("raw_vggt_vs_omega", "原图"), ("und_vggt_vs_omega", "去畸变")):
                c = sl["checks"][s][str(size)][key]
                if c is None:
                    L.append(f"| {s} | {size} | {cn} | – | – | – | – | – | – |")
                    continue
                L.append(f"| {s} | {size} | {cn} | {c['keyframes_identical']} | {c['partition_identical']} | {c['n_accepted_pairs'][0]} | {c['n_accepted_pairs'][1]} | "
                         f"{c['accepted_pairs_identical']} | {c['n_accepted_pairs_common']} |")

    L += ["", "## 自检", "",
          f"- 本脚本用同一函数重算的原图侧数字与 `runs/analysis_why/results.json` 逐项一致(最大绝对差 {res['raw_recompute_check_vs_analysis_why']['max_abs_diff']:.1e},"
          f"{res['raw_recompute_check_vs_analysis_why']['n_compared']} 项)。",
          "- 去畸变图上的子图窗口与稀疏帧集合与原图分析逐帧相同(`tools/analysis_undistort.py` 在前向前断言了 102 个 (后端, 任务) 的帧列表)。"]
    for b, v in A["determinism_und_desk_submap0"].items():
        L.append(f"- {cpu_label(b)} 去畸变图同一子图重复前向:位姿编码逐位相同={v['bitwise_identical_pose_enc']},深度逐位相同={v['bitwise_identical_depth_first']}。")
    if res["missing_cache_entries"]:
        L.append(f"- 缺失的前向缓存:{res['missing_cache_entries']}")
    verdict = OUT_ROOT / "verdict_zh.md"
    if verdict.is_file():
        L += ["", verdict.read_text().rstrip(), ""]
    path.write_text("\n".join(L) + "\n")


def cpu_label(b):
    return LABELS[b]


# ----------------------------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------------------------
def _style(ax, ygrid=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=10, length=3)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)


def write_figure(res, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullFormatter

    have = {f.name for f in fm.fontManager.ttflist}
    family = next((n for n in ("Noto Sans CJK JP", "Droid Sans Fallback") if n in have), "DejaVu Sans")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"], "axes.unicode_minus": False,
                         "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    A, d, sl = res["A"], res["decision"], res["slam"]
    S, ratios, per = A["summary_full_submaps"], A["paired_ratio_lists"], A["per_submap"]
    cal = res["protocol"]["focal_reference"]
    fig = plt.figure(figsize=(20, 12), dpi=100)
    gs = GridSpec(2, 2, figure=fig, left=0.055, right=0.985, top=0.80, bottom=0.075, hspace=0.50, wspace=0.14)
    seqs = SEQS_L
    row1_head, row2_head = 0.865, 0.425

    fig.text(0.055, 0.962, f"去畸变对照:镜头畸变{VERDICT_ZH[d['verdict']]} VGGT-Omega 的焦距偏大与平移变差(TUM fr1,TUM 官方标定去畸变)",
             fontsize=18, color=INK, fontweight="bold", va="center")
    sub = (f"判定依据:Omega 焦距偏差(对新 K)原图 {pct(abs(d['omega_focal_bias_raw_vs_old_K_pooled_signed']))} → 去畸变 "
           f"{pct(d['omega_focal_bias_und_vs_new_K_pooled_abs'])};逐子图 ATE/路径长比 Omega÷VGGT 的中位数 desk "
           f"{d['median_paired_ratio_raw']['desk']:.2f} → {d['ratio_desk']:.2f},desk2 {d['median_paired_ratio_raw']['desk2']:.2f} → {d['ratio_desk2']:.2f},"
           f"room {d['median_paired_ratio_raw']['room']:.2f} → {d['median_paired_ratio_und']['room']:.2f}(规则:偏差<5% 且比≤1.05 解释;偏差>8% 或比≥1.2 不解释)")
    fig.text(0.055, 0.925, sub, fontsize=12.5, color=INK2, va="center")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[b], markeredgecolor=SURFACE, markersize=10, label=LABELS[b])
               for b in BACKENDS]
    handles += [Line2D([0], [0], marker="o", color="none", markerfacecolor=SURFACE, markeredgecolor=INK2, markeredgewidth=1.6, markersize=9, label="原图(空心/浅色)"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=INK2, markeredgecolor=INK2, markersize=9, label="去畸变(实心/深色)")]
    fig.legend(handles=handles, loc="center right", bbox_to_anchor=(0.985, 0.962), ncol=4, frameon=False, fontsize=12)

    # (a) focal bias ----------------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    _style(ax)
    groups = seqs + ["pooled"]
    off = {"vggt": -0.21, "omega": 0.21}
    for gi, seq in enumerate(groups):
        for b in BACKENDS:
            x = gi + off[b]
            raw_rows = [m for s in (seqs if seq == "pooled" else [seq]) for m in per["raw"][s][b] if m["full"] and m["sim3_status"] == "ok"]
            und_rows = [m for s in (seqs if seq == "pooled" else [seq]) for m in per["und"][s][b] if m["full"] and m["sim3_status"] == "ok"]
            ax.scatter(np.full(len(raw_rows), x - 0.07), [m["fx_bias_vs_old"] * 100 for m in raw_rows], s=10, color=tint(COLORS[b], 0.35), alpha=0.8, edgecolor="none", zorder=2)
            ax.scatter(np.full(len(und_rows), x + 0.07), [m["fx_bias_vs_new"] * 100 for m in und_rows], s=10, color=COLORS[b], alpha=0.7, edgecolor="none", zorder=2)
            fr, fu = S["raw"][b][seq]["focal"]["fx_bias_vs_old_signed"] * 100, S["und"][b][seq]["focal"]["fx_bias_vs_new_signed"] * 100
            ax.plot([x - 0.07, x + 0.07], [fr, fu], color=INK2, lw=1.2, zorder=3)
            ax.scatter([x - 0.07], [fr], s=95, facecolor=SURFACE, edgecolor=COLORS[b], linewidth=2.2, zorder=4)
            ax.scatter([x + 0.07], [fu], s=95, facecolor=COLORS[b], edgecolor=SURFACE, linewidth=1.2, zorder=4)
            top = max([fr, fu] + [m["fx_bias_vs_old"] * 100 for m in raw_rows] + [m["fx_bias_vs_new"] * 100 for m in und_rows])
            ax.text(x, top + 0.9, f"{fr:+.1f}→{fu:+.1f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.axhspan(-5, 5, color="#2a78d6", alpha=0.07, zorder=0)
    ax.axhline(0, color=INK2, lw=0.9)
    ax.axhline(8, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(-0.5, 8.3, "不能解释线 +8%", ha="left", va="bottom", fontsize=9.5, color=INK2)
    ax.text(-0.5, -4.6, "解释区 ±5%", ha="left", va="bottom", fontsize=9.5, color=INK2)
    ax.set_ylim(-5.5, 20)
    ax.set_xticks(range(len(groups)), ["desk", "desk2", "room", "三序列合计"], fontsize=12)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylabel("子图均值 fx/图宽 相对真值焦距的偏差 (%)", fontsize=11, color=INK2)
    fig.text(0.055, row1_head, f"(a) 焦距偏差:原图对旧 K({cal['old_K_fx_over_width']:.3f})→ 去畸变对新 K({cal['new_K_fx_over_width']:.3f});"
             "粗点=均值,小点=各子图;左 VGGT、右 Omega", fontsize=13, color=INK, ha="left", va="bottom")

    # (b) paired ratio -----------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    _style(ax)
    for gi, seq in enumerate(seqs):
        for cond, x, col in (("raw", gi - 0.2, RAWGREY), ("und", gi + 0.2, INK2)):
            vals = [r["ratio"] for r in ratios[cond][seq]]
            rng = np.linspace(-0.07, 0.07, len(vals)) if len(vals) > 1 else [0.0]
            ax.scatter(x + np.array(rng), vals, s=34, facecolor=(SURFACE if cond == "raw" else col),
                       edgecolor=("#8a8984" if cond == "raw" else SURFACE), linewidth=1.6 if cond == "raw" else 0.8, zorder=3)
            med = float(np.median(vals))
            ax.hlines(med, x - 0.15, x + 0.15, color=(INK2 if cond == "und" else "#8a8984"), lw=3, zorder=4)
            worse = sum(1 for v in vals if v > 1)
            ax.text(x, 1.02, f"{med:.2f}\n{worse}/{len(vals)}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10, color=INK)
    ax.axhline(1.0, color=INK2, lw=0.9)
    for y, t, va in ((1.05, "解释线 ≤1.05", "top"), (1.2, "不能解释线 ≥1.2", "bottom")):
        ax.axhline(y, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.text(len(seqs) - 0.40, y, t, ha="left", va=va, fontsize=9.5, color=INK2,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.set_yscale("log")
    ax.set_yticks([0.7, 1, 1.5, 2, 3], ["0.7", "1", "1.5", "2", "3"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(range(len(seqs)), seqs, fontsize=12)
    ax.set_xlim(-0.55, len(seqs) - 0.45 + 0.6)
    ax.set_ylim(0.62, 3.3)
    ax.set_ylabel("Omega ÷ VGGT 的 ATE/路径长(同一子图,对数轴)", fontsize=11, color=INK2)
    fig.text(0.53, row1_head, "(b) 逐子图配对比:每序列左=原图(空心)、右=去畸变(实心),粗横线=中位数,点上方=中位数与 Omega 更差个数",
             fontsize=13, color=INK, ha="left", va="bottom")

    # (c) SLAM ATE -----------------------------------------------------------------------------
    subg = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.22)
    tab = slam_table(sl)
    pos = {("vggt", "raw"): -0.30, ("vggt", "und"): -0.10, ("omega", "raw"): 0.10, ("omega", "und"): 0.30}
    ymax = 0
    for s in seqs:
        for size in cpu.SIZES:
            for b in BACKENDS:
                for c in cpu.COND:
                    v = tab[s][size][b][c]
                    ymax = max(ymax, v or 0)
    axes_c = []
    for si, size in enumerate(cpu.SIZES):
        ax = fig.add_subplot(subg[0, si])
        axes_c.append(ax)
        _style(ax)
        for gi, s in enumerate(seqs):
            for b in BACKENDS:
                for c in cpu.COND:
                    v = tab[s][size][b][c]
                    x = gi + pos[(b, c)]
                    if v is None:
                        ax.text(x, 0.001, "无", ha="center", va="bottom", fontsize=8, color=INK2)
                        continue
                    ax.bar(x, v, width=0.17, color=(tint(COLORS[b], 0.5) if c == "raw" else COLORS[b]), edgecolor=SURFACE, linewidth=1.2, zorder=3)
                    ax.text(x, v + ymax * 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8.2, color=INK, rotation=90)
            if size == 32:  # the paper only reports the submap-32 protocol
                ax.hlines(cpu.PAPER_S32[s], gi - 0.42, gi + 0.42, color=INK, lw=1.6, ls=(0, (4, 2)), zorder=5)
        ax.set_xticks(range(len(seqs)), seqs, fontsize=12)
        ax.set_xlim(-0.55, len(seqs) - 0.45)
        ax.set_ylim(0, ymax * 1.28)
        ax.set_title(f"子图 {size}", fontsize=12, color=INK, loc="left")
        if si == 0:
            ax.set_ylabel("整条 SLAM 的 ATE RMSE (m)", fontsize=11, color=INK2)
        else:
            ax.text(0.99, 0.98, "虚线 = 论文值(子图 32 协议)\n浅色 = 原图,深色 = 去畸变", transform=ax.transAxes, ha="right", va="top",
                    fontsize=9.5, color=INK2)
    fig.text(0.055, row2_head, "(c) 整条 SLAM(回环开,seed 0):每序列四根柱 = VGGT 原图/去畸变、Omega 原图/去畸变", fontsize=13, color=INK, ha="left", va="bottom")

    # (d) relative rotation error: left = analysis_why definition, right = rotations projected to SO(3) first ------------
    subd = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.16)
    pos_d = {("vggt", "raw"): -0.33, ("vggt", "und"): -0.11, ("omega", "raw"): 0.11, ("omega", "und"): 0.33}
    ymax_d = 0
    for variant in ("orig", "proj"):
        ax = fig.add_subplot(subd[0, 0 if variant == "orig" else 1])
        _style(ax)
        for gi, s_ in enumerate(seqs):
            for b in BACKENDS:
                for c in cpu.COND:
                    rows = [m for m in per[c][s_][b] if m["full"] and m["sim3_status"] == "ok"]
                    vals = [(m["rel_rot_err_deg_mean"] if variant == "orig" else m["proj"]["rel_rot_err_deg_mean"]) for m in rows]
                    ymax_d = max(ymax_d, max(vals))
                    x = gi + pos_d[(b, c)]
                    rng = np.linspace(-0.045, 0.045, len(vals)) if len(vals) > 1 else [0.0]
                    col = COLORS[b]
                    ax.scatter(x + np.array(rng), vals, s=18, facecolor=(SURFACE if c == "raw" else col), edgecolor=col,
                               linewidth=1.3 if c == "raw" else 0.6, zorder=3)
                    med = float(np.median(vals))
                    ax.hlines(med, x - 0.09, x + 0.09, color=(tint(col, 0.2) if c == "raw" else col), lw=3, zorder=4)
                    ax.text(x, 1.015, f"{med:.2f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5, color=INK, rotation=90)
        ax.set_xticks(range(len(seqs)), seqs, fontsize=11)
        ax.set_xlim(-0.55, len(seqs) - 0.45)
        ax.set_ylim(0, 3.5)
        ax.set_title(("analysis_why 原定义" if variant == "orig" else "旋转矩阵先投影到 SO(3)") + "\n\n", fontsize=10.5, color=INK, loc="left")
        if variant == "orig":
            ax.set_ylabel("相邻帧相对旋转误差,子图内均值 (°)", fontsize=11, color=INK2)
            cr, cu = S["raw"]["vggt"]["pooled"], S["und"]["vggt"]["pooled"]
            ax.text(0.02, 0.97, f"VGGT 旋转矩阵非正交(bf16),相邻帧对角度被截成 0°:\n原图 {cr['n_rel_rot_pairs_clipped_to_zero']}/{cr['n_rel_rot_pairs']},"
                    f"去畸变 {cu['n_rel_rot_pairs_clipped_to_zero']}/{cu['n_rel_rot_pairs']}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8.5, color=INK2)
    fig.text(0.53, row2_head, "(d) 相对旋转误差(每点一个 17 帧子图,粗横线=中位数,顶部数字=中位数;每序列四列同 (c))", fontsize=13, color=INK, ha="left", va="bottom")

    fig.text(0.055, 0.02, "焦距偏差 = 子图内 fx 均值÷预处理图宽,对子图取均值后与真值焦距之比减 1(原图:旧 K fx/宽 0.8083;去畸变:新 K 0.8543)。"
             "子图窗口为原图分析的同一批 17 帧窗口(三序列 3 / 6 / 11 个完整子图);SLAM 的关键帧与划分在去畸变图上重新选取。",
             fontsize=10.5, color=INK2, ha="left", va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def report(args):
    an = json.loads((OUT / "analysis.json").read_text())
    sl = json.loads((OUT / "slam_summary.json").read_text())
    und_report = json.loads((OUT_ROOT / "undistort_report.json").read_text())
    res = build_results(an, sl, und_report)
    (OUT_ROOT / "results.json").write_text(json.dumps(cpu._clean(res), indent=1, ensure_ascii=False))
    write_markdown(res, OUT_ROOT / "results_zh.md")
    write_figure(res, OUT_ROOT / "去畸变对比.png")
    print(f"wrote {OUT_ROOT / 'results.json'}, results_zh.md, 去畸变对比.png; verdict {res['decision']['verdict']}")
    return 0
