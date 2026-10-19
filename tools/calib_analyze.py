#!/usr/bin/env python3
"""Calibration step 5 (CPU): apply the pre-declared rule (vos/calibration.RULE_TEXT) to the measured ratios.

Reads pairs_defs.json + ratios_vggt.json + ratios_omega.json; writes
  runs/calibration/pairs.json                       every pair with all ratios
  runs/calibration/summary.json                     all statistics
  runs/calibration/similarity_calibration.png       Chinese figure
  runs/calibration/negatives_top_sheet.png          N pairs with the highest ratios (contamination check)
  config/omega_similarity.json                      {layer, tau, ...} consumed by run_slam.py
No GPU. The rule is not edited here; the numbers only flow through it.

  OMP_NUM_THREADS=4 <python> tools/calib_analyze.py
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vos import env  # noqa: E402

env.prepare(upstream=False)

import numpy as np  # noqa: E402

from vos import calib_plots  # noqa: E402
from vos import calibration as cal  # noqa: E402

OUT = env.PROJECT / "runs" / "calibration"
SETS = ("P_adj", "P_loop", "N")


def merge(defs, vggt, omega):
    pairs = []
    for pair in defs["pairs"]:
        pid = pair["pair_id"]
        ratio, reverse = {}, {}
        for layer in cal.VGGT_LAYERS:
            ratio[f"vggt@{layer}"] = vggt["ratios"][pid][str(layer)]["solver"]
            reverse[f"vggt@{layer}"] = vggt["ratios"][pid][str(layer)]["reverse"]
        for layer in cal.OMEGA_LAYERS:
            ratio[f"omega@{layer}"] = omega["ratios"][pid][str(layer)]["solver"]
            reverse[f"omega@{layer}"] = omega["ratios"][pid][str(layer)]["reverse"]
        pairs.append(dict(pair, query=pair["image_late"], retrieved=pair["image_early"], ratio=ratio,
                          ratio_reverse_order=reverse))
    return pairs


def column(pairs, key, sets, order="ratio"):
    return np.array([p[order][key] for p in pairs if p["set"] in sets], dtype=np.float64)


def summarize(pairs, defs, vggt, omega, subsets):
    positives = ("P_adj", "P_loop")
    auc, ci, auc_adj, auc_loop = {}, {}, {}, {}
    for layer in cal.OMEGA_LAYERS:
        key = f"omega@{layer}"
        pos, neg = column(pairs, key, positives), column(pairs, key, ("N",))
        auc[layer] = cal.roc_auc(pos, neg)
        ci[layer] = cal.bootstrap_auc_ci(pos, neg)
        auc_adj[layer] = cal.roc_auc(column(pairs, key, ("P_adj",)), neg)
        auc_loop[layer] = cal.roc_auc(column(pairs, key, ("P_loop",)), neg)
    vggt_auc = {}
    for layer in cal.VGGT_LAYERS:
        key = f"vggt@{layer}"
        pos, neg = column(pairs, key, positives), column(pairs, key, ("N",))
        vggt_auc[layer] = {"pooled": cal.roc_auc(pos, neg), "ci95": cal.bootstrap_auc_ci(pos, neg),
                           "P_adj": cal.roc_auc(column(pairs, key, ("P_adj",)), neg),
                           "P_loop": cal.roc_auc(column(pairs, key, ("P_loop",)), neg)}

    l_star = cal.choose_layer(auc, anchor=20)
    layer_choice_stability = cal.bootstrap_layer_choice(
        {l: column(pairs, f"omega@{l}", positives) for l in cal.OMEGA_LAYERS},
        {l: column(pairs, f"omega@{l}", ("N",)) for l in cal.OMEGA_LAYERS})
    far = {layer: cal.far_at(column(pairs, f"vggt@{layer}", ("N",)), cal.VGGT_THRESHOLD) for layer in cal.VGGT_LAYERS}
    far_used = far[cal.VGGT_PRIMARY_LAYER]
    omega_neg = column(pairs, f"omega@{l_star}", ("N",))
    tau = cal.tau_from_far(omega_neg, far_used)
    tau_if_vggt20 = cal.tau_from_far(omega_neg, far[20])

    def sets_stats(key, threshold, order="ratio"):
        out = {s: cal.set_stats(column(pairs, key, (s,), order), threshold) for s in SETS}
        for name, ids in subsets.items():
            values = [p[order][key] for p in pairs if p["pair_id"] in set(ids)]
            out[name] = cal.set_stats(values, threshold) if values else None
        return out

    stats = {"vggt@21": sets_stats("vggt@21", cal.VGGT_THRESHOLD),
             "vggt@20": sets_stats("vggt@20", cal.VGGT_THRESHOLD),
             f"omega@{l_star}": sets_stats(f"omega@{l_star}", tau),
             "omega@19_at_placeholder_0.95": sets_stats("omega@19", 0.95)}
    reverse = {"vggt@21": sets_stats("vggt@21", cal.VGGT_THRESHOLD, "ratio_reverse_order"),
               f"omega@{l_star}": sets_stats(f"omega@{l_star}", tau, "ratio_reverse_order")}
    sensitivity = {}
    for name, ids in subsets.items():
        keep = set(ids)
        sub = [p for p in pairs if p["set"] != "P_loop" or p["pair_id"] in keep]
        auc_sub = {layer: cal.roc_auc(column(sub, f"omega@{layer}", positives), column(sub, f"omega@{layer}", ("N",)))
                   for layer in cal.OMEGA_LAYERS}
        l_sub = cal.choose_layer(auc_sub, anchor=20)
        neg_sub = column(pairs, f"omega@{l_sub}", ("N",))
        tau_sub = cal.tau_from_far(neg_sub, far_used)
        loop_vals = lambda key, order="ratio": [p[order][key] for p in pairs if p["pair_id"] in keep]  # noqa: E731
        sensitivity[name] = {
            "n_loop_pairs": len(keep), "auc_pooled": {str(k): v for k, v in auc_sub.items()}, "layer_star": l_sub,
            "tau": tau_sub,
            "vggt@21_pass_at_0.95": cal.set_stats(loop_vals("vggt@21"), cal.VGGT_THRESHOLD),
            "omega_at_rule_L*_and_tau": cal.set_stats(loop_vals(f"omega@{l_star}"), tau),
            "omega_at_subset_L*_and_tau": cal.set_stats(loop_vals(f"omega@{l_sub}"), tau_sub)}
    return {"layer_choice_bootstrap_frequency": {str(k): v for k, v in layer_choice_stability.items()},
            "sensitivity_subsets": sensitivity, "auc_pooled": auc, "auc_ci95": ci, "auc_P_adj_vs_N": auc_adj, "auc_P_loop_vs_N": auc_loop,
            "vggt_auc": vggt_auc, "layer_star": l_star,
            "layer_star_by_P_loop_auc_only_informational": cal.choose_layer(auc_loop, anchor=20),
            "FAR_vggt": far_used, "FAR_vggt_by_layer": {str(k): v for k, v in far.items()},
            "vggt_layer_used_for_far": cal.VGGT_PRIMARY_LAYER,
            "tau": tau, "tau_if_vggt_layer20_far": tau_if_vggt20,
            "omega_far_at_tau": cal.far_at(omega_neg, tau),
            "stats": stats, "stats_reverse_order": reverse}


def figure(pairs, summary, path, ring_ids=()):
    import matplotlib.pyplot as plt

    calib_plots.setup_font()
    l_star, tau = summary["layer_star"], summary["tau"]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.2), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    rng = np.random.default_rng(0)
    panels = [(axes[0], "vggt@21", cal.VGGT_THRESHOLD, f"VGGT 第{cal.VGGT_PRIMARY_LAYER}层(上游设定)",
               "阈值 0.95(上游硬编码)", summary["stats"]["vggt@21"]),
              (axes[1], f"omega@{l_star}", tau, f"VGGT-Omega 第{l_star}层(标定所选)",
               f"阈值 τ={tau:.3f}(对齐 VGGT 误通过率)", summary["stats"][f"omega@{l_star}"])]
    all_values = np.concatenate([column(pairs, "vggt@21", SETS), column(pairs, f"omega@{l_star}", SETS)])
    ylim = (min(all_values.min(), 0.0) - 0.0, all_values.max() * 1.06)
    for ax, key, thr, title, thr_label, stats in panels:
        for x, s in enumerate(SETS):
            values = column(pairs, key, (s,))
            color = calib_plots.SET_COLORS[s]
            ax.boxplot(values, positions=[x], widths=0.5, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.18, edgecolor=color, linewidth=1.2),
                       medianprops=dict(color=color, linewidth=2), whiskerprops=dict(color=color),
                       capprops=dict(color=color))
            xs = x + rng.uniform(-0.17, 0.17, values.size)
            ax.scatter(xs, values, s=16, color=color, alpha=0.85, edgecolors=calib_plots.SURFACE, linewidths=0.5,
                       zorder=3)
            if s == "P_loop" and ring_ids:
                ids = [q["pair_id"] for q in pairs if q["set"] == s]
                ringed = np.array([i in ring_ids for i in ids])
                ax.scatter(xs[ringed], values[ringed], s=70, facecolors="none", edgecolors=calib_plots.INK,
                           linewidths=1.3, zorder=4, label="圈 = 目视确认同一地点")
            st = stats[s]
            ax.text(x, ylim[0] - (ylim[1] - ylim[0]) * 0.095, f"通过 {st['n_pass']}/{st['n']}", ha="center",
                    va="top", fontsize=11, color=calib_plots.INK)
        ax.axhline(thr, color=calib_plots.INK, linestyle=(0, (5, 3)), linewidth=1.4, zorder=2)
        ax.text(-0.5, thr - (ylim[1] - ylim[0]) * 0.012, thr_label, ha="left", va="top", fontsize=9.5,
                color=calib_plots.INK2, zorder=4,
                bbox=dict(facecolor=calib_plots.SURFACE, edgecolor="none", alpha=0.85, pad=1.5))
        if ring_ids and ax is axes[0]:
            ax.text(-0.5, ylim[0] + (ylim[1] - ylim[0]) * 0.015, f"黑圈 = 目视确认同一地点({len(ring_ids)} 对,P_loop 内)",
                    ha="left", va="bottom", fontsize=9.5, color=calib_plots.INK2)
        ax.set_xticks(range(3))
        ax.set_xticklabels([calib_plots.SET_LABELS[s] for s in SETS], fontsize=10.5)
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=13, loc="left", color=calib_plots.INK)
        ax.grid(axis="y", color=calib_plots.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("原始相似度比值(两帧前向,查询帧在前)", fontsize=10.5)
    axes[1].tick_params(labelleft=False)

    ax = axes[2]
    layers = list(cal.OMEGA_LAYERS)
    auc = [summary["auc_pooled"][l] for l in layers]
    lo = [summary["auc_pooled"][l] - summary["auc_ci95"][l][0] for l in layers]
    hi = [summary["auc_ci95"][l][1] - summary["auc_pooled"][l] for l in layers]
    ax.errorbar(layers, auc, yerr=[lo, hi], color=calib_plots.INK, linewidth=1.8, marker="o", markersize=6,
                capsize=3, label="P_adj ∪ P_loop 对 N(选层依据,含 95% 自助区间)", zorder=3)
    ax.plot(layers, [summary["auc_P_loop_vs_N"][l] for l in layers], color=calib_plots.SET_COLORS["P_loop"],
            linewidth=1.6, marker="s", markersize=5, label="仅 P_loop 对 N(参考)", zorder=3)
    ax.axhline(summary["vggt_auc"][21]["pooled"], color=calib_plots.SET_COLORS["P_adj"], linestyle=(0, (5, 3)),
               linewidth=1.5, label=f"VGGT 第21层同口径 AUC = {summary['vggt_auc'][21]['pooled']:.3f}")
    ax.scatter([l_star], [summary["auc_pooled"][l_star]], s=210, facecolors="none", edgecolors=calib_plots.INK,
               linewidths=2, zorder=4)
    ax.annotate(f"L*={l_star}", (l_star, summary["auc_pooled"][l_star]), textcoords="offset points",
                xytext=(0, 16), ha="center", fontsize=11, color=calib_plots.INK)
    ax.set_xticks(layers)
    ax.set_xlabel("VGGT-Omega 全局帧间注意力块序号(20 为 register 块,不可用)", fontsize=10.5)
    ax.set_ylabel("ROC AUC", fontsize=10.5)
    ax.set_title("按层的区分能力", fontsize=13, loc="left", color=calib_plots.INK)
    ax.grid(color=calib_plots.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    low = min(min(summary["auc_ci95"][l][0] for l in layers), min(summary["auc_P_loop_vs_N"].values()))
    ax.set_ylim(low - 0.02, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9, frameon=False, ncol=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    p_loop_v = summary["stats"]["vggt@21"]["P_loop"]
    p_loop_o = summary["stats"][f"omega@{l_star}"]["P_loop"]
    fig.suptitle(
        f"回环相似度检验标定(仅 office_loop,{len(pairs)} 对):VGGT 误通过率 FAR={summary['FAR_vggt']:.3f} → "
        f"Omega 取 L*={l_star}、τ={tau:.3f};P_loop 通过率 VGGT {p_loop_v['n_pass']}/{p_loop_v['n']}"
        f" vs Omega {p_loop_o['n_pass']}/{p_loop_o['n']}",
        fontsize=14, color=calib_plots.INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument("--no-config", action="store_true", help="do not write config/omega_similarity.json")
    p.add_argument("--dir", type=Path, default=OUT, help="calibration directory (tests use a temporary one)")
    p.add_argument("--config", type=Path, default=env.SIM_CONFIG)
    args = p.parse_args()
    OUT = args.dir
    defs = json.loads((OUT / "pairs_defs.json").read_text())
    vggt = json.loads((OUT / "ratios_vggt.json").read_text())
    omega = json.loads((OUT / "ratios_omega.json").read_text())
    for name, blob in (("vggt", vggt), ("omega", omega)):
        if blob.get("status") != "complete":
            raise SystemExit(f"ratios_{name}.json is not complete: {blob.get('status')} {blob.get('error')}")
    pairs = merge(defs, vggt, omega)
    subsets_file = OUT / "sensitivity_subsets.json"
    subsets = json.loads(subsets_file.read_text())["subsets"] if subsets_file.is_file() else {}
    summary = summarize(pairs, defs, vggt, omega, subsets)
    l_star, tau = summary["layer_star"], summary["tau"]
    date = datetime.date.today().isoformat()

    (OUT / "pairs.json").write_text(json.dumps(
        {"dataset": cal.DATASET, "date": date, "rule_text": cal.RULE_TEXT, "n_keyframes": defs["n_keyframes"],
         "ratio_keys": "vggt@<layer> (bf16, VGGT_SPARK) and omega@<layer> (fp32 weights, bf16 autocast); "
                       "ratio = solver order (late/query frame first); ratio_reverse_order = sensitivity only",
         "pairs": pairs}, indent=1))
    summary_out = dict(summary, dataset=cal.DATASET, date=date,
                       regression={"vggt": vggt["regression"], "omega": omega["regression"],
                                   "omega_adapter_crosscheck": omega["adapter_crosscheck"],
                                   "omega_determinism_repeat_max_abs_diff": omega["determinism_repeat_max_abs_diff"]},
                       counts=defs["counts"], loop_partner_agreement_omega_loader=defs["loop_partner_agreement_omega_loader"],
                       p_loop_salad_below_0_95=defs["p_loop_salad_below_0.95"])
    (OUT / "summary.json").write_text(json.dumps(summary_out, indent=1))

    if not args.no_config:
        config = {
            "layer": l_star, "tau": tau, "dataset": cal.DATASET, "date": date, "rule_text": cal.RULE_TEXT,
            "FAR_vggt": summary["FAR_vggt"], "vggt_layer_used_for_far": summary["vggt_layer_used_for_far"],
            "FAR_vggt_by_layer": summary["FAR_vggt_by_layer"], "tau_if_vggt_layer20_far": summary["tau_if_vggt_layer20_far"],
            "omega_far_at_tau": summary["omega_far_at_tau"],
            "layer_choice_bootstrap_frequency": summary["layer_choice_bootstrap_frequency"],
            "pass_semantics": "a loop closure passes when raw image_match_ratio >= threshold",
            "auc_per_layer": {str(k): v for k, v in summary["auc_pooled"].items()},
            "auc_ci95_per_layer": {str(k): list(v) for k, v in summary["auc_ci95"].items()},
            "sensitivity_subsets_not_used_for_config": summary["sensitivity_subsets"],
            "per_set_stats": summary["stats"], "per_set_stats_reverse_order_sensitivity": summary["stats_reverse_order"],
            "sources": {"pairs": "runs/calibration/pairs.json", "summary": "runs/calibration/summary.json",
                        "figure": "runs/calibration/similarity_calibration.png"},
        }
        args.config.parent.mkdir(parents=True, exist_ok=True)
        args.config.write_text(json.dumps(config, indent=1))

    figure(pairs, summary, OUT / "similarity_calibration.png",
           ring_ids=set(subsets.get("P_loop_visual_same_place", [])))

    # contamination check: the negatives that score highest (never used to change the rule)
    kf = {k["image"]: k["path"] for k in json.loads((OUT / "keyframes.json").read_text())["keyframes"]}
    negs = [p_ for p_ in pairs if p_["set"] == "N"]
    top = sorted(negs, key=lambda q: -q["ratio"][f"omega@{l_star}"])[:6]
    top += [q for q in sorted(negs, key=lambda q: -q["ratio"]["vggt@21"])[:6] if q not in top]
    sheet = [dict(q, path_late=kf[q["image_late"]], path_early=kf[q["image_early"]]) for q in top]
    calib_plots.contact_sheet(
        sheet, OUT / "negatives_top_sheet.png",
        f"N 中比值最高的 {len(sheet)} 对(Omega 第{l_star}层前6 + VGGT 第21层前6,去重);核查负例是否混入同一地点",
        extra=lambda q: f"\nOmega@{l_star}={q['ratio'][f'omega@{l_star}']:.3f}  VGGT@21={q['ratio']['vggt@21']:.3f}")

    brief = {k: summary[k] for k in ("layer_star", "FAR_vggt", "FAR_vggt_by_layer", "tau", "tau_if_vggt_layer20_far",
                                     "omega_far_at_tau", "layer_star_by_P_loop_auc_only_informational")}
    brief["auc_pooled"] = {str(k): round(v, 4) for k, v in summary["auc_pooled"].items()}
    brief["auc_P_loop_vs_N"] = {str(k): round(v, 4) for k, v in summary["auc_P_loop_vs_N"].items()}
    brief["vggt_auc"] = summary["vggt_auc"]
    brief["layer_choice_bootstrap_frequency"] = summary["layer_choice_bootstrap_frequency"]
    brief["pass_P_loop"] = {key: (v["P_loop"]["n_pass"], v["P_loop"]["n"]) for key, v in summary["stats"].items()}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
