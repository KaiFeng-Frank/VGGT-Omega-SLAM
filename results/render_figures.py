#!/usr/bin/env python3
"""Render the published numerical summaries without datasets, models or a GPU."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
COLORS = {"vggt": "#286da8", "omega": "#c96225"}
LABELS = {"vggt": "VGGT", "omega": "VGGT-Ω"}


def read(name):
    return json.loads((ROOT / name).read_text())


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / (name + ".png"), dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / (name + ".svg"), bbox_inches="tight", facecolor="white",
                metadata={"Date": None})
    plt.close(fig)


def distortion_summary():
    data = read("undistortion/local_metrics.json")
    summary = data["A"]["summary_full_submaps"]
    paired = data["A"]["paired_omega_vs_vggt"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.5), constrained_layout=True)
    x = np.array([0, 1])
    for backend in ("vggt", "omega"):
        focal = [100 * summary[c][backend]["pooled"]["focal"][
            "fx_bias_vs_old_signed" if c == "raw" else "fx_bias_vs_new_signed"] for c in ("raw", "und")]
        rotation = [summary[c][backend]["pooled"]["proj"]["rel_rot_err_deg_mean"]["median"] for c in ("raw", "und")]
        axes[0].plot(x, focal, "o-", color=COLORS[backend], label=LABELS[backend])
        axes[2].plot(x, rotation, "o-", color=COLORS[backend], label=LABELS[backend])
        for xx, yy in zip(x, focal):
            axes[0].annotate(f"{yy:.2f}%", (xx, yy), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
        for xx, yy in zip(x, rotation):
            offset = -17 if backend == "vggt" and xx == 0 else 8
            axes[2].annotate(f"{yy:.3f}°", (xx, yy), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=9)
    for i, condition in enumerate(("raw", "und")):
        ratios = []
        for seq in ("desk", "desk2", "room"):
            rows = data["A"]["per_submap"][condition][seq]
            v = {r["update"]: r for r in rows["vggt"] if r["full"]}
            o = {r["update"]: r for r in rows["omega"] if r["full"]}
            ratios += [o[k]["proj"]["ate_over_gt_path"] / v[k]["proj"]["ate_over_gt_path"] for k in sorted(v)]
        axes[1].scatter(np.linspace(i - .13, i + .13, len(ratios)), ratios, color="#555555", s=16, alpha=.75)
        median = paired[condition]["pooled"]["proj"]["ate_over_gt_path"]["median_ratio_omega_over_vggt"]
        axes[1].plot([i - .23, i + .23], [median, median], color=COLORS["omega"], linewidth=2.8)
        axes[1].annotate(f"median {median:.3f}", (i, median), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
    axes[1].axhline(1, color="#999999", linestyle="--", linewidth=1)
    axes[0].set(title="Focal bias", ylabel="Mean signed fx error (%)", ylim=(0, 16))
    axes[1].set(title="Paired translation-shape error", ylabel="VGGT-Ω / VGGT ATE", ylim=(.5, 2.6))
    axes[2].set(title="Corrected relative rotation", ylabel="Median window-mean error (degrees)", ylim=(.4, 1.05))
    for ax in axes:
        ax.set_xticks(x, ["Raw", "Undistorted"])
        ax.set_xlim(-.4, 1.4)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="center right")
    fig.suptitle("20 matched full windows · archived decoded focal values and SO(3)-projected geometry", fontsize=11)
    save(fig, "undistortion_summary")


def principal_point_control():
    report = read("focal_cause/evidence_intervention_pairs.json")
    pairs = report["pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), constrained_layout=True)
    for row in pairs:
        axes[0].plot([0, 1], [100 * row["control"]["fx_bias"], 100 * row["centered"]["fx_bias"]], color=COLORS["omega"], alpha=.33, marker="o", markersize=3)
        axes[1].plot([0, 1], [1, row["ate_ratio"]], color="#555555", alpha=.33, marker="o", markersize=3)
    axes[0].plot([0, 1], [100 * report["pooled"]["fx_bias_control_mean"], 100 * report["pooled"]["fx_bias_centered_mean"]], "o-", color="#1b1b1b", linewidth=2.5, label="Window mean")
    axes[1].plot([0, 1], [1, report["pooled"]["ate_ratio_median"]], "o-", color="#1b1b1b", linewidth=2.5, label="Paired median")
    axes[0].set(title="Focal bias: −0.555 percentage points", ylabel="Signed fx error versus each case's K (%)")
    axes[1].set(title="Paired ATE: ×0.980", ylabel="ATE / matched control ATE")
    axes[1].axhline(1, color="#999999", linestyle="--", linewidth=1)
    for ax in axes:
        ax.set_xticks([0, 1], ["Original\nprincipal point", "Centered\nprincipal point"])
        ax.set_xlim(-.2, 1.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle("VGGT-Ω: identical 6% zoom, resolution and frame lists · 20 paired windows", fontsize=11)
    save(fig, "principal_point_control")


def slam_ate():
    report = read("undistortion/slam_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True, sharey=True)
    configs = [(s, n) for s in ("desk", "desk2", "room") for n in ("16", "32")]
    y = np.arange(len(configs))
    for ax, cond, title in zip(axes, ("raw", "und"), ("Raw images", "Undistorted images")):
        for backend, shift in (("vggt", -.18), ("omega", .18)):
            values = [report["runs"][s][n][backend][cond]["ate_rmse_m"] for s, n in configs]
            ax.barh(y + shift, values, height=.33, color=COLORS[backend], label=LABELS[backend])
        ax.set(title=title, xlabel="Sim(3)-aligned SLAM ATE (metres)", xlim=(0, .105))
        ax.set_yticks(y, [s + " / " + n for s, n in configs])
        ax.grid(axis="x", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("TUM fr1 sequence / submap size")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("12 paired configurations · 24 loop-enabled runs · one seed per run", fontsize=11)
    save(fig, "slam_ate")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "vggt-omega-slam-results"})
    distortion_summary()
    principal_point_control()
    slam_ate()
