"""Figures of the calibration (all text in Simplified Chinese). Headless: Agg backend, files only."""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
# categorical slots 1-3 of the reference palette (validated all-pairs in both modes)
SET_COLORS = {"P_adj": "#2a78d6", "P_loop": "#eb6834", "N": "#1baf7a"}
SET_LABELS = {"P_adj": "相邻关键帧 P_adj", "P_loop": "回环重访 P_loop", "N": "远距负例 N"}


def setup_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK JP", "Droid Sans Fallback"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        raise RuntimeError("no CJK font found (Noto Sans CJK JP / Droid Sans Fallback)")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID})
    return plt.rcParams["font.family"]


def contact_sheet(pairs, path, title, columns=3, thumb=(320, 180), extra=None):
    """Grid of pairs; each cell shows [late/query | early/retrieved] side by side.

    pairs : list of dicts with image_late/image_early (names), path_late/path_early, kf_late/kf_early,
            pair_id and optionally salad_l2.   extra: callable(pair) -> str appended to the cell title.
    """
    from PIL import Image

    setup_font()
    rows = int(np.ceil(len(pairs) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 6.2, rows * 2.55 + 0.9), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, pair in zip(axes.ravel(), pairs):
        tiles = [np.asarray(Image.open(pair[key]).convert("RGB").resize(thumb, Image.Resampling.BILINEAR))
                 for key in ("path_late", "path_early")]
        gap = np.full((thumb[1], 6, 3), 255, dtype=np.uint8)
        ax.imshow(np.concatenate([tiles[0], gap, tiles[1]], axis=1))
        label = (f"{pair['pair_id']}  关键帧{pair['kf_late']}(查询,{pair['image_late'][6:10]}) ↔ "
                 f"关键帧{pair['kf_early']}(检索,{pair['image_early'][6:10]})")
        if "salad_l2" in pair:
            label += f"\nSALAD 距离 {pair['salad_l2']:.3f}"
        if extra is not None:
            label += "  " + extra(pair)
        ax.set_title(label, fontsize=8.5, color=INK, loc="left")
    fig.suptitle(title, fontsize=13, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=110)
    plt.close(fig)
