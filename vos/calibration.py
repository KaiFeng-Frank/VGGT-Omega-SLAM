"""Pure (CPU, numpy-only) building blocks of the loop-closure similarity calibration.

The rule is pre-declared in RULE_TEXT and must not be edited after numbers have been seen; the
functions here only implement it. Dev data is office_loop only (TUM is never touched here).

Conventions
-----------
* K is the ordered keyframe list the SLAM actually sees (upstream LK selector, min_disparity 50).
  Pairs are index pairs (early, late) into K with early < late.
* Solver order: upstream builds lc_frames = (query, retrieved) where the query frame belongs to the
  NEWEST submap and the retrieved one to an OLDER submap. So every pair is evaluated as
  (late frame first, early frame second); the reverse order is only a sensitivity column.
"""
from __future__ import annotations

import math

import numpy as np

DATASET = "office_loop"
OMEGA_LAYERS = (12, 13, 15, 16, 17, 18, 19, 21, 22, 23)  # 'global' inter-frame blocks of VGGT-Omega
VGGT_LAYERS = (20, 21)
VGGT_PRIMARY_LAYER = 21  # what `VGGT()` really builds (see RULE_TEXT note); 20 is the Aggregator default
VGGT_THRESHOLD = 0.95  # solver.py: `if image_match_ratio < 0.95` -> reject
ADJ_CAP = 60
NEG_COUNT = 60
LOOP_FRACTION = 0.15
NEG_GAP_LO, NEG_GAP_HI = 0.30, 0.70
MIN_DISPARITY = 50.0

RULE_TEXT = (
    "Dev data = office_loop only (473 frames, TUM never used). K = keyframes of the upstream LK selector "
    "(min_disparity 50). P_adj = consecutive keyframe pairs (cap 60, evenly spaced). P_loop = for each keyframe "
    "in the last ceil(15%) of K, the keyframe in the first ceil(15%) of K with the smallest SALAD L2 distance "
    "(upstream ImageRetrieval on the VGGT-loader frames, i.e. the solver's own retrieval). N = pairs whose index "
    "gap is within [0.30, 0.70] x len(K), 60 pairs by an even deterministic stride over the lexicographically "
    "sorted candidates. Raw image_match_ratio from a 2-frame forward with the later (query) frame first, as the "
    "solver does. VGGT at its upstream setting, Omega at every global inter-frame block in "
    "{12,13,15,16,17,18,19,21,22,23}. L* = argmax ROC AUC of (P_adj U P_loop) vs N, ties -> layer closest to 20 "
    "(then the lower layer). FAR_vggt = fraction of N with VGGT raw ratio >= 0.95. tau = (1 - FAR_vggt) quantile "
    "(numpy linear interpolation) of Omega raw ratios on N at L*; FAR_vggt == 0 -> tau = max(Omega N ratios) + 1e-6. "
    "A loop closure passes when raw >= threshold (upstream rejects raw < threshold). "
    "Note on the VGGT layer: the rule text named layer 20, the Aggregator default, but main.py builds VGGT() whose "
    "constructor passes target_layer=21 to the Aggregator, so the upstream operating point is layer 21. FAR_vggt "
    "is taken at layer 21 and reported at layer 20 as well."
)


# ----------------------------------------------------------------------------------------------
# pair construction
# ----------------------------------------------------------------------------------------------
def evenly_spaced_indices(n_items, cap):
    """Deterministic evenly spaced subset of range(n_items) with exactly min(n_items, cap) members."""
    if cap <= 0:
        return []
    if n_items <= cap:
        return list(range(n_items))
    idx = np.round(np.linspace(0, n_items - 1, cap)).astype(int).tolist()
    if len(set(idx)) != cap:  # cannot happen for cap <= n_items (spacing >= 1) but never return fewer silently
        raise AssertionError("evenly spaced indices collided")
    return idx


def adjacent_pairs(n_kf, cap=ADJ_CAP):
    allp = [(i, i + 1) for i in range(n_kf - 1)]
    return [allp[k] for k in evenly_spaced_indices(len(allp), cap)]


def negative_pairs(n_kf, count=NEG_COUNT, lo=NEG_GAP_LO, hi=NEG_GAP_HI):
    """Pairs with lo*n <= j-i <= hi*n, lexicographically sorted, then an even stride to `count`."""
    eps = 1e-9
    cands = [(i, j) for i in range(n_kf) for j in range(i + 1, n_kf)
             if lo * n_kf - eps <= (j - i) <= hi * n_kf + eps]
    return [cands[k] for k in evenly_spaced_indices(len(cands), count)], len(cands)


def loop_group_size(n_kf, frac=LOOP_FRACTION):
    return int(math.ceil(frac * n_kf - 1e-9))


def loop_pairs(distance, frac=LOOP_FRACTION):
    """distance[a, b]: SALAD L2 distance between keyframes a and b (n x n).

    For each keyframe j of the last ceil(frac*n): the keyframe i in the first ceil(frac*n) with the
    smallest distance (first index on ties, np.argmin). Returns [(i, j)] with i < j.
    """
    n = distance.shape[0]
    m = loop_group_size(n, frac)
    if 2 * m > n:
        raise ValueError("first and last groups overlap")
    pairs = []
    for j in range(n - m, n):
        i = int(np.argmin(distance[j, :m]))
        pairs.append((i, j))
    return pairs


# ----------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------
def roc_auc(positive, negative):
    """P(pos > neg) + 0.5 P(pos == neg) (Mann-Whitney), exact over all pairs."""
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("AUC needs both classes")
    greater = (pos[:, None] > neg[None, :]).sum()
    equal = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * equal) / (pos.size * neg.size))


def bootstrap_auc_ci(positive, negative, n_boot=2000, seed=0, level=0.95):
    """Percentile bootstrap (positives and negatives resampled independently)."""
    rng = np.random.default_rng(seed)
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    values = np.empty(n_boot)
    for b in range(n_boot):
        values[b] = roc_auc(pos[rng.integers(0, pos.size, pos.size)], neg[rng.integers(0, neg.size, neg.size)])
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def choose_layer(auc_by_layer, anchor=20):
    """Argmax AUC; exact ties -> closest to `anchor`, then the lower layer."""
    best = max(auc_by_layer.values())
    tied = [layer for layer, value in auc_by_layer.items() if value >= best - 1e-12]
    return int(min(tied, key=lambda layer: (abs(layer - anchor), layer)))


def far_at(values, threshold):
    """Fraction of `values` that PASS (raw >= threshold)."""
    return float(np.mean(np.asarray(values, dtype=np.float64) >= threshold))


def tau_from_far(omega_negative, far):
    """Omega threshold that lets the same fraction of negatives pass as VGGT's operating point."""
    values = np.asarray(omega_negative, dtype=np.float64)
    if far <= 0.0:
        return float(values.max()) + 1e-6
    return float(np.quantile(values, 1.0 - far))


def set_stats(values, threshold):
    arr = np.asarray(values, dtype=np.float64)
    passed = int(np.count_nonzero(arr >= threshold))
    return {"n": int(arr.size), "min": float(arr.min()), "median": float(np.median(arr)),
            "max": float(arr.max()), "threshold": float(threshold), "n_pass": passed,
            "pass_rate": passed / arr.size}


def bootstrap_layer_choice(pos_by_layer, neg_by_layer, n_boot=2000, seed=0, anchor=20):
    """How often each layer wins the rule's argmax when the pairs are resampled (same indices for all layers).

    pos_by_layer / neg_by_layer: {layer: 1-D array}, all layers over the same pairs in the same order.
    Returns {layer: winning frequency}. A diagnostic of how sharp the L* decision is; not part of the rule.
    """
    rng = np.random.default_rng(seed)
    layers = sorted(pos_by_layer)
    n_pos = len(next(iter(pos_by_layer.values())))
    n_neg = len(next(iter(neg_by_layer.values())))
    wins = {layer: 0 for layer in layers}
    for _ in range(n_boot):
        ip, ineg = rng.integers(0, n_pos, n_pos), rng.integers(0, n_neg, n_neg)
        auc = {layer: roc_auc(np.asarray(pos_by_layer[layer])[ip], np.asarray(neg_by_layer[layer])[ineg])
               for layer in layers}
        wins[choose_layer(auc, anchor)] += 1
    return {layer: wins[layer] / n_boot for layer in layers}
