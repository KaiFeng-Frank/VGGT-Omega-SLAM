"""(f) Calibration building blocks: pair sets, AUC, layer choice, tau rule (CPU only)."""
import numpy as np
import pytest

from vos import calibration as cal


def test_evenly_spaced_is_exact_sorted_and_deterministic():
    for n, cap in [(149, 60), (60, 60), (10, 60), (4400, 60), (61, 60)]:
        idx = cal.evenly_spaced_indices(n, cap)
        assert len(idx) == min(n, cap) and idx == sorted(set(idx))
        assert idx[0] == 0 and (n <= cap or idx[-1] == n - 1)
        assert idx == cal.evenly_spaced_indices(n, cap)


def test_adjacent_pairs_are_consecutive_and_capped():
    pairs = cal.adjacent_pairs(150)
    assert len(pairs) == 60 and all(j == i + 1 for i, j in pairs) and pairs[0] == (0, 1) and pairs[-1] == (148, 149)
    assert cal.adjacent_pairs(10) == [(i, i + 1) for i in range(9)]


def test_negative_pairs_respect_gap_window_and_count():
    n = 150
    pairs, total = cal.negative_pairs(n)
    assert len(pairs) == 60 and total > 60 and len(set(pairs)) == 60
    gaps = np.array([j - i for i, j in pairs])
    assert gaps.min() >= 0.30 * n - 1e-9 and gaps.max() <= 0.70 * n + 1e-9
    assert all(i < j for i, j in pairs)
    # deterministic
    assert pairs == cal.negative_pairs(n)[0]
    # gap window boundaries are inclusive: n=10 -> gaps 3..7
    small, total_small = cal.negative_pairs(10, count=1000)
    assert total_small == len(small) == sum(10 - g for g in range(3, 8))


def test_loop_pairs_pick_best_match_in_first_group():
    n = 20
    m = cal.loop_group_size(n)
    assert m == 3  # ceil(0.15 * 20)
    rng = np.random.default_rng(0)
    dist = rng.uniform(0.8, 1.2, (n, n))
    dist = (dist + dist.T) / 2
    dist[19, 2] = dist[2, 19] = 0.1
    dist[18, 0] = dist[0, 18] = 0.2
    pairs = cal.loop_pairs(dist)
    assert [j for _, j in pairs] == [17, 18, 19]
    assert dict((j, i) for i, j in pairs)[19] == 2 and dict((j, i) for i, j in pairs)[18] == 0
    assert all(i < m for i, _ in pairs)
    with pytest.raises(ValueError):
        cal.loop_pairs(np.zeros((4, 4)), frac=0.6)


def test_loop_group_size_float_robust():
    assert cal.loop_group_size(100) == 15  # 0.15 * 100 is 15.000000000000002 in floating point
    assert cal.loop_group_size(20) == 3


def test_roc_auc_matches_definition_with_ties():
    assert cal.roc_auc([2, 3], [0, 1]) == 1.0
    assert cal.roc_auc([0, 1], [2, 3]) == 0.0
    assert cal.roc_auc([1, 1], [1, 1]) == 0.5
    assert cal.roc_auc([1, 3], [2]) == 0.5
    rng = np.random.default_rng(1)
    pos, neg = rng.normal(1, 1, 40), rng.normal(0, 1, 50)
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert cal.roc_auc(pos, neg) == pytest.approx(brute)
    lo, hi = cal.bootstrap_auc_ci(pos, neg, n_boot=200)
    assert lo < cal.roc_auc(pos, neg) < hi


def test_choose_layer_ties_prefer_closest_to_20_then_lower():
    assert cal.choose_layer({12: 0.9, 19: 0.95, 23: 0.8}) == 19
    assert cal.choose_layer({19: 0.95, 21: 0.95, 23: 0.8}) == 19  # equal distance -> lower
    assert cal.choose_layer({17: 0.95, 22: 0.95}) == 22  # |22-20| < |17-20|
    assert cal.choose_layer({15: 0.96, 19: 0.95}) == 15  # strict argmax wins over proximity


def test_tau_rule_matches_far_and_zero_far_case():
    neg = np.arange(60, dtype=float) / 60  # 0 .. 0.983
    assert cal.tau_from_far(neg, 0.0) == pytest.approx(neg.max() + 1e-6)
    for k in (1, 2, 3, 6):
        tau = cal.tau_from_far(neg, k / 60)
        assert cal.far_at(neg, tau) == pytest.approx(k / 60)
    # zero-FAR tau lets no negative pass
    assert cal.far_at(neg, cal.tau_from_far(neg, 0.0)) == 0.0


def test_set_stats_pass_rate_uses_ge():
    s = cal.set_stats([0.9, 0.95, 1.0, 1.2], 0.95)
    assert s["n_pass"] == 3 and s["pass_rate"] == 0.75 and s["min"] == 0.9 and s["max"] == 1.2


def test_bootstrap_layer_choice_prefers_the_clearly_best_layer():
    rng = np.random.default_rng(3)
    pos = {15: rng.normal(1.5, 0.2, 50), 19: rng.normal(0.9, 0.3, 50)}
    neg = {15: rng.normal(0.5, 0.2, 40), 19: rng.normal(0.7, 0.3, 40)}
    freq = cal.bootstrap_layer_choice(pos, neg, n_boot=100)
    assert freq[15] > 0.95 and abs(sum(freq.values()) - 1.0) < 1e-12
