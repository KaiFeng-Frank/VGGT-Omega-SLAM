"""CPU regressions for the cached confidence/scale audit, independent of real data."""
import numpy as np
import pytest

from tools.audit_backend_cache import confidence_mask, policy_mask, scale_metrics, valid_pixels


def test_each_submap_threshold_is_invariant_to_current_confidence_units():
    pc = np.array([4., 3., 2., 1.])
    cc = np.array([1., 2., 3., 4.])
    valid = np.ones(4, bool)
    prior = {"conf": pc, "submap_q25": 1.5}
    current = {"conf": cc, "submap_q25": 2.5}
    mask, _ = policy_mask("each_submap_q25", prior, current, valid, minimum_points=0)
    rescaled = {"conf": cc * 100, "submap_q25": 250.}
    mask_rescaled, _ = policy_mask("each_submap_q25", prior, rescaled, valid, minimum_points=0)
    legacy, _ = policy_mask("legacy_prior_q25", prior, current, valid, minimum_points=0)
    legacy_rescaled, _ = policy_mask("legacy_prior_q25", prior, rescaled, valid, minimum_points=0)
    assert np.array_equal(mask, mask_rescaled)
    assert not np.array_equal(legacy, legacy_rescaled)


def test_fallback_levels_are_reported_and_invalid_depth_is_not_used():
    pc = np.array([1., 2., 3., 4.])
    cc = np.ones(4)
    valid = np.array([True, True, False, True])
    mask, report = confidence_mask(pc, cc, 1.5, 10., valid, minimum_points=2)
    assert report["fallback_level"] == 1
    assert mask.tolist() == [False, True, False, True]
    mask, report = confidence_mask(pc, cc, 10., 10., valid, minimum_points=2)
    assert report["fallback_level"] == 2
    assert mask.tolist() == valid.tolist()
    assert valid_pixels([1., 0., 2., np.nan], [1., 1., 2., 1.], pc, cc).tolist() == [True, False, True, False]


def test_depth_ratio_matches_upstream_intrinsic_transfer_scale():
    from vos.why_utils import cam_points
    rng = np.random.default_rng(4)
    prior_depth = rng.uniform(.5, 3, (12, 16))
    current_depth = prior_depth / 1.7 * rng.uniform(.99, 1.01, (12, 16))
    prior_k = np.array([[13., 0, 7.7], [0, 11., 5.5], [0, 0, 1.]])
    current_k = np.array([[16., 0, 8.2], [0, 15., 6.1], [0, 0, 1.]])
    mask = rng.uniform(size=prior_depth.shape) > .3
    transformed_current = (np.linalg.inv(prior_k) @ current_k @ cam_points(current_depth, current_k).reshape(-1, 3).T).T
    prior_points = cam_points(prior_depth, prior_k).reshape(-1, 3)
    ratio = np.linalg.norm(prior_points[mask.ravel()], axis=1) / np.linalg.norm(transformed_current[mask.ravel()], axis=1)
    actual = scale_metrics(prior_depth, current_depth, mask, 1.7)
    assert actual["estimated_scale"] == pytest.approx(np.median(ratio), rel=2e-7)


def test_scale_error_uses_pose_groundtruth_and_is_not_optimized_per_mask():
    prior = np.array([2., 2., 2., 2.])
    current = np.array([1., 1., 1.2, 1.2])
    m = scale_metrics(prior, current, np.array([True, True, False, False]), 2.)
    assert m["abs_log_scale_error"] == pytest.approx(0)
    biased = scale_metrics(prior, current, np.array([False, False, True, True]), 2.)
    assert biased["abs_log_scale_error"] == pytest.approx(abs(np.log(1 / 1.2)))
    assert biased["true_scale"] == m["true_scale"] == 2.
