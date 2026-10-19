"""CPU-only counterexamples and geometry checks for opt-in backend policies."""
from types import SimpleNamespace

import numpy as np
import pytest

from vos.backend_policy import (BackendPolicy, ScaleEstimationError, estimate_overlap_scale,
                                make_factor_noise, make_solver_class)
from vos import backend_policy as bp


def _rays(n):
    # Positive-depth camera rays, varying in both image axes.
    u = np.linspace(-0.8, 0.9, n)
    return np.column_stack((u, np.sin(u), np.ones(n)))


def test_legacy_is_the_original_class_and_never_wraps_numerics():
    class Original:
        def add_edge(self, marker):
            return marker
    cls = make_solver_class(Original)
    marker = object()
    assert cls is Original and cls.add_edge is Original.add_edge
    assert cls().add_edge(marker) is marker
    with pytest.raises(ValueError, match="overrides require"):
        BackendPolicy(scale_confidence_percentile=50)


def test_independent_thresholds_reject_low_current_confidence_with_different_score_ranges():
    rays = _rays(240)
    # Correct pixels have scale 2. Grossly wrong depth has scale 20, but its
    # current-window confidence is below that window's cutoff. All current
    # scores exceed the *prior* window's numerical cutoff, exposing the old
    # cross-window comparison without relying on a replica of the algorithm.
    truth = np.r_[np.full(100, 2.0), np.full(140, 20.0)]
    pc = np.full(240, 2.0)
    cc = np.r_[np.full(100, 200.0), np.full(140, 100.0)]
    assert np.all(cc > 1.5)  # a reused prior cutoff would admit every outlier
    scale, diag = estimate_overlap_scale(rays, rays * truth[:, None], cc, pc, 150., 1.5)
    assert scale == pytest.approx(2.0)
    assert diag["n_scale_pixels"] == 100 and diag["fallback_level"] == 0
    assert diag["log_scale_mad"] == pytest.approx(0, abs=1e-15)


def test_scale_diagnostics_report_log_dispersion_in_known_units():
    rays = _rays(3)
    scale, diag = estimate_overlap_scale(rays, rays * np.array([1., 2., 4.])[:, None],
                                        np.ones(3), np.ones(3), 0., 0., min_points=1)
    assert scale == pytest.approx(2.)
    assert diag["log_scale_median"] == pytest.approx(np.log(2.))
    assert diag["log_scale_mad"] == pytest.approx(np.log(2.))


@pytest.mark.parametrize("prior_cut,current_cut,level", [(0., 2., 1), (2., 2., 2)])
def test_fallback_levels_are_explicit(prior_cut, current_cut, level):
    rays = _rays(8)
    scale, diag = estimate_overlap_scale(rays, 3 * rays, np.ones(8), np.ones(8),
                                        current_cut, prior_cut, min_points=4)
    assert scale == pytest.approx(3.)
    assert diag["fallback_level"] == level and diag["n_scale_pixels"] == 8


def test_nonfinite_zero_norm_and_invalid_confidence_pixels_do_not_enter_scale():
    rays = _rays(6)
    prior = rays * 2
    rays[1] = np.nan
    prior[2] = np.inf
    rays[3] = 0
    conf = np.array([2., 2., 2., 2., np.nan, -1.])
    scale, diag = estimate_overlap_scale(rays, prior, conf, conf, 1., 1., min_points=1)
    assert scale == pytest.approx(2.)
    assert diag["n_valid_pixels"] == diag["n_scale_pixels"] == 1
    with pytest.raises(ScaleEstimationError) as caught:
        estimate_overlap_scale(np.zeros((3, 3)), np.zeros((3, 3)), np.ones(3), np.ones(3), 0., 0.)
    assert caught.value.diagnostics["status"] == "no_valid_scale_pixels"
    assert caught.value.diagnostics["n_scale_pixels"] == 0


def test_bad_injected_scale_estimator_is_rejected():
    rays = _rays(4)
    with pytest.raises(ScaleEstimationError, match="invalid scale"):
        estimate_overlap_scale(rays, rays, np.ones(4), np.ones(4), 0., 0., min_points=1,
                               estimator=lambda *_: (float("nan"), None))


class FakeSubmap:
    def __init__(self, points, confidence, cutoff, poses=None, loop=False):
        self.points = np.asarray(points)[None]
        self.conf_masks = np.asarray(confidence)[None]
        self.cutoff = cutoff
        self.poses = np.asarray(poses if poses is not None else [np.eye(4)])
        self.proj_mats = np.repeat(np.eye(4)[None], len(self.poses), axis=0)
        self.loop = loop

    def get_frame_pointcloud(self, _frame):
        return self.points[0]

    def get_conf_masks_frame(self, _frame):
        return self.conf_masks[0]

    def get_conf_threshold(self):
        return self.cutoff

    def get_lc_status(self):
        return self.loop

    def get_all_poses(self):
        return self.poses


class FakeGraph:
    def __init__(self):
        self.intra_submap_noise = object()
        self.inner_submap_noise = object()
        self.nodes = {0: np.eye(4), 10: np.eye(4)}
        self.edges = []

    def get_homography(self, key):
        return self.nodes[key]

    def add_homography(self, key, matrix):
        self.nodes[key] = matrix.copy()

    def add_between_factor(self, previous, current, matrix, noise):
        self.edges.append((previous, current, matrix.copy(), noise))


class FakeBaseSolver:
    # __init__ must never be needed by offline prediction replay.
    def __init__(self):
        pytest.fail("policy replay instantiated a runtime/GPU solver")

    def add_edge(self, *args):
        self.original_call = args


def _solver(policy, prior, current):
    cls = make_solver_class(FakeBaseSolver, policy)
    solver = cls.__new__(cls)
    solver.map = SimpleNamespace(get_submap={0: prior, 10: current}.__getitem__)
    solver.graph = FakeGraph()
    return solver


def test_separate_scale_percentile_uses_entire_each_submap_without_mutating_cloud_cutoff():
    rays = _rays(8)
    ratios = np.r_[np.full(4, 2.), np.full(4, 20.)]
    # Stored cloud cutoffs retain all pixels. An independently configured
    # scale percentile selects only the four reliable measurements.
    prior = FakeSubmap(rays * ratios[:, None], np.arange(8., 0., -1), 0.)
    current = FakeSubmap(rays, np.arange(800., 0., -100), 0.)
    policy = BackendPolicy(name="omega_confidence", scale_confidence_percentile=50., min_scale_points=4)
    solver = _solver(policy, prior, current)
    solver.add_edge(10, 0, 0, 0)
    diag = solver.backend_diagnostics[0]
    assert diag["scale_factor"] == pytest.approx(2.) and diag["n_scale_pixels"] == 4
    assert diag["prior_conf_threshold"] == pytest.approx(4.5 + 1e-6)
    assert diag["current_conf_threshold"] == pytest.approx(450. + 1e-6)
    assert current.cutoff == prior.cutoff == 0.


def test_subclass_preserves_analytically_known_camera_chain_and_noise_identity():
    rays = _rays(120)
    prior = FakeSubmap(2 * rays, np.full(120, 2.), 1.)
    pose0, pose1 = np.eye(4), np.eye(4)
    pose1[0, 3] = -0.5  # second camera centre is +0.5 in its submap
    current = FakeSubmap(rays, np.full(120, 20.), 10., poses=[pose0, pose1])
    solver = _solver(BackendPolicy(name="omega_confidence"), prior, current)
    solver.add_edge(10, 0, 0, 0)
    # Scale 2 converts the +0.5 model-unit baseline into +1 prior unit.
    expected = np.diag([2., 2., 2., 1.])
    expected[0, 3] = 1.
    assert np.allclose(solver.graph.nodes[11], expected)
    assert len(solver.graph.edges) == 2
    assert solver.graph.edges[0][3] is solver.graph.intra_submap_noise
    assert solver.graph.edges[1][3] is solver.graph.inner_submap_noise
    assert solver.backend_diagnostics[0]["edge_kind"] == "cross_submap"


def test_frame_percentile_is_not_a_whole_submap_percentile():
    rays = _rays(8)
    ratios = np.r_[np.full(4, 2.), np.full(4, 20.)]
    prior = FakeSubmap(rays * ratios[:, None], np.arange(8., 0., -1), 0.)
    current = FakeSubmap(rays, np.arange(800., 0., -100), 0.)
    # A different frame has much higher scores. Its distribution must not
    # affect this overlap frame's explicit frame-percentile selection.
    prior.conf_masks = np.vstack([prior.conf_masks, np.full((1, 8), 100.)])
    current.conf_masks = np.vstack([current.conf_masks, np.full((1, 8), 8000.)])
    policy = BackendPolicy(name="omega_confidence", scale_confidence_percentile=50.,
                           scale_confidence_scope="frame", min_scale_points=4)
    solver = _solver(policy, prior, current)
    solver.add_edge(10, 0, 0, 0)
    diag = solver.backend_diagnostics[0]
    assert diag["scale_factor"] == pytest.approx(2.) and diag["n_scale_pixels"] == 4
    assert diag["prior_conf_threshold"] == pytest.approx(4.5 + 1e-6)
    assert diag["current_conf_threshold"] == pytest.approx(450. + 1e-6)
    assert diag["scale_confidence_scope"] == "frame"


def test_first_node_uses_original_anchor_path_and_loop_closure_adds_no_duplicate_nodes():
    rays = _rays(120)
    prior = FakeSubmap(2 * rays, np.full(120, 2.), 1.)
    current = FakeSubmap(rays, np.full(120, 20.), 10.)
    solver = _solver(BackendPolicy(name="omega_confidence"), prior, current)
    solver.add_edge(0, 0)
    assert solver.original_call == (0, 0, None, None, False)
    keys = set(solver.graph.nodes)
    solver.add_edge(10, 0, 0, 0, is_loop_closure=True)
    assert set(solver.graph.nodes) == keys and len(solver.graph.edges) == 1
    assert solver.backend_diagnostics[0]["edge_kind"] == "loop"


def test_default_noise_is_unchanged_and_missing_robust_api_is_an_error():
    marker = object()
    assert make_factor_noise(marker) is marker
    base = SimpleNamespace(sigmas=lambda: np.ones(1))
    minimal = SimpleNamespace(noiseModel=SimpleNamespace(Diagonal=SimpleNamespace(Sigmas=lambda x: x)))
    with pytest.raises(RuntimeError, match="Huber robust-noise API"):
        make_factor_noise(base, huber_delta=1., gtsam_module=minimal)


def test_real_gtsam_sigma_scaling_and_huber_downweight_outliers():
    gtsam = pytest.importorskip("gtsam", reason="optional real GTSAM noise API check")
    base = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5]))
    scaled = make_factor_noise(base, 2., gtsam_module=gtsam)
    assert np.allclose(scaled.sigmas(), [1.])
    # Doubling sigma divides squared Gaussian residual cost by four.
    residual = np.array([4.])
    assert np.dot(scaled.whiten(residual), scaled.whiten(residual)) == pytest.approx(
        np.dot(base.whiten(residual), base.whiten(residual)) / 4.)
    robust = make_factor_noise(scaled, huber_delta=1., gtsam_module=gtsam)
    assert np.linalg.norm(robust.whiten(np.array([0.5]))) == pytest.approx(0.5)
    assert np.linalg.norm(robust.whiten(residual)) < np.linalg.norm(scaled.whiten(residual))


def test_noise_multipliers_distinguish_regular_and_auxiliary_loop_factors():
    gtsam = pytest.importorskip("gtsam", reason="optional real GTSAM noise API check")
    rays = _rays(120)
    prior = FakeSubmap(2 * rays, np.full(120, 2.), 1.)
    current = FakeSubmap(rays, np.full(120, 20.), 10., poses=[np.eye(4), np.eye(4)])
    policy = BackendPolicy(name="omega_confidence", cross_submap_noise_multiplier=3., loop_noise_multiplier=5.)
    for loop in (False, True):
        current.loop = loop
        solver = _solver(policy, prior, current)
        solver.graph.intra_submap_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.]))
        solver.graph.inner_submap_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([2.]))
        solver.add_edge(10, 0, 0, 0)
        assert np.allclose(solver.graph.edges[0][3].sigmas(), [5. if loop else 3.])
        assert np.allclose(solver.graph.edges[1][3].sigmas(), [10. if loop else 2.])


def test_noise_only_calls_original_scale_path_and_restores_noise_on_failure(monkeypatch):
    gtsam = pytest.importorskip("gtsam", reason="optional real GTSAM noise API check")
    monkeypatch.setattr(bp, "estimate_overlap_scale", lambda *_a, **_k: pytest.fail("noise_only changed scale selection"))
    class Original:
        def add_edge(self, *args):
            self.captured = (args, self.graph.intra_submap_noise, self.graph.inner_submap_noise)
            if self.fail:
                raise RuntimeError("upstream error")
            return "original result"
    policy = BackendPolicy(name="noise_only", cross_submap_noise_multiplier=2., loop_noise_multiplier=4.)
    cls = make_solver_class(Original, policy, gtsam_module=gtsam)
    for loop in (False, True):
        solver = cls.__new__(cls)
        solver.map = SimpleNamespace(get_submap=lambda _key: SimpleNamespace(get_lc_status=lambda: loop))
        cross = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.]))
        inner = gtsam.noiseModel.Diagonal.Sigmas(np.array([3.]))
        solver.graph = SimpleNamespace(intra_submap_noise=cross, inner_submap_noise=inner)
        solver.fail = False
        assert solver.add_edge(10, 0, 0, 0) == "original result"
        assert np.allclose(solver.captured[1].sigmas(), [4. if loop else 2.])
        assert np.allclose(solver.captured[2].sigmas(), [12. if loop else 3.])
        assert solver.graph.intra_submap_noise is cross and solver.graph.inner_submap_noise is inner
        diag = solver.backend_diagnostics[-1]
        assert diag["internal_noise_sigma_multiplier"] == (4. if loop else 1.)
        assert diag["internal_huber_delta"] is None
        if not loop:
            assert solver.add_edge(0, 0) == "original result"
            assert solver.captured[1] is cross and solver.captured[2] is inner
            anchor = solver.backend_diagnostics[-1]
            assert anchor["edge_kind"] == "anchor" and anchor["noise_sigma_multiplier"] == 1.
            assert anchor["huber_delta"] is None
        solver.fail = True
        with pytest.raises(RuntimeError, match="upstream error"):
            solver.add_edge(10, 0, 0, 0)
        assert solver.graph.intra_submap_noise is cross and solver.graph.inner_submap_noise is inner


@pytest.mark.parametrize("kwargs", [{"scale_confidence_percentile": 25.},
                                    {"scale_confidence_scope": "frame", "scale_confidence_percentile": 25.},
                                    {"min_scale_points": 10}])
def test_noise_only_forbids_scale_selection_overrides(kwargs):
    with pytest.raises(ValueError, match="preserves upstream scale/mask/fallback"):
        BackendPolicy(name="noise_only", **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"scale_confidence_percentile": 101}, {"cross_submap_noise_multiplier": 0},
    {"loop_noise_multiplier": -1}, {"huber_delta": float("nan")}, {"min_scale_points": 0},
])
def test_invalid_policy_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        BackendPolicy(name="omega_confidence", **kwargs)
