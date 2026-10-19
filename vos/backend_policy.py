"""Opt-in backend ablations; the default returns the original Solver unchanged.

``omega_confidence`` changes confidence selection for overlap-scale estimation
and, when explicitly requested, cross-submap/loop factor noise. It is an
experimental candidate, not a demonstrated accuracy improvement. The scale
estimator and SL(4) geometry remain those of the pinned upstream pipeline.
``noise_only`` calls the original add_edge, temporarily replacing only noise
objects; all upstream confidence selection and fallback behavior is retained.

The add_edge control flow is adapted from MIT-SPARK/VGGT-SLAM, commit
35327ac28b7d193df9ccc39ba6346052bb6f1207, Copyright (c) 2025 MIT-SPARK,
under its BSD 2-Clause license; see licenses/VGGT-SLAM-BSD-2-Clause.txt.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

POLICY_NAMES = ("legacy", "omega_confidence", "noise_only")


@dataclass(frozen=True)
class BackendPolicy:
    name: str = "legacy"
    # Percentile over each submap or overlap frame, independently of cloud
    # filtering. None reuses each submap's stored confidence cutoff.
    scale_confidence_percentile: float | None = None
    scale_confidence_scope: str = "submap"
    # Multipliers apply to standard deviations, not variances/information.
    cross_submap_noise_multiplier: float = 1.0
    loop_noise_multiplier: float = 1.0
    huber_delta: float | None = None
    min_scale_points: int = 100

    def __post_init__(self):
        if self.name not in POLICY_NAMES:
            raise ValueError(f"Unknown backend policy {self.name!r}; choose {POLICY_NAMES}")
        q = self.scale_confidence_percentile
        if q is not None and (not math.isfinite(q) or not 0 <= q <= 100):
            raise ValueError("scale_confidence_percentile must be in [0, 100]")
        if self.scale_confidence_scope not in ("submap", "frame"):
            raise ValueError("scale_confidence_scope must be 'submap' or 'frame'")
        if self.scale_confidence_scope == "frame" and q is None:
            raise ValueError("Frame confidence scope requires an explicit scale_confidence_percentile")
        for value in (self.cross_submap_noise_multiplier, self.loop_noise_multiplier):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Noise standard-deviation multipliers must be finite and positive")
        if self.huber_delta is not None and (not math.isfinite(self.huber_delta) or self.huber_delta <= 0):
            raise ValueError("huber_delta must be finite and positive")
        if isinstance(self.min_scale_points, bool) or not isinstance(self.min_scale_points, int) or self.min_scale_points < 1:
            raise ValueError("min_scale_points must be a positive integer")
        if self.name == "legacy" and (
            q is not None or self.scale_confidence_scope != "submap"
            or self.cross_submap_noise_multiplier != 1 or self.loop_noise_multiplier != 1
            or self.huber_delta is not None or self.min_scale_points != 100
        ):
            raise ValueError("Backend overrides require an opt-in nonlegacy policy; legacy is unchanged")
        if self.name == "noise_only" and (q is not None or self.scale_confidence_scope != "submap"
                                           or self.min_scale_points != 100):
            raise ValueError("noise_only preserves upstream scale/mask/fallback; q, scope and min_scale_points overrides are forbidden")

    def to_dict(self):
        return asdict(self)


class ScaleEstimationError(RuntimeError):
    """An overlap has no usable scale estimate; diagnostics remain available."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _median_norm_scale(current_points, prior_points):
    # Same arithmetic as the pinned estimate_scale_pairwise; no metric scale
    # or confidence weighting is introduced by this policy.
    return np.median(np.linalg.norm(prior_points, axis=1) / np.linalg.norm(current_points, axis=1)), None


def estimate_overlap_scale(current_points, prior_points, current_conf, prior_conf,
                           current_threshold, prior_threshold, *, min_points=100, estimator=None):
    """Estimate scale with independently thresholded confidence and diagnostics.

    Points must already be in the corresponding overlap-camera coordinates.
    Level 0 uses both submaps' own high-confidence masks. If fewer than
    ``min_points`` survive, level 1 uses high prior confidence plus positive
    current confidence; level 2 accepts positive confidence on both sides.
    Every level also requires finite points/confidences and nonzero point
    norms. Level 2 may retain fewer than ``min_points``; this is reported.
    Zero valid points or a nonpositive/nonfinite estimate raises before a
    graph factor can be created. MAD is the unscaled natural-log-ratio MAD.
    """
    x, y = np.asarray(current_points), np.asarray(prior_points)
    if x.shape != y.shape or x.ndim < 2 or x.shape[-1] != 3:
        raise ValueError("Overlap point arrays must have matching (..., 3) shapes")
    x, y = x.reshape(-1, 3), y.reshape(-1, 3)
    cc, pc = np.asarray(current_conf).reshape(-1), np.asarray(prior_conf).reshape(-1)
    if len(cc) != len(x) or len(pc) != len(x):
        raise ValueError("Overlap confidence and point arrays must contain the same number of pixels")
    if not math.isfinite(float(current_threshold)) or not math.isfinite(float(prior_threshold)):
        raise ValueError("Overlap confidence cutoffs must be finite")
    if min_points < 1:
        raise ValueError("min_points must be positive")

    nx = np.linalg.norm(x.astype(np.float64), axis=1)
    ny = np.linalg.norm(y.astype(np.float64), axis=1)
    valid = (np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
             & np.isfinite(cc) & np.isfinite(pc) & (cc > 0) & (pc > 0)
             & np.isfinite(nx) & np.isfinite(ny) & (nx > 0) & (ny > 0))
    high_current, high_prior = cc > current_threshold, pc > prior_threshold
    masks = [valid & high_prior & high_current, valid & high_prior, valid]
    level = 0
    while level < 2 and np.count_nonzero(masks[level]) < min_points:
        level += 1
    mask = masks[level]
    count = int(np.count_nonzero(mask))
    diagnostic = {
        "n_total_pixels": int(len(x)), "n_valid_pixels": int(np.count_nonzero(valid)),
        "n_joint_confident_pixels": int(np.count_nonzero(masks[0])), "n_scale_pixels": count,
        "current_conf_threshold": float(current_threshold), "prior_conf_threshold": float(prior_threshold),
        "fallback_level": level,
        "fallback_name": ("joint_quantiles", "prior_quantile", "positive_confidence")[level],
        "minimum_requested_pixels": int(min_points), "below_requested_pixels": count < min_points,
        "scale_factor": None, "log_scale_median": None, "log_scale_mad": None,
    }
    if not count:
        diagnostic["status"] = "no_valid_scale_pixels"
        raise ScaleEstimationError("No finite positive-confidence overlap points for scale estimation", diagnostic)
    log_ratios = np.log(ny[mask]) - np.log(nx[mask])
    log_median = float(np.median(log_ratios))
    diagnostic.update(log_scale_median=log_median,
                      log_scale_mad=float(np.median(np.abs(log_ratios - log_median))))
    output = (estimator or _median_norm_scale)(x[mask], y[mask])
    scale = float(output[0] if isinstance(output, (tuple, list)) else output)
    if not math.isfinite(scale) or scale <= 0:
        diagnostic["status"] = "invalid_scale_estimate"
        raise ScaleEstimationError(f"Scale estimator returned invalid scale {scale}", diagnostic)
    diagnostic.update(status="ok", scale_factor=scale)
    return scale, diagnostic


def make_factor_noise(base_noise, multiplier=1.0, huber_delta=None, *, gtsam_module=None):
    """Scale diagonal sigmas and optionally robustify in whitened residual units.

    With multiplier 1 and no Huber, return the original object exactly. A
    requested but unavailable GTSAM robust API fails explicitly.
    """
    if multiplier == 1 and huber_delta is None:
        return base_noise
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Noise standard-deviation multiplier must be finite and positive")
    if huber_delta is not None and (not math.isfinite(huber_delta) or huber_delta <= 0):
        raise ValueError("huber_delta must be finite and positive")
    if gtsam_module is None:
        try:
            import gtsam as gtsam_module
        except ImportError as exc:
            raise RuntimeError("Factor-noise overrides require compatible GTSAM Python bindings") from exc
    try:
        gaussian = gtsam_module.noiseModel.Diagonal.Sigmas(np.asarray(base_noise.sigmas()) * multiplier)
        if huber_delta is None:
            return gaussian
        huber = gtsam_module.noiseModel.mEstimator.Huber.Create(huber_delta)
        return gtsam_module.noiseModel.Robust.Create(huber, gaussian)
    except AttributeError as exc:
        raise RuntimeError("This GTSAM build lacks the requested diagonal/Huber robust-noise API") from exc


def _confidence_threshold(submap, percentile, scope, frame_id):
    if percentile is None:
        return float(submap.get_conf_threshold())
    values = np.asarray(submap.get_conf_masks_frame(frame_id) if scope == "frame" else submap.conf_masks)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("Submap has no finite confidence values for a scale percentile")
    return float(np.percentile(finite, percentile) + 1e-6)


def make_solver_class(base_solver_class, policy=None, *, scale_estimator=None, gtsam_module=None):
    """Return the unchanged legacy class, or an opt-in Solver subclass.

    The subclass exposes ``backend_policy`` and a JSON-serializable
    ``backend_diagnostics`` list. Loop noise/Huber applies to every edge
    involving an auxiliary loop submap, including its internal two-view edge;
    ordinary submap internal edges and the first-node prior are unchanged.
    It can also be instantiated with ``__new__`` for CPU cache replay after
    explicitly assigning ``map`` and ``graph``; diagnostics/noise are lazy.
    """
    policy = policy or BackendPolicy()
    if policy.name == "legacy":
        return base_solver_class

    class PolicySolver(base_solver_class):
        backend_policy = policy

        def _policy_state(self):
            if "backend_diagnostics" not in self.__dict__:
                self.backend_diagnostics = []
            if "_backend_policy_noises" not in self.__dict__:
                self._backend_policy_noises = {
                    "cross": make_factor_noise(self.graph.intra_submap_noise, policy.cross_submap_noise_multiplier,
                                               policy.huber_delta, gtsam_module=gtsam_module),
                    "loop": make_factor_noise(self.graph.intra_submap_noise, policy.loop_noise_multiplier,
                                              policy.huber_delta, gtsam_module=gtsam_module),
                    "loop_inner": make_factor_noise(self.graph.inner_submap_noise, policy.loop_noise_multiplier,
                                                    policy.huber_delta, gtsam_module=gtsam_module),
                }
            return self._backend_policy_noises

        def add_edge(self, submap_id_curr, frame_id_curr, submap_id_prev=None, frame_id_prev=None, is_loop_closure=False):
            assert not (is_loop_closure and submap_id_prev is None), "Loop closure must have a previous submap"
            noises = self._policy_state()
            if policy.name == "noise_only":
                current = self.map.get_submap(submap_id_curr)
                prior = self.map.get_submap(submap_id_prev) if submap_id_prev is not None else None
                loop_edge = bool(is_loop_closure or current.get_lc_status()
                                 or (prior is not None and prior.get_lc_status()))
                previous_cross_noise = self.graph.intra_submap_noise
                previous_inner_noise = self.graph.inner_submap_noise
                try:
                    if prior is not None:
                        self.graph.intra_submap_noise = noises["loop" if loop_edge else "cross"]
                    if current.get_lc_status():
                        self.graph.inner_submap_noise = noises["loop_inner"]
                    # No reimplementation or interception of scale selection:
                    # every mask, fallback, point and estimator call is upstream.
                    result = super().add_edge(submap_id_curr, frame_id_curr, submap_id_prev,
                                               frame_id_prev, is_loop_closure)
                finally:
                    self.graph.intra_submap_noise = previous_cross_noise
                    self.graph.inner_submap_noise = previous_inner_noise
                self.backend_diagnostics.append({
                    "diagnostic_type": "noise_only", "status": "ok",
                    "prior_submap_id": submap_id_prev, "prior_frame_id": frame_id_prev,
                    "current_submap_id": submap_id_curr, "current_frame_id": frame_id_curr,
                    "edge_kind": "anchor" if prior is None else "loop" if loop_edge else "cross_submap",
                    "noise_sigma_multiplier": (1.0 if prior is None else policy.loop_noise_multiplier
                                               if loop_edge else policy.cross_submap_noise_multiplier),
                    "huber_delta": policy.huber_delta if prior is not None else None,
                    "internal_noise_sigma_multiplier": policy.loop_noise_multiplier if current.get_lc_status() else 1.0,
                    "internal_huber_delta": policy.huber_delta if current.get_lc_status() else None,
                    "upstream_scale_selection_unchanged": True,
                })
                return result
            if submap_id_prev is None:
                return super().add_edge(submap_id_curr, frame_id_curr, submap_id_prev, frame_id_prev, is_loop_closure)

            current = self.map.get_submap(submap_id_curr)
            prior = self.map.get_submap(submap_id_prev)
            loop_edge = bool(is_loop_closure or current.get_lc_status() or prior.get_lc_status())
            # Preserve the pinned backend's K[-1]/K[0] geometry and scale direction.
            transform = np.linalg.inv(prior.proj_mats[-1]) @ current.proj_mats[0]
            current_points = current.get_frame_pointcloud(frame_id_curr).reshape(-1, 3)
            current_points = (transform[:3, :3] @ current_points.T).T
            prior_points = prior.get_frame_pointcloud(frame_id_prev).reshape(-1, 3)
            context = {
                "prior_submap_id": int(submap_id_prev), "prior_frame_id": int(frame_id_prev),
                "current_submap_id": int(submap_id_curr), "current_frame_id": int(frame_id_curr),
                "edge_kind": "loop" if loop_edge else "cross_submap",
                "scale_confidence_percentile": policy.scale_confidence_percentile,
                "scale_confidence_scope": policy.scale_confidence_scope,
            }
            try:
                scale, diagnostics = estimate_overlap_scale(
                    current_points, prior_points, current.get_conf_masks_frame(frame_id_curr),
                    prior.get_conf_masks_frame(frame_id_prev),
                    _confidence_threshold(current, policy.scale_confidence_percentile,
                                          policy.scale_confidence_scope, frame_id_curr),
                    _confidence_threshold(prior, policy.scale_confidence_percentile,
                                          policy.scale_confidence_scope, frame_id_prev),
                    min_points=policy.min_scale_points, estimator=scale_estimator,
                )
            except ScaleEstimationError as exc:
                self.backend_diagnostics.append({**context, **exc.diagnostics})
                raise
            self.backend_diagnostics.append({**context, **diagnostics})
            h_scale = np.diag((scale, scale, scale, 1.0))
            relative_h = transform @ h_scale
            previous_node = submap_id_prev + frame_id_prev
            if not is_loop_closure:
                self.graph.add_homography(submap_id_curr + frame_id_curr,
                                          self.graph.get_homography(previous_node) @ relative_h)
            self.graph.add_between_factor(previous_node, submap_id_curr + frame_id_curr, relative_h,
                                          noises["loop" if loop_edge else "cross"])
            if is_loop_closure:
                return

            world_to_cam = current.get_all_poses()
            for index, pose in enumerate(world_to_cam):
                if index == 0:
                    continue
                h_inner = world_to_cam[index - 1] @ np.linalg.inv(pose)
                previous = submap_id_curr + index - 1
                self.graph.add_homography(submap_id_curr + index, self.graph.get_homography(previous) @ h_inner)
                inner_noise = noises["loop_inner"] if current.get_lc_status() else self.graph.inner_submap_noise
                self.graph.add_between_factor(previous, submap_id_curr + index, h_inner, inner_noise)

    PolicySolver.__name__ = f"{base_solver_class.__name__}WithOmegaConfidencePolicy"
    return PolicySolver
