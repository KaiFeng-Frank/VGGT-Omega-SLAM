"""VGGT-Omega as a drop-in replacement for the VGGT_SPARK model used by VGGT-SLAM 2.0.

`OmegaForSLAM.forward(images, compute_similarity=False)` returns the same keys with the same
shapes/semantics as VGGT_SPARK's `VGGT.forward`, so the unmodified `Solver.run_predictions` works:

    pose_enc   [B,S,9]       T(3) quaternion-xyzw(4) fov_h fov_w   (camera-from-world, OpenCV)
    depth      [B,S,H,W,1]
    depth_conf [B,S,H,W]
    images     [B,S,3,H,W]
    image_match_ratio      None, or (compute_similarity=True) 0.95 * raw / tau   (python float)
    image_match_ratio_raw  None, or the verbatim VGGT-SLAM 2.0 attention-similarity ratio

Upstream rejects a loop closure when `image_match_ratio < 0.95` (hard-coded in solver.py). Writing
image_match_ratio = 0.95 * raw / tau makes that test exactly `raw < tau`, with tau chosen for Omega.

The similarity is VGGT-SLAM 2.0's: at one global attention block the keys of the first frame's
patch tokens are attended by every token of both frames; see `get_similarity` (verbatim port of
vggt/models/aggregator.py of VGGT_SPARK 6e6e161, MIT-SPARK fork of facebookresearch/vggt).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import env

TOKEN_OFFSET = 17  # 1 camera token + 16 register tokens precede the patch tokens in VGGT-Omega
DEFAULT_LAYER = 19  # placeholder until calibration writes config/omega_similarity.json
DEFAULT_TAU = 0.95  # placeholder, same
UPSTREAM_TEST = 0.95  # solver.py: `if image_match_ratio < 0.95`


# ----------------------------------------------------------------------------------------------
# Verbatim port of VGGT_SPARK vggt/models/aggregator.py (commit 6e6e161): mean_top_quarter and
# get_similarity. Only the debugging print() was dropped; the arithmetic is unchanged.
# ----------------------------------------------------------------------------------------------
def mean_top_quarter(arr):
    # flatten to 1D
    flat = arr.ravel()
    # find 75th percentile threshold
    thresh = np.percentile(flat, 75)
    # select values >= threshold
    top_vals = flat[flat >= thresh]
    # return their mean
    return top_vals.mean()


def get_similarity(k, q, token_offset=5, image_height=21, image_width=37):
    num_imgs = 2
    tokens_per_img = q.shape[2] // num_imgs
    k = k[:, :, token_offset:tokens_per_img, :]

    attn = q @ k.transpose(-2, -1)
    attn = attn.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = attn.mean(dim=1)

    all_token_to_first_frame = attn[..., :tokens_per_img]
    all_token_to_second_frame = attn[..., tokens_per_img:]

    max_per_token_first_img = all_token_to_first_frame.max(dim=-1)[0]

    attn_second_frame_normalized = all_token_to_second_frame / (max_per_token_first_img.unsqueeze(-1) + 1e-8)
    ratio = attn_second_frame_normalized.max(dim=1)[0]

    ratio = ratio.float().detach().cpu().numpy()
    ratio = mean_top_quarter(ratio)

    return ratio


# ----------------------------------------------------------------------------------------------
# q/k reconstruction from the hooked qkv projection
# ----------------------------------------------------------------------------------------------
def rebuild_qkv(attn, qkv):
    """q, k, v exactly as `SelfAttention.compute_attention` forms them before rope/SDPA.

    attn : vggt_omega SelfAttention module (provides num_heads, qkv.in_features, q_norm/k_norm)
    qkv  : output of `attn.qkv`, [B, N, 3*C]
    returns q, k, v of shape [B, heads, N, head_dim]. Global inter-frame blocks are called with
    rope=None, so no rotary embedding is applied here (callers must not use this for rope blocks).
    """
    batch, tokens, _ = qkv.shape
    channels = attn.qkv.in_features
    qkv = qkv.reshape(batch, tokens, 3, attn.num_heads, channels // attn.num_heads)
    q, k, v = torch.unbind(qkv, 2)
    q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
    if attn.use_qk_norm:
        q = attn.q_norm(q)
        k = attn.k_norm(k)
    return q, k, v


def similarity_from_qkv(attn, qkv, token_offset=TOKEN_OFFSET):
    """Raw VGGT-SLAM 2.0 match ratio from a captured `attn.qkv` output (two frames, batch 1).

    All math in float32 from the captured projection; q and k are first rounded to the dtype the
    captured projection had (bf16 under the aggregator's autocast), which is what the real SDPA
    consumed. q is multiplied by attn.scale as VGGT_SPARK's Attention does for its visualisation path.
    """
    dtype = qkv.dtype
    with torch.autocast(device_type=qkv.device.type, enabled=False):
        q, k, _ = rebuild_qkv(attn, qkv.float())
        q = (q.to(dtype) * attn.scale).float()
        k = k.to(dtype).float()
        return float(get_similarity(k, q, token_offset=token_offset))


# ----------------------------------------------------------------------------------------------
def load_similarity_config(path=None):
    """Calibration result written by the calibration stage, or None. Keys: layer, tau."""
    path = Path(path) if path else env.SIM_CONFIG
    if not path.is_file():
        return None
    config = json.loads(path.read_text())
    if "layer" not in config or "tau" not in config:
        raise ValueError(f"{path} must contain 'layer' and 'tau'")
    config["_path"] = str(path)
    return config


def check_omega_checkpoint(checkpoint=env.OMEGA_CKPT, verify_sha256=False):
    checkpoint = Path(checkpoint)
    size = checkpoint.stat().st_size
    info = {"path": str(checkpoint), "bytes": size}
    if checkpoint.resolve() == env.OMEGA_CKPT.resolve() and size != env.OMEGA_CKPT_BYTES:
        raise ValueError(f"{checkpoint} is {size} bytes, expected {env.OMEGA_CKPT_BYTES}")
    if verify_sha256:  # ~10 s for 4.6 GB; off by default
        info["sha256"] = env.sha256(checkpoint)
        if info["sha256"] != env.OMEGA_SHA256:
            raise ValueError(f"unexpected VGGT-Omega SHA256 {info['sha256']}")
    return info


def load_omega(checkpoint=env.OMEGA_CKPT, device="cuda", verify_sha256=False):
    """Official recipe: fp32 weights (the heads run with autocast disabled), bf16 autocast inside."""
    from vggt_omega.models import VGGTOmega

    check_omega_checkpoint(checkpoint, verify_sha256)
    model = VGGTOmega().eval().float()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device)


class OmegaForSLAM(nn.Module):
    """Adapter: VGGTOmega with VGGT_SPARK's calling convention and loop-verification ratio."""

    def __init__(self, omega, layer=None, tau=None, config_path=None):
        super().__init__()
        self.omega = omega
        config = load_similarity_config(config_path)

        def pick(name, argument, default):
            """Per field: explicit argument > calibration file > placeholder default."""
            if argument is not None:
                return argument, "argument"
            if config is not None:
                return config[name], f"config file {config['_path']}"
            return default, "placeholder default (no calibration file)"

        layer, layer_source = pick("layer", layer, DEFAULT_LAYER)
        tau, tau_source = pick("tau", tau, DEFAULT_TAU)
        source = layer_source if layer_source == tau_source else f"layer: {layer_source}; tau: {tau_source}"
        self.layer = int(layer)
        self.tau = float(tau)
        if not self.tau > 0:
            raise ValueError("tau must be positive")
        types_ = omega.aggregator.inter_frame_attention_types
        if not 0 <= self.layer < len(types_):
            raise ValueError(f"similarity layer {self.layer} outside 0..{len(types_) - 1}")
        if types_[self.layer] != "global":
            raise AssertionError(f"inter-frame block {self.layer} is '{types_[self.layer]}', not 'global'")
        self.token_offset = omega.aggregator.patch_token_start
        assert self.token_offset == TOKEN_OFFSET, self.token_offset
        self.similarity_source = source
        self.similarity_config = config

    def describe(self):
        return {"layer": self.layer, "tau": self.tau, "token_offset": self.token_offset,
                "source": self.similarity_source,
                "upstream_test": f"image_match_ratio < {UPSTREAM_TEST}  <=>  raw < tau",
                "scaled_ratio": "0.95 * raw / tau",
                "block_type": self.omega.aggregator.inter_frame_attention_types[self.layer]}

    def scaled(self, raw):
        return UPSTREAM_TEST * float(raw) / self.tau

    def forward(self, images, compute_similarity=False):
        if images.dim() == 4:
            images = images.unsqueeze(0)
        captured = []
        handle = None
        if compute_similarity:
            attn = self.omega.aggregator.inter_frame_blocks[self.layer].attn
            handle = attn.qkv.register_forward_hook(lambda module, args, output: captured.append(output.detach()))
        try:
            out = self.omega(images)
        finally:
            if handle is not None:
                handle.remove()
        result = {"pose_enc": out["pose_enc"], "depth": out["depth"], "depth_conf": out["depth_conf"],
                  "images": out["images"], "image_match_ratio": None, "image_match_ratio_raw": None}
        if compute_similarity:
            if images.shape[1] != 2:
                raise ValueError("loop-closure similarity needs exactly two frames")
            if len(captured) != 1:
                raise RuntimeError(f"expected one qkv capture, got {len(captured)}")
            raw = similarity_from_qkv(self.omega.aggregator.inter_frame_blocks[self.layer].attn, captured[0],
                                      token_offset=self.token_offset)
            result["image_match_ratio_raw"] = raw
            result["image_match_ratio"] = self.scaled(raw)
        return result


def load_images_omega(paths):
    """Official VGGT-Omega loader: balanced token budget at 512 (384x688 for 16:9)."""
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    return load_and_preprocess_images(list(paths), mode="balanced", image_resolution=512)


def decode_pose(pose_enc, image_hw):
    """Official VGGT-Omega decoder: returns (extrinsic [B,S,3,4] camera-from-world, intrinsic [B,S,3,3])."""
    from vggt_omega.utils.pose_enc import encoding_to_camera

    return encoding_to_camera(pose_enc, image_hw)
