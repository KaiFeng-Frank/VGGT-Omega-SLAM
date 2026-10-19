"""VGGT (original 1B weights) through the VGGT_SPARK fork, loaded exactly like upstream main.py.

main.py:  model = VGGT(); model.load_state_dict(<checkpoint>); model.eval();
          model = model.to(torch.bfloat16); model = model.to(device)
Only the checkpoint source differs (local file instead of torch.hub.load_state_dict_from_url).
"""
from __future__ import annotations

from pathlib import Path

import torch

from . import env


def check_vggt_checkpoint(checkpoint=env.VGGT_CKPT, verify_sha256=False):
    checkpoint = Path(checkpoint)
    info = {"path": str(checkpoint), "bytes": checkpoint.stat().st_size}
    if verify_sha256:  # ~15 s for 5 GB; off by default
        info["sha256"] = env.sha256(checkpoint)
        if info["sha256"] != env.VGGT_SHA256:
            raise ValueError(f"unexpected VGGT SHA256 {info['sha256']}")
    return info


def load_vggt(checkpoint=env.VGGT_CKPT, device="cuda", verify_sha256=False):
    from vggt.models.vggt import VGGT

    check_vggt_checkpoint(checkpoint, verify_sha256)
    model = VGGT()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state)  # strict, as in main.py
    del state
    model.eval()
    model = model.to(torch.bfloat16)  # use half precision
    model = model.to(device)
    return model


def similarity_layer(model) -> int:
    """Global-attention block whose q/k VGGT-SLAM 2.0 uses for the match ratio.

    Note: `VGGT.__init__` defaults to target_layer=21 and passes it to the Aggregator (whose own
    default is 20), so main.py's `VGGT()` uses global block 21.
    """
    return int(model.aggregator.target_layer)


@torch.no_grad()
def raw_match_ratio(model, frames) -> float:
    """VGGT-SLAM 2.0 image_match_ratio for exactly two frames [2,3,H,W] (as predictions_lc does)."""
    if frames.shape[0] != 2:
        raise ValueError("loop-closure similarity needs exactly two frames")
    return float(model(frames, compute_similarity=True)["image_match_ratio"])
