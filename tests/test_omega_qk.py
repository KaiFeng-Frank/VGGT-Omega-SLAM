"""(b) q/k/v rebuilt from the hooked qkv reproduce the real attention; OmegaForSLAM hook plumbing."""
import json
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch", reason="attention integration tests need PyTorch (CPU is sufficient)")
import torch.nn as nn
import torch.nn.functional as F

SelfAttentionBlock = pytest.importorskip("vggt_omega.models.layers", reason="needs pinned Omega sources; run tools/setup_sources.py").SelfAttentionBlock
from vos import omega_backend as ob

DIM, HEADS = 64, 4


def make_block(seed=0, qk_norm=True):
    torch.manual_seed(seed)
    block = SelfAttentionBlock(dim=DIM, num_heads=HEADS, ffn_ratio=4.0, qkv_bias=True, proj_bias=True,
                               ffn_bias=True, init_values=1e-5, use_qk_norm=qk_norm, mask_k_bias=True).eval().float()
    with torch.no_grad():
        for p in block.parameters():  # non-trivial values everywhere (inits are zero/ones for several tensors)
            p.add_(0.1 * torch.randn_like(p))
    return block


def test_rebuilt_qkv_reproduces_block_attention_to_1e5():
    block = make_block()
    attn = block.attn
    captured = {}
    h1 = attn.qkv.register_forward_hook(lambda m, a, o: captured.__setitem__("qkv", o.detach()))
    h2 = attn.register_forward_hook(lambda m, a, o: captured.__setitem__("attn_out", o.detach()))
    x = torch.randn(1, 2 * 37, DIM)
    block(x, None)  # through the real Block.forward -> _forward_list -> attn -> qkv path
    h1.remove(); h2.remove()

    q, k, v = ob.rebuild_qkv(attn, captured["qkv"])
    assert q.shape == (1, HEADS, 74, DIM // HEADS)
    o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, 74, DIM)
    rebuilt = attn.proj(o)
    assert torch.allclose(rebuilt, captured["attn_out"], atol=1e-5, rtol=1e-5), \
        float((rebuilt - captured["attn_out"]).abs().max())
    # and against the module's own compute_attention
    assert torch.allclose(o, attn.compute_attention(captured["qkv"]), atol=1e-6)


def test_scale_convention_softmax_of_scaled_q_equals_sdpa():
    block = make_block(1)
    attn = block.attn
    qkv = attn.qkv(torch.randn(1, 50, DIM))
    q, k, v = ob.rebuild_qkv(attn, qkv)
    manual = torch.softmax((q * attn.scale) @ k.transpose(-2, -1), dim=-1) @ v
    assert torch.allclose(manual, F.scaled_dot_product_attention(q, k, v), atol=1e-5)
    assert attn.scale == pytest.approx((DIM // HEADS) ** -0.5)


def test_rebuild_applies_qk_norm_and_masked_k_bias():
    block = make_block(2)
    attn = block.attn
    assert attn.use_qk_norm and hasattr(attn.qkv, "bias_mask")
    x = torch.randn(1, 20, DIM)
    raw = attn.qkv(x)
    q, k, _ = ob.rebuild_qkv(attn, raw)
    # without the norm the result must differ -> the norm really is applied
    attn.use_qk_norm = False
    q0, k0, _ = ob.rebuild_qkv(attn, raw)
    attn.use_qk_norm = True
    assert not torch.allclose(q, q0) and not torch.allclose(k, k0)
    # the k third of the bias is masked to zero by LinearKMaskedBias
    manual = F.linear(x, attn.qkv.weight, attn.qkv.bias * attn.qkv.bias_mask)
    assert torch.allclose(raw, manual)


class FakeAggregator(nn.Module):
    def __init__(self, types_, tokens_per_frame):
        super().__init__()
        self.inter_frame_attention_types = types_
        self.patch_token_start = 17
        self.tokens_per_frame = tokens_per_frame
        self.inter_frame_blocks = nn.ModuleList([make_block(10 + i) for i in range(len(types_))])


class FakeOmega(nn.Module):
    """Runs real tiny SelfAttentionBlocks on global tokens so the hook sees the true call path."""

    def __init__(self, tokens_per_frame=17 + 30):
        super().__init__()
        types_ = ["global"] * 24  # the real VGGT-Omega layout
        for idx in (2, 6, 9, 14, 20):
            types_[idx] = "register"
        self.aggregator = FakeAggregator(types_, tokens_per_frame)

    def forward(self, images):
        b, s = images.shape[:2]
        torch.manual_seed(123)
        tokens = torch.randn(b, s * self.aggregator.tokens_per_frame, DIM)
        for block, kind in zip(self.aggregator.inter_frame_blocks, self.aggregator.inter_frame_attention_types):
            if kind == "global":
                tokens = block(tokens, None)
        return {"pose_enc": torch.zeros(b, s, 9), "depth": torch.ones(b, s, 4, 4, 1),
                "depth_conf": torch.ones(b, s, 4, 4), "images": images, "camera_and_register_tokens": tokens}


def test_omega_for_slam_forward_contract_and_hook_cleanup(tmp_path):
    model = ob.OmegaForSLAM(FakeOmega(), layer=3, tau=0.5, config_path=tmp_path / "missing.json")
    images = torch.rand(2, 3, 16, 16)
    out = model(images)  # [S,3,H,W] accepted
    assert set(out) == {"pose_enc", "depth", "depth_conf", "images", "image_match_ratio", "image_match_ratio_raw"}
    assert out["pose_enc"].shape == (1, 2, 9) and out["images"].shape == (1, 2, 3, 16, 16)
    assert out["image_match_ratio"] is None
    out = model(images, compute_similarity=True)
    raw, scaled = out["image_match_ratio_raw"], out["image_match_ratio"]
    assert isinstance(raw, float) and isinstance(scaled, float) and raw > 0
    assert scaled == pytest.approx(0.95 * raw / 0.5)
    assert (scaled < 0.95) == (raw < 0.5)
    attn = model.omega.aggregator.inter_frame_blocks[3].attn
    assert len(attn.qkv._forward_hooks) == 0  # the hook only lives for that forward
    # second call is reproducible
    assert model(images, compute_similarity=True)["image_match_ratio_raw"] == raw


def test_similarity_matches_independent_reference():
    """raw ratio == get_similarity(k, q*scale) computed by hand from the module's own tensors."""
    model = ob.OmegaForSLAM(FakeOmega(), layer=0, tau=1.0)
    attn = model.omega.aggregator.inter_frame_blocks[0].attn
    store = []
    h = attn.qkv.register_forward_hook(lambda m, a, o: store.append(o.detach()))
    out = model(torch.rand(1, 2, 3, 8, 8), compute_similarity=True)
    h.remove()
    q, k, _ = ob.rebuild_qkv(attn, store[-1])
    reference = ob.get_similarity(k, q * attn.scale, token_offset=17)
    assert out["image_match_ratio_raw"] == pytest.approx(float(reference), rel=1e-6)


def test_register_block_is_rejected():
    with pytest.raises(AssertionError, match="register"):
        ob.OmegaForSLAM(FakeOmega(), layer=2, tau=0.9)
    with pytest.raises(ValueError):
        ob.OmegaForSLAM(FakeOmega(), layer=99, tau=0.9)


def test_one_frame_similarity_is_refused():
    model = ob.OmegaForSLAM(FakeOmega(), layer=0, tau=0.9)
    with pytest.raises(ValueError, match="two frames"):
        model(torch.rand(1, 3, 3, 8, 8), compute_similarity=True)


def test_config_file_and_precedence(tmp_path):
    cfg = tmp_path / "sim.json"
    cfg.write_text(json.dumps({"layer": 5, "tau": 0.7}))
    m = ob.OmegaForSLAM(FakeOmega(), config_path=cfg)
    assert (m.layer, m.tau) == (5, 0.7) and str(cfg) in m.similarity_source
    m = ob.OmegaForSLAM(FakeOmega(), layer=0, tau=0.2, config_path=cfg)  # arguments win
    assert (m.layer, m.tau) == (0, 0.2)
    m = ob.OmegaForSLAM(FakeOmega(), tau=0.33, config_path=cfg)  # per-field: tau argument, layer from the file
    assert (m.layer, m.tau) == (5, 0.33) and "argument" in m.similarity_source and str(cfg) in m.similarity_source
    m = ob.OmegaForSLAM(FakeOmega(), config_path=tmp_path / "none.json")
    assert (m.layer, m.tau) == (ob.DEFAULT_LAYER, ob.DEFAULT_TAU) and "placeholder" in m.similarity_source
    cfg.write_text(json.dumps({"layer": 3}))
    with pytest.raises(ValueError):
        ob.OmegaForSLAM(FakeOmega(), config_path=cfg)
