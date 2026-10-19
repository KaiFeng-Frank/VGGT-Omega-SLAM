"""(c) VGGT-Omega's encoding_to_camera vs VGGT_SPARK's pose_encoding_to_extri_intri.

The task asks to REPORT whether they agree. The test asserts nothing about agreement itself (the
runner uses Omega's own decoder regardless) but writes the measured differences to
tests/out/pose_decoder_agreement.json and prints a verdict line (visible with `pytest -s`).
"""
import json
from pathlib import Path

import pytest
torch = pytest.importorskip("torch", reason="decoder integration tests need PyTorch (CPU is sufficient)")

pytest.importorskip("vggt.utils.pose_enc", reason="needs pinned VGGT_SPARK sources; run tools/setup_sources.py")
pytest.importorskip("vggt_omega.utils.pose_enc", reason="needs pinned Omega sources; run tools/setup_sources.py")
from vggt.utils.pose_enc import extri_intri_to_pose_encoding as vggt_encode
from vggt.utils.pose_enc import pose_encoding_to_extri_intri as vggt_decode
from vggt_omega.utils.pose_enc import encoding_to_camera as omega_decode
from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding as omega_encode

OUT = Path(__file__).parent / "out"


def random_encodings(seed, b=2, s=5):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(b, s, 3, generator=g)
    quat = torch.randn(b, s, 4, generator=g)  # not normalised on purpose: both decoders divide by |q|^2
    fov = 0.3 + 1.5 * torch.rand(b, s, 2, generator=g)
    return torch.cat([t, quat, fov], dim=-1)


def test_decoders_compared_and_reported():
    hw = (384, 688)
    worst_e = worst_k = 0.0
    quaternion_cases = []
    for seed in range(20):
        enc = random_encodings(seed)
        e_v, k_v = vggt_decode(enc, hw)
        e_o, k_o = omega_decode(enc, hw)
        assert e_v.shape == e_o.shape == (2, 5, 3, 4) and k_v.shape == k_o.shape == (2, 5, 3, 3)
        worst_e = max(worst_e, float((e_v - e_o).abs().max()))
        worst_k = max(worst_k, float((k_v - k_o).abs().max()))
    # round trip with each package's own encoder, decoded by the OTHER package's decoder
    g = torch.Generator().manual_seed(7)
    r = torch.linalg.qr(torch.randn(2, 5, 3, 3, generator=g))[0]
    r = r * torch.det(r)[..., None, None].sign()  # proper rotations
    extr = torch.cat([r, torch.randn(2, 5, 3, 1, generator=g)], dim=-1)
    intr = torch.zeros(2, 5, 3, 3)
    intr[..., 0, 0], intr[..., 1, 1], intr[..., 0, 2], intr[..., 1, 2], intr[..., 2, 2] = 600.0, 610.0, 344.0, 192.0, 1.0
    cross_vo = omega_decode(vggt_encode(extr, intr, hw), hw)
    cross_ov = vggt_decode(omega_encode(extr, intr, hw), hw)
    roundtrip_err = max(float((cross_vo[0] - extr).abs().max()), float((cross_ov[0] - extr).abs().max()))
    intr_err = max(float((cross_vo[1] - intr).abs().max()), float((cross_ov[1] - intr).abs().max()))
    agree = worst_e < 1e-6 and worst_k < 1e-3 and roundtrip_err < 1e-4
    report = {"max_abs_diff_extrinsic": worst_e, "max_abs_diff_intrinsic_px": worst_k,
              "cross_roundtrip_extrinsic_err": roundtrip_err, "cross_roundtrip_intrinsic_err_px": intr_err,
              "quaternion_order": "both xyzw (scalar-last): vggt/utils/rotation.py and vggt_omega/utils/rotation.py",
              "intrinsics": "both: principal point at (W/2, H/2), fx from fov_w, fy from fov_h",
              "decoders_agree": agree,
              "note": "the runner uses vggt_omega.utils.pose_enc.encoding_to_camera for backend omega regardless"}
    OUT.mkdir(exist_ok=True)
    (OUT / "pose_decoder_agreement.json").write_text(json.dumps(report, indent=2))
    print("\n[pose decoder agreement]", json.dumps(report))
    assert all(map(lambda v: v == v, [worst_e, worst_k, roundtrip_err]))  # finite, not NaN
