#!/usr/bin/env python3
"""GPU self-check of the Omega similarity hook on the real checkpoint (two frames, ~1 min).

Verifies, with the real weights under the real autocast, that
  1. capturing qkv does not change any output (compute_similarity True vs False are identical),
  2. q,k,v rebuilt from the hooked qkv reproduce the block's attention output (bf16 tolerance),
  3. prints the raw match ratio for a near pair and a far pair as a sanity read.

Launch (single CUDA process, lock inside):
  systemd-run --user --scope --quiet -p MemoryMax=12G env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 <python> tools/check_omega_hook.py --output runs/hook_check.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vos import env  # noqa: E402

env.prepare()

import torch  # noqa: E402

from vos import omega_backend as ob  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, default=env.OFFICE_ROOT)
    p.add_argument("--near", type=int, nargs=2, default=[1, 5], help="1-based frame numbers of a near pair")
    p.add_argument("--far", type=int, nargs=2, default=[1, 400], help="1-based frame numbers of a far pair")
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    free = env.claim_gpu(9000, "vggt-omega-slam hook-check")
    paths = {n: args.images / f"frame_{n:04d}.jpg" for n in {*args.near, *args.far}}
    model_core = ob.load_omega(env.OMEGA_CKPT, "cuda")
    model = ob.OmegaForSLAM(model_core, layer=args.layer, config_path=env.SIM_CONFIG)
    result = {"free_mb_at_claim": free, "similarity": model.describe()}

    def pair(a, b):
        return ob.load_images_omega([paths[a], paths[b]]).cuda()

    with torch.no_grad():
        near = pair(*args.near)
        plain = model(near)
        with_sim = model(near, compute_similarity=True)
        result["outputs_identical_with_and_without_hook"] = all(
            torch.equal(plain[k], with_sim[k]) for k in ("pose_enc", "depth", "depth_conf", "images"))

        # 2. rebuilt attention vs the block's real attention output
        attn = model_core.aggregator.inter_frame_blocks[model.layer].attn
        store = {}
        h1 = attn.qkv.register_forward_hook(lambda m, a, o: store.__setitem__("qkv", o.detach()))
        h2 = attn.register_forward_hook(lambda m, a, o: store.__setitem__("out", o.detach()))
        try:
            model_core(near)
        finally:
            h1.remove(); h2.remove()
        amp = torch.bfloat16
        with torch.autocast("cuda", dtype=amp):
            q, k, v = ob.rebuild_qkv(attn, store["qkv"])
            o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            b, n = store["qkv"].shape[:2]
            rebuilt = attn.proj(o.transpose(1, 2).reshape(b, n, -1))
        diff = (rebuilt.float() - store["out"].float()).abs()
        result["rebuilt_vs_block_attention"] = {"qkv_dtype": str(store["qkv"].dtype), "q_dtype": str(q.dtype),
            "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "ref_abs_mean": float(store["out"].float().abs().mean())}
        result["near_pair"] = {"frames": args.near, "raw": with_sim["image_match_ratio_raw"]}
        far = model(pair(*args.far), compute_similarity=True)
        result["far_pair"] = {"frames": args.far, "raw": far["image_match_ratio_raw"]}
    result["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    result["depth_shape"] = list(plain["depth"].shape)
    result["pose_enc_shape"] = list(plain["pose_enc"].shape)
    result["depth_conf_shape"] = list(plain["depth_conf"].shape)
    result["images_shape"] = list(plain["images"].shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
