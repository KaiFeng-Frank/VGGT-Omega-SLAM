#!/usr/bin/env python3
"""Calibration step 4 (GPU): raw VGGT-SLAM 2.0 image_match_ratio of every calibration pair, VGGT-Omega front-end,
at every candidate global inter-frame block in ONE forward per frame order.

The qkv projection of each candidate block is captured by a forward hook (read-only; outputs are
bit-identical with and without hooks, see tools/check_omega_hook.py) and turned into the ratio by the same
function the SLAM adapter uses (`omega_backend.similarity_from_qkv`). The multi-layer result is cross-checked
against `OmegaForSLAM(layer=L)` (the SLAM path) and against two earlier recorded values.

  systemd-run --user --scope --quiet -p MemoryMax=12G env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 <python> tools/calib_omega.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vos import env  # noqa: E402

env.prepare()

from vos import calibration as cal  # noqa: E402

OUT = env.PROJECT / "runs" / "calibration"
IMAGES = env.OFFICE_ROOT
# Recorded earlier on the same checkpoint at layer 19: tools/check_omega_hook.py and the Omega smoke SLAM run.
REGRESSION = [
    {"name": "hook_check near (1,5)", "first": "frame_0001.jpg", "second": "frame_0005.jpg", "layer": 19,
     "reference_raw": 1.2734313011169434, "reference_run": "runs/hook_check.json"},
    {"name": "hook_check far (1,400)", "first": "frame_0001.jpg", "second": "frame_0400.jpg", "layer": 19,
     "reference_raw": 0.9289451837539673, "reference_run": "runs/hook_check.json"},
    {"name": "smoke SLAM candidate (130,40)", "first": "frame_0130.jpg", "second": "frame_0040.jpg", "layer": 19,
     "reference_raw": 0.8330608010292053, "reference_run": "runs/smoke_omega_office150/metadata.json"},
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path, default=OUT / "pairs_defs.json")
    p.add_argument("--output", type=Path, default=OUT / "ratios_omega.json")
    p.add_argument("--gpu-need-mb", type=int, default=9000)
    args = p.parse_args()
    if args.output.exists():
        sys.exit(f"{args.output} exists")
    defs = json.loads(args.pairs.read_text())
    kf = json.loads((OUT / "keyframes.json").read_text())
    path_of = {k["image"]: k["path"] for k in kf["keyframes"]}

    free = env.claim_gpu(args.gpu_need_mb, "vggt-omega-slam calib_omega")  # lock before anything touches CUDA
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(0)
    from vos import omega_backend as ob

    started = time.perf_counter()
    core = ob.load_omega(env.OMEGA_CKPT, "cuda")
    load_s = time.perf_counter() - started
    aggregator = core.aggregator
    types_ = aggregator.inter_frame_attention_types
    layers = list(cal.OMEGA_LAYERS)
    for layer in layers:
        assert types_[layer] == "global", (layer, types_[layer])
    offset = aggregator.patch_token_start
    assert offset == ob.TOKEN_OFFSET

    captured = {layer: [] for layer in layers}

    def make_hook(layer, attn):
        def hook(module, args_, output):
            captured[layer].append(ob.similarity_from_qkv(attn, output.detach(), token_offset=offset))
        return hook

    handles = []
    for layer in layers:
        attn = aggregator.inter_frame_blocks[layer].attn
        handles.append(attn.qkv.register_forward_hook(make_hook(layer, attn)))

    cache = {}

    def frame(path):
        if path not in cache:
            cache[path] = ob.load_images_omega([path])[0]
        return cache[path]

    @torch.no_grad()
    def ratios(first_path, second_path):
        for values in captured.values():
            values.clear()
        images = torch.stack([frame(first_path), frame(second_path)]).unsqueeze(0).to("cuda")
        core(images)
        for layer, values in captured.items():
            if len(values) != 1:
                raise RuntimeError(f"layer {layer}: expected one qkv capture, got {len(values)}")
        return {layer: captured[layer][0] for layer in layers}

    result = {"schema": "vggt_omega_slam.calibration.omega.v1", "dataset": cal.DATASET, "layers": layers,
              "token_offset": offset, "input_shape": list(frame(next(iter(path_of.values()))).shape),
              "free_mb_at_claim": free, "model_load_seconds": load_s, "ratios": {}}

    # ---- checks -------------------------------------------------------------------------------
    regression = []
    for item in REGRESSION:
        got = ratios(str(IMAGES / item["first"]), str(IMAGES / item["second"]))[item["layer"]]
        regression.append(dict(item, raw_now=got, abs_diff_vs_reference=abs(got - item["reference_raw"])))
    first_pair = defs["pairs"][0]
    a, b = path_of[first_pair["image_late"]], path_of[first_pair["image_early"]]
    r1, r2 = ratios(a, b), ratios(a, b)
    result["determinism_repeat_max_abs_diff"] = max(abs(r1[k] - r2[k]) for k in layers)
    # the SLAM adapter path (OmegaForSLAM.forward hooks one layer itself); our multi-layer hooks stay
    # registered but only write into `captured`, which the adapter ignores.
    adapter = []
    images = torch.stack([frame(a), frame(b)]).unsqueeze(0).to("cuda")
    for layer in (19, 23):
        slam_model = ob.OmegaForSLAM(core, layer=layer, tau=0.95)
        with torch.no_grad():
            raw = slam_model(images, compute_similarity=True)["image_match_ratio_raw"]
        adapter.append({"layer": layer, "adapter_raw": raw, "multi_hook_raw": r1[layer], "abs_diff": abs(raw - r1[layer])})
    result["regression"] = regression
    result["adapter_crosscheck"] = adapter
    print("[regression]", json.dumps(regression), flush=True)
    print("[adapter]", json.dumps(adapter), flush=True)

    # ---- all pairs ---------------------------------------------------------------------------
    t0 = time.perf_counter()
    try:
        for n_done, pair in enumerate(defs["pairs"]):
            late, early = path_of[pair["image_late"]], path_of[pair["image_early"]]
            solver, reverse = ratios(late, early), ratios(early, late)
            result["ratios"][pair["pair_id"]] = {str(L): {"solver": solver[L], "reverse": reverse[L]} for L in layers}
            if n_done % 20 == 0:
                print(f"[progress] {n_done + 1}/{len(defs['pairs'])} {pair['pair_id']} "
                      f"L19 solver={solver[19]:.4f}", flush=True)
        result["status"] = "complete"
    except BaseException as exc:  # noqa: BLE001
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for handle in handles:
            handle.remove()
        result["pairs_seconds"] = time.perf_counter() - t0
        result["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        args.output.write_text(json.dumps(result, indent=1))
        print(f"[done] {len(result['ratios'])} pairs, {result['pairs_seconds']:.1f}s, status={result.get('status')}",
              flush=True)


if __name__ == "__main__":
    main()
