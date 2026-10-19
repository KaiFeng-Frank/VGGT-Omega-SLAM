#!/usr/bin/env python3
"""Calibration step 3 (GPU): raw VGGT-SLAM 2.0 image_match_ratio of every calibration pair, VGGT front-end.

VGGT is loaded exactly like upstream main.py (VGGT() -> bf16 -> cuda) and called like the solver's
loop-closure verification: model(lc_frames, compute_similarity=True) on 2 frames, query (late) frame first.
The similarity layer is VGGT_SPARK's `aggregator.target_layer`; VGGT() builds it with 21 (the Aggregator's
own default is 20). Both are evaluated: 21 is the upstream operating point, 20 is what the rule text named.
The reverse frame order is recorded as a sensitivity column only.

  systemd-run --user --scope --quiet -p MemoryMax=12G env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 <python> tools/calib_vggt.py
"""
import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vos import env  # noqa: E402

env.prepare()

from vos import calibration as cal  # noqa: E402

OUT = env.PROJECT / "runs" / "calibration"
IMAGES = env.OFFICE_ROOT
# Earlier SLAM smoke run (runs/smoke_vggt_office150): loop candidate query frame_0130 / retrieved frame_0040
REGRESSION = {"query": "frame_0130.jpg", "retrieved": "frame_0040.jpg", "layer": 21,
              "reference_raw": 0.6103094816207886, "reference_run": "runs/smoke_vggt_office150/metadata.json"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=Path, default=OUT / "pairs_defs.json")
    p.add_argument("--output", type=Path, default=OUT / "ratios_vggt.json")
    p.add_argument("--gpu-need-mb", type=int, default=7500)
    args = p.parse_args()
    if args.output.exists():
        sys.exit(f"{args.output} exists")
    defs = json.loads(args.pairs.read_text())
    kf = json.loads((OUT / "keyframes.json").read_text())
    path_of = {k["image"]: k["path"] for k in kf["keyframes"]}

    free = env.claim_gpu(args.gpu_need_mb, "vggt-omega-slam calib_vggt")  # lock before anything touches CUDA
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(0)
    from vggt.utils.load_fn import load_and_preprocess_images as load_fn
    from vos import vggt_backend as vb

    started = time.perf_counter()
    model = vb.load_vggt(env.VGGT_CKPT, "cuda")
    load_s = time.perf_counter() - started
    default_layer = vb.similarity_layer(model)
    cache = {}

    def frame(path):
        if path not in cache:
            cache[path] = load_fn([path])[0]
        return cache[path]

    devnull = open(os.devnull, "w")  # VGGT_SPARK prints debug lines inside forward

    @torch.no_grad()
    def ratio(first_path, second_path, layer):
        model.aggregator.target_layer = layer
        frames = torch.stack([frame(first_path), frame(second_path)]).to("cuda")
        with contextlib.redirect_stdout(devnull):
            return vb.raw_match_ratio(model, frames)

    result = {"schema": "vggt_omega_slam.calibration.vggt.v1", "dataset": cal.DATASET,
              "default_target_layer_of_VGGT()": default_layer, "layers": list(cal.VGGT_LAYERS),
              "input_shape": list(frame(next(iter(path_of.values()))).shape), "free_mb_at_claim": free,
              "model_load_seconds": load_s, "ratios": {}}
    assert default_layer == 21, default_layer

    # regression against the earlier SLAM run + determinism
    reg = ratio(str(IMAGES / REGRESSION["query"]), str(IMAGES / REGRESSION["retrieved"]), REGRESSION["layer"])
    reg2 = ratio(str(IMAGES / REGRESSION["query"]), str(IMAGES / REGRESSION["retrieved"]), REGRESSION["layer"])
    result["regression"] = dict(REGRESSION, raw_now=reg, raw_repeat=reg2, abs_diff_vs_reference=abs(reg - REGRESSION["reference_raw"]),
                                abs_diff_repeat=abs(reg - reg2))
    print("[regression]", json.dumps(result["regression"]), flush=True)

    t0 = time.perf_counter()
    try:
        for n_done, pair in enumerate(defs["pairs"]):
            late, early = path_of[pair["image_late"]], path_of[pair["image_early"]]
            entry = {}
            for layer in cal.VGGT_LAYERS:
                entry[str(layer)] = {"solver": ratio(late, early, layer), "reverse": ratio(early, late, layer)}
            result["ratios"][pair["pair_id"]] = entry
            if n_done % 20 == 0:
                print(f"[progress] {n_done + 1}/{len(defs['pairs'])} {pair['pair_id']} "
                      f"L21 solver={entry['21']['solver']:.4f}", flush=True)
        result["status"] = "complete"
    except BaseException as exc:  # noqa: BLE001 - keep what was computed, then fail loudly
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["pairs_seconds"] = time.perf_counter() - t0
        result["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        args.output.write_text(json.dumps(result, indent=1))
        print(f"[done] {len(result['ratios'])} pairs, {result['pairs_seconds']:.1f}s, status={result.get('status')}",
              flush=True)


if __name__ == "__main__":
    main()
