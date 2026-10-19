#!/usr/bin/env python3
"""Calibration step 2 (GPU, short): SALAD descriptors of all keyframes -> the pair sets P_adj / P_loop / N.

Uses the upstream ImageRetrieval (SALAD) exactly as the solver does: descriptors from
`get_batch_descriptors` on the loader-preprocessed frames, distance = torch.linalg.norm(a - b) (the
solver's retrieve_best_score_frame score; smaller = more similar). Pair selection uses the VGGT-loader
frames (the upstream pipeline's own retrieval input). Descriptors of the Omega-loader frames are also
computed, only to report how many P_loop partners would differ.

  systemd-run --user --scope --quiet -p MemoryMax=12G env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 <python> tools/calib_pairs.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vos import env  # noqa: E402

env.prepare()

import numpy as np  # noqa: E402

from vos import calibration as cal  # noqa: E402

OUT = env.PROJECT / "runs" / "calibration"
SALAD_NEED_MB = 1500  # standalone descriptor-memory estimate; the descriptors of 208 frames need far less


def descriptors(retrieval, torch, load_fn, paths, batch=16):
    """Solver path: loader tensor [S,3,H,W] on the GPU -> ImageRetrieval.get_batch_descriptors."""
    chunks = []
    for start in range(0, len(paths), batch):
        frames = torch.cat([load_fn([p]) for p in paths[start:start + batch]]).to("cuda")
        chunks.append(retrieval.get_batch_descriptors(frames).float())
    return torch.cat(chunks)


def distance_matrix(torch, desc):
    rows = [torch.linalg.norm(desc - desc[a], dim=-1) for a in range(desc.shape[0])]
    return torch.stack(rows).cpu().numpy().astype(np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keyframes", type=Path, default=OUT / "keyframes.json")
    p.add_argument("--output", type=Path, default=OUT / "pairs_defs.json")
    args = p.parse_args()
    if args.output.exists():
        sys.exit(f"{args.output} exists")
    kf = json.loads(args.keyframes.read_text())
    names = [k["image"] for k in kf["keyframes"]]
    paths = [k["path"] for k in kf["keyframes"]]
    n = len(paths)

    free = env.claim_gpu(SALAD_NEED_MB, "vggt-omega-slam calib_pairs (SALAD)")  # lock before anything touches CUDA
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(0)
    env.configure_offline_hub(torch, OUT / "torch_hub_pairs", env.SALAD_CKPT, env.DINOV2_SRC.resolve())
    from vggt_slam.loop_closure import ImageRetrieval
    from vggt.utils.load_fn import load_and_preprocess_images as load_vggt
    from vos import omega_backend as ob

    retrieval = ImageRetrieval()
    with torch.no_grad():
        d_vggt = descriptors(retrieval, torch, load_vggt, paths)
        d_omega = descriptors(retrieval, torch, ob.load_images_omega, paths)
        dist_vggt = distance_matrix(torch, d_vggt)
        dist_omega = distance_matrix(torch, d_omega)
    np.save(OUT / "salad_dist_vggt_loader.npy", dist_vggt)
    np.save(OUT / "salad_dist_omega_loader.npy", dist_omega)

    adj = cal.adjacent_pairs(n)
    loop = cal.loop_pairs(dist_vggt)
    loop_omega_loader = cal.loop_pairs(dist_omega)
    neg, n_neg_candidates = cal.negative_pairs(n)

    pairs = []
    for tag, items in (("adj", adj), ("loop", loop), ("neg", neg)):
        for k, (i, j) in enumerate(items):
            pairs.append({"pair_id": f"{tag}_{k:03d}", "set": {"adj": "P_adj", "loop": "P_loop", "neg": "N"}[tag],
                          "kf_early": i, "kf_late": j, "index_gap": j - i,
                          "image_early": names[i], "image_late": names[j],
                          "image_index_early": kf["keyframes"][i]["image_index"],
                          "image_index_late": kf["keyframes"][j]["image_index"],
                          "salad_l2": float(dist_vggt[i, j]), "salad_l2_omega_loader": float(dist_omega[i, j])})
    keys = [(q["kf_early"], q["kf_late"]) for q in pairs]
    m = cal.loop_group_size(n)
    result = {
        "dataset": cal.DATASET, "n_keyframes": n, "n_images": kf["n_images"], "loop_group_size": m,
        "first_group": [0, m - 1], "last_group": [n - m, n - 1],
        "salad": {"threshold_lc_thres": 0.95, "distance": "torch.linalg.norm(desc_a - desc_b)",
                  "pair_selection_frames": "vggt loader (crop/518), as the upstream solver",
                  "descriptor_dim": int(d_vggt.shape[1]),
                  "descriptor_norm_min_max": [float(d_vggt.norm(dim=-1).min()), float(d_vggt.norm(dim=-1).max())]},
        "counts": {"P_adj": len(adj), "P_loop": len(loop), "N": len(neg), "N_candidates_in_gap_window": n_neg_candidates,
                   "duplicate_pairs_across_sets": len(keys) - len(set(keys))},
        "loop_partner_agreement_omega_loader": {
            "same_partner": int(sum(a == b for a, b in zip(loop, loop_omega_loader))), "of": len(loop),
            "omega_loader_partners": [list(x) for x in loop_omega_loader]},
        "p_loop_salad_below_0.95": int(sum(dist_vggt[i, j] < 0.95 for i, j in loop)),
        "free_mb_at_claim": free, "pairs": pairs,
    }
    args.output.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, indent=1))


if __name__ == "__main__":
    main()
