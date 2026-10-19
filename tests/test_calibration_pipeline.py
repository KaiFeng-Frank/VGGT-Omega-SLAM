"""(g) The analysis stage end to end on synthetic ratios (CPU): rule application, outputs, figure, config."""
import json
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vos import calibration as cal

ROOT = Path(__file__).resolve().parents[1]
def test_analyze_applies_rule_and_writes_all_outputs(tmp_path):
    from PIL import Image

    # Rendering needs images, but calibration decisions use only the synthetic
    # ratios below. Generate fixtures so public CI needs no private sequence.
    frames = tmp_path / "images"
    frames.mkdir()
    entries = []
    for i in range(208):
        image = frames / f"frame_{i:04d}.png"
        Image.new("RGB", (32,24), ((i*7)%256, (i*13)%256, (i*19)%256)).save(image)
        entries.append({"kf":i,"image_index":i,"image":image.name,"path":str(image)})
    kf = {"n_keyframes":len(entries),"keyframes":entries,"image_dir":str(frames),"n_images":len(entries)}
    n = kf["n_keyframes"]
    names = [k["image"] for k in kf["keyframes"]]
    rng = np.random.default_rng(0)
    dist = rng.uniform(0.8, 1.2, (n, n))
    dist = (dist + dist.T) / 2
    sets = [("adj", "P_adj", cal.adjacent_pairs(n)), ("loop", "P_loop", cal.loop_pairs(dist)),
            ("neg", "N", cal.negative_pairs(n)[0])]
    pairs = []
    for tag, label, items in sets:
        for k, (i, j) in enumerate(items):
            pairs.append({"pair_id": f"{tag}_{k:03d}", "set": label, "kf_early": i, "kf_late": j, "index_gap": j - i,
                          "image_early": names[i], "image_late": names[j], "salad_l2": float(dist[i, j]),
                          "salad_l2_omega_loader": float(dist[i, j])})
    (tmp_path / "keyframes.json").write_text(json.dumps(kf))
    (tmp_path / "pairs_defs.json").write_text(json.dumps({
        "n_keyframes": n, "counts": {}, "loop_partner_agreement_omega_loader": {}, "p_loop_salad_below_0.95": 0,
        "pairs": pairs}))

    def column(base_pos, base_neg, noise, seed):
        g = np.random.default_rng(seed)
        out = {}
        for p in pairs:
            mean = base_neg if p["set"] == "N" else base_pos
            out[p["pair_id"]] = float(mean + noise * g.standard_normal())
        return out

    vggt21 = column(1.2, 0.6, 0.1, 1)
    vggt20 = column(1.1, 0.6, 0.1, 2)
    negs = [p["pair_id"] for p in pairs if p["set"] == "N"]
    for pid in negs[:3]:
        vggt21[pid] = 1.0  # FAR at layer 21 = 3/60
    vggt20[negs[0]] = 1.0  # FAR at layer 20 = 1/60
    omega = {}
    for layer in cal.OMEGA_LAYERS:
        gap = 0.9 if layer == 17 else 0.2  # layer 17 separates best
        omega[layer] = column(0.6 + gap, 0.6, 0.12, 10 + layer)

    def wrap(per_layer):
        return {pid: {str(layer): {"solver": values[pid], "reverse": values[pid] * 0.9}
                      for layer, values in per_layer.items()} for pid in vggt21}

    (tmp_path / "ratios_vggt.json").write_text(json.dumps({
        "status": "complete", "ratios": wrap({20: vggt20, 21: vggt21}), "regression": {}}))
    (tmp_path / "ratios_omega.json").write_text(json.dumps({
        "status": "complete", "ratios": wrap(omega), "regression": [], "adapter_crosscheck": [],
        "determinism_repeat_max_abs_diff": 0.0}))

    loop_ids = [p_["pair_id"] for p_ in pairs if p_["set"] == "P_loop"]
    (tmp_path / "sensitivity_subsets.json").write_text(json.dumps({"subsets": {"P_loop_visual_same_place": loop_ids[-8:]}}))
    env = dict(os.environ, OMP_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="")
    config = tmp_path / "cfg" / "omega_similarity.json"
    proc = subprocess.run([sys.executable, str(ROOT / "tools" / "calib_analyze.py"), "--dir", str(tmp_path),
                           "--config", str(config)], capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]

    cfg = json.loads(config.read_text())
    assert cfg["layer"] == 17
    assert cfg["FAR_vggt"] == pytest.approx(3 / 60) and cfg["vggt_layer_used_for_far"] == 21
    assert cfg["FAR_vggt_by_layer"]["20"] == pytest.approx(1 / 60)
    neg_omega = np.array([omega[17][pid] for pid in negs])
    assert cfg["tau"] == pytest.approx(np.quantile(neg_omega, 1 - 3 / 60))
    assert cfg["omega_far_at_tau"] == pytest.approx(3 / 60)
    assert cfg["tau_if_vggt_layer20_far"] > cfg["tau"]  # stricter operating point -> higher threshold
    assert set(cfg["auc_per_layer"]) == {str(l) for l in cal.OMEGA_LAYERS}
    assert cfg["per_set_stats"]["vggt@21"]["N"]["n_pass"] == 3
    assert cfg["dataset"] == "office_loop" and "rule_text" in cfg
    sens = cfg["sensitivity_subsets_not_used_for_config"]["P_loop_visual_same_place"]
    assert sens["n_loop_pairs"] == 8 and sens["layer_star"] == 17
    assert sens["omega_at_rule_L*_and_tau"]["n"] == 8 and cfg["per_set_stats"]["omega@17"]["P_loop_visual_same_place"]["n"] == 8

    merged = json.loads((tmp_path / "pairs.json").read_text())["pairs"]
    assert len(merged) == len(pairs) and "omega@17" in merged[0]["ratio"] and "vggt@20" in merged[0]["ratio_reverse_order"]
    for name in ("similarity_calibration.png", "negatives_top_sheet.png", "summary.json"):
        assert (tmp_path / name).stat().st_size > 1000, name

    # OmegaForSLAM must be able to consume the written file
    if importlib.util.find_spec("torch") is not None:
        from vos import omega_backend as ob
        loaded = ob.load_similarity_config(config)
        assert loaded["layer"] == 17 and loaded["tau"] == cfg["tau"]
