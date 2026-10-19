"""(d) ATE: a trajectory transformed by a known Sim(3) gives ATE ~ 0 and the scale is recovered."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import eval_ate

ROOT = Path(__file__).resolve().parents[1]


def random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def trajectory(rng, n=300):
    steps = rng.normal(scale=0.05, size=(n, 3))
    xyz = np.cumsum(steps, axis=0) + np.array([0.3, -0.2, 1.0])
    stamps = 1305031450.0 + np.arange(n) * 0.033333
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    return stamps, xyz, quat


def write_tum(path, stamps, xyz, quat):
    rows = np.column_stack([stamps, xyz, quat])
    Path(path).write_text("# test\n" + "\n".join(" ".join(f"{v:.9f}" for v in r) for r in rows) + "\n")


def test_known_sim3_gives_zero_ate_and_recovers_scale(tmp_path):
    rng = np.random.default_rng(0)
    stamps, gt_xyz, quat = trajectory(rng)
    s, R, t = 2.5, random_rotation(rng), np.array([5.0, -3.0, 0.7])
    est_xyz = s * (R @ gt_xyz.T).T + t  # est = Sim3(gt); aligning est onto gt needs scale 1/s
    write_tum(tmp_path / "gt.txt", stamps, gt_xyz, quat)
    write_tum(tmp_path / "est.txt", stamps + 0.004, est_xyz, quat)  # stamp offset inside the 0.01 s window
    result = eval_ate.evaluate(tmp_path / "gt.txt", tmp_path / "est.txt")
    assert result["status"] == "ok" and result["n_associated"] == 300
    assert result["ate_rmse_m"] < 1e-9
    assert result["scale"] == pytest.approx(1 / s, rel=1e-9)
    # SE(3)-only alignment cannot absorb the scale
    assert result["ate_rmse_se3_no_scale_m"] > 0.1


def test_noise_level_is_recovered(tmp_path):
    rng = np.random.default_rng(1)
    stamps, gt_xyz, quat = trajectory(rng, 2000)
    sigma = 0.01
    est_xyz = gt_xyz + rng.normal(scale=sigma, size=gt_xyz.shape)
    write_tum(tmp_path / "gt.txt", stamps, gt_xyz, quat)
    write_tum(tmp_path / "est.txt", stamps, est_xyz, quat)
    result = eval_ate.evaluate(tmp_path / "gt.txt", tmp_path / "est.txt")
    assert result["ate_rmse_m"] == pytest.approx(sigma * np.sqrt(3), rel=0.08)  # fit absorbs a little noise
    assert result["scale"] == pytest.approx(1.0, abs=0.01)


def test_association_follows_evo_semantics(tmp_path):
    # dense gt (100 Hz) vs sparse est with a duplicated stamp (VGGT-SLAM writes overlap frames twice)
    gt_t = 100.0 + np.arange(0, 100) * 0.01
    est_t = np.array([100.0, 100.3, 100.3, 100.7, 200.0])
    ref_idx, est_idx = eval_ate.associate(gt_t, est_t, 0.01)
    # gt stamps within 0.01 s of an est stamp; duplicates resolve to the first copy (np.argmin)
    assert 2 not in set(est_idx) and 1 in set(est_idx) and 4 not in set(est_idx)
    assert len(ref_idx) == len(est_idx) > 3
    assert all(abs(gt_t[i] - est_t[j]) <= 0.01 + 1e-12 for i, j in zip(ref_idx, est_idx))
    # max_diff 0: only exact matches survive
    r0, _ = eval_ate.associate(gt_t, est_t, 0.0)
    assert len(r0) < len(ref_idx)


def test_too_few_pairs_fails_cleanly(tmp_path):
    rng = np.random.default_rng(2)
    stamps, gt_xyz, quat = trajectory(rng, 10)
    write_tum(tmp_path / "gt.txt", stamps, gt_xyz, quat)
    write_tum(tmp_path / "est.txt", stamps + 50.0, gt_xyz, quat)
    result = eval_ate.evaluate(tmp_path / "gt.txt", tmp_path / "est.txt")
    assert result["status"] == "failed" and result["n_associated"] == 0


def test_cli_writes_json(tmp_path):
    rng = np.random.default_rng(3)
    stamps, gt_xyz, quat = trajectory(rng, 50)
    write_tum(tmp_path / "gt.txt", stamps, gt_xyz, quat)
    write_tum(tmp_path / "est.txt", stamps, 3.0 * gt_xyz + 1.0, quat)
    out = tmp_path / "ate.json"
    proc = subprocess.run([sys.executable, str(ROOT / "eval_ate.py"), str(tmp_path / "gt.txt"),
                           str(tmp_path / "est.txt"), "--json", str(out)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert data["ate_rmse_m"] < 1e-9 and data["scale"] == pytest.approx(1 / 3, rel=1e-9)
