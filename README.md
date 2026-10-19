# VGGT-Ω-SLAM

**A runnable, unofficial VGGT-Ω front end for VGGT-SLAM 2.0, with matched VGGT comparisons and camera-bias diagnostics.**

This research preview connects VGGT-Ω camera/depth predictions to the existing VGGT-SLAM 2.0 keyframe selection, submaps, SALAD retrieval, and SL(4) optimization. It includes a headless runner, calibrated attention-based loop verification, trajectory evaluation, fixed frame lists, and measured results on TUM RGB-D.

The project is independent of the VGGT-Ω and VGGT-SLAM authors. The released experiments use the original **VGGT-Ω-1B-512** checkpoint. They are not a reproduction of the newer 416-checkpoint benchmark or a claim of improved SLAM accuracy.

[Installation](docs/INSTALL.md) · [Reproduction](docs/REPRODUCE.md) · [Results and corrections](docs/RESULTS.md) · [System adaptation](docs/ADAPTATION.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

## What is included

- Switch between `--backend omega` and `--backend vggt` in the same SLAM runner.
- An Ω adapter using the official image loader and camera decoder, with attention-hook loop verification.
- TUM-format trajectories, run metadata, keyframe records, point-cloud export, and Sim(3)-aligned ATE evaluation.
- Fixed-window diagnostics for focal length, translation shape, lens distortion, principal point, input size, and numerical precision.
- Lossless prediction caches and CPU backend replay, opt-in confidence/noise policies, and a frozen development/validation sweep for keyframe spacing and submap size.
- Small result records and frame manifests. Download model weights and datasets from their original providers.

## Current findings

On **freiburg1_desk, freiburg1_desk2, and freiburg1_room**, Ω has a higher final ATE than VGGT in all 12 paired configurations: raw/undistorted images × submap size 16/32 × three sequences. These are single runs with seed 0.

![Sim(3)-aligned SLAM ATE across the 12 paired configurations](results/figures/slam_ate.png)

The focal-length overestimate persists after undistortion: approximately **+13.0% → +13.5%**. Across 20 matched submaps, the median translation-shape error ratio Ω/VGGT is approximately **1.298 → 1.246**, using SO(3)-projected camera geometry. These are observations about this checkpoint and these sequences, not a general ranking of the models.

An additional **104 successful diagnostic forwards** found:

| Control | Focal-length result |
|---|---|
| Full fp32, six matched windows | +13.966% → +13.945% |
| Centered principal point, equal zoom, 20 windows | +12.030% → +11.475% |
| 512×384 / 592×448 / 640×480 inputs | +14.917% / +13.469% / +12.834% |

**Correction:** the initial claim that Ω had roughly half the relative rotation error was a bf16/nonorthogonal-matrix metric artifact. After projection onto SO(3), adjacent-frame relative rotation errors are comparable: approximately **0.715° for VGGT and 0.731° for Ω** on the raw windows. Public geometry helpers include this correction and regression tests.

The remaining focal bias is present in the model's predicted FoV. Its training-level cause, and whether it causes the translation gap, remain unresolved. See [the result definitions and limitations](docs/RESULTS.md) before comparing numbers across metric versions.

## Run a sequence

First follow [installation](docs/INSTALL.md): fetch the pinned upstream sources, install a GTSAM build with **SL4**, and obtain the three checkpoints. Source paths, checkpoint paths, and dataset roots are configurable through `VOS_*` environment variables.

```bash
python tools/setup_sources.py
python run_slam.py --preflight-only --output runs/preflight

python run_slam.py data/tum_rgbd/rgbd_dataset_freiburg1_desk \
  --backend omega --submap-size 16 --max-loops 1 --seed 0 \
  --output runs/desk_omega

python run_slam.py data/tum_rgbd/rgbd_dataset_freiburg1_desk \
  --backend vggt --submap-size 16 --max-loops 1 --seed 0 \
  --output runs/desk_vggt
```

Output directories must be new. An available `groundtruth.txt` is evaluated automatically; predictions retain monocular scale until evaluation alignment. The standalone GPU lock serializes participating jobs and reserves memory for other applications.

The frozen loop-verification settings are layer **15** and threshold **0.3764**. They were calibrated on a separate development sequence, not these TUM sequences. That sequence is not distributed, so exact calibration-data replay is not included; the selected configuration, calibration procedure, and synthetic tests are provided.

## Repository map

| Path | Purpose |
|---|---|
| `run_slam.py`, `eval_ate.py` | Headless SLAM and trajectory evaluation |
| `vos/` | Front-end adapters, geometry, calibration, environment, GPU lock |
| `tools/` | Dependency setup, experiment drivers, and diagnostics |
| `config/omega_similarity.json` | Frozen Ω loop-verification configuration |
| `results/` | Published measurements and exact frame selections |
| `tests/` | CPU geometry, adapter, calibration, and runner checks |

## Attribution and use

Built on [VGGT-Ω](https://github.com/facebookresearch/vggt-omega), [VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM), [VGGT](https://github.com/facebookresearch/vggt), [SALAD](https://github.com/serizba/salad), [DINOv2](https://github.com/facebookresearch/dinov2), and [GTSAM](https://github.com/borglab/gtsam). Evaluation uses the [TUM RGB-D dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset).

Upstream components have different licenses; Ω code, weights, and derived results carry noncommercial research conditions. This repository does not relicense those materials. Read [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

中文：这是 **VGGT-Ω 接入 VGGT-SLAM 2.0 的非官方研究预览**。已公开接入代码、复现步骤、对照结果和原因排查；当前 TUM fr1 测试中 Ω 未超过 VGGT。焦距偏差根因仍在调查，早期旋转误差结论已订正。
