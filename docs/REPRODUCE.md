# Reproducing the experiments

The public snapshot separates immutable published measurements (`results/`) from newly generated output (`runs/`, ignored by Git). The original images, checkpoints, dense caches, and private development sequence are not included.

## Prerequisites

Complete [INSTALL.md](INSTALL.md). Set `VOS_TUM_ROOT` to a directory containing the downloaded and extracted `rgbd_dataset_freiburg1_desk`, `rgbd_dataset_freiburg1_desk2`, and `rgbd_dataset_freiburg1_room` directories. Each needs `rgb/`, `rgb.txt`, and `groundtruth.txt`.

```bash
export VOS_TUM_ROOT="$PWD/data/tum_rgbd"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Use the original Ω **512** checkpoint and its official balanced preprocessing to reproduce this release. Switching to the 416 reproduction checkpoint is a different experiment and requires the corresponding preprocessing; the runner does not silently switch between them.

## Full-sequence SLAM

These scripts run jobs sequentially and preserve separate output directories. `run_compare.sh` runs the three TUM sequences with loops enabled/disabled; the optional office sequence is skipped unless supplied. No private office sequence is needed for the TUM comparison.

```bash
bash tools/run_compare.sh
bash tools/run_submap32.sh
python tools/undistort_tum.py
bash tools/run_undistort_slam.sh

python tools/compare_summarize.py
python tools/submap32_summarize.py
python tools/analysis_undistort.py slam
```

`VOS_PYTHON` selects the Python executable for shell drivers. To opt into a systemd user-scope memory cap, set `VOS_SYSTEMD_MEMORY_MAX=12G`; it is not required by default. The runner exposes `--gpu-need-mb` and `--cuda-memory-limit-gb` for matching available hardware. Existing output directories are skipped by drivers and rejected by individual SLAM runs; check each run's `metadata.json` status.

## Fixed-window front-end and distortion analysis

`results/frame_manifest.json` contains the exact archived frame lists and partitions. The original regular windows contain 16 new frames plus one overlap frame; only the 20 full 17-frame windows enter the pooled local summaries. The 27 sparse sets vary sequence, frame count (8/16/32), and sampling offset. These are overlapping samples, not independent trials.

```bash
python tools/analysis_why.py forward
python tools/analysis_why.py analyze --no-figure

python tools/analysis_undistort.py forward
python tools/analysis_undistort.py analyze
python tools/analysis_undistort.py report
```

Run the SLAM comparison first for metadata and full-sequence context. The forward-stage regular partitions fall back to the published manifest when metadata is absent, while sparse jobs use the published fixed selections. Dense caches are regenerated under `runs/`; no inference is performed by `analyze`.

Geometry helpers now project decoded rotations to SO(3) in float64 before composing relative rotations or computing camera centers. Fresh numbers should be compared to the corrected metric version, not the historical unprojected bf16 values. [RESULTS.md](RESULTS.md) distinguishes the archived measurements and the later fp64 pose-encoding diagnostics.

## Focal-cause controls

After the undistorted forward cache is available:

```bash
python tools/probe_focal_cause.py forward
python tools/probe_focal_cause.py analyze
```

This performs 104 forwards: six baseline reproductions; 20 each for zoom with the original principal point, equal zoom with a centered principal point, 512×384 input, and 640×480 input; six full-fp32 windows; and six each at centered zoom factors 1.25 and 1.50. The camera head is retained and the depth head is omitted after strict checkpoint loading. Baseline pose encodings were bitwise identical to the original full-network cache on the experiment machine.

Principal-point effects must be measured **between the two equal-zoom cases**, not against the unzoomed baseline. Resolution controls also change the patch-rounded aspect ratio by about 0.90%. Zoom changes visible boundary content. The full-fp32 and larger-zoom controls use the same six preselected windows. These controls localize the remaining bias but do not identify a unique training-level cause.

## Evaluation and reproducibility scope

Whole-sequence ATE is computed after Sim(3) alignment and uses the timestamp association implemented in `eval_ate.py`. Local-window metrics use the association documented in `vos/why_utils.py`; the two association directions are not interchangeable. Whole-sequence runs are single seed-0 observations, and undistortion changes the selected keyframes.

The recorded GPU was an RTX 5070 Ti Laptop with 12 GB VRAM. Upstream commits, checkpoint hashes, software versions, and result definitions are included in the installation and result records. Packaging checks validate CPU tests and preflight against the recorded local dependencies; a fresh-machine install and every GPU experiment are not rerun as part of publishing this snapshot.
