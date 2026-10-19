# Installing VGGT-Ω-SLAM

The supported workflow is Linux with Python 3.11 and an NVIDIA CUDA GPU.
CPU tests and preflight checks can run without loading checkpoints; model
inference and the SLAM runner require CUDA. The recorded experiments used a
12 GB RTX 5070 Ti Laptop GPU. VRAM demand depends on image dimensions and
submap size; that observation is not a guarantee for every sequence.

Upstream code and checkpoints are obtained separately. Read
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the BSD, CC-BY-NC,
GPLv3, Apache, and FAIR license boundaries. Omega and its results are subject
to the FAIR noncommercial research terms.

## 1. Create an environment and install Python packages

```bash
git clone https://github.com/KaiFeng-Frank/VGGT-Omega-SLAM.git
cd VGGT-Omega-SLAM
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a matched PyTorch/torchvision pair for your GPU and driver using the
[official PyTorch instructions](https://pytorch.org/get-started/previous-versions/).
For example, this published CUDA 12.8 pair meets Omega's `torch>=2.6` requirement:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-dev.txt
```

The stable pair above is an installation example, not the stack used to produce
the archived experiment results. A fresh end-to-end installation of this pair
has not been validated by this project. The exact observed experiment stack
is recorded below. Keep a working torch/torchvision pair together rather than
installing the older `torch==2.3.1` requirements from VGGT-SLAM on top of Omega.

On a Linux installation without the relevant system libraries, Open3D/OpenCV
may also require `libgl1` and `libglib2.0-0` from your system package manager.
The calibration plots contain Chinese labels and require a CJK font, including
the synthetic calibration pipeline test. On Ubuntu/Debian, install it with
`sudo apt-get install fonts-noto-cjk` before running the full test suite.
There is no need to install SAM 3, Perception Encoder, RealSense bindings, or
the upstream interactive demos for this headless runner. DINOv2 can use its
ordinary PyTorch attention path; xFormers is not required by this integration.

## 2. Fetch the pinned source trees

```bash
python tools/setup_sources.py
```

The helper fetches these public source revisions under `deps/` and writes
provenance information. It does not fetch checkpoints or modify upstream code.
The runner imports the checked-out sources directly, so installing their old
requirements files or running the upstream all-features setup script is
unnecessary.

| Component | Repository | Pinned revision |
|---|---|---|
| SLAM backend | [MIT-SPARK/VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM) | `35327ac28b7d193df9ccc39ba6346052bb6f1207` |
| Original VGGT frontend | [MIT-SPARK/VGGT_SPARK](https://github.com/MIT-SPARK/VGGT_SPARK) | `6e6e16107b88e8e76c751826af10d4295d87ecd2` |
| SALAD retrieval | [Dominic101/salad](https://github.com/Dominic101/salad) | `33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d` |
| DINOv2 source | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | `7764ea0` |
| Omega frontend | [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega) | `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6` |

`VOS_DEPS` can select another dependency directory. Individual paths can be
overridden with `VOS_SLAM_SRC`, `VOS_VGGT_SRC`, `VOS_SALAD_SRC`,
`VOS_DINOV2_SRC`, and `VOS_OMEGA_SRC`. Preserve the pinned revisions and their
generated provenance when relocating them; preflight verifies the SLAM snapshot.

## 3. Verify GTSAM's SL(4) Python bindings

The factor graph needs `SL4`, `PriorFactorSL4`, and `BetweenFactorSL4`. An
importable `gtsam` package alone is insufficient. `requirements.txt` pins the
reference [gtsam-develop 4.3.2.dev202610020315](https://pypi.org/project/gtsam-develop/4.3.2.dev202610020315/)
wheel. PyPI publishes a Python 3.11 manylinux x86_64 wheel for that version;
wheel availability varies with interpreter and platform.

```bash
python - <<'PY'
import gtsam
required = ('SL4', 'PriorFactorSL4', 'BetweenFactorSL4')
missing = [name for name in required if not hasattr(gtsam, name)]
assert not missing, f'GTSAM is missing SL(4) bindings: {missing}'
print('GTSAM SL(4) bindings present')
PY
```

The official implementation is in
[borglab/gtsam, `gtsam/geometry/SL4.h`](https://github.com/borglab/gtsam/blob/develop/gtsam/geometry/SL4.h).
If no matching wheel exists, build a GTSAM revision containing these bindings
using its [Python wrapper guide](https://github.com/borglab/gtsam/blob/develop/python/README.md)
and [build instructions](https://github.com/borglab/gtsam/blob/develop/INSTALL.md),
then rerun the symbol check. Record the source revision and build options for
your reproduction. This repository has not validated a fresh source build.

When a separate compatible build is already installed outside the virtual
environment, set `VOS_PYLIB` to its Python package directory. Leave it unset
when GTSAM is installed in the active environment. Avoid installing both
`gtsam` and `gtsam-develop` into one environment because they expose the same
Python module name.

## 4. Obtain the three checkpoints

Use your own provider access and accept the corresponding terms. Download:

| Component | Source | Local example filename |
|---|---|---|
| VGGT-Omega 1B 512 | [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_512.pt) | `checkpoints/vggt_omega_1b_512.pt` |
| Original VGGT 1B | [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt) | `checkpoints/vggt_1b.pt` |
| DINOv2 SALAD | [official SALAD v1.0.0 release](https://github.com/serizba/salad/releases/tag/v1.0.0) | `checkpoints/dino_salad.ckpt` |

Point the runner at those files explicitly:

```bash
export VOS_CHECKPOINTS="$PWD/checkpoints"
export VOS_OMEGA_CKPT="$VOS_CHECKPOINTS/vggt_omega_1b_512.pt"
export VOS_VGGT_CKPT="$VOS_CHECKPOINTS/vggt_1b.pt"
export VOS_SALAD_CKPT="$VOS_CHECKPOINTS/dino_salad.ckpt"
```

The archived experiments and checkpoint checks use these SHA-256 values:

```text
c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934  vggt_omega_1b_512.pt
d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0  vggt_1b.pt
6b3f1720954293e83da6966c5cfcfc6713200d7fefadcca76fc51aeb80b3cada  dino_salad.ckpt
```

Omega 512 is the checkpoint used in the archived integration experiments.
The newer 416 reproduction checkpoint is a distinct model with different
preprocessing requirements and has not been validated by this integration;
it should not silently replace the 512 checkpoint in a reproduction. See the
[upstream checkpoint guidance](https://github.com/facebookresearch/vggt-omega/blob/main/reproduction.md)
when designing a new comparison.

Once sources and weights are local, inference runs offline. The SALAD checkpoint
already contains its DINOv2 weights: the wrapper initializes the local DINOv2
architecture with `pretrained=False`, then strictly loads the full SALAD state.
No separate DINOv2 checkpoint or model download is needed during a run.

## 5. Run preflight, tests, and a short sequence

```bash
python run_slam.py --preflight-only
python -m pytest -q
python run_slam.py /path/to/images --backend omega --output runs/omega_smoke --max-frames 150
```

Preflight and the CPU tests do not constitute an end-to-end GPU SLAM validation.
For a run, the output directory must not already exist. A TUM sequence directory
can be supplied in place of an image folder; its `rgb/` images are used and
`groundtruth.txt`, when present, is used only for the final trajectory evaluation.
Use `--verify-checkpoints` on a run to recompute the checkpoint hashes.

Run one CUDA job at a time. The runner's shared GPU lock protects other processes
using the same lock; its options use the `VOS_GPU_*` environment-variable prefix.
SLAM default settings and outputs are described in the main README.
On a shared workstation, point `VOS_GPU_LOCK` at the same lock used by the
other jobs and set `VOS_GPU_REQUIRE_IDLE=1`. The default reserve is 2048 MiB;
see [GPU execution and allocator margins](ADAPTATION.md#shared-gpu-execution-and-allocator-margin).

For scripts that locate datasets through environment variables, set `VOS_TUM_ROOT`
to the parent directory containing the `rgbd_dataset_freiburg1_*` sequence folders,
and set `VOS_OFFICE_ROOT` to the office image folder. `VOS_SIM_CONFIG` can select
a different similarity calibration file. The bundled calibration was fitted on
the office development sequence, independently of TUM.

## Observed experiment environment

These versions were read from the environment that generated the archived
experiments. This is a record of that environment, not a claim that the setup
recipe above has been tested on a clean machine. The PyTorch versions are
nightly builds and may no longer be available from their original index.

| Package | Observed version |
|---|---|
| Python | `3.11.14` |
| torch | `2.11.0.dev20260201+cu128` |
| torchvision | `0.25.0.dev20260201+cu128` |
| NumPy | `1.26.4` |
| Pillow | `10.4.0` |
| OpenCV | `4.10.0` |
| Open3D | `0.18.0` |
| GTSAM distribution | `gtsam-develop==4.3.2.dev202610020315` |
| einops / safetensors | `0.8.2` / `0.8.0` |
| SciPy / Matplotlib | `1.17.0` / `3.10.8` |
| pytorch-lightning / torchmetrics | `2.5.2` / `1.8.2` |
| pytorch-metric-learning | `2.8.1` |
| pytest | `9.0.2` |

The runtime records software versions, source revisions, checkpoint information,
configuration, and inference timings in each run's `metadata.json`.
