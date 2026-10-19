# Optional Ω backend adaptation and controlled replay

The runner exposes two experimental backend policies: `omega_confidence` for
confidence-selection changes, and `noise_only` for factor-noise changes while
preserving upstream scale selection. **The default is `legacy`.** With that
default, policy construction returns the pinned upstream Solver class itself;
non-default policy arguments are rejected instead of silently changing it.
Both optional policies use a project-owned Solver subclass and leave the fetched
upstream source files untouched.

This work tests how the existing SLAM system consumes Ω predictions. It does not
change the checkpoint, retrain the network, recalibrate predicted focal lengths,
or establish the training-level cause of the remaining focal bias. Numerical
camera corrections in the context diagnostic are evaluation corrections and are
not applied to the online SLAM predictions.

## Policy and command-line interface

The confidence policy has three independently controlled mechanisms:

1. **Select overlap pixels with confidence cutoffs from the corresponding
   submap or frame.** The legacy backend applies the prior submap's stored
   cutoff to both sides of a scale-estimation pair. The optional policy obtains
   separate prior/current cutoffs. If a percentile is supplied, it computes that
   percentile separately over each whole submap (`submap`) or over the individual
   frames participating in the edge (`frame`). If no percentile is supplied,
   it uses each submap's own stored cutoff, which was set by `--conf-threshold`.
2. **Change cross-submap and loop constraint noise.** Multipliers scale the
   Gaussian standard deviations, not variances or information matrices. A
   multiplier of two therefore reduces Gaussian information by a factor of
   four. The loop multiplier covers every edge involving an auxiliary loop
   submap, including its internal two-view edge. Ordinary submap internal edges
   and the first-node prior retain their existing noise models.
3. **Optionally use Huber robust noise.** The threshold is expressed in whitened
   residual units. It applies to the selected cross-submap/loop factors, including
   the auxiliary loop submap's internal edge. It does not rerun loop recognition
   or reject an accepted loop based on ground truth.

The overlap scale estimator and its scale direction remain the pinned backend's
median point-norm ratio. `omega_confidence` first requires finite points/confidences,
positive confidence, and nonzero point norms. Its fallback sequence is:

| Level | Pixels used |
|---|---|
| 0 | Both sides exceed their respective cutoffs |
| 1 | Prior side exceeds its cutoff; current side has positive confidence |
| 2 | Both sides have positive confidence |

The default requested minimum is 100 pixels. A fallback advances when too few
pixels remain; level 2 can retain fewer than 100 and reports that fact. Zero valid
pixels or a non-finite/non-positive scale raises an error before adding a factor.
Diagnostics record cutoffs, valid/selected counts, fallback level, scale, and the
natural-log scale-ratio median and unscaled MAD. MAD is a descriptive diagnostic,
not an automatically calibrated factor covariance.

`noise_only` calls the original upstream `add_edge` method, temporarily replacing
its factor-noise objects and restoring them afterward. It retains the exact
upstream confidence mask, scale points, estimator calls, and fallback behavior.
It rejects confidence-percentile/scope and minimum-scale-point overrides. This
separate mode permits noise experiments without bundling in a confidence-mask
change; unit multipliers with no Huber preserve the original noise objects.

The live runner and CPU replay expose the same policy fields with these names.
Policy choices are `legacy`, `omega_confidence`, and `noise_only`:

| Purpose | `run_slam.py` | `replay_slam_predictions.py` |
|---|---|---|
| Select policy | `--backend-policy` | `--policy` |
| Scale confidence percentile | `--scale-confidence-percentile` | `--scale-confidence-percentile` |
| Percentile scope | `--scale-confidence-scope submap\|frame` | `--scale-confidence-scope submap\|frame` |
| Cross-submap sigma multiplier | `--cross-noise-multiplier` | `--cross-submap-noise-multiplier` |
| Loop sigma multiplier | `--loop-noise-multiplier` | `--loop-noise-multiplier` |
| Huber threshold | `--edge-huber-delta` | `--huber-delta` |
| Cloud-filter percentile | `--conf-threshold` | `--conf-threshold` |
| Requested scale pixel count | Fixed default 100 | `--min-scale-points` |

`--conf-threshold` controls point-cloud filtering. An explicit
`--scale-confidence-percentile` controls overlap-scale selection independently.
With an omitted scale percentile, the optional policy reuses the stored
cloud-filter cutoffs as described above. The policy name is available with
either frontend for matched ablations; it does not itself change the frontend.

For example, this selects each submap's own 25th-percentile scale cutoff while
retaining unit noise multipliers and no Huber loss:

```bash
python run_slam.py "$VOS_TUM_ROOT/rgbd_dataset_freiburg1_desk" \
  --backend omega --backend-policy omega_confidence \
  --conf-threshold 25 --scale-confidence-percentile 25 \
  --scale-confidence-scope submap --submap-size 16 --max-loops 1 --seed 0 \
  --output runs/desk_omega_each_submap_q25
```

This is an experimental configuration example, not a claim that it improves
accuracy. The complete effective policy is saved in run metadata.

### Development replay findings

The `desk` legacy CPU replay reproduced all 57 ordered online trajectory rows
exactly, including quaternion signs and duplicate timestamps; four update counts
also matched. This establishes the local replay gate for these caches.

All five confidence-mask variants increased `desk` final Sim(3)-aligned ATE.
The development comparison used the same frozen predictions, two accepted
loops, 61 graph nodes, and six submaps:

| Scale mask | ATE RMSE (mm) | Change from legacy |
|---|---:|---:|
| Legacy | 43.178 | Reference |
| Each submap's stored q25 | 44.090 | +2.11% |
| Each participating frame's q10 | 44.658 | +3.43% |
| Each participating frame's q25 | 43.994 | +1.89% |
| Each participating frame's q50 | 44.426 | +2.89% |
| Each participating frame's q75 | 44.785 | +3.72% |

Legacy was retained in the selection pool and won the mask comparison. The
subsequent frozen backend candidate therefore uses `noise_only`: cross-submap
sigma ×2, loop sigma ×1, no Huber, with upstream scale/masks/fallbacks intact.
It changed `desk` replay ATE from 43.178 mm to 43.132 mm, a decrease of only
0.108%. This very small development change is not evidence of a useful general
improvement. Its external-sequence evaluation is a separate step under the
selection rule below; the default remains `legacy`.

For this frozen candidate, the live flags are
`--backend-policy noise_only --cross-noise-multiplier 2 --loop-noise-multiplier 1`;
the CPU replay flags are
`--policy noise_only --cross-submap-noise-multiplier 2 --loop-noise-multiplier 1`.
Do not add confidence-mask options to `noise_only`.

## Capture predictions once and replay the backend on CPU

Use `--prediction-cache` during an online run to save the complete input to
`Solver.add_points`, immediately after `run_predictions` and before map/graph
updates. The cache includes preprocessed images, cameras, depths, confidences,
accepted-loop data, frame names, and the actual regular-submap graph IDs. It uses
compressed NumPy arrays and JSON metadata, with `allow_pickle=False` on load.
Nested predictions, including VGGT's `pose_enc_list`, retain their list/tuple/
dictionary structure. BF16 tensors are stored as tagged `uint16` bit arrays and
restored to BF16 without a floating-point conversion. These optional metadata
fields extend the existing v1 format; older v1 caches remain readable.

```bash
python run_slam.py "$VOS_TUM_ROOT/rgbd_dataset_freiburg1_desk" \
  --backend omega --backend-policy legacy \
  --submap-size 16 --min-disparity 50 --conf-threshold 25 --max-loops 1 --seed 0 \
  --prediction-cache runs/desk_omega_cached/predictions \
  --output runs/desk_omega_cached
```

Replay rebuilds the submaps and SL(4) factor graph using those saved arrays. It
does not load a feed-forward model or retrieval model, recompute the accepted
loops, preprocess the images, query GPU state, or take a GPU lock. It disables
CUDA before importing runtime dependencies. GTSAM with the required SL(4)
bindings and the pinned source geometry are still required.

First verify the legacy replay against the online trajectory:

```bash
python tools/replay_slam_predictions.py \
  --cache-dir runs/desk_omega_cached/predictions \
  --reference-poses runs/desk_omega_cached/poses.tum \
  --groundtruth "$VOS_TUM_ROOT/rgbd_dataset_freiburg1_desk/groundtruth.txt" \
  --policy legacy --output runs/desk_omega_replay_legacy
```

Then evaluate an explicit backend policy using the same cache:

```bash
python tools/replay_slam_predictions.py \
  --cache-dir runs/desk_omega_cached/predictions \
  --reference-poses runs/desk_omega_cached/poses.tum \
  --groundtruth "$VOS_TUM_ROOT/rgbd_dataset_freiburg1_desk/groundtruth.txt" \
  --policy omega_confidence --scale-confidence-percentile 25 \
  --scale-confidence-scope submap \
  --output runs/desk_omega_replay_each_submap_q25
```

When `--reference-poses` is supplied for a nonlegacy run, the script first
replays the original backend internally and stops if verification fails. The
default reference tolerance is zero: it compares every ordered TUM row,
including duplicate overlap timestamps and quaternion signs, without alignment
or sign correction. If the online `metadata.json` is present next to the
reference, it also checks the recorded update/frame/graph/loop counts. A
nonzero `--reference-atol` explicitly relaxes numeric equality and is recorded.
Without `--reference-poses`, equivalence has not been checked.

Outputs include `summary.json`, `replay.json`, `poses.tum`, `run.log`, and
`backend_diagnostics.json`; a nonlegacy reference check also writes
`legacy_verification/`. Replay output directories must be fresh. The replay's
`--conf-threshold` must match the original run when verifying legacy behavior.

Replay estimates the effect of backend choices **conditional on a fixed set of
frontend predictions and accepted loops**. Changes to keyframe density, submap
size, network inference, or retrieval require new online runs. An improvement
in replay is not by itself an end-to-end runtime or accuracy validation.

Two NPZ formats serve different purposes:

| Cache | Producer | Consumer |
|---|---|---|
| Complete solver predictions | `run_slam.py --prediction-cache` | `replay_slam_predictions.py --cache-dir` |
| Named fixed-window cameras and boundary dense maps | `analysis_why.py forward`, `analysis_undistort.py forward` | Backend cache audit and context probe |

The historical fixed-window cache stores only selected dense maps and lacks the
complete accepted-loop solver input. It cannot substitute for a full replay
cache. `audit_backend_cache.py` uses those historical boundary maps for overlap
confidence diagnostics; it does not perform a full-SLAM intervention.

## Fixed-frame context diagnostic

`probe_submap_context.py` changes the context around identical evaluation
frames. Both backends retain their original checkpoints and preprocessing:
VGGT's pinned 518-width loader and Ω's balanced 512-pixel-budget loader. Each
window contains 17 keyframes. Moving its start forward by eight keyframes leaves
nine frames shared with the original window.

The published [frame manifest](../results/frame_manifest.json) defines the
original keyframe order. The default selection uses two eligible internal
boundaries per sequence, chosen uniformly by keyframe position before scoring:

| Sequence | Boundary keyframe indices | Adaptation role |
|---|---|---|
| `freiburg1_desk` | 16, 32 | Development |
| `freiburg1_desk2` | 16, 80 | Exploratory external-sequence validation |
| `freiburg1_room` | 16, 176 | Exploratory external-sequence validation |

Each pair is evaluated on the same nine frame names in the same chronological
order. Quaternion and FoV encodings are decoded again in float64, avoiding
rounded/nonorthogonal cached camera matrices. Both predicted trajectories and
ground truth use the same first shared frame as their local coordinate anchor.
Each common-frame trajectory is then aligned to the identical ground-truth
frames with Sim(3). The between-context shape residual also removes global
similarity, so a changed network reference frame is not counted as a geometry
error. All nine ground-truth associations must be valid; the script does not
silently drop a different first frame from one context.

The result records focal bias and its change, common-frame ATE, ATE/path length,
between-context shape change, and anchored orientation change. These are local
context diagnostics, not full-sequence SLAM scores or an estimate of independent
trial variability. The fixed-frame context results are not a pool for choosing
keyframe-density/submap-size settings.

### Raw-image context results

The raw-image probe completed all 12 new forwards, reusing the original named
baselines. Below, each ATE ratio is shifted/original on the identical nine
evaluation frames; values below one indicate a lower local ATE. The reported
ratio is the median of the two selected windows for that sequence. Focal change
is the mean change in Ω's horizontal focal bias, in percentage points.

| Sequence | VGGT local ATE ratio | Ω local ATE ratio | Ω windows with lower ATE | Ω focal-bias change |
|---|---:|---:|---:|---:|
| `desk` | 0.955 | 1.026 | 1/2 | +3.49 pp |
| `desk2` | 1.042 | 0.938 | 2/2 | −0.20 pp |
| `room` | 0.816 | 1.072 | 0/2 | −1.04 pp |

Moving the boundary by eight keyframes did not consistently improve Ω. Its
four external-sequence windows split two better/two worse, with median local
ATE ratio 1.015. Focal-bias changes also depended on the sequence. These few
dependent windows show context sensitivity; they do not establish that shifting
all SLAM boundaries would help or explain the model's focal bias. Detailed local
artifacts are under `runs/adaptation/context_raw/`; they are regenerated rather
than assumed to be distributed with the source tree.

For raw RGB and its matching baseline caches:

```bash
python tools/probe_submap_context.py design \
  --baseline-root runs/analysis_why \
  --data-root "$VOS_TUM_ROOT" --out runs/submap_context_raw

python tools/probe_submap_context.py forward \
  --baseline-root runs/analysis_why \
  --data-root "$VOS_TUM_ROOT" --out runs/submap_context_raw

python tools/probe_submap_context.py analyze \
  --baseline-root runs/analysis_why \
  --data-root "$VOS_TUM_ROOT" --out runs/submap_context_raw
```

`design` and `analyze` are CPU-only; `forward` is the GPU stage. The design saves
the exact executable commands, frame lists, calibration, and hashes of source
images/baseline caches. Successful forward results are resumable and checked
against the design. With all original baselines present, the default probe needs
12 new forwards: three sequences × two windows × two backends. Missing baseline
predictions add at most another 12 forwards.

For a rectified-image probe, use `runs/undistort/analysis` as the baseline root,
the parent directory containing the matching rectified sequence folders as
`--data-root`, and add
`--calibration runs/undistort/undistort_report.json` to each stage. This reads
`K_new` and the source dimensions for focal truth. The raw default calibration
must not be used for rectified images. Legacy caches do not record RGB-content
hashes, so matching the baseline's raw/rectified image variant remains the
caller's responsibility; new probe inputs are hash-checked after design creation.

## Adaptation split and selection rules

All three TUM sequences and their earlier results had been examined before
designing these adaptations. `desk2` and `room` are therefore **exploratory
external-sequence validation**, not unseen test data. The local experiment plan
is stored under `runs/adaptation/design.json`; generated `runs/` artifacts are
ignored by Git.

The confidence candidates are the legacy prior-cutoff rule, each submap's own
25th percentile, and each participating overlap frame's own percentile in
`{10, 25, 50, 75}`. The plan also compares cross/loop standard-deviation
multipliers in `{1, 2, 4}` and optional Huber threshold `1.345`, changing one
mechanism at a time before combining candidates. Backend selection uses `desk`
ATE only, then freezes one configuration for separate reporting on `desk2` and
`room`, including their pooled geometric-mean ATE ratio. No validation retuning
is part of that selection rule.

Frontend density/size candidates are minimum keyframe disparities `{25, 50, 75}`
pixels and `{16, 32}` new frames per submap. At most one additional density/size
configuration is selected on `desk` and carried unchanged to both validation
sequences. This requires new network predictions and cannot be evaluated by
reusing a complete cache from a different keyframe/window partition.

Changing a default requires **at least 2% lower ATE on each validation sequence**
against freshly reproduced, matching legacy runs. Until that criterion is established,
the experimental policy remains opt-in and `legacy` remains the default.

A weak result from one system adaptation does not establish that the pretrained
model would win after a different integration. It also does not establish a
general model ranking from these three sequences. Changes to confidence handling,
factor noise, keyframe spacing, and context answer separate questions. Positive,
negligible, and negative outcomes are retained with their controls; neither the
failed mask changes nor the mixed context results are removed from the evidence.

## Run the fixed frontend grid

Run these commands from the repository root. For both the context probe and
`sweep_frontend.py`, `--data-root` is the **parent of the sequence directories**:
for example, `$VOS_TUM_ROOT` contains
`rgbd_dataset_freiburg1_desk/rgb/` and its `groundtruth.txt`, with the other
sequence folders alongside it. Do not pass a single sequence directory or its
`rgb/` directory. `--sequence` accepts `desk`, `desk2`, `room`, or the full
sequence-folder name.

The sweep defaults to Ω and writes a CPU-only design without loading a model,
querying CUDA, or acquiring a GPU lock. Its development grid contains six
settings: submap size `{16, 32}` × minimum disparity `{25, 50, 75}`. The baseline
is `s16_d50`. All runs use the `legacy` backend policy; this grid tests frontend
size/density separately from the optional backend policy.

```bash
python tools/sweep_frontend.py \
  --data-root "$VOS_TUM_ROOT" --sequence desk \
  --out runs/frontend_development

python tools/sweep_frontend.py \
  --data-root "$VOS_TUM_ROOT" --sequence desk \
  --out runs/frontend_development --run
```

Only `--run` executes the scheduled commands. Child runs execute serially and
each acquires the shared GPU lock; the sweep parent does not claim GPU memory.
The child inherits `VOS_*` settings and caller thread limits. `--threads` accepts
one through four. Any external process-memory limit remains the caller's
responsibility.

The tool writes `design.json`, `results.json`, per-setting `logs/`, complete run
outputs under `runs/<setting>/`, and prediction caches under
`predictions/<setting>/`. Once all six settings have terminal records and a
consistent completed baseline exists, it writes `frozen_selection.json`.
Selection uses the lowest finite development ATE among completed feasible runs;
exact ties prefer the baseline, then fixed grid order. Failed or infeasible
runs are retained without an accuracy score. A process being killed alone is
not proof of an out-of-memory failure.

Use that frozen file for each external-sequence comparison. The first command
below only writes the validation design; the second executes it:

```bash
python tools/sweep_frontend.py \
  --data-root "$VOS_TUM_ROOT" --sequence desk2 \
  --out runs/frontend_validation_desk2 \
  --frozen-selection runs/frontend_development/frozen_selection.json

python tools/sweep_frontend.py \
  --data-root "$VOS_TUM_ROOT" --sequence desk2 \
  --out runs/frontend_validation_desk2 \
  --frozen-selection runs/frontend_development/frozen_selection.json --run
```

Repeat with `--sequence room` and a separate
`--out runs/frontend_validation_room`, keeping the same frozen file. With
`--frozen-selection`, only the baseline and the frozen selected setting are
scheduled; if the baseline won development, only that setting is scheduled.
Validation checks backend, checkpoint identity, and Ω similarity calibration,
and never selects or rewrites the development choice. Omitting the frozen file
would start another six-setting development sweep, which is outside this
validation design. These are the exploratory external sequences described
above, all of which had already been examined.

A matching completed `s16_d50` run may be reused with `--baseline-run` and its
optional `--baseline-cache`. A cache alone is insufficient: it has no completed
trajectory/ATE. An external completed baseline without a cache is recorded as
such. Reuse checks effective inference/evaluation arguments and, when a cache
is available, its frame lists. Use identical arguments/environment when resuming
a design. Failed or incomplete output directories are preserved rather than
retried in place; use a fresh output directory for a new attempt.

## Compare trajectories on identical time support

Changing keyframe disparity changes the timestamps used by the original per-run
ATE. `compare_trajectory_support.py` adds a CPU-only positional comparison on
the same ground-truth timestamps. Use the baseline and frozen candidate's
`poses.tum` files for the same sequence:

```bash
python tools/compare_trajectory_support.py \
  --baseline /path/to/baseline/poses.tum \
  --candidate /path/to/frozen_candidate/poses.tum \
  --groundtruth "$VOS_TUM_ROOT/rgbd_dataset_freiburg1_desk2/groundtruth.txt" \
  --out runs/frontend_validation_desk2/common_support.json
```

For each input, the tool drops non-finite rows, keeps the **first occurrence in
file order** of each duplicate timestamp, and then sorts by time. It takes the
intersection of both prediction time ranges and the ground-truth time range.
At every unique ground-truth timestamp in that interval, it linearly
interpolates each predicted position, without extrapolation, then independently
aligns both sets of positions to the same ground-truth positions with Sim(3).
It requires at least three common timestamps and reports alignment failures.
This does not evaluate orientation.

The JSON reports ATE, candidate/baseline ratio, common timestamp count and
coverage, duplicate counts, exact/interpolated sample counts, and interpolation
gap statistics. A second view uses exact shared prediction timestamps, retaining
the predicted positions and interpolating only ground truth. Fewer than 20
samples are flagged. Report coverage and large gaps alongside any ratio:
piecewise linear interpolation cannot recover the actual motion between sparse
keyframes.

This is a supplementary metric. Keep the original per-run ATE and the frozen
development selection; do not select another setting or retune parameters using
the common-time comparison or validation results.

## Shared GPU execution and allocator margin

All cooperating GPU jobs must use the same `VOS_GPU_LOCK` path. Set it to the
existing shared lock used by the other GPU jobs on your machine; replace the
example path below with that lock:

```bash
export VOS_GPU_LOCK="/path/to/existing/shared/gpu.lock"
export VOS_GPU_REQUIRE_IDLE=1
export VOS_GPU_RESERVE_MB=2048
```

Use the lock path already shared by other research jobs on your own machine.
Without an override, the project uses
`$XDG_CACHE_HOME/vggt-omega-slam/gpu.lock`, falling back to
`$HOME/.cache/vggt-omega-slam/gpu.lock`. An optional `VOS_GPU_WAIT_S` sets a wait
timeout. Lock acquisition occurs once per process, before CUDA work; the lock
is held until that process exits.

With `VOS_GPU_REQUIRE_IDLE=1`, startup waits for other compute PIDs to finish.
The runner checks again before regular frontend forwards, loop-verification
forwards, and SALAD descriptor forwards; the context probe checks before its
forwards too. If another compute process has appeared, the check fails. The
current process is excluded, and other jobs are left running. These additional
PID checks are opt-in through `VOS_GPU_REQUIRE_IDLE`; a shared lock is still
required for cooperating jobs to queue reliably.
`VOS_GPU_DEVICE` selects the matching `nvidia-smi` device when the logical CUDA
device mapping needs to be specified.

The allocator cap is calculated after CUDA context creation. In bytes, it is
the minimum of the requested cap, 95% of total device memory, and:

```text
(currently free MiB − VOS_GPU_RESERVE_MB − 256 MiB) × 2^20
```

The extra **256 MiB** provides headroom for cuBLAS/cuDNN and other allocations
outside PyTorch's allocator, in addition to the configured reserve. A
non-positive remaining budget fails explicitly. This is a budgeting margin and
a cooperative process lock; it is not a driver-level memory reservation against
unrelated applications. CPU preflight, context analysis, and prediction replay
do not query or reserve GPU memory.
