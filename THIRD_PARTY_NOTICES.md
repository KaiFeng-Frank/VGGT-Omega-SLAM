# VGGT-Ω-SLAM third-party notices

This is an independent integration and evaluation project. The following upstream
projects supply the reconstruction models, SLAM backend, and retrieval model;
their authors do not endorse this repository. Upstream source trees are fetched
separately by `tools/setup_sources.py`, kept at fixed revisions, and imported
without editing their Python source. Checkpoints and datasets are not included.

The license grant for original project contributions in [LICENSE](LICENSE)
does not replace the licenses below. This is a multi-license research stack,
including noncommercial and copyleft components, rather than an MIT-licensed
model or an unrestricted commercial package.

## Included adaptations

### VGGT-SLAM 2.0

- Upstream: [MIT-SPARK/VGGT-SLAM](https://github.com/MIT-SPARK/VGGT-SLAM/tree/35327ac28b7d193df9ccc39ba6346052bb6f1207).
- Revision: `35327ac28b7d193df9ccc39ba6346052bb6f1207`.
- Copyright: Copyright (c) 2025, MIT-SPARK.
- License: BSD 2-Clause; complete original text in
  [licenses/VGGT-SLAM-BSD-2-Clause.txt](licenses/VGGT-SLAM-BSD-2-Clause.txt).
- `run_slam.py` adapts the control sequence from upstream `main.py`, adding a
  selectable frontend, headless/offline operation, instrumentation, and evaluation.
  The non-rendering viewer and offline Torch Hub patterns in `vos/env.py` were
  initially extracted through a project-local wrapper of that backend and were
  adapted for this repository. The upstream keyframe/submap/SL(4) solver code is
  imported from the separate pinned source tree.

### VGGT / VGGT_SPARK

- Original model: [facebookresearch/vggt](https://github.com/facebookresearch/vggt),
  by Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian
  Rupprecht, and David Novotny.
- Integration fork: [MIT-SPARK/VGGT_SPARK](https://github.com/MIT-SPARK/VGGT_SPARK/tree/6e6e16107b88e8e76c751826af10d4295d87ecd2).
- Revision: `6e6e16107b88e8e76c751826af10d4295d87ecd2`.
- Source notice: Copyright (c) Meta Platforms, Inc. and affiliates. All rights
  reserved. MIT-SPARK supplies the SLAM-specific fork and attention-similarity
  additions.
- The **pinned fork's** `LICENSE.txt` is CC-BY-NC-4.0. Its complete original text
  is in [licenses/VGGT-SPARK-CC-BY-NC-4.0.txt](licenses/VGGT-SPARK-CC-BY-NC-4.0.txt).
  A different license on a newer upstream revision does not, by itself, change
  the license stated by this pinned snapshot.
- `vos/omega_backend.py::mean_top_quarter` and `get_similarity` are adapted from
  the fork's `vggt/models/aggregator.py`. The arithmetic is preserved; debugging
  output was removed, and the caller supplies Omega's token offset and calibrated
  acceptance threshold. These functions retain the upstream CC-BY-NC-4.0 terms.

### VGGT-Omega

- Upstream: [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega/tree/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6).
- Revision: `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6`.
- Source notice: Copyright (c) Meta Platforms, Inc. and affiliates. All rights
  reserved.
- License: FAIR Noncommercial Research License, v1, October 16, 2024, including
  the FAIR Acceptable Use Policy. The complete original text is in
  [licenses/VGGT-OMEGA-FAIR-NONCOMMERCIAL.txt](licenses/VGGT-OMEGA-FAIR-NONCOMMERCIAL.txt).
- The model, image preprocessing, and camera decoder are imported from that
  separate source tree. Q/K reconstruction in `vos/omega_backend.py` follows
  the upstream attention computation; to the extent that this adaptation is a
  derivative of those research materials, it retains their FAIR terms.
- The FAIR agreement restricts use of the research materials and their
  outputs/results to noncommercial research uses and requires acknowledgment
  when research results are submitted for publication. Model-generated
  results included here are not relicensed under the project's BSD grant.
  Access to a checkpoint is subject to its provider's access process and terms.

### evo-compatible trajectory evaluation

- Implementation reference: [MichaelGrupp/evo](https://github.com/MichaelGrupp/evo),
  by Michael Grupp and contributors, especially
  [`evo/core/geometry.py`](https://github.com/MichaelGrupp/evo/blob/master/evo/core/geometry.py)
  and [`evo/core/sync.py`](https://github.com/MichaelGrupp/evo/blob/master/evo/core/sync.py).
- `eval_ate.py::umeyama_alignment` and `associate` reproduce evo's alignment
  and nearest-timestamp conventions. The implementation uses vectorized
  covariance accumulation, simplified error handling, and NumPy index-array
  returns. To preserve the implementation reference's license boundary, these
  adapted helpers are distributed under GPL-3.0-or-later, not the original
  contributions' BSD grant. The underlying Umeyama algorithm is described in
  Shinji Umeyama, *Least-squares estimation of transformation parameters between
  two point patterns*, IEEE TPAMI, 1991.
- The source headers in evo state GPL version 3 or any later version. A complete
  copy of evo's GPL text is in
  [licenses/EVO-GPL-3.0.txt](licenses/EVO-GPL-3.0.txt). The rest of the evo package
  is neither bundled nor required at runtime.

## Separately obtained runtime dependencies

### SALAD

- Authors: Sergio Izquierdo and Javier Civera.
- Original upstream: [serizba/salad](https://github.com/serizba/salad).
- Fork used by VGGT-SLAM: [Dominic101/salad](https://github.com/Dominic101/salad/tree/33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d).
- Revision: `33ca9c0ca1e10cbb21efc0d6a5fcb6d45688e42d`.
- Its `LICENSE` contains the GNU GPL, version 3. The complete original text is
  preserved in [licenses/SALAD-GPL-3.0.txt](licenses/SALAD-GPL-3.0.txt).
  SALAD is imported from its separate source tree; its source and checkpoint
  are not incorporated into the project BSD grant. Any redistribution that
  includes SALAD must comply with its applicable GPL obligations.

### DINOv2

- Upstream: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2/tree/7764ea0).
- Revision: `7764ea0` (the setup script records the complete resolved revision).
- Copyright (c) Meta Platforms, Inc. and affiliates.
- License: Apache-2.0; complete snapshot text in
  [licenses/DINOV2-Apache-2.0.txt](licenses/DINOV2-Apache-2.0.txt).
- The snapshot also contains OpenAI CLIP code with its own MIT license,
  preserved in [licenses/DINOV2-CLIP-MIT.txt](licenses/DINOV2-CLIP-MIT.txt).
  The SLAM path uses the DINOv2 backbone through the separately loaded SALAD
  checkpoint; the entire source tree's original notices must be retained if
  that tree is redistributed.

### GTSAM

- Upstream: [borglab/gtsam](https://github.com/borglab/gtsam).
- Reference Python distribution: `gtsam-develop==4.3.2.dev202610020315`.
- GTSAM is distributed under its BSD license, with additional notices for
  included third-party components. See the upstream [LICENSE](https://github.com/borglab/gtsam/blob/develop/LICENSE)
  and [LICENSE.BSD](https://github.com/borglab/gtsam/blob/develop/LICENSE.BSD).
  No GTSAM source or binary is distributed in this repository. Installation and
  the required SL(4) binding checks are documented in [docs/INSTALL.md](docs/INSTALL.md).

Other Python packages listed in `requirements*.txt` retain their own licenses.
Installations and redistributions of those packages must retain their notices.
Dataset licenses and model-card terms remain with their respective providers.
