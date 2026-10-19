#!/usr/bin/env bash
# SLAM on the UNDISTORTED TUM fr1 frames (data/tum_undistorted): VGGT and VGGT-Omega inside the unmodified VGGT-SLAM 2.0,
# submap size 16 and 32, loops on, seed 0, every other parameter at the upstream default (Omega: frozen
# config/omega_similarity.json). 12 runs, strictly sequential, one CUDA process at a time (run_slam.py takes the vos GPU
# lock itself via --gpu-need-mb). CUDA allocator cap 10 GiB. A failed run is logged and the driver continues; an OOM is
# reported, the cap is never raised. Output dirs are never reused.
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${VOS_PYTHON:-python}"
source "$PROJECT/tools/run_env.sh"
UND=$PROJECT/data/tum_undistorted
OUT=$PROJECT/runs/undistort/slam
cd "$PROJECT" || exit 1
mkdir -p "$OUT"
declare -A NEED=( [vggt]=8500 [omega]=9500 )   # MB claimed from the GPU lock (+2048 MB always left for the desktop)
echo "[driver] start $(date -Is)"
for seq in desk desk2 room; do
  for size in 16 32; do
    for backend in vggt omega; do
      name="${seq}_${backend}_s${size}"
      if [ -e "$OUT/$name" ]; then echo "[driver] SKIP $name exists"; continue; fi
      echo "[driver] $(date -Is) BEGIN $name"
      run_env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
        OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VOS_GPU_WAIT_S=3600 \
        "$PY" run_slam.py "$UND/rgbd_dataset_freiburg1_${seq}" --backend "$backend" --submap-size "$size" --max-loops 1 \
        --seed 0 --min-disparity 50 --conf-threshold 25 --lc-thres 0.95 \
        --gpu-need-mb "${NEED[$backend]}" --cuda-memory-limit-gb 10 \
        --output "$OUT/$name" > "$OUT/$name.console.log" 2>&1
      echo "[driver] $(date -Is) END $name exit=$?"
    done
  done
done
echo "[driver] done $(date -Is)"
