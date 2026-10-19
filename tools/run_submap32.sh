#!/usr/bin/env bash
# Submap-32 runs (the VGGT-SLAM 2.0 paper protocol): VGGT vs VGGT-Omega inside the unmodified VGGT-SLAM 2.0 on
# TUM fr1 desk / desk2 / room, loops on, seed 0, every other parameter at the upstream default.
# Sequential, one CUDA process at a time (run_slam.py takes the vos GPU lock itself via --gpu-need-mb).
# CUDA allocator cap 10 GiB (run_slam.py --cuda-memory-limit-gb). A failed run is logged and the driver continues.
# Output dirs are never reused.
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${VOS_PYTHON:-python}"
source "$PROJECT/tools/run_env.sh"
TUM="${VOS_TUM_ROOT:-$PROJECT/data/tum_rgbd}"
OUT=$PROJECT/runs/submap32
cd "$PROJECT" || exit 1
mkdir -p "$OUT"

declare -A SEQDIR=( [fr1_desk]="$TUM/rgbd_dataset_freiburg1_desk"
                    [fr1_desk2]="$TUM/rgbd_dataset_freiburg1_desk2" [fr1_room]="$TUM/rgbd_dataset_freiburg1_room" )
declare -A NEED=( [vggt]=8500 [omega]=9500 )   # MB claimed from the GPU lock (+2048 MB always left for the desktop)
JOBS=( "fr1_desk vggt" "fr1_desk omega" "fr1_desk2 vggt" "fr1_desk2 omega" "fr1_room vggt" "fr1_room omega" )
echo "[driver] start $(date -Is)"
for job in "${JOBS[@]}"; do
  read -r seq backend <<<"$job"
  name="${seq}_${backend}_loops1"
  if [ -e "$OUT/$name" ]; then echo "[driver] SKIP $name exists"; continue; fi
  echo "[driver] $(date -Is) BEGIN $name"
  run_env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VOS_GPU_WAIT_S=1800 \
    "$PY" run_slam.py "${SEQDIR[$seq]}" --backend "$backend" --submap-size 32 --max-loops 1 --seed 0 \
    --min-disparity 50 --conf-threshold 25 --lc-thres 0.95 \
    --gpu-need-mb "${NEED[$backend]}" --cuda-memory-limit-gb 10 \
    --output "$OUT/$name" > "$OUT/$name.console.log" 2>&1
  echo "[driver] $(date -Is) END $name exit=$?"
done
echo "[driver] done $(date -Is)"
