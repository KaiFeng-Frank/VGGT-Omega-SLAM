#!/usr/bin/env bash
# Sequential comparison driver: VGGT vs VGGT-Omega inside the unmodified VGGT-SLAM 2.0.
# One CUDA process at a time (run_slam.py takes the vos GPU lock itself). A failed run is
# logged and the driver continues. Output dirs are never reused.
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${VOS_PYTHON:-python}"
source "$PROJECT/tools/run_env.sh"
TUM="${VOS_TUM_ROOT:-$PROJECT/data/tum_rgbd}"
OFFICE="${VOS_OFFICE_ROOT:-$PROJECT/data/office_loop}"
OUT=$PROJECT/runs/compare
mkdir -p "$OUT"
cd "$PROJECT" || exit 1

declare -A SEQDIR=( [office_loop]="$OFFICE" [fr1_desk]="$TUM/rgbd_dataset_freiburg1_desk"
                    [fr1_desk2]="$TUM/rgbd_dataset_freiburg1_desk2" [fr1_room]="$TUM/rgbd_dataset_freiburg1_room" )
JOBS=(
  "office_loop vggt 1" "office_loop omega 1"
  "fr1_desk vggt 1" "fr1_desk omega 1"
  "fr1_desk2 vggt 1" "fr1_desk2 omega 1"
  "fr1_room vggt 1" "fr1_room omega 1"
  "fr1_desk vggt 0" "fr1_desk omega 0"
  "fr1_desk2 vggt 0" "fr1_desk2 omega 0"
  "fr1_room vggt 0" "fr1_room omega 0"
)
echo "[driver] start $(date -Is)"
for job in "${JOBS[@]}"; do
  read -r seq backend loops <<<"$job"
  if [[ "$seq" == office_loop && ! -d "$OFFICE" ]]; then continue; fi
  name="${seq}_${backend}_loops${loops}"
  if [ -e "$OUT/$name" ]; then echo "[driver] SKIP $name exists"; continue; fi
  echo "[driver] $(date -Is) BEGIN $name"
  run_env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 "$PY" run_slam.py "${SEQDIR[$seq]}" --backend "$backend" \
    --max-loops "$loops" --seed 0 --output "$OUT/$name" > "$OUT/$name.console.log" 2>&1
  echo "[driver] $(date -Is) END $name exit=$?"
done
echo "[driver] done $(date -Is)"
