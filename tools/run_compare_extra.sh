#!/usr/bin/env bash
# Extra runs after the 14 requested ones: office_loop with loop closure off, so that the start-to-end gap proxy
# has an open-loop baseline. Sequential, one CUDA process at a time (run_slam.py takes the GPU lock).
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${VOS_PYTHON:-python}"
source "$PROJECT/tools/run_env.sh"
OFFICE="${VOS_OFFICE_ROOT:-$PROJECT/data/office_loop}"
[[ -d "$OFFICE" ]] || { echo "Set VOS_OFFICE_ROOT to your own calibration sequence." >&2; exit 1; }
OUT=$PROJECT/runs/compare
mkdir -p "$OUT"
cd "$PROJECT" || exit 1
for backend in vggt omega; do
  name="office_loop_${backend}_loops0"
  if [ -e "$OUT/$name" ]; then echo "[driver] SKIP $name exists"; continue; fi
  echo "[driver] $(date -Is) BEGIN $name"
  run_env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 "$PY" run_slam.py "$OFFICE" --backend "$backend" \
    --max-loops 0 --seed 0 --output "$OUT/$name" > "$OUT/$name.console.log" 2>&1
  echo "[driver] $(date -Is) END $name exit=$?"
done
echo "[driver] done $(date -Is)"
