#!/usr/bin/env bash
# GPU stages of the undistortion test, in order: (1) single-forward analysis on undistorted frames (both backends, one
# process), (2) the 12 SLAM runs. Sequential; each process takes the vos GPU lock itself.
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${VOS_PYTHON:-python}"
source "$PROJECT/tools/run_env.sh"
cd "$PROJECT" || exit 1
mkdir -p "$PROJECT/runs/undistort/analysis"
echo "[gpu-driver] forward start $(date -Is)"
run_env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VOS_GPU_WAIT_S=3600 \
  "$PY" tools/analysis_undistort.py forward > "$PROJECT/runs/undistort/analysis/forward.console.log" 2>&1
echo "[gpu-driver] forward end exit=$? $(date -Is)"
bash "$PROJECT/tools/run_undistort_slam.sh" > "$PROJECT/runs/undistort/slam_driver.log" 2>&1
echo "[gpu-driver] slam driver end exit=$? $(date -Is)"
