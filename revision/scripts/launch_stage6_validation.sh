#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
GPU_LIST="${STAGE6_GPUS:-0}"
PARALLEL_RUNS="${STAGE6_PARALLEL:-1}"
PIPELINE_LOG="${PROJECT_ROOT}/revision/logs/stage6_validation/pipeline.log"

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname -- "${PIPELINE_LOG}")"

echo "Running common metrics on the fixed 1,836-record validation set."
echo "GPUs: ${GPU_LIST}; parallel jobs: ${PARALLEL_RUNS}."
echo "The final test set remains locked."

conda run --no-capture-output -n boltz_finetune \
  env PYTHONPATH=boltz/src \
  python revision/scripts/run_stage6_validation.py \
  --gpus "${GPU_LIST}" \
  --parallel "${PARALLEL_RUNS}" \
  "$@" 2>&1 | tee -a "${PIPELINE_LOG}"
