#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
GPU_LIST="${STAGE5_GPUS:-0,1}"
PARALLEL_RUNS="${STAGE5_PARALLEL:-2}"
PIPELINE_LOG="${PROJECT_ROOT}/revision/logs/stage5/pipeline_screening.log"

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname -- "${PIPELINE_LOG}")"

echo "Starting the seven-run stage-5 core suite on GPUs ${GPU_LIST}."
echo "At most ${PARALLEL_RUNS} models will run concurrently."
echo "Pipeline log: ${PIPELINE_LOG}"

conda run --no-capture-output -n boltz_finetune \
  python revision/scripts/run_stage5_pipeline.py \
  --suite screening \
  --phase train \
  --gpus "${GPU_LIST}" \
  --parallel "${PARALLEL_RUNS}" \
  "$@" 2>&1 | tee -a "${PIPELINE_LOG}"
