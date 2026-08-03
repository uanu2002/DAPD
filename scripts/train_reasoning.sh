#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the base model}"
: "${DATASET:?Set DATASET to a training parquet file or dataset ID}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for LoRA checkpoints}"

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
NUM_PROCESSES=${NUM_PROCESSES:-4}
MICRO_BATCH=${MICRO_BATCH:-2}
MASTER_PORT=${MASTER_PORT:-29500}

if [[ -z "${GRAD_ACC:-}" ]]; then
  DENOMINATOR=$((NUM_PROCESSES * MICRO_BATCH))
  if ((32 % DENOMINATOR != 0)); then
    echo "NUM_PROCESSES * MICRO_BATCH must divide effective batch size 32" >&2
    exit 2
  fi
  GRAD_ACC=$((32 / DENOMINATOR))
fi

export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export USE_TF=0
export USE_FLAX=0
export TRL_EXPERIMENTAL_SILENCE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

cd "$ROOT"
exec torchrun \
  --nnodes 1 \
  --nproc_per_node "$NUM_PROCESSES" \
  --master_port "$MASTER_PORT" \
  train.py \
  --model "$MODEL_PATH" \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT_DIR" \
  --per-device-batch-size "$MICRO_BATCH" \
  --gradient-accumulation-steps "$GRAD_ACC"
