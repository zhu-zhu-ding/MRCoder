#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${TARGET_MODEL_PATH:-}}"
INPUT_JSONL="${INPUT_JSONL:-${REPO_ROOT}/benchmark/generation/DevEval-main/data/deveval_bm25.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/baseline}"
OUTPUT_NAME="${OUTPUT_NAME:-baseline_completion.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-0.95}"
N="${N:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a code completion assistant.}"

if [[ -z "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH is required. Set MODEL_PATH or TARGET_MODEL_PATH." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

python3 "${SCRIPT_DIR}/run_vllm_batch_inference.py" \
    --model-path "${MODEL_PATH}" \
    --input-file "${INPUT_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    --output-name "${OUTPUT_NAME}" \
    --batch-size "${BATCH_SIZE}" \
    --max-tokens "${MAX_TOKENS}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --n "${N}" \
    --system-prompt "${SYSTEM_PROMPT}" \
    $( [ -n "${LORA_PATH:-}" ] && echo --lora-path "${LORA_PATH}" ) \
    $( [ "${TRUST_REMOTE_CODE}" = "true" ] && echo "--trust-remote-code" )
