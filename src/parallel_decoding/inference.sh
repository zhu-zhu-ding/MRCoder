#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INFERENCE_TYPE="${INFERENCE_TYPE:-ar}"
DATA_FILE="${DATA_FILE:-${REPO_ROOT}/benchmark/generation/DevEval-main/data/deveval_bm25.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-${REPO_ROOT}/outputs/parallel_decoding/results.jsonl}"
TARGET_MODEL="${TARGET_MODEL:-${TARGET_MODEL_PATH:-}}"
DRAFT_MODEL="${DRAFT_MODEL:-${DRAFT_MODEL_PATH:-}}"

if [[ -z "${TARGET_MODEL}" || -z "${DRAFT_MODEL}" ]]; then
    echo "TARGET_MODEL and DRAFT_MODEL are required. Set them directly or export TARGET_MODEL_PATH and DRAFT_MODEL_PATH." >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_FILE}")"

python "${SCRIPT_DIR}/inference.py" \
    --inference_type "${INFERENCE_TYPE}" \
    --data_file "${DATA_FILE}" \
    --output_file "${OUTPUT_FILE}" \
    --target_model "${TARGET_MODEL}" \
    --draft_model "${DRAFT_MODEL}"
