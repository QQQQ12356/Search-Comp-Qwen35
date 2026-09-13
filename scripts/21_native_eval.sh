#!/usr/bin/env bash
# Qwen3.5 原生交互式 SearchAgent 评估（EM/F1）。
set -euo pipefail
source "$(dirname "$0")/common.sh"
PYTHON_BIN=$(resolve_python)

MODEL_PATH=${1:-outputs/models/native_qwen3_sft_v1/final}
RESULT_PATH=${2:-outputs/results/native_qwen3_sft_v1/predictions.jsonl}
shift $(( $# >= 2 ? 2 : $# ))
CORPUS=${CORPUS:-outputs/data/hotpotqa_corpus.jsonl}
MAX_QUESTIONS=${MAX_QUESTIONS:-200}
MAX_TURNS=${MAX_TURNS:-3}
TOP_K=${TOP_K:-3}
MAX_DOCS_TOKENS=${MAX_DOCS_TOKENS:-1024}
DO_SAMPLE=${DO_SAMPLE:-0}

mkdir -p outputs/results
SAMPLE_ARGS=()
if [ "$DO_SAMPLE" = "1" ]; then SAMPLE_ARGS=(--do_sample --temperature 0.8); fi

EXTRA_ARGS=("$@")
if [[ ${RESUME:-0} == 1 ]]; then EXTRA_ARGS+=(--resume); fi
LOG_PATH=${LOG_PATH:-$(new_log_path native_eval)}
run_logged "$LOG_PATH" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    "$PYTHON_BIN" -u -m search_comp.evaluation.native_interactive_eval \
    --model_path "$MODEL_PATH" \
    --corpus_path "$CORPUS" \
    --output_path "$RESULT_PATH" \
    --split validation \
    --max_questions "$MAX_QUESTIONS" \
    --max_turns "$MAX_TURNS" --topk "$TOP_K" \
    --max_docs_tokens "$MAX_DOCS_TOKENS" \
    "${SAMPLE_ARGS[@]}" "${EXTRA_ARGS[@]}"
echo "[eval] 完成 -> $RESULT_PATH"
