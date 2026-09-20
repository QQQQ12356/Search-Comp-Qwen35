#!/usr/bin/env bash
# Qwen3.5 Beacon 交互式 SearchAgent 评估（EM/F1）。
# 真实构建：语料库（不存在则构建）+ BM25 索引（BM25Retriever 内建），
# 真实交互式 Agent（think -> <search> -> <information> -> <answer>，Beacon 压缩生效）。
set -euo pipefail
source "$(dirname "$0")/common.sh"
PYTHON_BIN=$(resolve_python)

RATIO=${RATIO:-32}
MAX_QUESTIONS=${MAX_QUESTIONS:-1000}
MODEL_PATH=${1:-outputs/models/beacon_qwen3-4B-5000_ratio-${RATIO}_searchr1_intersect/final}
RESULT_PATH=${2:-outputs/results/beacon_qwen3-4B-ratio-${RATIO}_searchr1_intersect/distractor_final_predictions-${MAX_QUESTIONS}qa.jsonl}
shift $(( $# >= 2 ? 2 : $# ))
DATASET_NAME=${DATASET_NAME:-hotpotqa/hotpot_qa}
DATASET_CONFIG=${DATASET_CONFIG:-distractor}
CORPUS=${CORPUS:-outputs/data/hotpotqa_corpus.jsonl}
CORPUS_PER_SPLIT=${CORPUS_PER_SPLIT:-7405}
MAX_TURNS=${MAX_TURNS:-3}
TOP_K=${TOP_K:-3}


mkdir -p outputs/data outputs/results

echo "[beacon-eval] 语料库（不存在则构建）: $CORPUS"
if [ ! -f "$CORPUS" ]; then
  "$PYTHON_BIN" -u -m search_comp.data.build_corpus \
      --output_path "$CORPUS" --splits validation --max_per_split "$CORPUS_PER_SPLIT"
fi
echo "[beacon-eval] BM25 索引 + 交互式 Agent 评估（Beacon 压缩生效）"

EXTRA_ARGS=("$@")
if [[ ${RESUME:-0} == 1 ]]; then EXTRA_ARGS+=(--resume); fi
LOG_PATH=${LOG_PATH:-$(new_log_path beacon_eval)}
run_logged "$LOG_PATH" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} \
  "$PYTHON_BIN" -u -m search_comp.evaluation.beacon_interactive_eval \
  --model_path "$MODEL_PATH" --corpus_path "$CORPUS" --output_path "$RESULT_PATH" \
  --split validation --dataset_name "$DATASET_NAME" --dataset_config "$DATASET_CONFIG" \
  --max_questions "$MAX_QUESTIONS" --max_turns "$MAX_TURNS" --topk "$TOP_K" \
  --max_docs_tokens "${MAX_DOCS_TOKENS:-1024}" "${EXTRA_ARGS[@]}"
echo "[beacon-eval] 完成 -> $RESULT_PATH"
