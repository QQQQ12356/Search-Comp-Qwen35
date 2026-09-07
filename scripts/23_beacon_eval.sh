#!/usr/bin/env bash
# Qwen3.5 Beacon 交互式 SearchAgent 评估（EM/F1）。
# 真实构建：语料库（不存在则构建）+ BM25 索引（BM25Retriever 内建），
# 真实交互式 Agent（think -> <search> -> <information> -> <answer>，Beacon 压缩生效）。
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
ENV=${CONDA_ENV:-search-comp-qwen3.5}

MODEL_PATH=${1:-outputs/models/beacon_qwen3_searchr1_v1/final}
RESULT_PATH=${2:-outputs/results/beacon_qwen3_searchr1_v1/predictions.jsonl}
CORPUS=${CORPUS:-outputs/data/hotpotqa_corpus.jsonl}
CORPUS_PER_SPLIT=${CORPUS_PER_SPLIT:-5000}
MAX_QUESTIONS=${MAX_QUESTIONS:-100}
MAX_TURNS=${MAX_TURNS:-3}
TOP_K=${TOP_K:-3}

mkdir -p outputs/data outputs/results

echo "[beacon-eval] 语料库（不存在则构建）: $CORPUS"
if [ ! -f "$CORPUS" ]; then
  conda run -n "$ENV" python -m search_comp.data.build_corpus \
      --output_path "$CORPUS" --splits train,validation --max_per_split "$CORPUS_PER_SPLIT"
fi
echo "[beacon-eval] BM25 索引 + 交互式 Agent 评估（Beacon 压缩生效）"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  conda run -n "$ENV" python -m search_comp.evaluation.beacon_interactive_eval \
  --model_path "$MODEL_PATH" --corpus_path "$CORPUS" --output_path "$RESULT_PATH" \
  --split validation --max_questions "$MAX_QUESTIONS" --max_turns "$MAX_TURNS" --topk "$TOP_K"
echo "[beacon-eval] 完成 -> $RESULT_PATH"