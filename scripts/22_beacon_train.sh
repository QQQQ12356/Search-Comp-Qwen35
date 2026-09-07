#!/usr/bin/env bash
# Qwen3.5 Beacon 搜索轨迹 SFT（tqdm 可视化）。
# 1) 构建检索语料（不存在时）；2) 构建交互式搜索轨迹；3) 训练。
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
ENV=${CONDA_ENV:-search-comp-qwen3.5}

CONFIG=${1:-configs/train/beacon_qwen3.5.yaml}
CORPUS=${CORPUS:-outputs/data/hotpotqa_corpus.jsonl}
TRAIN_DATA=${TRAIN_DATA:-outputs/data/hotpotqa_train_interactive.jsonl}
CORPUS_PER_SPLIT=${CORPUS_PER_SPLIT:-5000}
MAX_TRAIN=${MAX_TRAIN:-2000}
MAX_DOCS_TOKENS=${MAX_DOCS_TOKENS:-1024}
TOP_K=${TOP_K:-3}

mkdir -p outputs/data outputs/models

echo "[beacon] 1) 语料（不存在则构建）: $CORPUS"
if [ ! -f "$CORPUS" ]; then
  conda run -n "$ENV" python -m search_comp.data.build_corpus \
      --output_path "$CORPUS" --splits train,validation --max_per_split "$CORPUS_PER_SPLIT"
fi

echo "[beacon] 2) 构建交互式搜索轨迹 -> $TRAIN_DATA"
conda run -n "$ENV" python -m search_comp.data.build_interactive_data \
    --corpus_path "$CORPUS" --output_path "$TRAIN_DATA" --split train \
    --topk "$TOP_K" --max_questions "$MAX_TRAIN" --max_docs_tokens "$MAX_DOCS_TOKENS"

echo "[beacon] 3) 开始训练（tqdm 可视化）"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    conda run -n "$ENV" python -m search_comp.trainer.beacon_trainer --config "$CONFIG"
echo "[beacon] 训练完成"