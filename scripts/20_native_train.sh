#!/usr/bin/env bash
# Qwen3.5 原生搜索轨迹 SFT（标准 Trainer + tqdm 可视化）。
# 1) 构建交互式搜索轨迹数据；2) 标准 Trainer 训练并可视化到终端。
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
ENV=${CONDA_ENV:-search-comp-qwen3.5}
# 直接用该 conda 环境 python（避免 conda run 的子进程与输出缓冲，保证实时进度）
PYTHON=${PYTHON:-"$(conda info --base)/envs/${ENV}/bin/python"}
[ -x "$PYTHON" ] || PYTHON=python

CONFIG=${1:-configs/train/native_qwen3.5.yaml}
MODEL=${MODEL:-Qwen/Qwen3.5-2B}
CORPUS=${CORPUS:-outputs/data/hotpotqa_corpus.jsonl}
CORPUS_SPLIT=${CORPUS_SPLIT:-train,validation}
CORPUS_PER_SPLIT=${CORPUS_PER_SPLIT:-5000}
TRAIN_DATA=${TRAIN_DATA:-outputs/data/hotpotqa_train_interactive.jsonl}
TOP_K=${TOP_K:-3}
MAX_TRAIN=${MAX_TRAIN:-2000}
MAX_DOCS_TOKENS=${MAX_DOCS_TOKENS:-1024}
MAX_LENGTH=${MAX_LENGTH:-8192}

mkdir -p outputs/data outputs/models

echo "[native] 语料（不存在则构建）: $CORPUS"
if [ ! -f "$CORPUS" ]; then
  "$PYTHON" -u -m search_comp.milestones.build_small_corpus \
      --output_path "$CORPUS" --splits "$CORPUS_SPLIT" --max_per_split "$CORPUS_PER_SPLIT"
fi

echo "[native] 构建交互式搜索轨迹 -> $TRAIN_DATA"
"$PYTHON" -u -m search_comp.data.build_interactive_data \
    --corpus_path "$CORPUS" --output_path "$TRAIN_DATA" --split train \
    --topk "$TOP_K" --max_questions "$MAX_TRAIN" --max_docs_tokens "$MAX_DOCS_TOKENS"

echo "[native] 开始训练（标准 Trainer，可视化到终端）"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    "$PYTHON" -u -m search_comp.trainer.native_trainer --config "$CONFIG"
echo "[native] 训练完成"