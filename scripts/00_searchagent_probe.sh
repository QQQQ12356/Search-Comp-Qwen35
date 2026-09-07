#!/usr/bin/env bash
# Milestone-1：Qwen3.5-2B SearchAgent 检索能力探针。
# 1) 构建小规模 HotpotQA 语料 + 简单 BM25 索引；2) 原生生成跑若干条轨迹，
# 验证基础模型是否支持 think → <search> → observe → <answer> 检索行为。
set -euo pipefail
cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
ENV=${CONDA_ENV:-search-comp-qwen3.5}

MODEL=${MODEL:-Qwen/Qwen3.5-2B}
CORPUS=${CORPUS:-outputs/data/hotpotqa_corpus_small.jsonl}
OUTPUT=${OUTPUT:-outputs/results/searchagent_probe/predictions.jsonl}
CORPUS_SPLIT=${CORPUS_SPLIT:-train,validation}
CORPUS_PER_SPLIT=${CORPUS_PER_SPLIT:-200}
MAX_QUESTIONS=${MAX_QUESTIONS:-10}
TOP_K=${TOP_K:-3}
MAX_TURNS=${MAX_TURNS:-3}
MAX_DOCS_TOKENS=${MAX_DOCS_TOKENS:-1024}
DO_SAMPLE=${DO_SAMPLE:-0}

mkdir -p outputs/data outputs/results

echo "[probe] 构建小规模语料 -> $CORPUS"
conda run -n "$ENV" python -m search_comp.milestones.build_small_corpus \
    --output_path "$CORPUS" \
    --splits "$CORPUS_SPLIT" \
    --max_per_split "$CORPUS_PER_SPLIT"

echo "[probe] 运行 SearchAgent 探针 (max_questions=$MAX_QUESTIONS)"
SAMPLE_FLAG=""
if [ "$DO_SAMPLE" = "1" ]; then SAMPLE_FLAG="--do_sample --temperature 1.0"; fi
# shellcheck disable=SC2086
conda run -n "$ENV" python -m search_comp.milestones.searchagent_probe \
    --model_path "$MODEL" \
    --corpus_path "$CORPUS" \
    --output_path "$OUTPUT" \
    --split validation \
    --max_questions "$MAX_QUESTIONS" \
    --topk "$TOP_K" \
    --max_turns "$MAX_TURNS" \
    --max_docs_tokens "$MAX_DOCS_TOKENS" \
    $SAMPLE_FLAG

echo "[probe] 完成 -> $OUTPUT"