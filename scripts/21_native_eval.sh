#!/usr/bin/env bash
# Qwen3.5 原生交互式 SearchAgent 评估（EM/F1）。
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
ENV=${CONDA_ENV:-search-comp-qwen3.5}
# 直接用该 conda 环境 python（避免 conda run 的子进程与输出缓冲，保证实时进度）
PYTHON=${PYTHON:-"$(conda info --base)/envs/${ENV}/bin/python"}
[ -x "$PYTHON" ] || PYTHON=python

MODEL_PATH=${1:-outputs/models/native_qwen3_sft_v1/final}
RESULT_PATH=${2:-outputs/results/native_qwen3_sft_v1/predictions.jsonl}
CORPUS=${CORPUS:-outputs/data/hotpotqa_corpus.jsonl}
MAX_QUESTIONS=${MAX_QUESTIONS:-200}
MAX_TURNS=${MAX_TURNS:-3}
TOP_K=${TOP_K:-3}
MAX_DOCS_TOKENS=${MAX_DOCS_TOKENS:-1024}
DO_SAMPLE=${DO_SAMPLE:-0}

mkdir -p outputs/results
SAMPLE_FLAG=""
if [ "$DO_SAMPLE" = "1" ]; then SAMPLE_FLAG="--do_sample --temperature 0.8"; fi

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    "$PYTHON" -u -m search_comp.evaluation.native_interactive_eval \
    --model_path "$MODEL_PATH" \
    --corpus_path "$CORPUS" \
    --output_path "$RESULT_PATH" \
    --split validation \
    --max_questions "$MAX_QUESTIONS" \
    --max_turns "$MAX_TURNS" --topk "$TOP_K" \
    --max_docs_tokens "$MAX_DOCS_TOKENS" \
    $SAMPLE_FLAG
echo "[eval] 完成 -> $RESULT_PATH"