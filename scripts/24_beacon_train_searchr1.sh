#!/usr/bin/env bash
# Qwen3.5 Beacon 训练 —— Search-R1 官方 SFT 轨迹（messages 格式，无需构建数据）。
# 只压缩 <information> 检索文档块；损失只在 assistant 生成片段上计算。
set -euo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
ENV=${CONDA_ENV:-search-comp-qwen3.5}
# 直接用该 conda 环境 python（避免 conda run 的子进程与输出缓冲，保证实时进度）
PYTHON=${PYTHON:-"$(conda info --base)/envs/${ENV}/bin/python"}
[ -x "$PYTHON" ] || PYTHON=python
CONFIG=${1:-configs/train/beacon_qwen3.5_searchr1.yaml}
mkdir -p outputs/models
echo "[beacon-searchr1] 训练数据: qwen3-4b-instruct-sft.jsonl（messages 格式）"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} \
  "$PYTHON" -u -m search_comp.trainer.beacon_trainer --config "$CONFIG"
echo "[beacon-searchr1] 训练完成"