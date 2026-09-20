#!/usr/bin/env bash
# Qwen3.5 Beacon 训练 —— Search-R1 官方 SFT 轨迹（messages 格式，无需构建数据）。
# 只压缩 <information> 检索文档块；损失只在 assistant 生成片段上计算。
set -euo pipefail
source "$(dirname "$0")/common.sh"
# PyTorch 2.7.0+cu126 自带 cuDNN 9.5。若系统里另有旧版 cuDNN 通过
# LD_LIBRARY_PATH 抢先加载，会覆盖 PyTorch 自带版本并导致运行时崩溃。
# 需要屏蔽时导出要剔除的目录（多个用 ':' 分隔），例如：
#   export SHADOWED_LD_PATHS=/opt/cudnn9.0/lib
# 未设置时不改动 LD_LIBRARY_PATH。
if [[ -n ${SHADOWED_LD_PATHS:-} && -n ${LD_LIBRARY_PATH:-} ]]; then
  IFS=: read -r -a _ld_library_paths <<< "$LD_LIBRARY_PATH"
  _kept_ld_library_paths=()
  for _ld_library_path in "${_ld_library_paths[@]}"; do
    if [[ ":${SHADOWED_LD_PATHS}:" == *":${_ld_library_path}:"* ]]; then
      echo "[beacon-searchr1] 已从 LD_LIBRARY_PATH 剔除: $_ld_library_path"
      continue
    fi
    _kept_ld_library_paths+=("$_ld_library_path")
  done
  LD_LIBRARY_PATH=$(IFS=:; printf '%s' "${_kept_ld_library_paths[*]:-}")
  export LD_LIBRARY_PATH
fi
PYTHON_BIN=$(resolve_python)
CONFIG=${1:-configs/train/beacon_qwen35_searchr1.yaml}
if [[ $# -gt 0 ]]; then shift; fi
mkdir -p outputs/models
echo "[beacon-searchr1] 训练数据: Search-R1 messages 轨迹（默认 outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl）"
LOG_PATH=${LOG_PATH:-$(new_log_path beacon_searchr1_train)}
run_logged "$LOG_PATH" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  "$PYTHON_BIN" -u -m search_comp.trainer.beacon_trainer --config "$CONFIG" "$@"
echo "[beacon-searchr1] 训练完成"
