#!/usr/bin/env bash
# 用法：bash scripts/26_export_excel.sh OUTPUT.xlsx RESULT1.jsonl [RESULT2.jsonl ...]
# 汇总评测预测 JSONL（plain / beacon 均可）为 Excel 对比表。
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [[ $# -lt 2 ]]; then
  echo "用法: $0 OUTPUT.xlsx RESULT1.jsonl [RESULT2.jsonl ...]" >&2
  exit 2
fi

OUTPUT_PATH=$1
shift
PYTHON_BIN=$(resolve_python)
LOG_PATH=${LOG_PATH:-$(new_log_path export_excel)}
ARGS=()
for result_path in "$@"; do
  ARGS+=(--result_path "$result_path")
done

run_logged "$LOG_PATH" "$PYTHON_BIN" -u -m search_comp.evaluation.export_excel \
    "${ARGS[@]}" --output_path "$OUTPUT_PATH"
