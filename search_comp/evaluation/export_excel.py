"""把一份或多份评测预测 JSONL 汇总导出为 Excel 对比表。

复用 :func:`search_comp.evaluation.statistics.summarize_file` 计算指标，
每个 JSONL 一行，字段包括：测试样本数、总体 EM/F1、格式正确率、
格式正确子集 EM/F1、检索轮次与压缩比。

用法::

    python -m search_comp.evaluation.export_excel \
        --result_path a/predictions.jsonl --result_path b/predictions.jsonl \
        --output_path outputs/results/summary.xlsx
"""

from __future__ import annotations

import argparse

import pandas as pd

from .statistics import summarize_file

#: Excel 列（指标字段名与 summarize_results 输出保持一致）
EXPORT_COLUMNS = {
    "result_path": "结果文件",
    "samples": "测试样本数",
    "overall_em": "总体EM",
    "overall_f1": "总体F1",
    "format_correct_rate": "格式正确率",
    "formatted_em": "格式正确EM",
    "formatted_f1": "格式正确F1",
    "average_turns": "平均检索轮次",
    "effective_information_compression_ratio": "压缩比",
}


def export_excel(result_paths: list[str], output_path: str) -> pd.DataFrame:
    """汇总各 result JSONL 并写出 Excel；返回汇总 DataFrame。"""
    if not result_paths:
        raise ValueError("至少需要一个 --result_path")
    summaries = [summarize_file(path) for path in result_paths]
    frame = pd.DataFrame(summaries).rename(columns=EXPORT_COLUMNS)
    frame = frame[list(EXPORT_COLUMNS.values())]
    frame.to_excel(output_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总评测 JSONL 并导出 Excel")
    parser.add_argument(
        "--result_path", action="append", required=True,
        help="预测 JSONL，可重复传入以比较多个实验",
    )
    parser.add_argument("--output_path", required=True, help="Excel 输出路径（.xlsx）")
    args = parser.parse_args()

    frame = export_excel(args.result_path, args.output_path)
    print(frame.to_string(index=False))
    print(f"[export-excel] 已保存 -> {args.output_path}")


if __name__ == "__main__":
    main()
