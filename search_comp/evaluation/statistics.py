"""汇总 SearchAgent 预测文件的质量、搜索行为、耗时与压缩统计。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from collections import Counter
from typing import Any, Iterable

from .em_f1 import compute_metrics, extract_answer
from ..utils.runtime import write_json

#: 视为「未按格式作答」的占位预测（协议层面模型未闭合 <answer>）
NO_ANSWER_PLACEHOLDERS = {"", "[无作答]"}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} 不是合法 JSONL") from error
    return records


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(index, 0)]


def _is_format_correct(row: dict[str, Any]) -> bool:
    """判断该样本是否按协议闭合了 ``<answer>...</answer>``。

    优先用原始输出 ``output`` 判定（两条评测路径都会写入）；
    兼容缺失 ``output`` 的旧记录，退化为检查预测占位符。
    """
    output = row.get("output")
    if output is not None:
        return extract_answer(str(output)) is not None
    return str(row.get("prediction", "")).strip() not in NO_ANSWER_PLACEHOLDERS


def summarize_results(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """计算统一统计；缺失的可选字段不会影响 EM/F1。

    指标分两层：

    - **总体**（``overall_em`` / ``overall_f1``）：全部样本参与，
      未按格式作答的样本计 0 分，反映端到端真实效果；
    - **格式正确子集**（``formatted_em`` / ``formatted_f1``）：仅统计
      闭合了 ``<answer>...</answer>`` 的样本，衡量「会作答时的质量」，
      与 ``format_correct_rate``（格式正确率）配套。
    """
    rows = list(records)
    scored = [
        (str(row.get("id", "")), str(row.get("prediction", "")), str(row.get("ground_truth", "")))
        for row in rows
        if row.get("ground_truth") is not None
    ]
    formatted = [
        (str(row.get("id", "")), str(row.get("prediction", "")), str(row.get("ground_truth", "")))
        for row in rows
        if row.get("ground_truth") is not None and _is_format_correct(row)
    ]
    overall = compute_metrics(scored)
    formatted_quality = compute_metrics(formatted)
    turns = [int(row.get("turns", 0) or 0) for row in rows]
    latencies = [float(row["latency_seconds"]) for row in rows if row.get("latency_seconds") is not None]
    information_tokens = [int(row.get("information_tokens", 0) or 0) for row in rows]
    beacon_tokens = [int(row.get("beacon_tokens", 0) or 0) for row in rows]
    total_information = sum(information_tokens)
    total_beacons = sum(beacon_tokens)
    format_correct = sum(1 for row in rows if _is_format_correct(row))
    turns_histogram = {
        str(turn): count
        for turn, count in sorted(Counter(turns).items())
    }
    return {
        # 兼容别名：em / f1 即总体指标
        "em": overall["em"],
        "f1": overall["f1"],
        "valid_count": overall["valid_count"],
        "overall_em": overall["em"],
        "overall_f1": overall["f1"],
        "format_correct_samples": format_correct,
        "format_correct_rate": format_correct / len(rows) if rows else 0.0,
        "formatted_em": formatted_quality["em"],
        "formatted_f1": formatted_quality["f1"],
        "formatted_count": formatted_quality["valid_count"],
        "samples": len(rows),
        "answered": format_correct,
        "answer_rate": format_correct / len(rows) if rows else 0.0,
        "search_samples": sum(turn > 0 for turn in turns),
        "search_rate": sum(turn > 0 for turn in turns) / len(rows) if rows else 0.0,
        "multi_turn_samples": sum(turn >= 2 for turn in turns),
        "average_turns": mean(turns) if turns else 0.0,
        "turns_histogram": turns_histogram,
        "latency_seconds": {
            "total": sum(latencies),
            "mean": mean(latencies) if latencies else 0.0,
            "median": median(latencies) if latencies else 0.0,
            "p95": _percentile(latencies, 0.95),
        },
        "information_tokens": total_information,
        "beacon_tokens": total_beacons,
        "effective_information_compression_ratio": (
            total_information / total_beacons if total_beacons else None
        ),
    }


def summarize_file(path: str | Path) -> dict[str, Any]:
    summary = summarize_results(load_jsonl(path))
    summary["result_path"] = str(path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 SearchAgent 评测 JSONL")
    parser.add_argument(
        "--result_path", action="append", required=True,
        help="预测 JSONL，可重复传入以比较多个实验",
    )
    parser.add_argument("--output_path", required=True, help="汇总 JSON 输出路径")
    args = parser.parse_args()

    summaries = [summarize_file(path) for path in args.result_path]
    payload = {"experiments": summaries}
    write_json(args.output_path, payload)
    for summary in summaries:
        print(
            f"[stats] {summary['result_path']} samples={summary['samples']} "
            f"总体EM={summary['overall_em']:.4f} 总体F1={summary['overall_f1']:.4f} | "
            f"格式正确率={summary['format_correct_rate']:.1%} "
            f"子集EM={summary['formatted_em']:.4f} 子集F1={summary['formatted_f1']:.4f} | "
            f"search={summary['search_rate']:.1%} turns={summary['average_turns']:.2f} "
            f"turns分布={summary['turns_histogram']} "
            f"compression={summary['effective_information_compression_ratio']}",
            flush=True,
        )
    print(f"[stats] 汇总已保存 -> {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
