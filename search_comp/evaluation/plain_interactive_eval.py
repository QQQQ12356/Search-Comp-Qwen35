"""Qwen3.5 纯文本（无 Beacon）交互式 SearchAgent 评估（EM/F1）。

对应用户要求的「推理测试文件」：加载训练后的纯文本模型，在 HotpotQA
validation 上跑真实检索 Agent（think → <search> → <information> → <answer>），
计算 EM / F1。与 Beacon 的 :mod:`search_comp.evaluation.beacon_interactive_eval`
做同源对比——唯一区别是本路径**不做压缩**，模型直接吃完整上下文。

复用现有模块：检索、推理与指标分别来自
:class:`search_comp.data.retrieval.BM25Retriever`、
:func:`search_comp.milestones.searchagent_probe.run_searchagent_probe`、
:func:`search_comp.evaluation.statistics.summarize_results`。

输出 JSONL 到 ``output_path``，打印逐条结果与最终指标。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from tqdm import tqdm

from ..data.retrieval import BM25Retriever
from ..milestones.searchagent_probe import run_searchagent_probe
from ..models.plain_qwen3 import load_sft_model, load_sft_tokenizer
from .statistics import summarize_results
from ..utils.runtime import append_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5 纯文本交互式 SearchAgent 评估")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_questions", type=int, default=200)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--max_docs_tokens", type=int, default=1024)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", action="store_true",
        help="保留已有 JSONL，并跳过其中已完成的 id",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.manual_seed(args.seed)

    # 加载训练后的纯文本模型（既支持基础多模态 checkpoint，也支持 LoRA 合并后
    # 的 Qwen3_5ForCausalLM checkpoint）。
    tokenizer = load_sft_tokenizer(args.model_path)
    model = load_sft_model(args.model_path)

    retriever = BM25Retriever(args.corpus_path)

    from datasets import load_dataset

    hp = load_dataset("hotpot_qa", "distractor", split=args.split)
    if args.max_questions:
        hp = hp.select(range(args.max_questions))

    completed = {}
    if args.resume and os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as existing_file:
            for line in existing_file:
                if line.strip():
                    row = json.loads(line)
                    completed[str(row.get("id"))] = row
    elif os.path.exists(args.output_path):
        os.remove(args.output_path)

    print(f"\n[plain-eval] 总题数={len(hp)} 已完成={len(completed)}", flush=True)
    results = list(completed.values())
    progress = tqdm(hp, desc="plain-eval", ncols=100)
    for ex in progress:
        example_id = str(ex["id"])
        if example_id in completed:
            continue
        started_at = time.time()
        r = run_searchagent_probe(
            model, tokenizer, retriever, str(ex["question"]),
            max_turns=args.max_turns, topk=args.topk,
            max_docs_tokens=args.max_docs_tokens,
            do_sample=args.do_sample, temperature=args.temperature,
            verbosity=0,
        )
        r["id"] = example_id
        r["question"] = str(ex["question"]).strip()
        r["ground_truth"] = str(ex["answer"]).strip()
        r["latency_seconds"] = round(time.time() - started_at, 4)
        regions = r.get("regions", [])
        r["information_tokens"] = sum(end - start for start, end in regions)
        r["generated_tokens"] = len(
            tokenizer(str(r.get("output", "")), add_special_tokens=False).input_ids
        )
        results.append(r)
        append_jsonl(args.output_path, r)
        progress.set_postfix(turns=r.get("turns", 0), pred=str(r.get("prediction", ""))[:24])

    metrics = summarize_results(results)
    metric_path = os.path.splitext(args.output_path)[0] + "_metrics.json"
    write_json(metric_path, {**metrics, "evaluation_config": vars(args)})

    print(f"\n=== 结果 ===")
    print(f"EM={metrics['em']:.3f}  F1={metrics['f1']:.3f}  (valid={metrics.get('valid_count')})")
    print(
        f"触发搜索: {metrics['search_samples']} ({metrics['search_rate']:.0%}) | "
        f"多轮搜索(≥2): {metrics['multi_turn_samples']} | "
        f"平均轮次 {metrics['average_turns']:.2f}"
    )
    print(f"结果 -> {args.output_path}\n指标 -> {metric_path}")


if __name__ == "__main__":
    main()