"""Qwen3.5 原生交互式 SearchAgent 评估（EM/F1）。

复用 SearchAgent 探针的分段生成逻辑（think → <search> → observe → <answer>），
但改为：加载**训练后的文本模型**，在 HotpotQA validation 上跑检索 Agent 并
计算 EM / F1。输出 JSONL 到 ``output_path``，打印逐条结果与最终指标。
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from ..data.retrieval import BM25Retriever
from ..milestones.searchagent_probe import decode_until, run_searchagent_probe
from .em_f1 import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5 原生交互式 SearchAgent 评估")
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
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.manual_seed(args.seed)

    # 加载模型（优先文本 ForCausalLM，否则退化到多模态原生）
    try:
        from ..milestones.qwen35_text import load_text_tokenizer, load_text_causal_model

        tokenizer = load_text_tokenizer(args.model_path)
        model = load_text_causal_model(args.model_path)
    except Exception:  # noqa: BLE001
        from ..milestones.qwen35_native import load_chat_model, load_tokenizer

        tokenizer = load_tokenizer(args.model_path)
        model = load_chat_model(args.model_path)

    retriever = BM25Retriever(args.corpus_path)

    from datasets import load_dataset

    hp = load_dataset("hotpot_qa", "distractor", split=args.split)
    if args.max_questions:
        hp = hp.select(range(args.max_questions))

    print(f"\n=== Qwen3.5 原生交互式 SearchAgent 评估（{len(hp)} 题）===")
    results = []
    for i, ex in enumerate(hp):
        r = run_searchagent_probe(
            model, tokenizer, retriever, str(ex["question"]),
            max_turns=args.max_turns, topk=args.topk,
            max_docs_tokens=args.max_docs_tokens,
            do_sample=args.do_sample, temperature=args.temperature,
            verbosity=0,
        )
        r["id"] = str(ex["id"])
        r["ground_truth"] = str(ex["answer"]).strip()
        results.append(r)
        if (i + 1) % 20 == 0 or i == len(hp) - 1:
            print(f"  已生成 {i + 1}/{len(hp)}")

    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    triples = [(r["id"], r["prediction"], r["ground_truth"]) for r in results if r["ground_truth"]]
    metrics = compute_metrics(triples)
    metric_path = os.path.splitext(args.output_path)[0] + "_metrics.json"
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    n_search = sum(1 for r in results if r["turns"] > 0)
    n_multi = sum(1 for r in results if r["turns"] > 1)
    print(f"\n=== 结果 ===")
    print(f"EM={metrics['em']:.3f}  F1={metrics['f1']:.3f}  (valid={metrics.get('valid_count')})")
    print(f"触发搜索: {n_search} ({n_search/max(len(results),1):.0%}) | 多轮搜索(≥2): {n_multi} | 平均轮次 {sum(r['turns'] for r in results)/max(len(results),1):.2f}")
    print(f"结果 -> {args.output_path}\n指标 -> {metric_path}")


if __name__ == "__main__":
    main()