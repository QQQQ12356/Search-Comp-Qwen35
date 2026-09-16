"""Qwen3.5 Beacon 交互式 SearchAgent 评估（EM/F1）。

对训练后的 Beacon 模型跑 Search-R1 风格交互式搜索：模型用 ``beacon_generate``
分段生成 ``think → <search> → <information> → <answer>``，检索文档块在生成时被
压缩为 beacon K/V。输出 JSONL 并计算 EM/F1。

用法::

    python -m search_comp.evaluation.beacon_interactive_eval \
        --model_path outputs/models/beacon_qwen3_sft_v1/final \
        --corpus_path outputs/data/hotpotqa_corpus.jsonl \
        --output_path outputs/results/beacon_qwen3_v1/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from ..data.build_sft_data import truncate_docs_by_tokens
from ..data.retrieval import BM25Retriever, format_docs_as_reference
from ..data.trajectory import (
    INFO_PREFIX,
    INFO_SUFFIX,
    build_search_chat_prompt,
    extract_search_query,
)
from ..models.beacon_config import BeaconConfig
from .em_f1 import extract_answer
from .statistics import summarize_results
from ..utils.runtime import append_jsonl, write_json

# 模型未闭合 <answer>...</answer> 时的占位答案（按协议视为未作答，而非整段轨迹兜底）
NO_ANSWER = "[无作答]"


def run_beacon_agent(model, tokenizer, retriever, question, max_turns=3, topk=3,
                     max_docs_tokens=1024, max_new_tokens_per_turn=256) -> Dict[str, Any]:
    question = str(question).strip()
    if not question.endswith("?"):
        question += "?"

    chat_prefix = build_search_chat_prompt(question, add_generation_prompt=True)
    context_ids = tokenizer(chat_prefix, add_special_tokens=False).input_ids
    regions: List[tuple] = []
    queries: List[str] = []
    turns = 0

    device = next(model.parameters()).device
    for _ in range(max_turns):
        ids = torch.tensor([context_ids], dtype=torch.long, device=device)
        attn = torch.ones(1, len(context_ids), dtype=torch.long, device=device)
        gen_ids = model.beacon_generate(
            input_ids=ids, attention_mask=attn, regions=regions,
            max_new_tokens=max_new_tokens_per_turn,
            stop_texts=["</search>", "</answer>"], tokenizer=tokenizer,
        )
        gen_tokens = gen_ids[0].tolist()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        context_ids = context_ids + gen_tokens

        if "</search>" in gen_text:
            query = extract_search_query(gen_text)
            if not query:
                break
            queries.append(query)
            turns += 1
            retrieved = retriever.retrieve(query, topk=topk)
            docs_text = truncate_docs_by_tokens(
                format_docs_as_reference(retrieved), tokenizer, max_docs_tokens
            )
            ip = tokenizer(INFO_PREFIX, add_special_tokens=False).input_ids
            di = tokenizer(docs_text, add_special_tokens=False).input_ids
            isuf = tokenizer(INFO_SUFFIX, add_special_tokens=False).input_ids
            docs_start = len(context_ids) + len(ip)
            context_ids = context_ids + ip + di + isuf
            regions.append((docs_start, docs_start + len(di)))
        else:
            break

    assistant_text = tokenizer.decode(
        context_ids[len(tokenizer(chat_prefix, add_special_tokens=False).input_ids):],
        skip_special_tokens=True,
    )
    prediction = extract_answer(assistant_text) or NO_ANSWER
    information_tokens = sum(end - start for start, end in regions)
    ratio = model.beacon_config.beacon_ratio
    beacon_tokens = sum(max(1, (end - start + ratio - 1) // ratio) for start, end in regions)
    return {
        "prediction": prediction,
        "turns": turns,
        "queries": queries,
        "output": assistant_text,
        "regions": regions,
        "context_tokens": len(context_ids),
        "information_tokens": information_tokens,
        "beacon_tokens": beacon_tokens,
        "generated_tokens": len(tokenizer(assistant_text, add_special_tokens=False).input_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5 Beacon 交互式 SearchAgent 评估")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_questions", type=int, default=100)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_docs_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens_per_turn", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", action="store_true",
        help="保留已有 JSONL，并跳过其中已完成的 id",
    )
    args = parser.parse_args()

    from ..models.beacon_qwen3 import load_beacon_qwen3_5
    from ..milestones.qwen35_text import load_text_tokenizer

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.manual_seed(args.seed)
    tokenizer = load_text_tokenizer(args.model_path)
    model = load_beacon_qwen3_5(args.model_path)  # 从保存的 config 读取 beacon 字段
    model.eval()

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
    print(f"[beacon-eval] 总题数={len(hp)} 已完成={len(completed)}", flush=True)
    results = list(completed.values())
    pbar = tqdm(hp, desc="eval", ncols=100)
    for ex in pbar:
        example_id = str(ex["id"])
        if example_id in completed:
            continue
        started_at = time.time()
        r = run_beacon_agent(model, tokenizer, retriever, ex["question"],
                             max_turns=args.max_turns, topk=args.topk,
                             max_docs_tokens=args.max_docs_tokens,
                             max_new_tokens_per_turn=args.max_new_tokens_per_turn)
        r["id"] = example_id
        r["ground_truth"] = str(ex["answer"]).strip()
        r["latency_seconds"] = round(time.time() - started_at, 4)
        results.append(r)
        append_jsonl(args.output_path, r)
        pbar.set_postfix(turns=r["turns"], pred=r["prediction"][:30])

    m = summarize_results(results)
    mp = os.path.splitext(args.output_path)[0] + "_metrics.json"
    write_json(mp, {**m, "evaluation_config": vars(args)})
    print(f"[beacon-eval] EM={m['em']:.3f} F1={m['f1']:.3f} -> {args.output_path}")


if __name__ == "__main__":
    main()
