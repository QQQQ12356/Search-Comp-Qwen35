"""构建交互式搜索的 SFT 轨迹数据。

利用 HotpotQA 的 ``supporting_facts``（金标准支撑文档标题）构造多轮搜索轨迹：

- **turn 1**：query = 问题，docs = 金标准支撑文档 + BM25 检索结果（金标准在前）。
- **turn 2**（仅 bridge 型且第 2 个支撑文档标题出现在第 1 个文档正文中）：
  query = 第 2 个支撑文档标题，docs = 该文档 + BM25 检索结果。
  该启发式保证了轨迹的"真实性"：模型在第 1 轮检索结果中发现了缺失实体
  （第 2 个文档的标题），从而发起第 2 次搜索。
- think 文本为模板化生成，最终给出金标准答案。

输出每行：
``{id, question, answer, turns: [{query, docs}], thinks: [...], final_think}``
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from ..milestones.qwen35_native import load_tokenizer
from .retrieval import BM25Retriever, format_docs_as_reference
from .build_sft_data import truncate_docs_by_tokens

#: 每轮检索返回的最大文档数（对齐 Search-R1 的 topk=3）
DEFAULT_TOPK = 3


def _clean_question(question: str) -> str:
    """清洗问题：去掉结尾问号后补回，保证格式统一。"""
    q = str(question).strip()
    if q.endswith("?"):
        q = q[:-1].strip()
    return q + "?"


def _lookup_gold_docs(
    ex: Dict[str, Any], corpus_by_title: Dict[str, Dict[str, str]]
) -> List[Dict[str, str]]:
    """按 supporting_facts 标题取金标准文档（保持顺序、去重）。"""
    titles: List[str] = []
    for t in ex["supporting_facts"]["title"]:
        if t not in titles:
            titles.append(t)
    docs = []
    for t in titles:
        if t in corpus_by_title:
            docs.append(corpus_by_title[t])
    return docs


def _merge_docs(
    gold_docs: List[Dict[str, str]], bm25_docs: List[Dict[str, str]], topk: int
) -> List[Dict[str, str]]:
    """金标准文档在前 + BM25 结果，按 id 去重，截断到 topk。"""
    seen = set()
    pool: List[Dict[str, str]] = []
    for d in list(gold_docs) + list(bm25_docs):
        if d["id"] not in seen:
            seen.add(d["id"])
            pool.append(d)
    return pool[:topk]


def build_trajectory(
    ex: Dict[str, Any],
    retriever: BM25Retriever,
    corpus_by_title: Dict[str, Dict[str, str]],
    topk: int,
    max_docs_tokens: Optional[int],
    tokenizer,
) -> Optional[Dict[str, Any]]:
    """为单个 HotpotQA 样本构造搜索轨迹。

    Args:
        ex: HotpotQA 样本。
        retriever: BM25 检索器。
        corpus_by_title: ``title -> 文档`` 的映射（供金标准文档查询）。
        topk: 每轮检索文档数。
        max_docs_tokens: 每轮 docs 文本最大 token 数（截断）。
        tokenizer: 用于截断计算。

    Returns:
        轨迹样本 dict；金标准文档缺失时返回 None（跳过）。
    """
    question = _clean_question(ex["question"])
    answer = str(ex["answer"]).strip()
    if not answer:
        return None

    gold_docs = _lookup_gold_docs(ex, corpus_by_title)
    if not gold_docs:
        return None

    # 2-turn 启发式：bridge 型 + ≥2 个支撑文档 + t2 标题出现在 t1 正文中
    two_turn = False
    if ex.get("type") == "bridge" and len(gold_docs) >= 2:
        t1_doc = gold_docs[0]
        t2_title = gold_docs[1]["title"]
        if t2_title.lower() in t1_doc["text"].lower():
            two_turn = True

    turns: List[Dict[str, str]] = []
    thinks: List[str] = []

    # turn 1：query = 问题；docs = t1（+t2 若 1-turn）金标准 + BM25
    think1 = "I need to search for relevant information to answer this question."
    gold_for_turn1 = gold_docs[:1] if two_turn else gold_docs
    bm25_1 = retriever.retrieve(question, topk=topk)
    docs1 = format_docs_as_reference(_merge_docs(gold_for_turn1, bm25_1, topk))
    if max_docs_tokens and tokenizer is not None:
        docs1 = truncate_docs_by_tokens(docs1, tokenizer, max_docs_tokens)
    turns.append({"query": question, "docs": docs1})
    thinks.append(think1)

    # turn 2：query = t2 标题
    if two_turn:
        t2_doc = gold_docs[1]
        query2 = t2_doc["title"]
        think2 = f"The search results mention {query2}. I need more information about {query2}."
        bm25_2 = retriever.retrieve(query2, topk=topk)
        docs2 = format_docs_as_reference(_merge_docs([t2_doc], bm25_2, topk))
        if max_docs_tokens and tokenizer is not None:
            docs2 = truncate_docs_by_tokens(docs2, tokenizer, max_docs_tokens)
        turns.append({"query": query2, "docs": docs2})
        thinks.append(think2)

    final_think = f"Based on the search results, the answer is {answer}."
    return {
        "id": str(ex["id"]),
        "question": question,
        "answer": answer,
        "turns": turns,
        "thinks": thinks,
        "final_think": final_think,
    }


def build_interactive_data(
    dataset: Any,
    retriever: BM25Retriever,
    corpus_by_title: Dict[str, Dict[str, str]],
    output_path: str,
    topk: int = DEFAULT_TOPK,
    max_questions: Optional[int] = None,
    max_docs_tokens: Optional[int] = None,
    tokenizer=None,
    seed: int = 42,
) -> int:
    """为数据集构造交互式轨迹并保存 JSONL。

    Returns:
        生成的样本数。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if max_questions is not None:
        dataset = dataset.shuffle(seed=seed).select(range(max_questions))

    count = 0
    n_two_turn = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in dataset:
            try:
                sample = build_trajectory(
                    ex, retriever, corpus_by_title, topk, max_docs_tokens, tokenizer
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[build_interactive_data] 跳过 {ex.get('id', '?')}: {exc}")
                continue
            if sample is None:
                continue
            if len(sample["turns"]) >= 2:
                n_two_turn += 1
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

    print(
        f"[build_interactive_data] 生成样本 {count}（其中 2-turn {n_two_turn}） -> {output_path}"
    )
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建交互式搜索 SFT 轨迹数据")
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--max_docs_tokens", type=int, default=1024)
    args = parser.parse_args()

    tokenizer = load_tokenizer("Qwen/Qwen3.5-2B")
    retriever = BM25Retriever(args.corpus_path)
    # title -> doc 映射（含 id/title/text）
    corpus_by_title: Dict[str, Dict[str, str]] = {}
    for doc in retriever.docs:
        corpus_by_title.setdefault(doc["title"], doc)

    hp = load_dataset("hotpot_qa", "distractor", split=args.split)
    build_interactive_data(
        hp,
        retriever,
        corpus_by_title,
        args.output_path,
        topk=args.topk,
        max_questions=args.max_questions,
        max_docs_tokens=args.max_docs_tokens,
        tokenizer=tokenizer,
    )
