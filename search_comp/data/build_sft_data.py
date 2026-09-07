"""构建 RAG + Beacon 的 SFT 训练数据。

流程：
1. 加载 HotpotQA 数据集（distractor 配置）。
2. 对每个问题用 :class:`BM25Retriever` 检索 top-k 文档（训练阶段离线预检索）。
3. 组织为 ``{id, question, docs, answer}`` 的 JSONL 样本。

prompt 模板（与 Search-R1 的 RAG 格式对齐）：
    "Answer the given question ... Question: {question}\\n<information>\\n{docs}\\n</information>"

压缩区为 ``<information>`` 与 ``</information>`` 之间的文档文本。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

from datasets import load_dataset

from .retrieval import BM25Retriever, format_docs_as_reference

PROMPT_TEMPLATE = """Answer the given question with some potentially useful context. \
You should analyze the question carefully, evaluate the given context (which may or may not be useful), and then generate an accurate and well-reasoned response. \
You should first have a reasoning process in mind and then provide the answer. \
Show your reasoning in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> Beijing </answer>.
Question: {question}
<information>
{docs}
</information>"""


def make_prompt(question: str, docs_text: str) -> str:
    """构造带检索上下文的 prompt。

    Args:
        question: 问题文本。
        docs_text: 格式化后的文档文本（不含 ``<information>`` 标签）。

    Returns:
        完整 prompt 字符串。
    """
    return PROMPT_TEMPLATE.format(question=question, docs=docs_text)


def make_response(answer: str) -> str:
    """构造模型的期望响应（think + answer 标签）。

    Args:
        answer: 标准答案。

    Returns:
        ``<think>...</think><answer>...</answer>`` 形式的响应。
    """
    reasoning = f"Based on the retrieved documents, the answer is {answer}."
    return f"<think>{reasoning}</think><answer>{answer}</answer>"


def truncate_docs_by_tokens(docs_text: str, tokenizer, max_tokens: int) -> str:
    """按 token 数截断文档文本（按 Doc 块截断，保留前面的文档）。

    Args:
        docs_text: 格式化后的文档文本。
        tokenizer: HuggingFace tokenizer。
        max_tokens: 允许的最大 token 数。

    Returns:
        截断后的文档文本。
    """
    lines = docs_text.split("\n")
    kept_lines = []
    total = 0
    for line in lines:
        n = len(tokenizer(line, add_special_tokens=False).input_ids)
        if total + n > max_tokens and kept_lines:
            break
        kept_lines.append(line)
        total += n
    return "\n".join(kept_lines)


def build_sft_data(
    dataset: Any,
    retriever: BM25Retriever,
    topk: int,
    output_path: str,
    max_questions: Optional[int] = None,
    max_docs_tokens: Optional[int] = None,
    tokenizer=None,
    seed: int = 42,
) -> int:
    """为数据集中的每个问题检索 top-k 文档并生成 SFT 样本。

    Args:
        dataset: HuggingFace 数据集（含 ``question``, ``answer``, ``id``）。
        retriever: BM25 检索器。
        topk: 每个问题检索的文档数。
        output_path: 输出 JSONL 路径。
        max_questions: 最多处理的问题数（用于快速测试）。
        seed: 随机采样种子。

    Returns:
        生成的样本数。

    Raises:
        OSError: 输出目录写入失败时抛出。
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if max_questions is not None:
        dataset = dataset.shuffle(seed=seed).select(range(max_questions))

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in dataset:
            question = str(ex["question"]).strip()
            if question.endswith("?"):
                question = question[:-1].strip()
            answer = str(ex["answer"]).strip()
            if not answer:
                continue

            try:
                retrieved = retriever.retrieve(question, topk=topk)
            except Exception as exc:  # noqa: BLE001
                print(f"[build_sft_data] 检索失败，跳过 {ex.get('id', '?')}: {exc}")
                continue

            docs_text = format_docs_as_reference(retrieved)
            # 可选：按 token 数截断文档，限制序列长度（控制显存）
            if max_docs_tokens and tokenizer is not None:
                docs_text = truncate_docs_by_tokens(
                    docs_text, tokenizer, max_docs_tokens
                )
            sample = {
                "id": str(ex["id"]),
                "question": question + "?",
                "docs": docs_text,
                "answer": answer,
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

    print(f"[build_sft_data] 生成样本数: {count} -> {output_path}")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 RAG + Beacon SFT 数据")
    parser.add_argument(
        "--corpus_path", type=str, required=True, help="语料 JSONL 路径"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="输出 JSONL 路径"
    )
    parser.add_argument("--split", type=str, default="train", help="HotpotQA split")
    parser.add_argument("--topk", type=int, default=10, help="每个问题检索文档数")
    parser.add_argument(
        "--max_questions", type=int, default=None, help="最多处理问题数"
    )
    parser.add_argument(
        "--max_docs_tokens", type=int, default=None, help="文档区最大 token 数（截断）"
    )
    args = parser.parse_args()

    from ..milestones.qwen35_native import load_tokenizer

    tokenizer = (
        load_tokenizer("Qwen/Qwen3.5-2B") if args.max_docs_tokens else None
    )

    retriever = BM25Retriever(args.corpus_path)
    hp = load_dataset("hotpot_qa", "distractor", split=args.split)
    build_sft_data(
        dataset=hp,
        retriever=retriever,
        topk=args.topk,
        output_path=args.output_path,
        max_questions=args.max_questions,
        max_docs_tokens=args.max_docs_tokens,
        tokenizer=tokenizer,
    )
