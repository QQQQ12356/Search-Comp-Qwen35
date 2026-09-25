"""从 Search-R1 训练轨迹 jsonl 构建检索语料（JSONL）。

训练用 ``*-instruct-sft.jsonl`` 里 ``<information>`` 块的文档是模型训练时**实际看到的
检索结果**（NQ + HotpotQA 混合、文档 ID 为数值）。让在线 BM25 评估能对训练题目召回这些
文档，最省事且与训练严格对齐的做法就是直接从这些块抽取文档建库——一次覆盖 NQ 与 HotpotQA
训练文档，无需另找 NQ 文档源。

用法::

    python -m search_comp.data.build_corpus_searchr1 \\
        --jsonl_path outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl \\
        --output_path outputs/data/searchr1_corpus.jsonl \\
        [--merge_base outputs/data/hotpotqa_corpus.jsonl]

语料 schema 与 ``retrieval.build_corpus_from_hotpotqa`` 一致：每行 ``{id, title, text}``，
``text = f"{title}\\n{正文}"``，BM25Retriever 可直接读取。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, Iterable, List

#: 文档头：``[Document N] (ID: <数值>, Score: <浮点>)``。
_DOC_HEADER = re.compile(r"\[Document\s+\d+\]\s*\(ID:\s*([^,)]+)\s*,\s*Score:\s*([\d.eE+-]+)\)")

#: 文档头作为分隔符（下一个文档头或块结束）。
_DOC_SPLIT = re.compile(r"(?=\[Document\s+\d+\]\s*\(ID:)")

#: 剥离块外层 ``<information>`` / ``</information>`` 标签。
_INFO_PREFIX_RE = re.compile(r"^\s*<information>\s*", re.DOTALL)
_INFO_SUFFIX_RE = re.compile(r"\s*</information>\s*$", re.DOTALL)


def parse_document_blocks(info_content: str) -> List[Dict[str, str]]:
    """把一个 ``<information>`` 块内容解析为文档字典列表。

    Args:
        info_content: ``<information>... </information>`` 的完整内容（含/不含外层标签皆可）。

    Returns:
        ``[{id, title, text}, ...]``；无文档时返回空列表。
    """
    content = _INFO_PREFIX_RE.sub("", info_content or "")
    content = _INFO_SUFFIX_RE.sub("", content)
    if not content.strip():
        return []

    docs: List[Dict[str, str]] = []
    for part in _DOC_SPLIT.split(content):
        part = part.strip()
        if not part:
            continue
        header = _DOC_HEADER.match(part)
        if header is None:
            continue
        body = part[header.end():].strip()
        lines = [ln for ln in body.split("\n") if ln.strip()]
        if not lines:
            continue
        title = lines[0].strip().strip('"').strip()
        body = "\n".join(lines[1:]).strip()
        text = f"{title}\n{body}"
        doc_id = header.group(1).strip()
        if not doc_id:
            doc_id = f"{title}|||{body}"
        docs.append(
            {
                "id": doc_id,
                "title": title,
                "text": text,
            }
        )
    return docs


def _iter_docs_from_jsonl(jsonl_path: str) -> Iterable[Dict[str, str]]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for msg in obj.get("messages", []):
                if msg.get("role") == "user" and "<information>" in msg.get("content", ""):
                    yield from parse_document_blocks(msg["content"])


def build_corpus_from_searchr1_jsonl(
    jsonl_path: str,
    output_path: str,
    merge_base: str | None = None,
) -> int:
    """从训练 jsonl 抽文档建去重语料；可选合并既有语料（按 title+text 去重）。

    Returns:
        合并去重后的语料段落数。
    """
    seen: Dict[tuple, Dict[str, str]] = {}

    def _add(doc: Dict[str, str]) -> None:
        key = (doc["title"], doc["text"])
        seen.setdefault(key, doc)

    if merge_base:
        with open(merge_base, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    _add(json.loads(line))

    for doc in _iter_docs_from_jsonl(jsonl_path):
        _add(doc)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in seen.values():
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"[build_corpus] 语料段落数: {len(seen)} -> {output_path}")
    return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description="从 Search-R1 训练 jsonl 构建检索语料")
    parser.add_argument("--jsonl_path", type=str, required=True, help="训练 jsonl 路径")
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/data/searchr1_corpus.jsonl",
        help="语料 JSONL 输出路径",
    )
    parser.add_argument(
        "--merge_base",
        type=str,
        default=None,
        help="既有语料路径（如 hotpotqa_corpus.jsonl），合并去重",
    )
    args = parser.parse_args()
    build_corpus_from_searchr1_jsonl(args.jsonl_path, args.output_path, args.merge_base)


if __name__ == "__main__":
    main()