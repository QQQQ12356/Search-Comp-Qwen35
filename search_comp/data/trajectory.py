"""交互式搜索轨迹的共享模板与序列构建。

训练数据与推理共用本模块，保证 ``<information>`` 块、prompt 模板、token 序列
构造**完全一致**：

- :data:`SEARCH_INSTRUCTION`：SearchAgent 搜索协议正文（放在 system 消息）。
- :data:`INFO_PREFIX` / :data:`INFO_SUFFIX`：``<information>`` 块的前后缀。
- :func:`build_assistant_segments`：把样本拆成 ``(text, kind)`` 片段，
  kind ∈ {"gen", "info_prefix", "docs", "info_suffix"}。
- :func:`build_sequence_ids`：**逐片段 tokenize 并拼接**，返回 token id 序列、
  docs 压缩区 token 区间、gen（有损失）token 区间。
- :func:`format_information_block` / :func:`extract_search_query`：推理辅助。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

#: Search-R1 风格的搜索协议正文。该说明属于长期行为约束，因此放在 system 消息。
SEARCH_INSTRUCTION = """Answer the given question. You must conduct reasoning inside <thinking> and </thinking> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>."""

#: 不含搜索协议的基础 system 文本，供提示词对照实验使用。
BASE_SYSTEM_PROMPT = "You are a helpful and harmless assistant."

#: 主训练和推理路径使用的 system 消息：基础角色说明 + 完整搜索协议。
SYSTEM_PROMPT = f"{BASE_SYSTEM_PROMPT}\n\n{SEARCH_INSTRUCTION}"


#: <information> 块前后缀（与 Search-R1 rollout 的 next_obs 格式一致）
INFO_PREFIX = "\n\n<information>"
INFO_SUFFIX = "</information>\n\n"

_SEARCH_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL)


def format_information_block(docs_text: str) -> str:
    """把检索文档文本包装成 ``<information>`` 块。"""
    return f"{INFO_PREFIX}{docs_text}{INFO_SUFFIX}"


def extract_search_query(text: str) -> str:
    """从模型输出中提取最后一个 ``<search>...</search>`` 的 query。"""
    matches = list(_SEARCH_PATTERN.finditer(text))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def build_search_chat_prompt(question: str, add_generation_prompt: bool = True) -> str:
    """手动构造 SearchAgent 的 ChatML 提示（不自动插入空 ``<think></think>`` 块）。

    与 :mod:`searchr1_dataset` 的手动 ChatML 渲染一致，让模型从零生成
    ``<think>...</think><search>...</search>``（原生推理标签，无空块）。

    Args:
        question: 问题文本。
        add_generation_prompt: 是否追加 ``<|im_start|>assistant\n``。

    Returns:
        ChatML 提示字符串。
    """
    question = str(question).strip()
    text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    text += f"<|im_start|>user\n{question}<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"
    return text


# ----------------------------------------------------------------------
# 样本结构
# ----------------------------------------------------------------------
def build_assistant_segments(sample: Dict[str, Any]) -> List[Tuple[str, str]]:
    """把样本拆成 ``(text, kind)`` 片段序列。

    Args:
        sample: ``{id, question, answer, turns: [{query, docs}], thinks: [...],
            final_think}``。

    Returns:
        有序片段列表；kind 取值：
        - "gen"：模型生成的文本（think / search query / answer），计算损失。
        - "docs"：检索文档文本，对应一个压缩区。
        - "info_prefix" / "info_suffix"：``<information>`` 标签，环境提供，不压缩不计损失。
    """
    segments: List[Tuple[str, str]] = []
    for i, turn in enumerate(sample["turns"]):
        think = sample["thinks"][i]
        segments.append(
            (f"<think>{think}</think>\n<search>{turn['query']}</search>", "gen")
        )
        segments.append((INFO_PREFIX, "info_prefix"))
        segments.append((turn["docs"], "docs"))
        segments.append((INFO_SUFFIX, "info_suffix"))
    segments.append(
        (
            f"<think>{sample['final_think']}</think>\n<answer>{sample['answer']}</answer>",
            "gen",
        )
    )
    return segments


def build_sequence_ids(
    tokenizer, chat_input: str, sample: Dict[str, Any], max_length: int = 8192
) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """逐片段 tokenize 并拼接，返回训练/推理统一的 token 序列。

    Args:
        tokenizer: HuggingFace tokenizer。
        chat_input: ``apply_chat_template(..., add_generation_prompt=True)`` 的结果。
        sample: 交互式样本。
        max_length: 最大 token 数（超出则截断，并丢弃被截断的压缩区/损失区）。

    Returns:
        ``(ids, doc_regions, gen_spans)``：
        - ``ids``: 拼接后的 token id 列表。
        - ``doc_regions``: 各 ``<information>`` 文档块的 token 区间 ``[(start, end)]``。
        - ``gen_spans``: 各模型生成片段的 token 区间 ``[(start, end)]``（用于损失掩码）。
    """
    parts: List[Tuple[str, str]] = [(chat_input, "chat")]
    parts += build_assistant_segments(sample)

    ids: List[int] = []
    doc_regions: List[Tuple[int, int]] = []
    gen_spans: List[Tuple[int, int]] = []
    cursor = 0
    for text, kind in parts:
        seg_ids = tokenizer(text, add_special_tokens=False).input_ids
        if kind == "docs":
            doc_regions.append((cursor, cursor + len(seg_ids)))
        elif kind == "gen":
            gen_spans.append((cursor, cursor + len(seg_ids)))
        ids.extend(seg_ids)
        cursor += len(seg_ids)

    # 截断处理
    if len(ids) > max_length:
        ids = ids[:max_length]
        doc_regions = [(s, e) for s, e in doc_regions if e <= max_length]
        gen_spans = [(s, e) for s, e in gen_spans if e <= max_length]
    return ids, doc_regions, gen_spans


def build_loss_labels(
    seq_len: int, ids: List[int], gen_spans: List[Tuple[int, int]]
) -> List[int]:
    """构造损失掩码：只在 gen（模型生成）片段上计算损失，其余为 -100。

    Args:
        seq_len: 序列长度。
        ids: token id 列表（gen 片段处填入自身 id，其余 -100）。
        gen_spans: 模型生成片段的 token 区间。

    Returns:
        labels 列表。
    """
    labels = [-100] * seq_len
    for s, e in gen_spans:
        labels[s:e] = ids[s:e]
    return labels
