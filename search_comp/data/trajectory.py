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
- :func:`format_document_blocks` / :func:`parse_document_blocks`：``<information>``
  内检索文档的统一渲染与解析（训练数据与在线检索评测共用同一格式）。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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


# ----------------------------------------------------------------------
# <information> 内的检索文档：训练与评测统一的格式
# ----------------------------------------------------------------------
# 训练侧（Search-R1 轨迹的 ``<information>`` 块、交互式轨迹的 ``turns[].docs``）与
# 评测侧（BM25 在线检索结果）必须把文档渲染成**同一种文本**，否则模型在评测时看到
# 的是训练分布外的格式。本模块是该格式的唯一来源::
#
#     Doc 1 DeLorean time machine
#     The DeLorean time machine is a fictional automobile-based time travel device ...
#
#     Doc 2 Chelsea Handler
#     Seymour Handler, a used car dealer. ...
#
# 规则：每篇文档以 ``Doc <序号> <标题>`` 开头且标题独占一行；正文原样保留，仅折叠
# 开头连续重复的标题（至少保留一次，见 :func:`collapse_repeated_title`）；文档之间
# 空行分隔；不再输出 ``[Document N] (ID: ..., Score: ...)`` 或 ``(Title: ...)`` 检索
# 元信息。

#: 统一格式的文档序号前缀。
DOC_INDEX_PREFIX = "Doc "
#: 文档之间的分隔（空行）。
DOC_SEPARATOR = "\n\n"

#: ``<information>...</information>`` 整块匹配；组 1 为块内文档文本（即压缩区）。
INFORMATION_PATTERN = re.compile(r"\s*<information>(.*?)</information>\s*", re.DOTALL)

#: 历史格式 ``[Document N] (ID: <数值>, Score: <浮点>)`` 的文档头与切分点。
_LEGACY_DOC_HEADER = re.compile(
    r"\[Document\s+\d+\]\s*\(ID:\s*([^,)]+)\s*,\s*Score:\s*([\d.eE+-]+)\)"
)
_LEGACY_DOC_SPLIT = re.compile(r"(?=\[Document\s+\d+\]\s*\(ID:)")
#: 历史格式 ``Doc N (Title: <标题>) <标题>\n<正文>`` 的文档头。
_TITLED_DOC_HEADER = re.compile(r"Doc\s+\d+\s*\(Title:\s*(.*?)\)\s*", re.DOTALL)
#: 统一格式 ``Doc N <标题>`` 的文档头。
_INDEXED_DOC_HEADER = re.compile(r"Doc\s+\d+\s+")
#: 统一格式的切分点：块首或空行之后的 ``Doc N ``，避免正文里的同形文本被误切。
_INDEXED_DOC_SPLIT = re.compile(r"(?m)(?=(?:\A|\n\n)Doc \d+ )")


def document_body(title: str, text: str) -> str:
    """从语料 ``text``（``<标题>\\n<正文>``）中取出正文。

    Args:
        title: 文档标题。
        text: 语料里的文档全文（标题行 + 正文）。

    Returns:
        去掉标题行的正文；``text`` 不以标题行开头时原样返回（去首尾空白）。
    """
    heading = (title or "").strip()
    if not heading:
        return (text or "").strip()
    prefix = f"{heading}\n"
    if (text or "").startswith(prefix):
        return text[len(prefix):].strip()
    return (text or "").strip()


def collapse_repeated_title(title: str, body: str) -> str:
    """把正文开头**连续重复**的标题折叠成一个，至少保留第一次出现。

    检索片段里存在正文以标题开头重复两次的情况（``Anil Kumble Anil Kumble ( born ...``）。
    全部剥掉会把句子切成 ``( born ...`` 这种语法破碎的开头并丢掉标题，因此这里只折叠
    连续重复，绝不删到正文以标点开头。标题后紧跟字母数字时不算重复（``Matoma`` 是
    ``Matomaa`` 的前缀）。
    """
    heading = (title or "").strip()
    if not heading:
        return body
    stripped = body.lstrip()
    rest = stripped
    count = 0
    while rest.lower().startswith(heading.lower()):
        remainder = rest[len(heading):]
        if remainder[:1].isalnum():
            break
        count += 1
        rest = remainder.lstrip()
    if count <= 1:
        return body
    first = stripped[:len(heading)]
    return f"{first} {rest}" if rest else first


def format_document_block(index: int, title: str, text: str) -> str:
    """渲染单篇文档：``Doc <index> <标题>`` + 换行 + 正文。

    正文除「折叠开头连续重复的标题」外**原样保留**：数据里约 40% 的正文以标题开头
    （维基风格），其中 17% 是标题连续重复两次，折叠成一个既保留标题又不会把句子切成
    ``( born 1970) ...`` 这种语法破碎的开头；详见 :func:`collapse_repeated_title`。
    """
    heading = (title or "").strip()
    body = collapse_repeated_title(heading, document_body(heading, text))
    if not heading:
        return f"{DOC_INDEX_PREFIX}{index} {body}".rstrip()
    if not body:
        return f"{DOC_INDEX_PREFIX}{index} {heading}"
    return f"{DOC_INDEX_PREFIX}{index} {heading}\n{body}"


def format_document_blocks(docs: Sequence[Mapping[str, Any]]) -> str:
    """把 ``[{title, text}, ...]`` 渲染为统一格式的文档块。

    空文档（``text`` 为空）直接跳过，序号按保留的文档连续编号。
    """
    blocks: List[str] = []
    for doc in docs:
        text = str(doc.get("text", "") or "").strip()
        if not text:
            continue
        blocks.append(
            format_document_block(len(blocks) + 1, str(doc.get("title", "") or ""), text)
        )
    return DOC_SEPARATOR.join(blocks)


def _parse_legacy_documents(content: str) -> List[Dict[str, str]]:
    """解析 ``[Document N] (ID: ..., Score: ...)`` + 带引号标题行的历史格式。"""
    docs: List[Dict[str, str]] = []
    for part in _LEGACY_DOC_SPLIT.split(content):
        part = part.strip()
        if not part:
            continue
        header = _LEGACY_DOC_HEADER.match(part)
        if header is None:
            continue
        lines = [line for line in part[header.end():].strip().split("\n") if line.strip()]
        if not lines:
            continue
        title = lines[0].strip().strip('"').strip()
        body = "\n".join(lines[1:]).strip()
        docs.append(
            {
                "id": header.group(1).strip() or f"{title}|||{body}",
                "title": title,
                "text": f"{title}\n{body}",
            }
        )
    return docs


def _parse_indexed_documents(content: str) -> List[Dict[str, str]]:
    """解析 ``Doc N <标题>`` 统一格式与 ``Doc N (Title: ...)`` 历史格式。"""
    docs: List[Dict[str, str]] = []
    for part in _INDEXED_DOC_SPLIT.split(content):
        part = part.strip()
        if not part:
            continue
        titled = _TITLED_DOC_HEADER.match(part)
        if titled is not None:
            title = titled.group(1).strip().strip('"').strip()
            rest = part[titled.end():].strip()
        else:
            indexed = _INDEXED_DOC_HEADER.match(part)
            if indexed is None:
                continue
            rest = part[indexed.end():].strip()
            title = rest.split("\n", 1)[0].strip()
        if not rest:
            continue
        body = rest.split("\n", 1)[1].strip() if "\n" in rest else ""
        docs.append(
            {
                "id": f"{title}|||{body}" if title else body,
                "title": title,
                "text": f"{title}\n{body}" if body else title,
            }
        )
    return docs


def parse_document_blocks(content: str) -> List[Dict[str, str]]:
    """把 ``<information>`` 块内容解析为 ``[{id, title, text}, ...]``。

    同时接受历史格式（``[Document N] (ID: ..., Score: ...)``、
    ``Doc N (Title: ...)``）与 :func:`format_document_blocks` 的统一格式，
    因此旧数据文件与旧语料都能继续解析。``text`` 始终是 ``<标题>\\n<正文>``。

    Args:
        content: ``<information>...</information>`` 的内容（含或不含外层标签）。

    Returns:
        文档字典列表；无可识别文档时返回空列表。
    """
    matched = INFORMATION_PATTERN.fullmatch(content or "")
    if matched is not None:
        content = matched.group(1)
    content = (content or "").strip()
    if not content:
        return []
    if "[Document" in content:
        return _parse_legacy_documents(content)
    return _parse_indexed_documents(content)
