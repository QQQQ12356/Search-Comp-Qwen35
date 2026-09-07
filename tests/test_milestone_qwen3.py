"""Qwen3.5 迁移里程碑的烟雾测试。

覆盖已交付并验证的路径（不依赖尚未完成的混合 Beacon 训练）：

1. Qwen3.5 tokenizer 能渲染 `` thinking`` / `` response`` 推理特殊 token 与
   SearchAgent ``<search>`` 指令模板。
2. 轨迹序列构建（think→search→<information>→answer）的压缩区 / 损失区定位。
3. ``extract_search_query`` / ``extract_answer`` / EM/F1 指标。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from search_comp.data.trajectory import (
    SEARCH_INSTRUCTION,
    build_assistant_segments,
    build_loss_labels,
    build_sequence_ids,
    extract_search_query,
)
from search_comp.evaluation.em_f1 import (
    compute_metrics,
    extract_answer,
    normalize_answer,
)


@pytest.fixture(scope="module")
def tokenizer():
    from search_comp.milestones.qwen35_native import load_tokenizer

    return load_tokenizer("Qwen/Qwen3.5-2B")


def test_chat_template_has_reasoning_tokens(tokenizer):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": SEARCH_INSTRUCTION.format(question="Who founded Google?")}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # 生成提示以 ``<think>`` / ``</think>`` 起始（Qwen3.5 原生推理格式，与 SearchAgent 标签对齐）
    assert "<think>" in text and "</think>" in text
    ids = tokenizer(text, add_special_tokens=False).input_ids
    assert tokenizer.eos_token_id in (248046,)  # eos = <|im_end|>
    dec = tokenizer.decode(ids, skip_special_tokens=False)
    assert "<think>" in dec and "</think>" in dec


def _sample():
    return {
        "id": "x",
        "question": "Who founded Google?",
        "answer": "Larry Page and Sergey Brin",
        "turns": [
            {"query": "Google founders", "docs": "Doc 1 (Title: Google) Google was founded by Larry Page."},
            {"query": "Larry Page", "docs": "Doc 2 (Title: Larry Page) Larry Page co-founded Google."},
        ],
        "thinks": ["I need to search.", "I need more info."],
        "final_think": "Based on the results the answer is known.",
    }


def test_sequence_build_regions_and_loss(tokenizer):
    s = _sample()
    chat_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": SEARCH_INSTRUCTION.format(question=s["question"])}],
        tokenize=False, add_generation_prompt=True,
    )
    ids, regions, gen_spans = build_sequence_ids(tokenizer, chat_input, s, max_length=4096)
    assert len(ids) > 0
    # 两个 <information> 文档压缩区
    assert len(regions) == 2
    for rs, re in regions:
        assert rs < re < len(ids)
    labels = build_loss_labels(len(ids), ids, gen_spans)
    # 压缩区标签为 -100（文档不参与损失）
    for rs, re in regions:
        assert all(v == -100 for v in labels[rs:re])
    # gen 区有监督
    assert any(labels[gs:ge] != -100 for gs, ge in gen_spans)


def test_extract_and_metrics():
    assert extract_search_query("a<search>foo bar</search>b") == "foo bar"
    assert extract_search_query("no search") == ""
    # last <answer> wins
    assert extract_answer("x<answer>A</answer><answer>B</answer>") == "B"
    assert normalize_answer("  The  Paris, France  ") == "paris france"
    m = compute_metrics([("1", "Paris", "paris"), ("2", "Rome", "Tokyo")])
    assert m["em"] == 0.5  # 1/2 exact
    assert 0.0 < m["f1"] < 1.0
    assert m["valid_count"] == 2


def test_sample_jsonl_shape():
    # 构建出的交互数据 jsonl 形状与 dataset 期望一致
    sample = _sample()
    turns = [t for t in sample["turns"]]
    segs = build_assistant_segments(sample)
    assert any(kind == "docs" for _, kind in segs)
    assert all(("query" in t and "docs" in t) for t in turns)
    assert json.loads(json.dumps(sample))["answer"] == sample["answer"]