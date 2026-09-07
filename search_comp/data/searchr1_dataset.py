"""Search-R1 SFT 数据集（messages 格式）与 collator。

从搜索/检索增强模型的 SFT 轨迹（``messages`` 对话格式）直接构造
beacon 训练样本，**不再**用规则生成轨迹。数据来源：

    /home/huangzj/proj/searchagent/Search-R1-SFT-jsonl/{backbone}-instruct-sft.jsonl

每行形如（Search-R1 原文格式，含系统 / 用户 / 助手多轮）::

    {"messages": [
        {"role": "system", "content": "You are a helpful and harmless assistant."},
        {"role": "user", "content": "Answer the given question. ... Question: <q>?"},
        {"role": "assistant", "content": "<thinking> ... </thinking>\\n<search> <query> </search>"},
        {"role": "user", "content": "<information> [Document 1] ... </information>"},
        {"role": "assistant", "content": "<thinking> ... </thinking>\\n<answer> <answer> </answer>"},
    ]}

设计要点：

- 用 ``apply_chat_template`` **逐条消息增量编码**：第 ``i`` 条消息的 token 区间
  长度 = 前缀 ``[0..i]`` 编码长度 − 前缀 ``[0..i-1]`` 编码长度，从而精确对齐
  chat 模板的角色标记 / 分隔符 / eos，无需手拼模板。
- **loss（labels）**：只对 ``role == "assistant"`` 的 token 计算（think / search /
  answer 均为模型输出，整体作为 SFT 目标）。
- **压缩区（regions）**：``role == "user"`` 且内容以 ``<information>`` 开头
  的整个消息块对应的 token 区间，就是 beacon 要压缩的文档区。
- 样本结构多变（turn 数、文档长度都不同），且需逐样本对齐压缩区与 loss 区间，
  因此**仅支持 batch_size=1**（用梯度累积扩大有效 batch）。

其他字段（非 ``<information>`` 的 user / system / 助手消息）token 保持原样但不
计损失、不压缩。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

#: 标记文档块的 user 消息前缀（用于识别压缩区）
_INFO_MARKER = "<information>"
_SEARCH_PATTERN = re.compile(r"<search>")
_ANSWER_PATTERN = re.compile(r"<answer>")


class SearchR1SFTDataset(Dataset):
    """把 Search-R1 ``messages`` 格式的 SFT 轨迹转成 beacon 训练样本。

    Args:
        data_path: JSONL 路径，每行 ``{messages: [{role, content}, ...]}``。
        tokenizer: HuggingFace tokenizer（Qwen2.5 系列）。
        max_length: 最大 token 数（超出截断，超出的压缩/loss 区间被丢弃）。
    """

    def __init__(
        self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 8192
    ):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[Dict[str, Any]] = []
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        # 规范化：若整行已是 dict 且含 "messages"
                        if "messages" not in obj:
                            raise ValueError(f"缺少 messages 字段: {line[:80]}")
                        self.samples.append(obj)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"加载数据 {data_path} 失败: {exc}") from exc
        if not self.samples:
            raise RuntimeError(f"数据为空: {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """把一条 messages 会话编码为 ``input_ids`` / ``labels`` / ``regions``。

        Returns:
            dict：
            - ``input_ids``: 拼接后的 token id 列表。
            - ``labels``: loss 掩码，assistant token 处为自身 id，其余 -100。
            - ``regions``: 各 ``<information>`` 文档块的 token 区间。
            - ``n_turns`` / ``n_searches``: 用于统计（交互轮数、搜索次数）。
        """
        msgs = sample["messages"]
        if not msgs or msgs[0].get("role") != "system":
            raise ValueError(f"messages 应以 system 消息开头: {sample.get('id', '?')}")

        input_ids: List[int] = []
        labels: List[int] = []
        regions: List[Tuple[int, int]] = []
        n_searches = 0

        # 手动 ChatML 逐条渲染：绕过 chat 模板对 <think> 的 reasoning 抽取
        # （模板会把中间搜索轮的 <think> 思考剥离、并为 assistant 自动插入空
        # <think></think> 块），改用模型原生 <think>/</think> 特殊 token，保证
        # 训练序列与推理格式一致、无空块。
        for msg in msgs:
            role = msg.get("role")
            content = msg.get("content", "") or ""
            if role == "assistant":
                # 训练数据里的 <thinking> 统一改为原生 <think> 推理标签
                content = content.replace("<thinking>", "<think>").replace("</thinking>", "</think>")
            text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            seg_ids = self.tokenizer(text, add_special_tokens=False).input_ids

            start = len(input_ids)
            end = start + len(seg_ids)

            if role == "assistant":
                # 模型输出整体作为 SFT 目标（think + search + answer）
                labels.extend(seg_ids)
                n_searches += len(_SEARCH_PATTERN.findall(content))
            elif role == "user" and content.lstrip().startswith(_INFO_MARKER):
                # <information> 文档块 → beacon 压缩区
                regions.append((start, end))
                labels.extend([-100] * len(seg_ids))
            else:
                # system / 普通 user（问题）→ 不计损失、不压缩
                labels.extend([-100] * len(seg_ids))

            input_ids.extend(seg_ids)

        # 截断保护：丢弃被截断的压缩区 / loss 区间
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
            regions = [(s, e) for s, e in regions if e <= self.max_length]

        n_turns = sum(1 for m in msgs if m.get("role") == "assistant")
        return {
            "input_ids": input_ids,
            "labels": labels,
            "regions": regions,
            "n_turns": n_turns,
            "n_searches": n_searches,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._tokenize(self.samples[idx])


class SearchR1Collator:
    """Search-R1 数据 collator（仅支持 batch_size=1）。

    Args:
        tokenizer: HuggingFace tokenizer（用于 pad_token_id，其实 bs=1 不会 pad）。
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("tokenizer 缺少 pad_token，请先设置 pad_token")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """打包一个样本。

        Returns:
            dict：``input_ids``(1,L), ``attention_mask``(1,L), ``labels``(1,L),
            ``regions``（压缩区 token 区间列表）。
        """
        if len(features) != 1:
            raise ValueError(
                f"Search-R1 数据仅支持 batch_size=1（当前 {len(features)}）。"
                "请用 grad_accum_steps 扩大有效 batch。"
            )
        f = features[0]
        input_ids = torch.tensor([f["input_ids"]], dtype=torch.long)
        labels = torch.tensor([f["labels"]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "regions": f["regions"],
        }
