"""SFT 数据集与批处理 collator。

- :class:`BeaconSFTDataset`：加载 JSONL 样本，tokenize 并计算：
  - ``input_ids`` / ``labels``（只对答案部分计算损失）。
  - ``compress_start`` / ``compress_end``（文档压缩区在 token 序列中的位置）。
- :class:`BeaconDataCollator`：把 batch 内样本按区域统一 padding
  （question 区、文档区对齐到窗口整数倍、suffix 区），使整个 batch 共享
  同一组 ``compress_start/compress_end``，满足 :class:`BeaconMemory` 的要求。

prompt 使用 Qwen2.5 的 chat 模板包装（user 消息内含 ``<information>`` 文档块），
response 直接拼接在 assistant 标记之后。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .build_sft_data import make_response


class BeaconSFTDataset(Dataset):
    """Beacon RAG SFT 数据集。

    Args:
        data_path: JSONL 路径，每行 ``{id, question, docs, answer}``。
        tokenizer: HuggingFace tokenizer。
        max_length: 序列最大 token 数（超长截断）。
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
                        self.samples.append(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"加载数据 {data_path} 失败: {exc}") from exc
        if not self.samples:
            raise RuntimeError(f"数据为空: {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize_parts(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """tokenize 并计算压缩区与损失区。

        Returns:
            dict，含 ``input_ids``, ``labels``, ``compress_start``, ``compress_end``,
            ``prefix_len``, ``docs_len``, ``suffix_len``。
        """
        tok = self.tokenizer
        prompt = (
            "Answer the given question with some potentially useful context. "
            "Show your reasoning in <think> </think> tags and return the final answer "
            "in <answer> </answer> tags.\n"
            f"Question: {sample['question']}\n"
            "<information>\n"
            f"{sample['docs']}\n"
            "</information>"
        )
        response = make_response(sample["answer"])

        # chat 模板包装（user 消息含文档，assistant 标记后接 response）
        chat_input = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_str = chat_input + response

        full_ids = tok(full_str, add_special_tokens=False).input_ids
        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]

        # 压缩区：<information>\n 之后的文档部分（到 </information> 之前）。
        # 用 docs 在 chat_input 中的真实字符位置定位，避免把 chat 模板尾部
        # 算入压缩起点（见 code review H-1）。
        docs_pos = chat_input.index(sample["docs"])
        before_docs = chat_input[:docs_pos]
        compress_start = len(tok(before_docs, add_special_tokens=False).input_ids)
        docs_tokens = tok(sample["docs"], add_special_tokens=False).input_ids
        compress_end = compress_start + len(docs_tokens)
        # 截断保护：若 max_length 截断落在文档区，压缩区终点不越界
        compress_end = min(compress_end, len(full_ids))

        # 损失区：response 开始位置
        response_start = full_str.index(response)
        loss_start = len(
            tok(full_str[:response_start], add_special_tokens=False).input_ids
        )

        labels = [-100] * len(full_ids)
        labels[loss_start:] = full_ids[loss_start:]

        return {
            "input_ids": full_ids,
            "labels": labels,
            "compress_start": compress_start,
            "compress_end": compress_end,
            "prefix_len": compress_start,
            "docs_len": compress_end - compress_start,
            "suffix_len": len(full_ids) - compress_end,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._tokenize_parts(self.samples[idx])


def _align_up(x: int, window: int) -> int:
    """向上取整到 window 的整数倍。"""
    return ((x + window - 1) // window) * window


class BeaconDataCollator:
    """把样本按区域统一 padding，保证 batch 共享同一 compress_start/compress_end。

    Args:
        tokenizer: HuggingFace tokenizer（提供 pad_token_id）。
        beacon_window: 文档窗口大小（文档区 padding 对齐到其整数倍）。
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, beacon_window: int = 1024):
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("tokenizer 缺少 pad_token，请先设置 pad_token")
        self.beacon_window = beacon_window

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """批处理。

        Args:
            features: ``__getitem__`` 返回的 dict 列表。

        Returns:
            dict：``input_ids``, ``attention_mask``, ``labels``,
            ``compress_start``（int）, ``compress_end``（int）。
        """
        max_prefix = max(f["prefix_len"] for f in features)
        max_docs = max(f["docs_len"] for f in features)
        max_docs_aligned = _align_up(max_docs, self.beacon_window)
        max_suffix = max(f["suffix_len"] for f in features)

        compress_start = max_prefix
        compress_end = max_prefix + max_docs_aligned

        batch_ids, batch_labels = [], []
        for f in features:
            ids, labels = f["input_ids"], f["labels"]
            cs, ce = f["compress_start"], f["compress_end"]
            prefix, docs, suffix = ids[:cs], ids[cs:ce], ids[ce:]
            lprefix, ldocs, lsuffix = labels[:cs], labels[cs:ce], labels[ce:]

            prefix = prefix + [self.pad_token_id] * (max_prefix - len(prefix))
            docs = docs + [self.pad_token_id] * (max_docs_aligned - len(docs))
            suffix = suffix + [self.pad_token_id] * (max_suffix - len(suffix))

            lprefix = lprefix + [-100] * (max_prefix - len(lprefix))
            ldocs = ldocs + [-100] * (max_docs_aligned - len(ldocs))
            lsuffix = lsuffix + [-100] * (max_suffix - len(lsuffix))

            batch_ids.append(prefix + docs + suffix)
            batch_labels.append(lprefix + ldocs + lsuffix)

        input_ids = torch.tensor(batch_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor(batch_labels, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "compress_start": compress_start,
            "compress_end": compress_end,
        }
