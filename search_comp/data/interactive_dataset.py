"""交互式搜索 SFT 数据集与 collator。

- :class:`InteractiveSFTDataset`：加载 JSONL 轨迹样本，用 :func:`build_sequence_ids`
  逐片段 tokenize 拼接，返回 ``input_ids`` / ``labels``（只在模型生成片段上算损失）/
  ``regions``（各 ``<information>`` 文档压缩区 token 区间）。
- :class:`InteractiveCollator`：批量打包。交互式样本结构多变（turn 数、各区域长度
  均不同），且 batch 内所有样本需共享同一组压缩区，因此**仅支持 batch_size=1**
  （用梯度累积扩大有效 batch）。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .trajectory import build_loss_labels, build_search_chat_prompt, build_sequence_ids


class InteractiveSFTDataset(Dataset):
    """交互式搜索 SFT 数据集。

    Args:
        data_path: JSONL 路径，每行 ``{id, question, answer, turns, thinks, final_think}``。
        tokenizer: HuggingFace tokenizer。
        max_length: 最大 token 数。
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

    def _tokenize(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """tokenize 一个样本。

        Returns:
            dict，含 ``input_ids``, ``labels``, ``regions``, ``n_turns``。
        """
        question = sample["question"]
        chat_input = build_search_chat_prompt(question, add_generation_prompt=True)
        ids, doc_regions, gen_spans = build_sequence_ids(
            self.tokenizer, chat_input, sample, max_length=self.max_length
        )
        labels = build_loss_labels(len(ids), ids, gen_spans)
        return {
            "input_ids": ids,
            "labels": labels,
            "regions": doc_regions,
            "n_turns": len(sample["turns"]),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._tokenize(self.samples[idx])


class InteractiveCollator:
    """交互式数据 collator（仅支持 batch_size=1）。

    Args:
        tokenizer: HuggingFace tokenizer（用于 pad_token_id）。
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("tokenizer 缺少 pad_token，请先设置 pad_token")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """打包一个样本。

        Args:
            features: ``__getitem__`` 返回的 dict 列表（长度必须为 1）。

        Returns:
            dict：``input_ids``(1,L), ``attention_mask``(1,L), ``labels``(1,L),
            ``regions``（压缩区列表，token 区间）。
        """
        if len(features) != 1:
            raise ValueError(
                f"交互式数据仅支持 batch_size=1（当前 {len(features)}）。"
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
