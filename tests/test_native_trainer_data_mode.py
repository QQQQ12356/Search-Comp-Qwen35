"""原生 SFT 训练器的 data_mode 分支与 searchr1 collator 单元测试。

覆盖：

- ``build_dataset`` 按 ``data_mode`` 选择 Search-R1 / 交互式数据集，未知值报错；
- ``collate_for_searchr1`` 把 ``SearchR1SFTDataset`` 返回的 Python list 转成
  ``(1, L)`` tensor，并丢弃 ``regions`` 等非模型键（否则 Trainer 会透传给
  ``model.forward()`` 报错）；
- searchr1 collator 仅支持 ``batch_size=1``（与 Beacon 路径一致）；
- 交互式 collator 仍按右侧 padding 支持 ``batch > 1``（回归保护）。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from search_comp.data.searchr1_dataset import SearchR1SFTDataset
from search_comp.trainer.native_trainer import (
    NativeSearchSFTDataset,
    build_collator,
    build_dataset,
    collate_for_native,
    collate_for_searchr1,
)

MSG_SAMPLE = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Answer the question. Question: Who founded Google?"},
        {
            "role": "assistant",
            "content": "<thinking>\nI need to search.\n</thinking>\n<search>Google founders</search>",
        },
        {
            "role": "user",
            "content": "<information>[Document 1] Google was founded by Larry Page.</information>",
        },
        {
            "role": "assistant",
            "content": "<thinking>\nNow I know.\n</thinking>\n<answer> Larry Page </answer>",
        },
    ]
}


@pytest.fixture(scope="module")
def tokenizer():
    from search_comp.milestones.qwen35_native import load_tokenizer

    tok = load_tokenizer("Qwen/Qwen3.5-2B")
    tok.pad_token = tok.eos_token
    return tok


def _searchr1_item():
    """模拟 ``SearchR1SFTDataset.__getitem__`` 的返回：Python list + 元数据键。"""
    return {
        "input_ids": [1, 2, 3, 4],
        "labels": [-100, -100, 2, 3],
        "regions": [(2, 3)],
        "n_turns": 2,
        "n_searches": 1,
    }


def _write(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


class TestBuildDataset:
    def test_searchr1_mode_builds_searchr1_dataset(self, tmp_path, tokenizer):
        data = tmp_path / "sft.jsonl"
        _write(str(data), MSG_SAMPLE)
        ds = build_dataset("searchr1", str(data), tokenizer, max_length=8192)
        assert isinstance(ds, SearchR1SFTDataset)
        assert len(ds) == 1

    def test_interactive_mode_builds_native_dataset(self, tmp_path, tokenizer):
        data = tmp_path / "traj.jsonl"
        _write(str(data), {"question": "Who founded Google?"})
        ds = build_dataset("interactive", str(data), tokenizer, max_length=8192)
        assert isinstance(ds, NativeSearchSFTDataset)
        assert len(ds) == 1

    def test_unknown_mode_raises(self, tmp_path, tokenizer):
        with pytest.raises(ValueError, match="data_mode"):
            build_dataset("bogus", str(tmp_path / "unused.jsonl"), tokenizer, 8192)


class TestBuildCollator:
    def test_searchr1_mode_returns_searchr1_collator(self):
        assert build_collator("searchr1") is collate_for_searchr1

    def test_interactive_mode_returns_native_collator(self):
        assert build_collator("interactive") is collate_for_native

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="data_mode"):
            build_collator("bogus")


class TestSearchR1Collator:
    def test_converts_lists_to_single_batch_tensors(self):
        batch = collate_for_searchr1([_searchr1_item()])
        assert batch["input_ids"].dtype == torch.long
        assert batch["input_ids"].shape == (1, 4)
        assert batch["input_ids"].tolist() == [[1, 2, 3, 4]]
        assert batch["attention_mask"].tolist() == [[1, 1, 1, 1]]
        assert batch["labels"].tolist() == [[-100, -100, 2, 3]]

    def test_drops_keys_the_model_does_not_accept(self):
        batch = collate_for_searchr1([_searchr1_item()])
        assert set(batch) == {"input_ids", "attention_mask", "labels"}

    def test_rejects_batch_size_above_one(self):
        with pytest.raises(ValueError, match="batch_size=1"):
            collate_for_searchr1([_searchr1_item(), _searchr1_item()])


class TestSearchR1CollatorMatchesDataset:
    def test_labels_and_ids_are_passed_through_unchanged(self, tokenizer, tmp_path):
        data = tmp_path / "sft.jsonl"
        _write(str(data), MSG_SAMPLE)
        item = SearchR1SFTDataset(str(data), tokenizer)[0]

        batch = collate_for_searchr1([item])

        assert batch["input_ids"][0].tolist() == item["input_ids"]
        assert batch["labels"][0].tolist() == item["labels"]


class TestInteractiveCollatorStillBatches:
    def test_right_pads_ids_and_fills_labels_with_ignore_index(self):
        batch = collate_for_native(
            [
                {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "attention_mask": torch.ones(3, dtype=torch.long),
                    "labels": torch.tensor([-100, 2, 3]),
                },
                {
                    "input_ids": torch.tensor([4, 5]),
                    "attention_mask": torch.ones(2, dtype=torch.long),
                    "labels": torch.tensor([4, 5]),
                },
            ]
        )
        assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
        assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
        assert batch["labels"].tolist() == [[-100, 2, 3], [4, 5, -100]]
