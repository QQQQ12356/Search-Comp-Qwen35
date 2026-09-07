"""Qwen3.5 原生（无 Beacon）搜索轨迹 SFT 训练入口。

基于 **transformers 标准 Trainer** 做搜索轨迹监督微调：模型只需学会
``think → <search>query</search> → 观察 <information> → <answer>`` 行为。

数据与 Beacon 版共用 ``search_comp.data.trajectory`` 的序列构建逻辑
（chat 模板 + 生成片段损失掩码），因此训练出的模型可直接丢给交互式
SearchAgent 推理。

标准 Trainer 自带 tqdm 进度条、loss 实时显示、checkpoint 与日志——
满足「训练过程可视化到终端、参考标准 Trainer 训练方式」。

用法::

    python -m search_comp.trainer.native_trainer --config configs/train/native_qwen3.5.yaml
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from ..data.trajectory import (
    SEARCH_INSTRUCTION,
    build_loss_labels,
    build_sequence_ids,
)


class NativeSearchSFTDataset(Dataset):
    """把交互式搜索轨迹 jsonl 转成标准 (input_ids, attention_mask, labels)。

    每样本的 chat 前缀来自 ``apply_chat_template(user=SEARCH_INSTRUCTION)``，
    标签只在模型生成片段（think / <search> / <answer>）上计算损失，其余 -100。
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 8192):
        self.samples: List[Dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        q = str(sample["question"])
        chat_input = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": SEARCH_INSTRUCTION.format(question=q)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids, _regions, gen_spans = build_sequence_ids(
            self.tokenizer, chat_input, sample, max_length=self.max_length
        )
        labels = build_loss_labels(len(ids), ids, gen_spans)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_for_native(batch: List[Dict[str, torch.Tensor]]):
    """标准右侧 padding collator（用于原生 Trainer，bs 可 > 1）。"""
    input_ids = [b["input_ids"] for b in batch]
    max_len = max(t.shape[0] for t in input_ids)
    pad_id = batch[0].get("pad_token_id", 0)
    pad = lambda t, v: torch.cat(
        [t, torch.full((max_len - t.shape[0],), v, dtype=torch.long)]
    )
    ids = torch.stack([pad(t, pad_id) for t in input_ids])
    attn = torch.stack([pad(t, 0) for t in [b["attention_mask"] for b in batch]])
    labels = torch.stack(
        [pad(t, -100) for t in [b["labels"] for b in batch]]
    )
    return {"input_ids": ids, "attention_mask": attn, "labels": labels}


def main_train(config_path: str) -> None:
    import yaml

    from ..milestones.qwen35_text import load_text_tokenizer

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    tokenizer = load_text_tokenizer(cfg["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token

    from ..milestones.qwen35_text import load_text_causal_model

    model = load_text_causal_model(cfg["model_name_or_path"])

    train_ds = NativeSearchSFTDataset(
        cfg["train_data_path"], tokenizer, max_length=cfg.get("max_length", 8192)
    )

    exp_dir = os.path.join(cfg["output_dir"], "models", cfg["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)
    ckpt_dir = os.path.join(exp_dir, "final")

    # 24GB 单卡微调 2B 模型：用 8-bit Adam 优化器 + 梯度检查点以控制显存
    use_8bit = cfg.get("optim", "adamw_torch") == "adamw_bnb_8bit"
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    args = TrainingArguments(
        output_dir=exp_dir,
        per_device_train_batch_size=cfg.get("per_device_batch_size", 1),
        gradient_accumulation_steps=cfg.get("grad_accum_steps", 8),
        learning_rate=cfg.get("learning_rate", 5e-5),
        weight_decay=cfg.get("weight_decay", 0.01),
        num_train_epochs=cfg.get("num_epochs", 1),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        logging_steps=cfg.get("log_freq_steps", 5),
        save_steps=cfg.get("save_freq_steps", 200),
        save_total_limit=2,
        save_strategy="steps",
        prediction_loss_only=True,
        remove_unused_columns=False,
        bf16=cfg.get("use_bf16", True),
        optim="adamw_bnb_8bit" if use_8bit else cfg.get("optim", "adamw_torch"),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        report_to="none",
        disable_tqdm=False,
        seed=cfg.get("seed", 42),
        logging_first_step=True,
        length_column_name="n_tokens",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collate_for_native,
    )
    print(f"\n[Qwen3] 开始搜索轨迹 SFT：{len(train_ds)} 样本, 输出到 {exp_dir}\n")
    trainer.train()

    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    print(f"[Qwen3] 已保存最终模型 -> {ckpt_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3.5 原生搜索轨迹 SFT")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main_train(args.config)