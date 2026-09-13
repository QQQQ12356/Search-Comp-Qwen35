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
import time
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from ..utils.runtime import (
    count_parameters,
    load_yaml_config,
    prepare_run_artifacts,
    require_keys,
    resolve_experiment_dir,
    write_json,
)
from ..utils.trainer_callbacks import JsonlMetricsCallback

from ..data.trajectory import (
    build_loss_labels,
    build_search_chat_prompt,
    build_sequence_ids,
)


class NativeSearchSFTDataset(Dataset):
    """把交互式搜索轨迹 jsonl 转成标准 (input_ids, attention_mask, labels)。

    每样本的 chat 前缀由 ``build_search_chat_prompt`` 统一构造：搜索协议位于
    system，首个 user 消息只包含问题文本。
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
        chat_input = build_search_chat_prompt(q, add_generation_prompt=True)
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


def main_train(
    config_path: str,
    overrides=(),
    resume_from_checkpoint: str | None = None,
) -> None:
    from ..milestones.qwen35_text import load_text_tokenizer

    cfg = load_yaml_config(config_path, overrides)
    require_keys(cfg, ("model_name_or_path", "train_data_path", "output_dir", "exp_name"))

    tokenizer = load_text_tokenizer(cfg["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token

    from ..milestones.qwen35_text import load_text_causal_model

    model = load_text_causal_model(cfg["model_name_or_path"])

    train_ds = NativeSearchSFTDataset(
        cfg["train_data_path"], tokenizer, max_length=cfg.get("max_length", 8192)
    )
    eval_ds = None
    if cfg.get("val_data_path"):
        eval_ds = NativeSearchSFTDataset(
            cfg["val_data_path"], tokenizer, max_length=cfg.get("max_length", 8192)
        )

    exp_dir = str(resolve_experiment_dir(cfg))
    prepare_run_artifacts(exp_dir, cfg, config_path, overrides)
    ckpt_dir = os.path.join(exp_dir, "final")

    # 24GB 单卡微调 2B 模型：用 8-bit Adam 优化器 + 梯度检查点以控制显存
    use_8bit = cfg.get("optim", "adamw_torch") == "adamw_bnb_8bit"
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    args = TrainingArguments(
        output_dir=exp_dir,
        per_device_train_batch_size=cfg.get("per_device_batch_size", 1),
        gradient_accumulation_steps=cfg.get("grad_accum_steps", 8),
        learning_rate=cfg.get("learning_rate", 5e-5),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        weight_decay=cfg.get("weight_decay", 0.01),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        num_train_epochs=cfg.get("num_epochs", 1),
        max_steps=cfg.get("max_train_steps", -1) or -1,
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        logging_steps=cfg.get("logging_steps", cfg.get("log_freq_steps", 5)),
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
        include_num_input_tokens_seen=True,
        length_column_name="n_tokens",
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.get("eval_steps", cfg.get("save_freq_steps", 200)),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        run_name=cfg["exp_name"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_for_native,
        callbacks=[JsonlMetricsCallback(exp_dir, append=resume_from_checkpoint is not None)],
    )
    parameter_stats = count_parameters(model)
    write_json(
        os.path.join(exp_dir, "dataset_summary.json"),
        {
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds) if eval_ds is not None else 0,
            **parameter_stats,
        },
    )
    print(
        f"\n[native-train] 样本={len(train_ds)} 可训练参数="
        f"{parameter_stats['trainable_parameters'] / 1e6:.2f}M 输出={exp_dir}\n",
        flush=True,
    )
    started_at = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    summary = {
        "total_steps": trainer.state.global_step,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "final_model_path": ckpt_dir,
        "train_metrics": train_result.metrics,
        **parameter_stats,
    }
    write_json(os.path.join(exp_dir, "train_summary.json"), summary)
    print(f"[native-train] 已保存最终模型 -> {ckpt_dir}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3.5 原生搜索轨迹 SFT")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="覆盖 YAML 参数，可重复，例如 --set learning_rate=1e-5",
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="Trainer checkpoint 路径；原生训练支持完整断点续训",
    )
    args = parser.parse_args()
    main_train(args.config, args.overrides, args.resume_from_checkpoint)
