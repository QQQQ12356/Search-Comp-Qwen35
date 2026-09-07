"""Qwen3.5 Beacon 搜索轨迹 SFT 训练入口（transformers.Trainer，实时进度）。

在交互式搜索轨迹上训练 Beacon 压缩模型：

- 只压缩检索到的 ``<information>`` 文档块（``regions``），文档 K/V 压缩为 beacon。
- 损失只在模型生成片段（think / <search> / <answer>）上计算，不含检索内容。
- 用 ``transformers.Trainer`` 驱动训练：实时 tqdm 进度条、按 ``logging_steps`` 打印
  损失/学习率/耗时，``max_steps`` 天然支持"按训练步数训练"。

注意：Trainer 默认保存只写 LoRA adapter，无法被 ``load_beacon_qwen3_5`` 直接加载，
因此关闭 Trainer 自带保存（``save_strategy="no"``），改由自定义 callback 合并保存为
标准 ``BeaconQwen3_5ForCausalLM`` checkpoint（``checkpoint-<N>`` / ``final``）。

用法::

    python -m search_comp.trainer.beacon_trainer --config configs/train/beacon_qwen3.5_searchr1.yaml
"""

from __future__ import annotations

import json
import os
import time

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

from ..data.interactive_dataset import InteractiveCollator, InteractiveSFTDataset
from ..models.beacon_config import BeaconConfig
from ..milestones.qwen35_text import load_text_tokenizer


class _SaveBeaconCheckpoints(TrainerCallback):
    """按步数合并保存模型为标准 Beacon checkpoint（LoRA 需 merge）。

    Trainer 默认 save 只存 LoRA adapter，不能被 ``load_beacon_qwen3_5`` 直接加载，
    故 ``save_strategy="no"``，由本回调统一合并保存。

    Args:
        exp_dir: 输出目录（``output_dir/models/<exp_name>``）。
        save_freq: 每多少优化器步保存一次。
        model: 训练中的模型（可能是 PEFT 包装）。
        tokenizer: 用于随 checkpoint 保存。
        use_lora: 是否 LoRA（决定保存前是否 merge）。
        max_steps: 按步数训练的终止步数；为 None 时按 epoch 终止。
    """

    def __init__(self, exp_dir: str, save_freq: int, model, tokenizer,
                 use_lora: bool, max_steps) -> None:
        self.exp_dir = exp_dir
        self.save_freq = save_freq
        self.model = model
        self.tokenizer = tokenizer
        self.use_lora = use_lora
        self.max_steps = max_steps

    def on_step_end(self, args, state, control, **kwargs):
        gs = state.global_step
        if gs > 0 and gs % self.save_freq == 0:
            ckpt = os.path.join(self.exp_dir, f"checkpoint-{gs}")
            _save_model(self.model, self.tokenizer, ckpt, self.use_lora)
            print(f"[beacon] 已保存 checkpoint -> {ckpt}", flush=True)
        if self.max_steps is not None and gs >= self.max_steps:
            control.should_training_stop = True  # 精确在 max_steps 处停止

    def on_train_end(self, args, state, control, **kwargs):
        final_dir = os.path.join(self.exp_dir, "final")
        _save_model(self.model, self.tokenizer, final_dir, self.use_lora)
        print(f"[beacon] 训练结束，模型已保存到 {final_dir}", flush=True)


def main_train(config_path: str) -> None:
    import yaml

    from ..models.beacon_qwen3 import load_beacon_qwen3_5

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))

    tokenizer = load_text_tokenizer(cfg["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token

    beacon_cfg = BeaconConfig.from_dict(cfg.get("beacon", {}))
    model = load_beacon_qwen3_5(cfg["model_name_or_path"], beacon_config=beacon_cfg)
    model.train()
    # 梯度检查点：降低激活显存（对 linear_attention 层的 O(seq) 中间量尤其关键）
    model._use_gradient_checkpointing = cfg.get("gradient_checkpointing", True)

    # 可选 LoRA：冻结基础模型，仅在注意力/MLP 投影上训练低秩适配器，
    # beacon 参数保持全量可训练。显著降低优化器/梯度显存（约 -10GB）。
    use_lora = cfg.get("use_lora", False)
    if use_lora:
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        # beacon 投影（beacon_*_proj / beacon_embed_tokens）保持全量可训练。
        # 关键：get_peft_model 会把所有基础参数 requires_grad=False（含 beacon），
        # 若不重新开启，loss 将不连接任何可训练参数，backward 报
        # "element 0 ... does not require grad"。
        for name, param in model.named_parameters():
            if "beacon" in name:
                param.requires_grad = True
        beacon_trainable = [
            n for n, p in model.named_parameters() if "beacon" in n and p.requires_grad
        ]
        if not beacon_trainable:
            raise RuntimeError(
                "LoRA 模式下未找到任何 require_grad 的 beacon 压缩参数；"
                "请检查 beacon_config.enable_beacon 与模型结构。"
            )
        model.print_trainable_parameters()

    data_mode = cfg.get("data_mode", "interactive")
    if data_mode == "searchr1":
        from ..data.searchr1_dataset import SearchR1Collator, SearchR1SFTDataset

        train_ds = SearchR1SFTDataset(
            cfg["train_data_path"], tokenizer, max_length=cfg.get("max_length", 8192)
        )
        collator = SearchR1Collator(tokenizer)  # bs=1
        print(f"[beacon] data_mode=searchr1，{len(train_ds)} 条 Search-R1 SFT 轨迹", flush=True)
    else:
        train_ds = InteractiveSFTDataset(
            cfg["train_data_path"], tokenizer, max_length=cfg.get("max_length", 8192)
        )
        collator = InteractiveCollator(tokenizer)  # bs=1
        print(f"[beacon] data_mode=interactive，{len(train_ds)} 条交互式轨迹", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(
        f"[beacon] 可训练参数 {n_train/1e6:.2f}M / 总 "
        f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M",
        flush=True,
    )

    # 24GB 单卡：优先 8-bit Adam 控制优化器显存（回退普通 AdamW）
    try:
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            trainable, lr=cfg.get("learning_rate", 5e-5), weight_decay=cfg.get("weight_decay", 0.01)
        )
        print("[beacon] 使用 8-bit AdamW 优化器", flush=True)
    except ImportError:
        optimizer = torch.optim.AdamW(
            trainable, lr=cfg.get("learning_rate", 5e-5), weight_decay=cfg.get("weight_decay", 0.01)
        )

    grad_accum = cfg.get("grad_accum_steps", 8)
    save_freq = cfg.get("save_freq_steps", 200)
    logging_steps = cfg.get("logging_steps", 5)
    max_train_steps = cfg.get("max_train_steps", None)

    exp_dir = os.path.join(cfg["output_dir"], "models", cfg["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)

    if max_train_steps is not None:
        max_train_steps = int(max_train_steps)
        print(f"[beacon] 按训练步数训练：max_train_steps={max_train_steps} "
              f"(steps/epoch≈{len(train_ds)//grad_accum})", flush=True)
    else:
        print(f"[beacon] 按 epoch 训练：num_epochs={cfg.get('num_epochs', 1)} "
              f"(steps/epoch≈{len(train_ds)//grad_accum})", flush=True)

    args = TrainingArguments(
        output_dir=exp_dir,
        per_device_train_batch_size=1,          # collator 强制 bs=1，靠 grad_accum 扩大有效 batch
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.get("learning_rate", 5e-5),
        weight_decay=cfg.get("weight_decay", 0.01),
        # 二选一：max_steps>0 则按步数训练（覆盖 num_epochs）；否则按 epoch
        max_steps=max_train_steps if max_train_steps is not None else -1,
        num_train_epochs=cfg.get("num_epochs", 1),
        logging_steps=logging_steps,
        log_level="info",
        save_strategy="no",                     # 由 _SaveBeaconCheckpoints 合并保存
        report_to=[],                            # 不写 wandb/tensorboard
        remove_unused_columns=False,             # 保留 collator 的 regions 透传给 forward
        seed=cfg.get("seed", 42),
        disable_tqdm=False,                      # 实时进度条
        fp16=False,
        bf16=False,                              # 参数本身为 bf16，交由 GPU 原生计算
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    # 恒定学习率（与原始手写循环一致）；显式传入使 Trainer 不再自建调度器
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[
            _SaveBeaconCheckpoints(exp_dir, save_freq, model, tokenizer, use_lora, max_train_steps)
        ],
    )

    t0 = time.time()
    print("[beacon] 开始训练 ...", flush=True)
    trainer.train()

    final_dir = os.path.join(exp_dir, "final")
    with open(os.path.join(exp_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"total_steps": trainer.state.global_step,
             "elapsed_seconds": round(time.time() - t0, 1)},
            f, indent=2,
        )
    print(f"[beacon] 训练完成，训练步数={trainer.state.global_step}，耗时 {time.time()-t0:.1f}s，"
          f"模型保存到 {final_dir}")


def _save_model(model, tokenizer, save_dir: str, use_lora: bool) -> None:
    """保存模型：LoRA 时先合并权重，存成标准 BeaconQwen3_5ForCausalLM checkpoint。"""
    if use_lora:
        merged = model.merge_and_unload()
        merged.save_pretrained(save_dir)
    else:
        model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3.5 Beacon 搜索轨迹 SFT")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main_train(args.config)