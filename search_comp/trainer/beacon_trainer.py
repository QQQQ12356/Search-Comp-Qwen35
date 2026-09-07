"""Qwen3.5 Beacon 搜索轨迹 SFT 训练入口（tqdm 可视化）。

在交互式搜索轨迹上训练 Beacon 压缩模型：

- 只压缩检索到的 ``<information>`` 文档块（``regions``），文档 K/V 压缩为 beacon。
- 损失只在模型生成片段（think / <search> / <answer>）上计算，不含检索内容。

Beacon 模型 ``forward`` 返回 ``(loss, batch_loss)`` 元组（非 Trainer 兼容的
ModelOutput），因此用轻量手写训练循环 + tqdm 进度条可视化（单卡）。

用法::

    python -m search_comp.trainer.beacon_trainer --config configs/train/beacon_qwen3.5.yaml
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.interactive_dataset import InteractiveCollator, InteractiveSFTDataset
from ..models.beacon_config import BeaconConfig
from ..milestones.qwen35_text import load_text_tokenizer


def main_train(config_path: str) -> None:
    import yaml

    from ..models.beacon_qwen3 import load_beacon_qwen3_5

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))

    tokenizer = load_text_tokenizer(cfg["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token

    beacon_cfg = BeaconConfig.from_dict(cfg.get("beacon", {}))
    model = load_beacon_qwen3_5(cfg["model_name_or_path"], beacon_config=beacon_cfg)
    model.to(device).train()
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
        # beacon 投影（beacon_*_proj / beacon_embed_tokens）保持全量可训练
        for name, param in model.named_parameters():
            if "beacon" in name:
                param.requires_grad = True
        model.print_trainable_parameters()

    data_mode = cfg.get("data_mode", "interactive")
    if data_mode == "searchr1":
        from ..data.searchr1_dataset import SearchR1Collator, SearchR1SFTDataset

        train_ds = SearchR1SFTDataset(
            cfg["train_data_path"], tokenizer, max_length=cfg.get("max_length", 8192)
        )
        collator = SearchR1Collator(tokenizer)  # bs=1
        print(f"[beacon] data_mode=searchr1，{len(train_ds)} 条 Search-R1 SFT 轨迹")
    else:
        train_ds = InteractiveSFTDataset(
            cfg["train_data_path"], tokenizer, max_length=cfg.get("max_length", 8192)
        )
        collator = InteractiveCollator(tokenizer)  # bs=1
        print(f"[beacon] data_mode=interactive，{len(train_ds)} 条交互式轨迹")
    dataloader = DataLoader(train_ds, batch_size=1, collate_fn=collator, shuffle=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"[beacon] 可训练参数 {n_train/1e6:.2f}M / 总 {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # 24GB 单卡：优先 8-bit Adam 控制优化器显存（回退普通 AdamW）
    try:
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            trainable, lr=cfg.get("learning_rate", 5e-5), weight_decay=cfg.get("weight_decay", 0.01)
        )
        print("[beacon] 使用 8-bit AdamW 优化器")
    except ImportError:
        optimizer = torch.optim.AdamW(
            trainable, lr=cfg.get("learning_rate", 5e-5), weight_decay=cfg.get("weight_decay", 0.01)
        )
    grad_accum = cfg.get("grad_accum_steps", 8)
    num_epochs = cfg.get("num_epochs", 1)
    total_steps = len(dataloader) * num_epochs // grad_accum

    exp_dir = os.path.join(cfg["output_dir"], "models", cfg["exp_name"])
    os.makedirs(exp_dir, exist_ok=True)

    global_step = 0
    t0 = time.time()
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"epoch {epoch+1}/{num_epochs}", ncols=100)
        running = 0.0
        for step, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            regions = batch["regions"]

            loss, _ = model(
                input_ids=ids, attention_mask=attn, labels=labels, compress_regions=regions
            )
            if loss is None:
                continue
            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                running += loss.item() * grad_accum
                pbar.set_postfix(loss=f"{running:.4f}", lr=f"{cfg.get('learning_rate',5e-5):.1e}", step=global_step)
                running = 0.0
                if global_step % cfg.get("save_freq_steps", 200) == 0:
                    ckpt = os.path.join(exp_dir, f"checkpoint-{global_step}")
                    _save_model(model, tokenizer, ckpt, use_lora)
                    print(f"[beacon] 已保存 checkpoint -> {ckpt}")

    final_dir = os.path.join(exp_dir, "final")
    _save_model(model, tokenizer, final_dir, use_lora)
    with open(os.path.join(exp_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"total_steps": global_step, "elapsed_seconds": round(time.time() - t0, 1)}, f, indent=2)
    print(f"[beacon] 训练完成，模型保存到 {final_dir}")


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