"""模型加载/保存工具。

负责：
- 按配置加载 Qwen2 基础模型并挂载 beacon 参数。
- 保存带 beacon 参数的模型（config.json 中含 beacon_* 字段）。
- 参数冻结（可选：只训练 beacon 参数）。
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel

from .beacon_config import BeaconConfig
from .beacon_qwen2 import BeaconQwen2ForCausalLM

#: 架构名 -> beacon 模型类的映射（当前仅支持 Qwen2 系列）
#: 包含原始架构名与保存后的 beacon 架构名（BeaconQwen2ForCausalLM）
BEACON_ARCH_CLASSES = {
    "Qwen2ForCausalLM": BeaconQwen2ForCausalLM,
    "BeaconQwen2ForCausalLM": BeaconQwen2ForCausalLM,
}


def _normalize_arch(arch: str) -> str:
    """把保存后的 beacon 架构名归一化为可识别的架构名。"""
    if "Qwen2ForCausalLM" in arch:
        return "Qwen2ForCausalLM"
    return arch


def load_model(
    model_name_or_path: str,
    beacon_config: Optional[BeaconConfig] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "eager",
    device_map: Optional[str] = None,
    trust_remote_code: bool = False,
) -> PreTrainedModel:
    """加载带 beacon 压缩的模型。

    Args:
        model_name_or_path: 本地目录或 HF hub 上的 Qwen2 系列模型。
        beacon_config: beacon 超参数；若为 None 则从模型 config 中读取，
            不存在时使用默认值。
        torch_dtype: 权重数据类型（默认 bf16）。
        attn_implementation: 注意力实现（beacon 前向强制使用 eager 以保证正确性）。
        device_map: 设备映射（单卡直接传 device 或 None）。
        trust_remote_code: 是否信任远程代码。

    Returns:
        已挂载 beacon 参数的 ``BeaconQwen2ForCausalLM``。

    Raises:
        ValueError: 模型架构不受支持时抛出。
        FileNotFoundError: 模型路径不存在时抛出。
    """
    if not os.path.exists(model_name_or_path):
        # 允许传入 HF hub 模型名（下载）
        print(f"[load_model] 模型路径不存在，尝试从 HF hub 加载: {model_name_or_path}")

    try:
        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"无法加载模型配置 {model_name_or_path}，请检查路径或网络。"
        ) from exc

    arch = config.architectures[0] if config.architectures else None
    arch = _normalize_arch(arch) if arch else None
    if arch not in BEACON_ARCH_CLASSES:
        raise ValueError(
            f"不支持的模型架构: {arch}。当前支持: {list(BEACON_ARCH_CLASSES.keys())}"
        )

    # 关键：把 beacon 配置合并进 config **再实例化模型**，
    # 否则 BeaconQwen2Attention.__init__ 读不到 beacon_param，beacon 投影不会创建。
    if beacon_config is not None:
        beacon_config.merge_into_config(config)

    model_cls = BEACON_ARCH_CLASSES[arch]
    try:
        model = model_cls.from_pretrained(
            model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"加载模型 {model_name_or_path} 失败: {exc}") from exc

    # 覆盖/写入 beacon 配置
    if beacon_config is not None:
        model.beacon_config = beacon_config
        model.memory = model.memory.__class__(model.config, beacon_config)

    if device_map is not None:
        model = model.to(device_map)
    return model


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = False):
    """加载 tokenizer，并确保已知特殊 token 存在。"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def freeze_llm_except_beacon(model: PreTrainedModel) -> None:
    """冻结 LLM 主体参数，只训练 beacon 参数（beacon_embed + beacon_*_proj）。

    Args:
        model: 已挂载 beacon 参数的模型。
    """
    for name, param in model.named_parameters():
        is_beacon = ("beacon" in name) or (name == "lm_head.weight")
        param.requires_grad = is_beacon


def get_trainable_param_stats(model: PreTrainedModel) -> dict:
    """统计可训练参数数量，返回 dict。"""
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return {"total_params": total, "trainable_params": trainable}


def save_model(model: PreTrainedModel, tokenizer, output_dir: str) -> None:
    """保存模型、tokenizer 与配置到目录。

    Args:
        model: beacon 模型。
        tokenizer: tokenizer。
        output_dir: 输出目录（不存在则创建）。

    Raises:
        OSError: 目录写入失败时抛出。
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
    except Exception as exc:  # noqa: BLE001
        raise OSError(f"保存模型到 {output_dir} 失败: {exc}") from exc
