"""把复合损失的各个组成分量并入 Trainer 的日志。

Beacon 的损失是「主 CE + 续写 + 读出蒸馏」三项之和，但 ``Trainer`` 默认只记录
``outputs.loss`` 一个标量，看不出各项各占多少。模型在
:attr:`~search_comp.models.beacon_qwen3.BeaconQwen3_5ForCausalLM._last_loss_parts`
里按名留痕（detach 后的标量），本模块的 :class:`LossComponentTrainer` 把它按
**log 窗口**对 micro-batch 取算术平均后写进 ``logs``：

- 终端打印的损失行；
- ``trainer_metrics.jsonl``（:class:`~search_comp.utils.trainer_callbacks.JsonlMetricsCallback`）；
- ``trainer_state_summary.json`` 的 ``log_history``。

三处都会带上 ``ce_loss`` / ``cont_loss`` / ``readout_loss``。未激活的项恒为 ``0``，
因此日志的键集合稳定，不会随配置变化。

只在训练日志行（含 ``loss`` 键）注入；eval 行（``eval_loss``）不注入，eval 前向也
不计入窗口。分量以张量形式累加、只在 log 时取值，不额外触发 GPU 同步。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from transformers import Trainer

#: 模型上存放复合损失分量的属性名。
LOSS_PARTS_ATTRIBUTE = "_last_loss_parts"


def find_beacon_module(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    """解开 LoRA/PEFT 包装，找到承载损失分量的 beacon 模型。

    ``Trainer`` 拿到的是 PEFT 包装后的模型（``PeftModel.base_model.model`` 才是
    :class:`BeaconQwen3_5ForCausalLM`），沿途的每层都可能是别的类型。

    Args:
        model: Trainer 正在训练的模型，可能是 :class:`~peft.PeftModel`。

    Returns:
        带 :data:`LOSS_PARTS_ATTRIBUTE` 的 beacon 模型；模型不含 beacon 时返回 ``None``
        （此时日志里就不出现分量，不影响训练）。
    """
    from ..models.beacon_qwen3 import BeaconQwen3_5ForCausalLM

    seen = set()
    current: Optional[torch.nn.Module] = model
    while current is not None and id(current) not in seen:
        if isinstance(current, BeaconQwen3_5ForCausalLM):
            return current
        seen.add(id(current))
        # PeftModel.base_model（LoraModel）-> .model；显式判空，避免依赖模块的真值。
        nxt = getattr(current, "base_model", None)
        if nxt is None:
            nxt = getattr(current, "model", None)
        current = nxt
    return None


class LossComponentTrainer(Trainer):
    """在训练日志行里额外打印复合损失的每个分量。

    用法与 :class:`transformers.Trainer` 完全一致，直接替换即可。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._loss_part_totals: Dict[str, torch.Tensor] = {}
        self._loss_part_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        # eval 前向（model.eval()）不污染下一个训练日志窗口。
        if model.training:
            self._accumulate_loss_parts(model)
        return loss

    def _accumulate_loss_parts(self, model) -> None:
        """把本次前向的分量累加进当前 log 窗口。"""
        module = find_beacon_module(model)
        parts = getattr(module, LOSS_PARTS_ATTRIBUTE, None) if module is not None else None
        if not parts:
            return
        for name, value in parts.items():
            total = self._loss_part_totals.get(name)
            self._loss_part_totals[name] = value if total is None else total + value
        self._loss_part_count += 1

    def log(self, logs: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """把窗口内的分量均值写进 ``logs`` 并清零，再交给父类打印。"""
        # 训练日志行带 loss；eval 行只有 eval_loss，不应混入训练分量。
        if self._loss_part_count and "loss" in logs:
            for name, total in self._loss_part_totals.items():
                logs[name] = float(total) / self._loss_part_count
            self._loss_part_totals = {}
            self._loss_part_count = 0
        return super().log(logs, *args, **kwargs)
