"""Activate Beacon 压缩机制在 Qwen3.5 上的移植（只压缩检索到的内容）。

背景：``Qwen/Qwen3.5-2B`` 是混合架构 —— 24 层中仅 6 层（``layer_types`` 为
``full_attention``）带标准 QKV 注意力缓存，其余 18 层为 ``linear_attention``
（GatedDeltaNet，固定尺寸循环状态）。

语义（与参考实现 [[beacon_qwen2]] / [[beacon_memory]] 完全一致）：

- **只压缩检索内容**：由 ``regions``（各 ``<information>`` 块 token 区间）指定，
  按 ``beacon_window`` 分 chunk；append 在末尾追加 beacon，intersect 在 chunk
  内每 ``beacon_ratio`` 个 token 后插入一个 beacon。chunk 结束仅缓存 beacon。
- **损失不含检索内容**：labels 全局预移位后，文档与 beacon 位置为 -100，
  只在模型生成片段（think / <search> / <answer>）上计算损失。
- **混合层都只提交 Beacon 载体**：full_attention 层只持久化 beacon K/V；
  linear_attention 层把压缩窗的原生 DeltaNet/卷积状态视为临时 reader 状态，
  窗结束后丢弃，并仅由 beacon 激活经可训练 writer 重建固定尺寸循环状态。

``enable_beacon=False`` 退化为原生 ``Qwen3_5ForCausalLM`` 前向。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5ForCausalLM,
    apply_rotary_pos_emb,
)

from .beacon_config import BeaconConfig
from .beacon_question_memory import QuestionMemoryMixin
from .modeling_utils import beacon_intersect_order, cat_tensor, slice_tensor


class BeaconLinearStateWriter(nn.Module):
    """仅从 Beacon 激活更新 GatedDeltaNet 的持久循环状态。"""

    def __init__(self, config, rank: int):
        super().__init__()
        self.num_heads = config.linear_num_value_heads
        self.key_dim = config.linear_key_head_dim
        self.value_dim = config.linear_value_head_dim
        self.norm = nn.LayerNorm(config.hidden_size)
        self.down = nn.Linear(config.hidden_size, rank, bias=False)
        self.up = nn.Linear(
            rank,
            self.num_heads * (self.key_dim + self.value_dim + 2),
        )
        with torch.no_grad():
            self.up.bias.zero_()
            self.up.bias[-2 * self.num_heads:-self.num_heads].fill_(4.0)

    def forward(self, beacons, previous):
        batch = beacons.shape[0]
        projected = self.up(F.silu(self.down(self.norm(beacons))))
        key, value, decay, beta = projected.split(
            [
                self.num_heads * self.key_dim,
                self.num_heads * self.value_dim,
                self.num_heads,
                self.num_heads,
            ],
            dim=-1,
        )
        key = key.reshape(batch, -1, self.num_heads, self.key_dim).float()
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        value = value.reshape(batch, -1, self.num_heads, self.value_dim).float()
        if previous is None:
            state = torch.zeros(
                batch,
                self.num_heads,
                self.key_dim,
                self.value_dim,
                device=beacons.device,
                dtype=torch.float32,
            )
        else:
            state = previous.float().clone()
        for token_idx in range(beacons.shape[1]):
            state = state * decay[:, token_idx].float().sigmoid()[..., None, None]
            token_key = key[:, token_idx]
            prediction = (state * token_key.unsqueeze(-1)).sum(-2)
            update = (value[:, token_idx] - prediction) * beta[:, token_idx].float().sigmoid().unsqueeze(-1)
            state = state + token_key.unsqueeze(-1) * update.unsqueeze(-2)
        return state


# ======================================================================
# Beacon Attention（full_attention 层）
# ======================================================================
class QuestionBeaconLinearStateWriter(QuestionMemoryMixin, BeaconLinearStateWriter):
    def __init__(self, config, rank):
        super().__init__(config, rank)
        self.init_question_memory(config.hidden_size, rank)


class BeaconQwen3Attention(Qwen3_5Attention):
    """带 beacon 投影切换的 Qwen3.5 注意力（仅 full_attention 层使用）。

    ``q_proj`` 输出 ``num_heads * head_dim * 2``（query + 门控），beacon_q_proj
    同宽，门控随 query 一起被 ``torch.where`` 切换。K/V 缓存保存 **RoPE 之前**
    的 key，每个窗口对 ``past + current`` 整体重施加 mRoPE。
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.beacon_param = getattr(config, "beacon_param", "").split()

        if "q" in self.beacon_param:
            self.beacon_q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim * 2, bias=config.attention_bias
            )
            self.beacon_q_proj.weight.data.zero_()
        if "k" in self.beacon_param:
            self.beacon_k_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
            )
            self.beacon_k_proj.weight.data.zero_()
        if "v" in self.beacon_param:
            self.beacon_v_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
            )
            self.beacon_v_proj.weight.data.zero_()
        if "o" in self.beacon_param:
            self.beacon_o_proj = nn.Linear(
                self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
            )
            self.beacon_o_proj.weight.data.zero_()

    # ------------------------------------------------------------------
    def _qkv(self, hidden_states, beacon_size, beacon_indices):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cur = hidden_states.shape[1]
        bi = beacon_indices[-cur:] if beacon_size > 0 else None

        if beacon_size > 0 and "q" in self.beacon_param:
            oq = self.q_proj(hidden_states)
            bq = self.beacon_q_proj(hidden_states)
            qg = torch.where((bi == 0)[:, None], oq, bq)
        else:
            qg = self.q_proj(hidden_states)
        query_states, gate = torch.chunk(qg.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)

        if beacon_size > 0 and "k" in self.beacon_param:
            ok = self.k_proj(hidden_states)
            bk = self.beacon_k_proj(hidden_states)
            key_states = torch.where((bi == 0)[:, None], ok, bk)
        else:
            key_states = self.k_proj(hidden_states)
        if beacon_size > 0 and "v" in self.beacon_param:
            ov = self.v_proj(hidden_states)
            bv = self.beacon_v_proj(hidden_states)
            value_states = torch.where((bi == 0)[:, None], ov, bv)
        else:
            value_states = self.v_proj(hidden_states)

        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states, gate, input_shape

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        **kwargs,
    ):
        """分派：beacon 窗口路径 vs 原生前向。

        - 传 ``past_key_value``（4 元组）时走 beacon 窗口路径（由 BeaconMemory 驱动）。
        - 否则委托原生 ``Qwen3_5Attention.forward``（``enable_beacon=False`` 时
          整条序列正常前向，保证与原生等价）。
        """
        if isinstance(past_key_value, tuple) and len(past_key_value) == 4:
            return self._beacon_forward(hidden_states, position_embeddings, attention_mask, past_key_value)
        return super().forward(
            hidden_states, position_embeddings=position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values, **kwargs,
        )

    def _beacon_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Tuple,
    ) -> Tuple[torch.Tensor, Tuple]:
        """窗口前向。

        Args:
            hidden_states: ``(bsz, cur+beacon, hidden)`` 当前窗口。
            position_embeddings: ``(cos, sin)``，覆盖 ``[0, mem+cur+beacon)`` 全长。
            attention_mask: 4D 因果掩码 ``(bsz, 1, cur+beacon, mem+cur+beacon)``。
            past_key_value: 4 元组 ``(key, value, beacon_size, beacon_indices)``，
                key/value 为 **RoPE 之前** 的历史 K/V。

        Returns:
            ``(attn_output, new_past)``；new_past = 当前**增量**、RoPE 之前的
            ``(key, value, beacon_size, beacon_indices)``。
        """
        past_key, past_value, beacon_size, beacon_indices = past_key_value
        cur_len = hidden_states.shape[1]

        query_states, key_states, value_states, gate, input_shape = self._qkv(
            hidden_states, beacon_size, beacon_indices
        )

        # 增量、RoPE 之前的 K/V（供记忆缓存，下个窗口整体重施加 RoPE）
        new_past = (key_states, value_states, beacon_size, beacon_indices)

        if past_key is not None:
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        cos, sin = position_embeddings
        # query 用当前窗口位置（末尾 cur_len 位），key 用全局位置（全长）。
        # apply_rotary_pos_emb 内部会自行 unsqueeze，这里无需额外扩展。
        q_cos = cos[:, -cur_len:, :]
        q_sin = sin[:, -cur_len:, :]
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, q_cos, q_sin)
        key_states, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin)

        # 组扩展
        g = self.num_key_value_groups
        if g > 1:
            key_states = key_states.repeat_interleave(g, dim=1)
            value_states = value_states.repeat_interleave(g, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(*input_shape, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        if "o" in self.beacon_param and beacon_size > 0:
            oo = self.o_proj(attn_output)
            bo = self.beacon_o_proj(attn_output)
            bi = beacon_indices[-cur_len:]
            attn_output = torch.where((bi == 0)[:, None], oo, bo)
        else:
            attn_output = self.o_proj(attn_output)
        return attn_output, new_past


# ======================================================================
# Beacon 窗口状态机（只压缩检索内容，线性层显式管理压缩状态）
# ======================================================================
class _Qwen3BeaconMemory:
    """把整条序列切窗处理的状态机。

    - full_attention 层：``_cache[layer_idx] = (K, V)`` 持久缓存（RoPE 之前），
      compress 窗口只保留 beacon K/V，keep 窗口全量保留。
    - linear_attention 层：显式保存循环状态与卷积尾状态；压缩窗只提交由
      beacon writer 生成的循环状态，卷积尾状态清空，杜绝原始文档残留。
    """

    def __init__(self, model, beacon_config: BeaconConfig):
        self.model = model
        self.text = model.model
        self.config = model.config
        self.beacon = beacon_config
        self.layer_types = self.config.layer_types
        self.num_layers = self.config.num_hidden_layers
        # 每层 full-attn 缓存 (K, V)，RoPE 之前
        self._cache: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = [
            (None, None) for _ in range(self.num_layers)
        ]
        self._linear_recurrent = {}
        self._linear_conv = {}
        self.reset()

    def reset(self):
        self._pos = 0
        self._seg_idx = 0
        self._segments: List[Tuple[int, int, str]] = []
        self._cache = [(None, None) for _ in range(self.num_layers)]
        self._linear_recurrent = {}
        self._linear_conv = {}
        self._batch_loss = None
        self._valid_num = None
        self._store = None
        self._step_beacon_indices = None
        self._question = None
        self._question_input_ids = None
        self._readout_losses = []

    @property
    def finish(self) -> bool:
        return self._pos >= self._seq_len

    # ------------------------------------------------------------------
    def prepare(self, input_ids, labels, regions):
        self._seq_len = input_ids.shape[1]
        self._device = input_ids.device
        self._input_ids = input_ids
        # labels 全局预移位：位置 i 监督 token i+1（跨窗口因果监督）
        if labels is not None:
            labels = torch.cat([labels[:, 1:], labels.new_full((labels.shape[0], 1), -100)], dim=1)
        self._labels = labels

        regions = [(int(s), int(e)) for s, e in regions]
        cursor = 0
        for s, e in regions:
            if s > cursor:
                self._segments.append((cursor, s, "keep"))
            self._segments.append((s, e, "compress"))
            cursor = e
        if cursor < self._seq_len:
            self._segments.append((cursor, self._seq_len, "keep"))
        self._seg_idx = 0

    # ------------------------------------------------------------------
    def step(self):
        while self._seg_idx < len(self._segments) and self._segments[self._seg_idx][1] <= self._pos:
            self._seg_idx += 1
        if self._pos >= self._seq_len:
            raise RuntimeError("序列已处理完毕")

        seg_start, seg_end, mode = self._segments[self._seg_idx]
        window = self.beacon.beacon_window
        start = self._pos

        if mode == "keep":
            end = min(start + window, seg_end)
            beacon_size = 0
            store = "cache"
        else:
            end = min(start + window, seg_end)
            remaining = end - start
            if remaining == window:
                beacon_size = self.beacon.beacon_size_per_window
            else:
                beacon_size = max(1, (remaining + self.beacon.beacon_ratio - 1) // self.beacon.beacon_ratio)
            store = "beacon"

        input_ids = self._input_ids[:, start:end]
        labels = self._labels[:, start:end] if self._labels is not None else None

        if beacon_size > 0:
            input_ids = torch.cat(
                [input_ids, input_ids.new_full((input_ids.shape[0], beacon_size), self.config.vocab_size)], dim=1
            )
            if labels is not None:
                labels = torch.cat([labels, labels.new_full((labels.shape[0], beacon_size), -100)], dim=1)

        cur_len = input_ids.shape[1]
        if beacon_size > 0:
            self._step_beacon_indices = torch.cat(
                [torch.zeros(cur_len - beacon_size, dtype=torch.long, device=self._device),
                 torch.ones(beacon_size, dtype=torch.long, device=self._device)]
            )
        else:
            self._step_beacon_indices = None

        if beacon_size > 0 and self.beacon.beacon_pos == "intersect":
            order = beacon_intersect_order(end - start, self.beacon.beacon_ratio, self._device)
            input_ids = input_ids[:, order]
            self._step_beacon_indices = self._step_beacon_indices[order]
            if labels is not None:
                labels = labels[:, order]

        # past：full-attn 层 4 元组
        mem_size = 0
        past = []
        for layer_idx in range(self.num_layers):
            if self.layer_types[layer_idx] == "full_attention":
                ck, cv = self._cache[layer_idx]
                if ck is not None:
                    mem_size = ck.shape[2]
                past.append((layer_idx, (ck, cv, beacon_size, self._step_beacon_indices)))
        # mem_size 取任一 full-attn 层缓存长度（每层长度一致）
        for layer_idx in range(self.num_layers):
            if self.layer_types[layer_idx] == "full_attention":
                ck, _ = self._cache[layer_idx]
                if ck is not None:
                    mem_size = ck.shape[2]
                break

        # 位置：全局单调（含 beacon 占位），用于 mRoPE
        total = mem_size + cur_len
        pos_ids = torch.arange(total, device=self._device).expand(3, total).unsqueeze(1)
        # rotary_emb 只依赖 ref 的 device/dtype（不依赖内容），用空张量避免
        # beacon token id（=vocab_size）越界 embed_tokens。
        ref = torch.zeros(
            1, 1, self.config.hidden_size,
            device=self._device, dtype=self.text.embed_tokens.weight.dtype,
        )
        cos, sin = self.text.rotary_emb(ref, pos_ids)

        attn_mask = self._make_4d_causal_mask(1, cur_len, mem_size, self._device)

        self._store = store
        self._pos = end
        return input_ids, labels, attn_mask, (cos, sin), past

    # ------------------------------------------------------------------
    def update_memory(self, new_past):
        """new_past: list of (layer_idx, (key, value, beacon_size, indices)) 增量、RoPE 前。"""
        for layer_idx, (key, value, _bs, _indices) in new_past:
            ck, cv = self._cache[layer_idx]
            if self._store == "beacon":
                sel = self._step_beacon_indices.bool()
                bk = slice_tensor(key, index=sel, dim=2)
                bv = slice_tensor(value, index=sel, dim=2)
                self._cache[layer_idx] = (cat_tensor([ck, bk], dim=2), cat_tensor([cv, bv], dim=2))
            else:
                self._cache[layer_idx] = (cat_tensor([ck, key], dim=2), cat_tensor([cv, value], dim=2))

    # ------------------------------------------------------------------
    def update_loss(self, batch_loss, valid_num):
        if self._batch_loss is None:
            self._batch_loss = batch_loss * valid_num
            self._valid_num = valid_num
        else:
            self._batch_loss = self._batch_loss + batch_loss * valid_num
            self._valid_num = self._valid_num + valid_num

    def output(self):
        if self._batch_loss is None:
            return None
        total = self._valid_num.sum()
        if total.item() == 0:
            return self._batch_loss.sum() * 0.0
        return self._batch_loss.sum() / total

    # ------------------------------------------------------------------
    def _make_4d_causal_mask(self, batch, q_len, mem_size, device):
        min_value = torch.finfo(torch.float32).min
        full_len = mem_size + q_len
        mask = torch.full((batch, 1, q_len, full_len), min_value, dtype=torch.float32, device=device)
        if mem_size > 0:
            mask[:, :, :, :mem_size] = 0.0
        causal = torch.tril(torch.ones(q_len, q_len, device=device, dtype=torch.bool))
        cur = torch.where(causal, torch.zeros((), device=device), torch.full((), min_value, device=device))
        mask[:, :, :, mem_size:] = cur.unsqueeze(0).unsqueeze(0)
        return mask


# ======================================================================
# ForCausalLM
# ======================================================================
class BeaconQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    """带 beacon 压缩的 Qwen3.5 因果 LM（只压缩检索内容，损失不含检索内容）。"""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.beacon_config = BeaconConfig.from_model_config(config)
        self.model.beacon_embed_tokens = nn.Embedding(1, config.hidden_size)
        self.model.beacon_embed_tokens.weight.data.zero_()
        self.model.beacon_linear_writers = nn.ModuleDict()
        for idx, layer in enumerate(self.model.layers):
            layer_type = getattr(config, "layer_types", [None] * config.num_hidden_layers)[idx]
            if layer_type == "full_attention":
                layer.self_attn = BeaconQwen3Attention(config, idx)
            elif layer_type == "linear_attention":
                writer_type = QuestionBeaconLinearStateWriter if self.beacon_config.beacon_question_memory_v1 else BeaconLinearStateWriter
                self.model.beacon_linear_writers[str(idx)] = writer_type(
                    config,
                    self.beacon_config.beacon_linear_writer_rank,
                )
        self.post_init()
        self._init_beacon_params()

    def _init_beacon_params(self):
        cfg = self.config
        src_id = cfg.eos_token_id
        with torch.no_grad():
            if src_id is not None and src_id < self.model.embed_tokens.num_embeddings:
                self.model.beacon_embed_tokens.weight.data.copy_(self.model.embed_tokens.weight.data[src_id])
            else:
                self.model.beacon_embed_tokens.weight.data.normal_(0, cfg.hidden_size ** -0.5)
        params = self.beacon_config.beacon_param.split()
        for writer in self.model.beacon_linear_writers.values():
            with torch.no_grad():
                writer.up.bias.zero_()
                writer.up.bias[-2 * writer.num_heads:-writer.num_heads].fill_(4.0)
                if isinstance(writer, QuestionBeaconLinearStateWriter):
                    writer.question_up.weight.zero_()
        for idx, layer in enumerate(self.model.layers):
            if getattr(cfg, "layer_types", [None] * cfg.num_hidden_layers)[idx] != "full_attention":
                continue
            attn = layer.self_attn
            with torch.no_grad():
                for p in params:
                    b = getattr(attn, f"beacon_{p}_proj", None)
                    o = getattr(attn, f"{p}_proj", None)
                    if b is None or o is None:
                        continue
                    b.weight.data.copy_(o.weight.data)
                    if b.bias is not None and o.bias is not None:
                        b.bias.data.copy_(o.bias.data)

    def set_beacon_config(self, beacon_config: BeaconConfig):
        if beacon_config.beacon_question_memory_v1 != self.beacon_config.beacon_question_memory_v1:
            raise ValueError("beacon_question_memory_v1 改变模型结构，必须在构造模型前配置")
        for writer in self.model.beacon_linear_writers.values():
            if writer.down.out_features != beacon_config.beacon_linear_writer_rank:
                raise ValueError(
                    "checkpoint 的 beacon_linear_writer_rank 与请求配置不一致: "
                    f"{writer.down.out_features} != {beacon_config.beacon_linear_writer_rank}"
                )
        self.beacon_config = beacon_config
        beacon_config.merge_into_config(self.config)
        return self

    # ------------------------------------------------------------------
    def forward(self, input_ids=None, attention_mask=None, labels=None,
                compress_regions=None, compress_start=None, compress_end=None,
                regions=None, question_input_ids=None, **kwargs):
        if not self.beacon_config.enable_beacon:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        if compress_regions is None and compress_start is not None and compress_end is not None:
            compress_regions = [(int(compress_start), int(compress_end))]
        if compress_regions is None:
            # collator 返回的 batch 键名可能是 regions（Trainer 经 remove_unused_columns=False 透传）
            compress_regions = regions
        if compress_regions is None:
            raise ValueError("beacon 模式必须提供 compress_regions 或 compress_start/compress_end")
        use_cpu_offload = (
            self.training
            and labels is not None
            and self.beacon_config.beacon_cpu_offload_activations
            and (
                self.beacon_config.beacon_cpu_offload_threshold == 0
                or input_ids.shape[1] >= self.beacon_config.beacon_cpu_offload_threshold
            )
        )
        if use_cpu_offload:
            with torch.autograd.graph.save_on_cpu(pin_memory=input_ids.is_cuda):
                loss, logits = self._beacon_forward(input_ids, labels, compress_regions, question_input_ids)
        else:
            loss, logits = self._beacon_forward(input_ids, labels, compress_regions, question_input_ids)
        # 返回 ModelOutput，兼容 transformers.Trainer（取 outputs["loss"]）
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # ------------------------------------------------------------------
    def _embed(self, input_ids, beacon_size):
        """分开嵌入普通 token 与 beacon token（beacon id == vocab_size）。"""
        if beacon_size > 0:
            bi = self._mem._step_beacon_indices
            cur = input_ids.shape[1]
            bi = bi[-cur:] if bi is not None else None
            if bi is not None and bi.any():
                ordinal_ids = input_ids[:, bi == 0]
                beacon_ids = input_ids[:, bi > 0]
                ordinal_emb = self.model.embed_tokens(ordinal_ids)
                beacon_emb = self.model.beacon_embed_tokens(beacon_ids - self.config.vocab_size)
                emb = beacon_emb.new_zeros(*input_ids.shape, beacon_emb.shape[-1])
                emb[:, bi == 0] = ordinal_emb
                emb[:, bi > 0] = beacon_emb
                return emb
        return self.model.embed_tokens(input_ids)

    def _native_forward(
        self,
        win_ids,
        win_labels,
        attn_mask,
        position_embeddings,
        past,
        last_logits_only=False,
    ):
        """单窗口前向。past: list of (layer_idx, 4元组)。返回 (new_past, logits)。"""
        beacon_size = past[0][1][2] if past else 0
        emb = self._embed(win_ids, beacon_size)

        hidden = emb
        new_past = []
        for idx, layer in enumerate(self.model.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            if self._mem.layer_types[idx] == "full_attention":
                pkv = next(pv for li, pv in past if li == idx)
                out, npv = layer.self_attn(
                    hidden, position_embeddings=position_embeddings,
                    attention_mask=attn_mask, past_key_value=pkv,
                )
                new_past.append((idx, npv))
            else:
                previous_recurrent = self._mem._linear_recurrent.get(idx)
                previous_conv = self._mem._linear_conv.get(idx)
                reader_initial_recurrent = previous_recurrent
                reader_initial_conv = previous_conv
                if self._mem._store == "beacon":
                    reader_initial_recurrent = previous_recurrent.clone() if previous_recurrent is not None else None
                    reader_initial_conv = previous_conv.clone() if previous_conv is not None else None
                out, reader_recurrent, reader_conv = self._linear_attention_forward(
                    layer.linear_attn,
                    hidden,
                    reader_initial_recurrent,
                    reader_initial_conv,
                )
            hidden = residual + out
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
            if self._mem.layer_types[idx] == "linear_attention":
                if self._mem._store == "beacon":
                    beacon_mask = self._mem._step_beacon_indices.bool()
                    beacon_hidden = hidden[:, beacon_mask]
                    writer = self.model.beacon_linear_writers[str(idx)]
                    if self.beacon_config.beacon_question_memory_v1:
                        written = writer(beacon_hidden, previous_recurrent, self._mem._question)
                        self._mem._linear_recurrent[idx] = written
                        if self.training and win_labels is not None and self.beacon_config.beacon_readout_distill_weight > 0:
                            self._mem._readout_losses.append(
                                writer.readout_loss(written, reader_recurrent, self._mem._question)
                            )
                    else:
                        self._mem._linear_recurrent[idx] = writer(beacon_hidden, previous_recurrent)
                    self._mem._linear_conv.pop(idx, None)
                else:
                    self._mem._linear_recurrent[idx] = reader_recurrent
                    self._mem._linear_conv[idx] = reader_conv

        hidden = self.model.norm(hidden)
        logits = None

        if win_labels is not None:
            valid_num = (win_labels != -100).sum(-1)
            if valid_num.sum().item() > 0:
                batch_loss = self._sparse_window_loss(hidden, win_labels, valid_num)
                self._mem.update_loss(batch_loss, valid_num)
        elif last_logits_only:
            logits = self.lm_head(hidden[:, -1:, :]).float()
        else:
            logits = self.lm_head(hidden).float()
        return new_past, logits

    def _sparse_window_loss(self, hidden, labels, valid_num):
        """只为非 ``-100`` 位置计算分块词表交叉熵。"""
        chunk_size = self.beacon_config.beacon_loss_chunk_size
        checkpoint_loss = self.beacon_config.beacon_checkpoint_loss and self.training
        batch_losses = []
        for batch_idx in range(hidden.shape[0]):
            selected_hidden = hidden[batch_idx, labels[batch_idx] != -100]
            selected_labels = labels[batch_idx, labels[batch_idx] != -100]
            loss_sum = hidden.new_zeros((), dtype=torch.float32)
            for start in range(0, selected_hidden.shape[0], chunk_size):
                chunk_hidden = selected_hidden[start:start + chunk_size]
                chunk_labels = selected_labels[start:start + chunk_size]

                def loss_function(current_hidden, current_labels):
                    chunk_logits = self.lm_head(current_hidden).float()
                    return F.cross_entropy(chunk_logits, current_labels, reduction="sum")

                if checkpoint_loss and chunk_hidden.requires_grad:
                    chunk_loss = torch.utils.checkpoint.checkpoint(
                        loss_function,
                        chunk_hidden,
                        chunk_labels,
                        use_reentrant=False,
                    )
                else:
                    chunk_loss = loss_function(chunk_hidden, chunk_labels)
                loss_sum = loss_sum + chunk_loss
            batch_losses.append(loss_sum / valid_num[batch_idx].clamp(min=1))
        return torch.stack(batch_losses)

    @staticmethod
    def _linear_attention_forward(module, hidden, previous_recurrent, previous_conv):
        """Qwen3.5 GatedDeltaNet 的显式、可微分状态前向。"""
        batch, length, _ = hidden.shape
        projected = module.in_proj_qkv(hidden).transpose(1, 2)
        if previous_conv is not None:
            conv_input = torch.cat([previous_conv, projected], dim=-1)
        else:
            conv_input = projected
        padded = F.pad(conv_input, (module.conv_kernel_size - conv_input.shape[-1], 0))
        next_conv = padded[..., -module.conv_kernel_size:]
        if module.causal_conv1d_fn is not None:
            mixed = module.causal_conv1d_fn(
                x=conv_input,
                weight=module.conv1d.weight.squeeze(1),
                bias=module.conv1d.bias,
                activation=module.activation,
            )
        else:
            mixed = F.silu(module.conv1d(conv_input)[:, :, :conv_input.shape[-1]])
        mixed = mixed[:, :, -length:].transpose(1, 2)

        query, key, value = mixed.split([module.key_dim, module.key_dim, module.value_dim], dim=-1)
        query = query.reshape(batch, length, -1, module.head_k_dim)
        key = key.reshape(batch, length, -1, module.head_k_dim)
        value = value.reshape(batch, length, -1, module.head_v_dim)
        repeat = module.num_v_heads // module.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
        decay = -module.A_log.float().exp() * F.softplus(
            module.in_proj_a(hidden).float() + module.dt_bias
        )
        output, next_recurrent = module.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=decay,
            beta=module.in_proj_b(hidden).sigmoid(),
            initial_state=previous_recurrent,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        gate = module.in_proj_z(hidden).reshape(-1, module.head_v_dim)
        output = module.norm(output.reshape(-1, module.head_v_dim), gate)
        output = module.out_proj(output.reshape(batch, length, module.value_dim))
        return output, next_recurrent, next_conv

    def _prepare_question(self, question_input_ids):
        if not self.beacon_config.beacon_question_memory_v1:
            return
        if self._mem._question is not None:
            if question_input_ids is not None:
                question_input_ids = question_input_ids[:, :self.beacon_config.beacon_question_max_tokens]
                if not torch.equal(question_input_ids.to(self._mem._question_input_ids.device), self._mem._question_input_ids):
                    raise ValueError("复用缓存时不能切换 question，请使用 reuse_cache=False")
            return
        if question_input_ids is None or question_input_ids.ndim != 2 or question_input_ids.shape[0] != 1 or question_input_ids.shape[1] == 0:
            raise ValueError("question memory v1 首次调用需要非空 question_input_ids (1, length)")
        question_input_ids = question_input_ids[:, :self.beacon_config.beacon_question_max_tokens]
        self._mem._question_input_ids = question_input_ids.detach().clone()
        self._mem._question = self.model.embed_tokens(question_input_ids.to(self.model.embed_tokens.weight.device))

    def _beacon_forward(self, input_ids, labels, regions, question_input_ids=None):
        self._mem = _Qwen3BeaconMemory(self, self.beacon_config)
        self._prepare_question(question_input_ids)
        self._mem.prepare(input_ids, labels, regions)

        last_logits = None
        while not self._mem.finish:
            win_ids, win_labels, attn_mask, position_embeddings, past = self._mem.step()
            new_past, _logits = self._native_forward(win_ids, win_labels, attn_mask, position_embeddings, past)
            last_logits = _logits
            self._mem.update_memory(new_past)

        loss = self._mem.output()
        if loss is None:
            loss = self.model.beacon_embed_tokens.weight.sum() * 0.0
        if self._mem._readout_losses:
            loss = loss + self.beacon_config.beacon_readout_distill_weight * torch.stack(self._mem._readout_losses).mean()
            self._mem._readout_losses = []
        return loss, last_logits

    # ------------------------------------------------------------------
    # 生成 / 推理
    # ------------------------------------------------------------------
    def _mem_full_cache_past(self):
        """构造 decode 用的 full-attn 层 past 4 元组列表。"""
        past = []
        for idx in range(self.config.num_hidden_layers):
            if self._mem.layer_types[idx] == "full_attention":
                ck, cv = self._mem._cache[idx]
                past.append((idx, (ck, cv, 0, None)))
        return past

    def _mem_cache_len(self) -> int:
        """full-attn 层缓存长度（用于位置编码/掩码）。"""
        for idx in range(self.config.num_hidden_layers):
            if self._mem.layer_types[idx] == "full_attention":
                ck, _ = self._mem._cache[idx]
                if ck is not None:
                    return ck.shape[2]
        return 0

    def _rope_for(self, total, device):
        """计算 mRoPE cos/sin，覆盖 ``[0, total)`` 全局文本位置。"""
        pos_ids = torch.arange(total, device=device).expand(3, total).unsqueeze(1)
        ref = torch.zeros(
            1, 1, self.config.hidden_size,
            device=device, dtype=self.model.embed_tokens.weight.dtype,
        )
        return self.model.rotary_emb(ref, pos_ids)

    def prefill_and_get_cache(self, input_ids, regions, return_last_logits=False,
                              reuse_cache=False, question_input_ids=None):
        """把整条上下文（question + 各检索文档块）编码为 beacon K/V。

        ``reuse_cache=True`` 时保留已有持久状态，只处理新增的 ``input_ids``；
        此时 ``regions`` 使用相对于新增输入的 token 区间。

        Returns:
            ``return_last_logits=False`` 时返回 :class:`_Qwen3BeaconMemory`（其
            ``_cache`` / ``_linear_recurrent`` / ``_linear_conv`` 已就绪）；``True`` 时返回
            ``(memory, last_logits)``。
        """
        if reuse_cache:
            if not hasattr(self, "_mem"):
                raise ValueError("复用缓存前必须先执行首次 prefill")
            self._mem._pos = 0
            self._mem._seg_idx = 0
            self._mem._segments = []
        else:
            self._mem = _Qwen3BeaconMemory(self, self.beacon_config)
        self._prepare_question(question_input_ids)
        self._mem.prepare(input_ids, None, regions)

        last_logits = None
        while not self._mem.finish:
            win_ids, _, attn_mask, position_embeddings, past = self._mem.step()
            new_past, logits = self._native_forward(
                win_ids,
                None,
                attn_mask,
                position_embeddings,
                past,
                last_logits_only=True,
            )
            self._mem.update_memory(new_past)
            # 每个窗口末位 logits = 对下一 token 的预测；prefill 结束即首个新 token 的预测
            last_logits = logits[:, -1, :]

        if return_last_logits:
            return self._mem, last_logits
        return self._mem

    def decode_step(self, last_token_ids):
        """单步自回归解码：对 ``last_token_ids`` (bsz,1) 取下一个 token 的 logits。"""
        mem_size = self._mem_cache_len()
        total = mem_size + 1
        device = last_token_ids.device
        cos, sin = self._rope_for(total, device)
        attn_mask = self._mem._make_4d_causal_mask(1, 1, mem_size, device)
        past = self._mem_full_cache_past()

        self._mem._store = "cache"
        new_past, logits = self._native_forward(
            last_token_ids,
            None,
            attn_mask,
            (cos, sin),
            past,
            last_logits_only=True,
        )
        self._mem.update_memory(new_past)
        return logits[:, -1, :]

    def beacon_generate(self, input_ids, attention_mask=None, regions=None,
                        compress_start=None, compress_end=None, max_new_tokens=256,
                        do_sample=False, temperature=1.0, top_p=1.0,
                        eos_token_ids=None, stop_texts=None, tokenizer=None,
                        reuse_cache=False, question_input_ids=None):
        """Beacon 模式生成：先编码上下文为 beacon K/V，再自回归生成答案。

        参数与参考 :meth:`beacon_qwen2.beacon_generate` 一致；``regions`` 指定
        各 ``<information>`` 检索文档块，其余（question/模板尾部/生成）为 keep。
        ``reuse_cache=True`` 时只传入新增输入和相对压缩区，接续已有缓存生成。
        """
        if regions is None and compress_start is not None and compress_end is not None:
            regions = [(int(compress_start), int(compress_end))]
        if regions is None:
            raise ValueError("必须提供 regions 或 compress_start/compress_end")

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self._beacon_generate_loop(
                    input_ids, regions, max_new_tokens, do_sample, temperature, top_p,
                    eos_token_ids, stop_texts, tokenizer, reuse_cache, question_input_ids,
                )
        finally:
            self.train(was_training)

    def _beacon_generate_loop(self, input_ids, regions, max_new_tokens, do_sample,
                              temperature, top_p, eos_token_ids, stop_texts, tokenizer,
                              reuse_cache=False, question_input_ids=None):
        _mem, logits = self.prefill_and_get_cache(
            input_ids, regions, return_last_logits=True, reuse_cache=reuse_cache,
            question_input_ids=question_input_ids,
        )
        eos = eos_token_ids if eos_token_ids is not None else [self.config.eos_token_id]
        generated = []

        for _ in range(max_new_tokens):
            if do_sample:
                logits = logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    logits = logits.masked_fill(
                        remove.scatter(-1, sorted_indices, remove), float("-inf")
                    )
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated.append(next_token)
            logits = self.decode_step(next_token)
            if next_token.item() in eos:
                break
            if stop_texts and tokenizer is not None:
                gen_text = tokenizer.decode(
                    torch.cat(generated, dim=1)[0].tolist(), skip_special_tokens=False
                )
                if any(st in gen_text for st in stop_texts):
                    break

        if not generated:
            return torch.empty((1, 0), dtype=torch.long, device=input_ids.device)
        return torch.cat(generated, dim=1)


def load_beacon_qwen3_5(model_name_or_path, beacon_config=None,
                        torch_dtype=torch.bfloat16, device_map="auto"):
    """装载 BeaconQwen3_5ForCausalLM。

    两种输入形态：
    1. ``architectures`` 含 ``BeaconQwen3_5ForCausalLM``（如训练后 ``_save_model`` 保存的
       checkpoint）：直接 ``from_pretrained`` 加载，保留训练好的 beacon 压缩参数。
    2. 原始 HF 多维包装器（``Qwen3_5ForConditionalGeneration``）：提取文本主干权重后构建，
       beacon 参数按配置重新初始化。
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if "BeaconQwen3_5ForCausalLM" in (cfg.architectures or []):
        if not hasattr(cfg, "beacon_linear_writer_rank"):
            raise RuntimeError(
                "该 checkpoint 来自旧版 Qwen3.5 Beacon：它保留了原始 information 的 "
                "linear-attention DynamicCache，且不含已训练的 BeaconLinearStateWriter。"
                "为避免静默使用随机 writer，必须用当前实现重新训练 Beacon 参数。"
            )
        model = BeaconQwen3_5ForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, device_map="cpu"
        )
        if beacon_config is not None:
            model.set_beacon_config(beacon_config)
        if device_map == "auto":
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(torch_dtype)
        model.eval()
        return model

    from ..milestones.qwen35_text import load_text_causal_model

    base = load_text_causal_model(model_name_or_path, torch_dtype=torch_dtype, device_map="cpu")
    text_config = base.config
    if beacon_config is not None:
        beacon_config.merge_into_config(text_config)
    model = BeaconQwen3_5ForCausalLM(text_config)
    model.load_state_dict(base.state_dict(), strict=False)
    if device_map == "auto":
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(torch_dtype)
    if beacon_config is not None:
        model.set_beacon_config(beacon_config)
    model.eval()
    return model
