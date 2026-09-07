"""Activate Beacon 压缩机制在 Qwen3.5 上的移植（只压缩检索到的内容）。

背景：``Qwen/Qwen3.5-2B`` 是混合架构 —— 24 层中仅 6 层（``layer_types`` 为
``full_attention``）带标准 QKV 注意力缓存，其余 18 层为 ``linear_attention``
（GatedDeltaNet，固定尺寸循环状态）。

语义（与参考实现 [[beacon_qwen2]] / [[beacon_memory]] 完全一致）：

- **只压缩检索内容**：由 ``regions``（各 ``<information>`` 块 token 区间）指定，
  文档块按 ``beacon_window`` 切窗，每窗末尾追加 ``window // beacon_ratio`` 个
  beacon，beacon K/V 进入持久缓存，原始文档 K/V 丢弃。
- **损失不含检索内容**：labels 全局预移位后，文档与 beacon 位置为 -100，
  只在模型生成片段（think / <search> / <answer>）上计算损失。
- **Beacon 只作用于 full_attention 层**：6 个 full_attention 层做 K/V 压缩；
  linear_attention 层用 transformers ``DynamicCache`` 跨窗口承接循环状态
  （其内存天然有界，无需压缩）。

``enable_beacon=False`` 退化为原生 ``Qwen3_5ForCausalLM`` 前向。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5ForCausalLM,
    apply_rotary_pos_emb,
)

from .beacon_config import BeaconConfig
from .modeling_utils import cat_tensor, compute_loss, slice_tensor


# ======================================================================
# Beacon Attention（full_attention 层）
# ======================================================================
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
# Beacon 窗口状态机（只压缩检索内容，线性层用 DynamicCache 承接状态）
# ======================================================================
class _Qwen3BeaconMemory:
    """把整条序列切窗处理的状态机。

    - full_attention 层：``_cache[layer_idx] = (K, V)`` 持久缓存（RoPE 之前），
      compress 窗口只保留 beacon K/V，keep 窗口全量保留。
    - linear_attention 层：``DynamicCache`` 跨窗口承接 GatedDeltaNet 循环状态。
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
        self._linear_cache: Optional[DynamicCache] = None
        self.reset()

    def reset(self):
        self._pos = 0
        self._seg_idx = 0
        self._segments: List[Tuple[int, int, str]] = []
        self._cache = [(None, None) for _ in range(self.num_layers)]
        self._linear_cache = DynamicCache(config=self.config)
        self._batch_loss = None
        self._valid_num = None
        self._store = None
        self._step_beacon_indices = None

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
        for idx, layer in enumerate(self.model.layers):
            if getattr(config, "layer_types", [None] * config.num_hidden_layers)[idx] == "full_attention":
                layer.self_attn = BeaconQwen3Attention(config, idx)
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
        self.beacon_config = beacon_config
        return self

    # ------------------------------------------------------------------
    def forward(self, input_ids=None, attention_mask=None, labels=None,
                compress_regions=None, compress_start=None, compress_end=None, **kwargs):
        if not self.beacon_config.enable_beacon:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        if compress_regions is None and compress_start is not None and compress_end is not None:
            compress_regions = [(int(compress_start), int(compress_end))]
        if compress_regions is None:
            raise ValueError("beacon 模式必须提供 compress_regions 或 compress_start/compress_end")
        return self._beacon_forward(input_ids, labels, compress_regions)

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

    def _native_forward(self, win_ids, win_labels, attn_mask, position_embeddings, past):
        """单窗口前向。past: list of (layer_idx, 4元组)。返回 (new_past, logits)。"""
        beacon_size = past[0][1][2] if past else 0
        emb = self._embed(win_ids, beacon_size)

        use_gc = self.training and getattr(self, "_use_gradient_checkpointing", False)
        hidden = emb
        new_past = []
        linear_cache = self._mem._linear_cache if not use_gc else None
        for idx, layer in enumerate(self.model.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            if self._mem.layer_types[idx] == "full_attention":
                pkv = next(pv for li, pv in past if li == idx)
                if use_gc:
                    out, npv = torch.utils.checkpoint.checkpoint(
                        layer.self_attn, hidden, position_embeddings, attn_mask, pkv,
                        use_reentrant=False,
                    )
                else:
                    out, npv = layer.self_attn(
                        hidden, position_embeddings=position_embeddings,
                        attention_mask=attn_mask, past_key_value=pkv,
                    )
                new_past.append((idx, npv))
            else:
                if use_gc:
                    # 线性注意力层：训练时用 cache_params=None（与 transformers 的
                    # GradientCheckpointingLayer 一致，避免 DeltaNet 状态在重计算中
                    # 不一致），并做梯度检查点以释放 O(seq×chunk×head) 中间量。
                    out = torch.utils.checkpoint.checkpoint(
                        layer.linear_attn, hidden, None, use_reentrant=False,
                    )
                else:
                    out = layer.linear_attn(hidden, cache_params=linear_cache)
            hidden = residual + out
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden

        hidden = self.model.norm(hidden)
        logits = self.lm_head(hidden).float()

        if win_labels is not None:
            _loss, token_loss = compute_loss(logits, win_labels, shift=False)
            valid_num = (win_labels != -100).sum(-1).clamp(min=1)
            batch_loss = token_loss.sum(-1) / valid_num
            self._mem.update_loss(batch_loss, valid_num)
        return new_past, logits

    def _beacon_forward(self, input_ids, labels, regions):
        self._mem = _Qwen3BeaconMemory(self, self.beacon_config)
        self._mem.prepare(input_ids, labels, regions)

        while not self._mem.finish:
            win_ids, win_labels, attn_mask, position_embeddings, past = self._mem.step()
            new_past, _logits = self._native_forward(win_ids, win_labels, attn_mask, position_embeddings, past)
            self._mem.update_memory(new_past)

        loss = self._mem.output()
        return loss, loss

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

    def prefill_and_get_cache(self, input_ids, regions, return_last_logits=False):
        """把整条上下文（question + 各检索文档块）编码为 beacon K/V。

        Returns:
            ``return_last_logits=False`` 时返回 :class:`_Qwen3BeaconMemory`（其
            ``_cache`` / ``_linear_cache`` 已就绪）；``True`` 时返回
            ``(memory, last_logits)``。
        """
        self._mem = _Qwen3BeaconMemory(self, self.beacon_config)
        self._mem.prepare(input_ids, None, regions)

        last_logits = None
        while not self._mem.finish:
            win_ids, _, attn_mask, position_embeddings, past = self._mem.step()
            new_past, logits = self._native_forward(win_ids, None, attn_mask, position_embeddings, past)
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
        new_past, logits = self._native_forward(last_token_ids, None, attn_mask, (cos, sin), past)
        self._mem.update_memory(new_past)
        return logits[:, -1, :]

    def beacon_generate(self, input_ids, attention_mask=None, regions=None,
                        compress_start=None, compress_end=None, max_new_tokens=256,
                        do_sample=False, temperature=1.0, top_p=1.0,
                        eos_token_ids=None, stop_texts=None, tokenizer=None):
        """Beacon 模式生成：先编码上下文为 beacon K/V，再自回归生成答案。

        参数与参考 :meth:`beacon_qwen2.beacon_generate` 一致；``regions`` 指定
        各 ``<information>`` 检索文档块，其余（question/模板尾部/生成）为 keep。
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
                    eos_token_ids, stop_texts, tokenizer,
                )
        finally:
            self.train(was_training)

    def _beacon_generate_loop(self, input_ids, regions, max_new_tokens, do_sample,
                              temperature, top_p, eos_token_ids, stop_texts, tokenizer):
        _mem, logits = self.prefill_and_get_cache(input_ids, regions, return_last_logits=True)
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
            if next_token.item() in eos:
                break
            if stop_texts and tokenizer is not None:
                gen_text = tokenizer.decode(
                    torch.cat(generated, dim=1)[0].tolist(), skip_special_tokens=False
                )
                if any(st in gen_text for st in stop_texts):
                    break

            logits = self.decode_step(next_token)

        if not generated:
            return torch.empty((1, 0), dtype=torch.long, device=input_ids.device)
        return torch.cat(generated, dim=1)


def load_beacon_qwen3_5(model_name_or_path, beacon_config=None,
                        torch_dtype=torch.bfloat16, device_map="auto"):
    """从头 checkpoint 装载 BeaconQwen3_5ForCausalLM（文本主干权重）。"""
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