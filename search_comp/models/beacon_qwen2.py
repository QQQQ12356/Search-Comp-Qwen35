"""带 Activate Beacon 压缩机制的 Qwen2 因果语言模型。

通过子类化 transformers 4.43 的 Qwen2 组件，只覆写与 beacon 相关的部分：

- :class:`BeaconQwen2Attention`：为 beacon token 引入独立的 ``beacon_q/k/v/o_proj``
  投影，用 ``torch.where`` 按 ``beacon_indices`` 切换；K/V cache 保存 **RoPE 之前**
  的 key，每个窗口前向时对 ``past + current`` 整体施加 RoPE。
- :class:`BeaconQwen2Model`：新增 ``beacon_embed_tokens``，前向时把普通 token 与
  beacon token 分开嵌入。
- :class:`BeaconQwen2ForCausalLM`：以 :class:`BeaconMemory` 滑动窗口处理整条序列，
  question 区保留、document 区压缩、answer 区解码，损失只在 answer 区计算。

推理时通过 :meth:`prefill_and_get_cache` 把 question+文档编码为 beacon K/V，
再通过 :meth:`beacon_generate` 生成答案。
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple, Union

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2ForCausalLM,
    Qwen2Model,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.modeling_outputs import BaseModelOutputWithPast

from .beacon_config import BeaconConfig
from .beacon_memory import BeaconMemory
from .modeling_utils import cat_tensor, compute_loss, init_beacon_params


# ======================================================================
# Attention
# ======================================================================
class BeaconQwen2Attention(Qwen2Attention):
    """带 beacon 投影切换的 Qwen2 注意力模块。

    ``beacon_indices`` 标记当前输入中的 beacon 位置（1=beacon），beacon 位置
    使用独立的 ``beacon_q/k/v/o_proj`` 投影，普通位置使用原始投影。
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        #: 需要为 beacon 引入独立投影的类型（来自 model.config.beacon_param）
        self.beacon_param = getattr(config, "beacon_param", "").split()

        # 创建独立的 beacon 投影，初始化为 0（随后由 init_beacon_params 复制原始投影）
        if "q" in self.beacon_param:
            self.beacon_q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=self.q_proj.bias is not None,
            )
            self.beacon_q_proj.weight.data.zero_()
            self.beacon_q_proj._is_hf_initialized = True
        if "k" in self.beacon_param:
            self.beacon_k_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=self.k_proj.bias is not None,
            )
            self.beacon_k_proj.weight.data.zero_()
            self.beacon_k_proj._is_hf_initialized = True
        if "v" in self.beacon_param:
            self.beacon_v_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=self.v_proj.bias is not None,
            )
            self.beacon_v_proj.weight.data.zero_()
            self.beacon_v_proj._is_hf_initialized = True
        if "o" in self.beacon_param:
            self.beacon_o_proj = nn.Linear(
                self.num_heads * self.head_dim,
                self.hidden_size,
                bias=self.o_proj.bias is not None,
            )
            self.beacon_o_proj.weight.data.zero_()
            self.beacon_o_proj._is_hf_initialized = True

    # ------------------------------------------------------------------
    def qkv_proj_with_beacon(
        self,
        hidden_states: torch.Tensor,
        beacon_size: int,
        beacon_indices: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """按 beacon_indices 切换 Q/K/V 投影。

        Args:
            hidden_states: 当前窗口的隐状态 ``(bsz, cur, hidden)``。
            beacon_size: 当前输入中 beacon 的数量（>0 时启用 beacon 投影）。
            beacon_indices: 标记 beacon 位置的 0/1 张量（长度覆盖当前输入）。

        Returns:
            ``(query, key, value)`` 投影结果。
        """
        if beacon_size > 0:
            cur_beacon_indices = beacon_indices[-hidden_states.shape[1] :]
            if "q" in self.beacon_param:
                ordinal_q = self.q_proj(hidden_states)
                beacon_q = self.beacon_q_proj(hidden_states)
                query_states = torch.where(
                    (cur_beacon_indices == 0)[:, None], ordinal_q, beacon_q
                )
            else:
                query_states = self.q_proj(hidden_states)
            if "k" in self.beacon_param:
                ordinal_k = self.k_proj(hidden_states)
                beacon_k = self.beacon_k_proj(hidden_states)
                key_states = torch.where(
                    (cur_beacon_indices == 0)[:, None], ordinal_k, beacon_k
                )
            else:
                key_states = self.k_proj(hidden_states)
            if "v" in self.beacon_param:
                ordinal_v = self.v_proj(hidden_states)
                beacon_v = self.beacon_v_proj(hidden_states)
                value_states = torch.where(
                    (cur_beacon_indices == 0)[:, None], ordinal_v, beacon_v
                )
            else:
                value_states = self.v_proj(hidden_states)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        return query_states, key_states, value_states

    def o_proj_with_beacon(
        self,
        attn_output: torch.Tensor,
        beacon_size: int,
        beacon_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """按 beacon_indices 切换 O 投影。"""
        if beacon_size > 0 and "o" in self.beacon_param:
            cur_beacon_indices = beacon_indices[-attn_output.shape[1] :]
            ordinal_o = self.o_proj(attn_output)
            beacon_o = self.beacon_o_proj(attn_output)
            return torch.where((cur_beacon_indices == 0)[:, None], ordinal_o, beacon_o)
        return self.o_proj(attn_output)

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple]]:
        """Beacon 感知的前向。

        ``past_key_value`` 为 4 元组 ``(key, value, beacon_size, beacon_indices)``，
        其中 key/value 是 **RoPE 之前** 的历史 K/V（每窗口整体重施加 RoPE）。

        Returns:
            ``(attn_output, attn_weights, past_key_value)``；返回的 K/V 为当前
            **增量** 且 **RoPE 之前** 的 K/V。
        """
        bsz, q_len, _ = hidden_states.size()
        past_key, past_value, beacon_size, beacon_indices = past_key_value

        kv_seq_len = q_len
        if past_key is not None:
            kv_seq_len += past_key.shape[2]

        query_states, key_states, value_states = self.qkv_proj_with_beacon(
            hidden_states, beacon_size, beacon_indices
        )
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # 返回 RoPE 之前的增量 K/V（供 Memory 缓存，下个窗口整体重施加 RoPE）
        past_key_value_out = (key_states, value_states, beacon_size, beacon_indices)

        if past_key is not None:
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        # 施加 RoPE：query 用当前位置，key 用全局位置（past + current）
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        position_ids_q = position_ids[:, -q_len:]
        query_states, _ = apply_rotary_pos_emb(
            query_states, query_states, cos, sin, position_ids_q
        )
        key_states, _ = apply_rotary_pos_emb(
            key_states, key_states, cos, sin, position_ids
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .reshape(bsz, q_len, self.hidden_size)
        )
        attn_output = self.o_proj_with_beacon(attn_output, beacon_size, beacon_indices)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value_out


# ======================================================================
# Decoder Layer
# ======================================================================
# 注：BeaconQwen2Model 直接在各层上替换 self_attn 为 BeaconQwen2Attention，
# 不定义独立的 DecoderLayer 子类（见 BeaconQwen2Model.__init__）。


# ======================================================================
# Model
# ======================================================================
class BeaconQwen2Model(Qwen2Model):
    """带 beacon 嵌入与逐层 beacon 注意力的 Qwen2 模型主体。"""

    def __init__(self, config):
        super().__init__(config)
        #: beacon token 的独立嵌入表（1 行，初始化自 eos embedding）
        self.beacon_embed_tokens = nn.Embedding(1, config.hidden_size, self.padding_idx)
        self.beacon_embed_tokens._is_hf_initialized = True
        # 替换每一层的注意力为 beacon 版本
        for idx, layer in enumerate(self.layers):
            layer_idx = getattr(layer, "layer_idx", idx)
            layer.self_attn = BeaconQwen2Attention(config, layer_idx)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """Beacon 感知的模型前向。

        只处理**单个窗口/块**的输入（由 :class:`BeaconMemory` 逐窗口调用）。
        ``past_key_values`` 为每层 4 元组 ``(key, value, beacon_size, beacon_indices)``。

        Returns:
            ``BaseModelOutputWithPast``，其中 ``past_key_values`` 为每层当前
            **增量** 的 (key, value, beacon_size, beacon_indices)（RoPE 之前）。
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("input_ids 与 inputs_embeds 必须二选一")

        # ---- 分开嵌入普通 token 与 beacon token ----
        beacon_size = past_key_values[0][2] if past_key_values is not None else 0
        if beacon_size > 0:
            cur_beacon_indices = past_key_values[0][3][-input_ids.shape[1] :]
            ordinal_ids = input_ids[:, cur_beacon_indices == 0]
            beacon_ids = input_ids[:, cur_beacon_indices > 0]
            ordinal_embeds = self.embed_tokens(ordinal_ids)
            beacon_embeds = self.beacon_embed_tokens(
                beacon_ids - self.config.vocab_size
            )
            inputs_embeds = beacon_embeds.new_zeros(
                *input_ids.shape, beacon_embeds.shape[-1]
            )
            inputs_embeds[:, cur_beacon_indices == 0] = ordinal_embeds
            inputs_embeds[:, cur_beacon_indices > 0] = beacon_embeds
        else:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_decoder_cache] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
        )


# ======================================================================
# ForCausalLM
# ======================================================================
class BeaconQwen2ForCausalLM(Qwen2ForCausalLM):
    """带 beacon 压缩的 Qwen2 因果语言模型（SFT/推理入口）。"""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        # 用 BeaconQwen2Model 替换 super 中创建的 Qwen2Model
        super().__init__(config)
        self.model = BeaconQwen2Model(config)
        self.post_init()
        #: beacon 配置
        self.beacon_config = BeaconConfig.from_model_config(config)
        self.memory = BeaconMemory(config, self.beacon_config)
        # 初始化 beacon 参数（beacon 投影从原始投影复制，beacon 嵌入从 eos 复制）
        # 否则直接构造（不经 from_pretrained）时 beacon 投影全为 0，无法学习
        init_beacon_params(self)

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """加载权重并初始化 beacon 参数与 Memory。

        beacon 参数若 checkpoint 中**缺失**（如加载基础模型）则从 eos/原始投影
        初始化；若 checkpoint 中已存在（加载微调后的模型）则保留不覆盖，
        避免丢失训练成果（见 code review C-1）。
        """
        kwargs.update(output_loading_info=True)
        model, loading_info = super().from_pretrained(*args, **kwargs)
        config = model.config
        model.beacon_config = BeaconConfig.from_model_config(config)
        model.memory = BeaconMemory(config, model.beacon_config)
        init_beacon_params(model, missing_keys=loading_info.get("missing_keys"))
        return model

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def _native_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values: List[Tuple],
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = True,
    ):
        """对单个窗口/块做一次前向，返回 logits 与增量 K/V。

        Returns:
            ``(logits, new_past_key_values, batch_loss, token_loss)``。
            - ``new_past_key_values``: 每层增量 (key, value, beacon_size, beacon_indices)。
            - ``batch_loss``: 本块逐样本平均损失，无标签时为 None。
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        batch_loss = None
        if labels is not None:
            # labels 已在 memory.prepare 中全局预移位（跨窗口因果监督），此处不再 shift
            loss, token_loss = compute_loss(logits, labels, shift=False)
            valid_num = (labels != -100).sum(-1).clamp(min=1)
            batch_loss = token_loss.sum(-1) / valid_num
        return logits, outputs.past_key_values, batch_loss

    def _beacon_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.LongTensor],
        regions: List[Tuple[int, int]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """用 Memory 滑动窗口处理整条序列。

        Args:
            input_ids: ``(1, seq_len)`` 整条序列。
            attention_mask: ``(1, seq_len)``。
            labels: ``(1, seq_len)``，-100 忽略。
            regions: 压缩区列表 ``[(start, end), ...]``（各 ``<information>`` 文档块）。

        Returns:
            ``(loss, batch_loss)``。
        """
        self.memory.reset()
        self.memory.prepare(input_ids, attention_mask, labels, regions=regions)

        while not self.memory.finish:
            win_ids, win_mask, win_pos, win_past, win_labels = self.memory.step()
            _logits, new_past, batch_loss = self._native_forward(
                input_ids=win_ids,
                attention_mask=win_mask,
                position_ids=win_pos,
                past_key_values=win_past,
                labels=win_labels,
            )
            self.memory.update_memory(new_past)
            if win_labels is not None and batch_loss is not None:
                valid_num = (win_labels != -100).sum(-1)
                self.memory.update_loss(batch_loss, valid_num)

        loss, batch_loss = self.memory.output(None)
        return loss, batch_loss

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        compress_regions: Optional[List[Tuple[int, int]]] = None,
        compress_start: Optional[int] = None,
        compress_end: Optional[int] = None,
        compress_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """前向入口。

        beacon 模式下通过 ``compress_regions``（多压缩区，交互式多轮搜索）或
        ``compress_start/compress_end``（单压缩区）指定文档压缩区。

        Args:
            input_ids: ``(batch, seq_len)``。
            attention_mask: ``(batch, seq_len)``。
            labels: ``(batch, seq_len)``，-100 忽略。
            compress_regions: 压缩区列表 ``[(start, end), ...]``（token 区间）。
            compress_start / compress_end: 单压缩区 [start, end) 的全局位置（兼容）。
            compress_mask: 形状 ``(batch, seq_len)``，1=文档压缩 token（兼容）。

        Returns:
            beacon 模式返回 ``(loss, batch_loss)`` 元组；退化模式返回标准输出。
        """
        beacon_cfg = self.beacon_config
        if not beacon_cfg.enable_beacon:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        # 解析压缩区：regions 优先，其次单区域，最后 mask
        if compress_regions is None:
            if compress_start is not None and compress_end is not None:
                compress_regions = [(int(compress_start), int(compress_end))]
            elif compress_mask is not None:
                positions = compress_mask[0].nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    compress_regions = [
                        (
                            int(positions[0].item()),
                            int(positions[-1].item() + 1),
                        )
                    ]
        if compress_regions is None:
            raise ValueError(
                "beacon 模式下必须提供 compress_regions、"
                "compress_start/compress_end 或 compress_mask"
            )
        # 允许空 regions（整条序列按 keep 处理，如交互式第一轮或截断后无压缩区）

        # batch 内的所有样本共享同一区域结构（由 collator 保证）
        loss, batch_loss = self._beacon_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            regions=compress_regions,
        )
        return loss, batch_loss

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def prefill_and_get_cache(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        regions: Optional[List[Tuple[int, int]]] = None,
        compress_start: Optional[int] = None,
        compress_end: Optional[int] = None,
        return_last_logits: bool = False,
    ):
        """把整条上下文（question + 各检索文档块）编码为 beacon K/V。

        Args:
            input_ids: ``(1, seq_len)`` 整条上下文。
            attention_mask: ``(1, seq_len)``。
            regions: 压缩区列表（各 ``<information>`` 文档块）。
            compress_start / compress_end: 单压缩区兼容写法。
            return_last_logits: 是否返回上下文末位位置的 logits（即对第一个
                新生成 token 的预测），用于生成首步，避免重复注入末位 token
                （见 code review H-1）。

        Returns:
            ``return_last_logits=False`` 时返回每层 4 元组
            ``(key, value, beacon_size=0, None)`` 的 past_key_values；
            ``True`` 时返回 ``(past_key_values, last_logits)``。
        """
        if regions is None:
            if compress_start is None or compress_end is None:
                raise ValueError("必须提供 regions 或 compress_start/compress_end")
            regions = [(int(compress_start), int(compress_end))]

        self.memory.reset()
        self.memory.prepare(input_ids, attention_mask, labels=None, regions=regions)

        last_logits = None
        while not self.memory.finish:
            win_ids, win_mask, win_pos, win_past, _ = self.memory.step()
            logits, new_past, _ = self._native_forward(
                input_ids=win_ids,
                attention_mask=win_mask,
                position_ids=win_pos,
                past_key_values=win_past,
                labels=None,
            )
            # 每个窗口末位的 logits = 对下一个 token 的预测；prefill 结束时即
            # 上下文末位（第一个新生成 token）的预测
            last_logits = logits[:, -1, :]
            self.memory.update_memory(new_past)

        past = []
        for layer_idx in range(self.config.num_hidden_layers):
            cache_k, cache_v = self.memory._cache[layer_idx]
            past.append((cache_k, cache_v, 0, None))
        if return_last_logits:
            return past, last_logits
        return past

    def decode_step(
        self, last_token_ids: torch.LongTensor, past_key_values: List[Tuple]
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        """单步自回归解码。

        Args:
            last_token_ids: ``(batch, 1)`` 上一步生成的 token。
            past_key_values: 累积的 K/V（RoPE 之前）。

        Returns:
            ``(logits_of_last_token, new_past_key_values)``。
        """
        mem_size = (
            past_key_values[0][0].shape[2] if past_key_values[0][0] is not None else 0
        )
        # 全局位置 0..mem+1：key 按缓存索引整体施加 RoPE
        position_ids = torch.arange(
            0, mem_size + 1, device=last_token_ids.device, dtype=torch.long
        ).unsqueeze(0)
        # 单 token 无历史时不需要掩码；有历史时全可关注
        attention_mask = None
        bsz = last_token_ids.shape[0]
        if mem_size > 0:
            attention_mask = torch.zeros(
                (bsz, 1, 1, mem_size + 1),
                device=last_token_ids.device,
                dtype=torch.float32,
            )

        logits, new_past, _ = self._native_forward(
            input_ids=last_token_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=None,
        )
        # 累积 K/V
        updated_past = []
        for layer_idx, (key, value, _bs, _bi) in enumerate(new_past):
            prev_key, prev_value = (
                past_key_values[layer_idx][0],
                past_key_values[layer_idx][1],
            )
            new_key = cat_tensor([prev_key, key], dim=2)
            new_value = cat_tensor([prev_value, value], dim=2)
            updated_past.append((new_key, new_value, 0, None))
        return logits[:, -1, :], updated_past

    def beacon_generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        regions: Optional[List[Tuple[int, int]]] = None,
        compress_start: Optional[int] = None,
        compress_end: Optional[int] = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_ids: Optional[List[int]] = None,
        stop_texts: Optional[List[str]] = None,
        tokenizer: Optional[Any] = None,
    ) -> torch.LongTensor:
        """Beacon 模式生成：先编码上下文为 beacon K/V，再自回归生成。

        Args:
            input_ids: ``(1, seq_len)``，包含 question + 各检索文档块 + 模板尾部。
            attention_mask: ``(1, seq_len)``。
            regions: 压缩区列表（各 ``<information>`` 文档块）。
            compress_start / compress_end: 单压缩区兼容写法。
            max_new_tokens: 生成的最大新 token 数。
            do_sample: 是否采样（否则贪心）。
            temperature: 采样温度。
            top_p: nucleus 采样参数。
            eos_token_ids: 提前停止的 token id 列表。
            stop_texts: 文本级停止序列（如 ``["</search>"]``、``["</answer>"]``），
                生成文本出现任一即停止（用于交互式搜索的分段生成）。
            tokenizer: 用于解码生成文本以检测 stop_texts；为 None 时仅按 eos 停止。

        Returns:
            ``(1, gen_len)`` 生成的 token 序列（含停止序列，若触发）。
        """
        if regions is None:
            if compress_start is None or compress_end is None:
                raise ValueError("必须提供 regions 或 compress_start/compress_end")
            regions = [(int(compress_start), int(compress_end))]

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self._beacon_generate_loop(
                    input_ids,
                    attention_mask,
                    regions,
                    max_new_tokens,
                    do_sample,
                    temperature,
                    top_p,
                    eos_token_ids,
                    stop_texts,
                    tokenizer,
                )
        finally:
            self.train(was_training)

    def _beacon_generate_loop(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        regions: List[Tuple[int, int]],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        eos_token_ids: Optional[List[int]],
        stop_texts: Optional[List[str]],
        tokenizer: Optional[Any],
    ) -> torch.LongTensor:
        """beacon 生成循环（内部使用，已包 no_grad）。

        首 token 直接从 prefill 返回的上下文末位 logits 采样（不重复注入末位
        上下文 token，见 code review H-1）；后续 token 用 decode_step 逐位生成。
        """
        past, logits = self.prefill_and_get_cache(
            input_ids, attention_mask, regions=regions, return_last_logits=True
        )

        eos = eos_token_ids if eos_token_ids is not None else [self.config.eos_token_id]
        generated = []

        for _ in range(max_new_tokens):
            if do_sample:
                logits = logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cum_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    logits = logits.masked_fill(indices_to_remove, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated.append(next_token)
            if next_token.item() in eos:
                break

            # 文本级停止序列检查（如 </search> / </answer>）
            if stop_texts and tokenizer is not None:
                gen_text = tokenizer.decode(
                    torch.cat(generated, dim=1)[0].tolist(),
                    skip_special_tokens=False,
                )
                if any(st in gen_text for st in stop_texts):
                    break

            # 用刚生成的 token 计算下一个位置的 logits
            logits, past = self.decode_step(next_token, past)

        if not generated:
            return torch.empty((1, 0), dtype=torch.long, device=input_ids.device)
        return torch.cat(generated, dim=1)
