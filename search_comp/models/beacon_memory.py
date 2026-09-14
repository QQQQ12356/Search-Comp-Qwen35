"""Beacon 滑动窗口记忆状态机。

本模块实现了 Activate Beacon 的滑动窗口处理逻辑，面向 RAG / 交互式搜索场景。
序列被划分为若干**片段（segment）**：

- **keep 片段**（指令、问题、think 推理、search 标签、答案等）：正常前向，
  K/V 全部保留到持久缓存 ``cache``。
- **compress 片段**（``<information>`` 检索文档块）：append 按 ``beacon_window``
  切分并追加 beacon；intersect 每 ratio 个 token 后插入一个 beacon。
  beacon 通过自注意力聚合窗口信息，抽取 beacon K/V 保留到 ``cache``，
  **原始文档 K/V 随即丢弃**（不参与后续解码）；尾部不足一片也压缩。
  intersect 在 chunk 内保留完整因果可见性，chunk 结束后仅缓存 beacon。

交互式搜索（多轮 ``<search>`` → ``<information>``）会产生**多个** compress
片段，通过 ``regions: List[(start, end)]`` 传入。所有片段按处理顺序依次
追加到 ``cache``，保证位置单调连续。

K/V 缓存保存的是 **RoPE 之前** 的 key（与 activation_beacon 一致），
每个窗口前向时对整个 ``past + current`` 重新施加 RoPE，从而支持全局位置。
"""

from __future__ import annotations

import torch
from typing import List, Optional, Sequence, Tuple

from .modeling_utils import beacon_intersect_order, cat_tensor, slice_tensor


class BeaconMemory:
    """Beacon 滑动窗口状态机。

    Args:
        model_config: HuggingFace 模型 config（含 num_hidden_layers, vocab_size 等）。
        beacon_config: :class:`BeaconConfig`，beacon 超参数。
    """

    def __init__(self, model_config, beacon_config):
        self.config = model_config
        self.beacon = beacon_config
        self.num_layers = model_config.num_hidden_layers
        #: 注意力掩码的浮点 dtype（与模型权重 dtype 对齐，默认 float32）
        self._dtype = getattr(model_config, "torch_dtype", torch.float32)
        if self._dtype is None or not torch.is_floating_point(
            torch.zeros(1, dtype=self._dtype)
        ):
            self._dtype = torch.float32
        self.reset()

    # ------------------------------------------------------------------
    # 状态管理
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """初始化一条新序列的状态。"""
        #: 全局游标（下一个待处理 token 的位置）
        self._pos = 0
        self._step_idx = 0
        #: 当前步的存储策略（"cache" / "beacon"），见 update_memory
        self._store: Optional[str] = None
        #: 持久缓存（keep K/V + beacon K/V + 未压缩尾部 K/V），每层 (K, V)。
        #: 所有内容按处理顺序追加，位置单调连续。
        self._cache: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = [
            (None, None) for _ in range(self.num_layers)
        ]
        #: 当前步输入中 beacon token 的位置掩码（1=beacon）
        self._step_beacon_indices: Optional[torch.Tensor] = None
        self.all_input_ids: Optional[torch.Tensor] = None
        self.all_attention_mask: Optional[torch.Tensor] = None
        self.all_labels: Optional[torch.Tensor] = None
        #: 压缩区列表（token 区间，闭开）
        self._regions: List[Tuple[int, int]] = []
        #: 片段计划：[(start, end, mode)]，mode ∈ {"keep", "compress"}
        self._segments: List[Tuple[int, int, str]] = []
        self._seg_idx = 0
        #: 累积损失
        self._batch_loss: Optional[torch.Tensor] = None
        self._valid_num: Optional[torch.Tensor] = None

    @property
    def finish(self) -> bool:
        """是否已处理完整个序列。"""
        return self.all_input_ids is None or self._pos >= self.all_sequence_length

    @property
    def all_sequence_length(self) -> int:
        """整个序列的长度。"""
        return self.all_input_ids.shape[1]

    @property
    def min_value(self) -> float:
        """当前 dtype 的最小值（用于注意力掩码）。"""
        return torch.finfo(self.dtype).min

    @property
    def dtype(self) -> torch.dtype:
        """注意力掩码使用的浮点 dtype（与模型权重一致）。"""
        return self._dtype

    # ------------------------------------------------------------------
    # 准备与步进
    # ------------------------------------------------------------------
    def prepare(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        regions: Optional[Sequence[Tuple[int, int]]] = None,
        compress_start: Optional[int] = None,
        compress_end: Optional[int] = None,
    ) -> None:
        """准备一条序列。

        Args:
            input_ids: 形状 ``(1, seq_len)`` 的整条序列 token id。
            attention_mask: 形状 ``(1, seq_len)``，全 1 序列。
            labels: 形状 ``(1, seq_len)``，-100 表示不计算损失。
            regions: 压缩区列表 ``[(start, end), ...]``（token 区间，闭开），
                对应各个 ``<information>`` 文档块。必须升序且互不重叠。
            compress_start / compress_end: 单压缩区的兼容写法（与 regions
                二选一，regions 优先）。

        Raises:
            ValueError: regions 为空 / 重叠 / 越界时抛出。
        """
        # 兼容单区域写法
        if regions is None:
            if compress_start is None or compress_end is None:
                raise ValueError("必须提供 regions 或 compress_start/compress_end")
            regions = [(compress_start, compress_end)]
        regions = [(int(s), int(e)) for s, e in regions]

        self._device = input_ids.device
        self.all_input_ids = input_ids
        self.all_attention_mask = (
            attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        )
        # labels 全局预移位：position i 的标签 = 原序列 i+1 的标签，从而保证
        # **跨窗口**的因果监督（前一个窗口末位 logit 预测下一个窗口首个 token，
        # 见 code review M-2）。模型侧 compute_loss 使用 shift=False。
        if labels is not None:
            labels = torch.cat(
                [labels[:, 1:], labels.new_full((labels.shape[0], 1), -100)], dim=1
            )
        self.all_labels = labels

        seq_len = self.all_sequence_length
        # 校验压缩区：升序、不重叠、在界内
        prev_end = 0
        for s, e in regions:
            if e <= s:
                raise ValueError(f"压缩区 ({s}, {e}) 非法（end 必须大于 start）")
            if s < prev_end:
                raise ValueError(f"压缩区 ({s}, {e}) 与前一个区域重叠")
            if e > seq_len:
                raise ValueError(f"压缩区 ({s}, {e}) 越界（序列长度 {seq_len}）")
            prev_end = e
        self._regions = regions

        # 构建片段计划：compress 区之间/前后的部分都是 keep
        self._segments = []
        cursor = 0
        for s, e in regions:
            if s > cursor:
                self._segments.append((cursor, s, "keep"))
            self._segments.append((s, e, "compress"))
            cursor = e
        if cursor < seq_len:
            self._segments.append((cursor, seq_len, "keep"))
        self._seg_idx = 0

    def step(self):
        """产生当前滑动窗口的输入。

        Returns:
            ``(input_ids, attention_mask, position_ids, past_key_values, labels)``：
            - ``input_ids``: 当前窗口 token（含追加的 beacon token）。
            - ``attention_mask``: 4D 因果掩码 ``(bsz, 1, cur, mem+cur)``。
            - ``position_ids``: 全局单调位置 ``(bsz, mem+cur)``。
            - ``past_key_values``: 每层 4 元组 ``(key, value, beacon_size, beacon_indices)``，
              key/value 为 **RoPE 之前** 的历史 K/V。
            - ``labels``: 当前窗口标签（beacon 位置为 -100）。
        """
        # 跳过已完成的片段
        while (
            self._seg_idx < len(self._segments)
            and self._segments[self._seg_idx][1] <= self._pos
        ):
            self._seg_idx += 1

        seq_len = self.all_sequence_length
        if self._pos >= seq_len:
            raise RuntimeError("序列已处理完毕，请检查 memory.finish")

        seg_start, seg_end, mode = self._segments[self._seg_idx]
        window = self.beacon.beacon_window
        start = self._pos

        # ---- 根据片段模式确定窗口切分与存储策略 ----
        if mode == "keep":
            # keep 片段：按窗口切分（限制单步显存），全部入持久缓存
            end = min(start + window, seg_end)
            beacon_size = 0
            store = "cache"
        else:
            # compress 片段：满窗口追加 window//ratio 个 beacon；
            # 不满一个窗口的尾部也按剩余 token 数按比例生成 beacon（ceil），
            # 保证任意长度的文档块都被压缩（见 code review M-3）。
            end = min(start + window, seg_end)
            remaining = end - start
            if remaining == window:
                beacon_size = self.beacon.beacon_size_per_window
                store = "beacon"
            elif remaining > 0:
                ratio = self.beacon.beacon_ratio
                beacon_size = max(1, (remaining + ratio - 1) // ratio)
                store = "beacon"
            else:
                beacon_size = 0
                store = "cache"

        # ---- 切出当前窗口的原始 token ----
        input_ids = self.all_input_ids[:, start:end].to(self._device)
        attention_mask = self.all_attention_mask[:, start:end].to(self._device)
        if self.all_labels is not None:
            labels = self.all_labels[:, start:end].to(self._device)
        else:
            labels = None

        # ---- 满窗口时追加 beacon token ----
        if beacon_size > 0:
            beacon_token = self.config.vocab_size
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.full(
                        (input_ids.shape[0], beacon_size),
                        beacon_token,
                        dtype=input_ids.dtype,
                        device=self._device,
                    ),
                ],
                dim=1,
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    attention_mask.new_ones((input_ids.shape[0], beacon_size)),
                ],
                dim=1,
            )
            if labels is not None:
                labels = torch.cat(
                    [labels, labels.new_full((labels.shape[0], beacon_size), -100)],
                    dim=1,
                )

        # ---- beacon_indices：标记当前输入中的 beacon 位置 ----
        if beacon_size > 0:
            cur_len = input_ids.shape[1]
            self._step_beacon_indices = torch.cat(
                [
                    torch.zeros(
                        cur_len - beacon_size, dtype=torch.long, device=self._device
                    ),
                    torch.ones(beacon_size, dtype=torch.long, device=self._device),
                ]
            )
        else:
            self._step_beacon_indices = None

        if beacon_size > 0 and self.beacon.beacon_pos == "intersect":
            order = beacon_intersect_order(end - start, self.beacon.beacon_ratio, self._device)
            input_ids = input_ids[:, order]
            attention_mask = attention_mask[:, order]
            self._step_beacon_indices = self._step_beacon_indices[order]
            if labels is not None:
                labels = labels[:, order]

        # ---- 构造 past_key_values：past = cache（RoPE 前 K/V，按处理顺序） ----
        past_key_values: List[Tuple] = []
        mem_size = 0
        for layer_idx in range(self.num_layers):
            cache_k, cache_v = self._cache[layer_idx]
            if cache_k is not None:
                mem_size = cache_k.shape[2]
            past_key_values.append(
                (cache_k, cache_v, beacon_size, self._step_beacon_indices)
            )

        # ---- 位置编码与注意力掩码 ----
        # 全局位置 0..mem+cur-1：key 按缓存索引整体施加 RoPE（与 activation_beacon 一致），
        # query 只用当前部分（position_ids 的后 cur_len 位）。
        cur_len = input_ids.shape[1]
        position_ids = (
            torch.arange(0, mem_size + cur_len, device=self._device)
            .unsqueeze(0)
            .expand(input_ids.shape[0], -1)
        )

        attn_mask = self._make_4d_causal_mask(
            batch_size=input_ids.shape[0],
            query_len=cur_len,
            mem_size=mem_size,
            device=self._device,
        )

        # ---- 更新游标与存储策略 ----
        self._store = store
        self._pos = end
        self._step_idx += 1

        return input_ids, attn_mask, position_ids, past_key_values, labels

    # ------------------------------------------------------------------
    # 更新记忆
    # ------------------------------------------------------------------
    def update_memory(self, past_key_values: List[Tuple]) -> None:
        """根据当前步的存储策略更新缓存。

        Args:
            past_key_values: 每层 4 元组 ``(key, value, beacon_size, beacon_indices)``，
                其中 key/value 为当前步 **增量** 的、RoPE 之前的 K/V。
        """
        store = self._store
        for layer_idx, (key, value, _beacon_size, _indices) in enumerate(
            past_key_values
        ):
            cache_k, cache_v = self._cache[layer_idx]

            if store == "beacon":
                # 压缩满窗口：只抽取 beacon K/V 进入持久缓存，原始 K/V 丢弃
                beacon_idx = self._step_beacon_indices.bool()
                beacon_key = slice_tensor(key, index=beacon_idx, dim=2)
                beacon_value = slice_tensor(value, index=beacon_idx, dim=2)
                self._cache[layer_idx] = (
                    cat_tensor([cache_k, beacon_key], dim=2),
                    cat_tensor([cache_v, beacon_value], dim=2),
                )
            elif store == "cache":
                # keep 片段 / 不满窗口的压缩尾部：全部 K/V 进入持久缓存
                self._cache[layer_idx] = (
                    cat_tensor([cache_k, key], dim=2),
                    cat_tensor([cache_v, value], dim=2),
                )
            else:
                raise RuntimeError(f"未知的存储策略: {store}")

    # ------------------------------------------------------------------
    # 损失累积
    # ------------------------------------------------------------------
    def update_loss(self, batch_loss: torch.Tensor, valid_num: torch.Tensor) -> None:
        """累积各窗口的损失（避免 in-place 操作以保留梯度图）。

        Args:
            batch_loss: 当前步的 batch 平均损失（已除以本步有效 token 数）。
            valid_num: 当前步的有效 token 数，形状 ``(batch,)``。
        """
        if self._batch_loss is None:
            self._batch_loss = batch_loss * valid_num
            self._valid_num = valid_num
        else:
            self._batch_loss = self._batch_loss + batch_loss * valid_num
            self._valid_num = self._valid_num + valid_num

    def output(
        self, logits: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """汇总累积损失，返回 ``(loss, batch_loss)``。

        Args:
            logits: 最后一层输出的 logits（用于生成，可传 None）。

        Returns:
            ``(loss, batch_loss)``：标量总损失与逐样本损失；无有效标签时均为 None。
        """
        if self._batch_loss is None or self._valid_num is None:
            return None, None
        total_valid = self._valid_num.sum()
        if total_valid.item() == 0:
            # 整条序列无有效标签（如全 -100），返回 0 损失并保留梯度图，避免 0/0 NaN
            return self._batch_loss.sum() * 0.0, None
        loss = self._batch_loss.sum() / total_valid
        batch_loss = self._batch_loss / self._valid_num.clamp(min=1)
        return loss, batch_loss

    # ------------------------------------------------------------------
    # 掩码构造
    # ------------------------------------------------------------------
    def _make_4d_causal_mask(
        self, batch_size: int, query_len: int, mem_size: int, device: torch.device
    ) -> torch.Tensor:
        """构造 4D 因果注意力掩码。

        形状 ``(batch, 1, query_len, mem_size + query_len)``：
        - 历史 ``mem_size`` 列全部置 0（可关注）。
        - 当前部分为下三角因果掩码（0=可关注，min_value=屏蔽）。
        append 的 beacon 关注整个窗口；intersect 的 beacon 关注当前 chunk
        内此前的全部普通 token 和 beacon。历史缓存不包含已压缩 chunk 的原始 token。

        Returns:
            掩码张量。
        """
        min_value = self.min_value
        full_len = mem_size + query_len
        mask = torch.full(
            (batch_size, 1, query_len, full_len),
            min_value,
            dtype=torch.float32,
            device=device,
        )
        if mem_size > 0:
            mask[:, :, :, :mem_size] = 0.0
        # 当前部分下三角因果
        causal = torch.tril(
            torch.ones(query_len, query_len, device=device, dtype=torch.bool)
        )
        current = torch.where(
            causal,
            torch.zeros((), dtype=torch.float32, device=device),
            torch.full((), min_value, dtype=torch.float32, device=device),
        )
        mask[:, :, :, mem_size:] = current.unsqueeze(0).unsqueeze(0)
        return mask
