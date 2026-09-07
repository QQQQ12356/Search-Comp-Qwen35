"""Beacon 压缩机制的配置数据类。

本模块定义了 Activate Beacon 长上下文压缩机制的全部超参数，
作为独立的数据类，方便从 YAML / dict 初始化并与 HuggingFace
模型 config 互相转换。

参考论文: Soaring from 4K to 400K: Extending LLM's Context with
Activation Beacon (https://arxiv.org/abs/2401.03462)。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class BeaconConfig:
    """Beacon 压缩机制的超参数配置。

    核心思路：把检索到的文档按窗口切分，每 ``beacon_ratio`` 个 token
    生成 1 个 beacon token。beacon token 通过自注意力聚合所在窗口的
    信息（软提示压缩），后续解码只关注 beacon 的 K/V cache，原始文档
    的 K/V 在生成 beacon 后即被丢弃。
    """

    #: 是否启用 beacon 压缩机制（False 时退化为普通 causal LM）
    enable_beacon: bool = True
    #: 滑动窗口大小（token 数）。文档区按此大小切分窗口
    beacon_window: int = 1024
    #: 窗口步长（token 数）。本实现使用 append 模式且 stride == window，无重叠
    beacon_stride: int = 1024
    #: 压缩率：每多少个原始 token 生成 1 个 beacon。例：64 表示每 64 token -> 1 beacon
    beacon_ratio: int = 64
    #: beacon 的注意力模式。仅支持 "full-coverage"（beacon 关注窗口内全部 token）
    beacon_attn: str = "full-coverage"
    #: 为 beacon 引入哪些独立投影矩阵，取值 "q"/"k"/"v"/"o" 的任意组合（空格分隔）
    beacon_param: str = "q k v"
    #: beacon 嵌入向量的初始化来源："eos" / "bos"
    beacon_embed_init: str = "eos"
    #: 序列开头始终保留的 attention sink token 数（本实现用 skip_first 保留 question 区，置 0）
    beacon_sink_size: int = 0
    #: beacon 是否能关注历史 beacon（跨窗口记忆）
    beacon_attend_prev: bool = True
    #: beacon 放置方式。仅支持 "append"（beacon 追加在窗口末尾）
    beacon_pos: str = "append"
    #: 生成 / 推理时使用的压缩率（覆盖训练时的 beacon_ratio）
    eval_beacon_ratio: Optional[int] = None

    def __post_init__(self) -> None:
        """校验参数合法性。"""
        if self.enable_beacon:
            assert (
                self.beacon_window >= self.beacon_stride
            ), f"beacon_window({self.beacon_window}) 必须 >= beacon_stride({self.beacon_stride})"
            assert (
                self.beacon_ratio > 0
            ), f"beacon_ratio 必须为正数，当前为 {self.beacon_ratio}"
            assert (
                self.beacon_window % self.beacon_ratio == 0
            ), f"beacon_window({self.beacon_window}) 必须能被 beacon_ratio({self.beacon_ratio}) 整除"
            assert (
                self.beacon_attn == "full-coverage"
            ), f"当前实现仅支持 full-coverage 注意力模式，收到 {self.beacon_attn}"
            assert (
                self.beacon_pos == "append"
            ), f"当前实现仅支持 append 放置模式，收到 {self.beacon_pos}"
            valid_params = {"q", "k", "v", "o"}
            for p in self.beacon_param.split():
                assert p in valid_params, f"beacon_param 含非法投影类型: {p}"

    # ------------------------------------------------------------------
    # 序列化 / 反序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """转换为 dict（用于写入模型 config 与超参数记录）。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeaconConfig":
        """从 dict 构造，忽略未知字段。"""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    # ------------------------------------------------------------------
    # 派生属性
    # ------------------------------------------------------------------
    @property
    def beacon_size_per_window(self) -> int:
        """每个满窗口生成的 beacon 数量 = window // ratio。"""
        return self.beacon_window // self.beacon_ratio

    def merge_into_config(self, model_config: Any) -> "BeaconConfig":
        """把 beacon 超参写入 HuggingFace model.config。

        字段名保持原样（如 ``beacon_window``），与 Qwen2Config 默认字段无冲突，
        保存权重时会一并写入 config.json。

        Args:
            model_config: HuggingFace PretrainedConfig 对象。

        Returns:
            写入后的自身引用。
        """
        for name, value in self.to_dict().items():
            setattr(model_config, name, value)
        return self

    @classmethod
    def from_model_config(cls, model_config: Any) -> "BeaconConfig":
        """从 HuggingFace model.config 读取 beacon 字段。"""
        data = {f.name: getattr(model_config, f.name, f.default) for f in fields(cls)}
        return cls.from_dict(data)
