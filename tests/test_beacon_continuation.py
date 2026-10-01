"""子文档压缩区切分 + 非首窗续写损失。

覆盖：

- :func:`split_document_regions` 的边界定位、单文档与降级路径；
- :func:`build_sequence_ids` 在 ``doc_region_split`` 下产出逐子文档的压缩区；
- 续写损失：只有一个窗口的压缩区是**严格 no-op**；跨两个窗口时才产生损失与梯度；
- 推理路径（``labels is None``）不产生任何续写损失。
"""

import os

import torch

from search_comp.data.trajectory import build_sequence_ids, split_document_regions
from search_comp.models.beacon_config import BeaconConfig
from search_comp.models.beacon_qwen3 import _Qwen3BeaconMemory
from test_beacon_qwen35_linear_memory import _tiny_model


DEVICE = os.environ.get("BEACON_TEST_DEVICE", "cpu")


class _Encoded(dict):
    """既支持 ``.get`` 也支持 ``.input_ids`` 属性访问的返回对象。"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _CharTokenizer:
    """字符级假分词器：token 与字符一一对应，便于精确断言区间。"""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        encoded = _Encoded(input_ids=list(range(len(text))))
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return encoded


class _NoOffsetTokenizer(_CharTokenizer):
    """不支持 offset mapping 的分词器（触发整块降级）。"""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return _Encoded(input_ids=list(range(len(text))))


#: 两篇子文档：``Doc 1 A`` + 空行 + ``Doc 2 B``（长度 16）。
TWO_DOCS = "Doc 1 A\n\nDoc 2 B"

SAMPLE = {
    "id": "sample-0",
    "question": "Q",
    "answer": "A",
    "turns": [{"query": "q1", "docs": TWO_DOCS}],
    "thinks": ["t1"],
    "final_think": "t2",
}


# ----------------------------------------------------------------------
# 切分
# ----------------------------------------------------------------------
def test_split_document_regions_locates_document_start():
    # 边界落在第二篇文档的首字符（9），而不是分隔空行的起点（7）。
    assert split_document_regions(TWO_DOCS, _CharTokenizer()) == [(0, 9), (9, 16)]


def test_split_document_regions_single_document_is_one_region():
    assert split_document_regions("Doc 1 only", _CharTokenizer()) == [(0, 10)]


def test_split_document_regions_falls_back_without_offset_mapping():
    assert split_document_regions(TWO_DOCS, _NoOffsetTokenizer()) == [(0, 16)]


def test_split_document_regions_empty_text():
    assert split_document_regions("", _CharTokenizer()) == []


def test_split_document_regions_covers_block_without_gaps():
    regions = split_document_regions(TWO_DOCS, _CharTokenizer())
    assert regions[0][0] == 0
    assert regions[-1][1] == len(TWO_DOCS)
    assert all(end == nxt[0] for (_, end), nxt in zip(regions, regions[1:]))


def test_build_sequence_ids_splits_regions_per_document():
    _, regions, _, _ = build_sequence_ids(
        _CharTokenizer(), "chat", SAMPLE, doc_region_split=True
    )
    assert len(regions) == 2
    assert regions[0][1] == regions[1][0]
    assert regions[1][1] - regions[0][0] == len(TWO_DOCS)


def test_build_sequence_ids_keeps_single_region_by_default():
    _, regions, _, _ = build_sequence_ids(_CharTokenizer(), "chat", SAMPLE)
    assert len(regions) == 1
    assert regions[0][1] - regions[0][0] == len(TWO_DOCS)


# ----------------------------------------------------------------------
# 续写损失
# ----------------------------------------------------------------------
#: 12 个 token；``beacon_window=4``、``beacon_ratio=2``（见 ``_tiny_model``）。
IDS = torch.tensor([[10, 11, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31]], device=DEVICE)


def _make_model(weight: float):
    """同种子构造两个逐位相同的模型，只有续写权重不同。"""
    torch.manual_seed(7)
    model = _tiny_model()
    model.beacon_config.beacon_continuation_loss_weight = weight
    model.beacon_config.beacon_continuation_tokens = 2
    return model.eval()


def _loss(model, regions, labels):
    return model(input_ids=IDS, labels=labels, regions=regions).loss


def test_continuation_is_strict_noop_without_later_windows():
    """4 token 的压缩区只有一个窗口 → 不产生续写目标，损失逐位不变。"""
    labels = IDS.clone()
    labels[:, :8] = -100
    baseline = _loss(_make_model(0.0), [(2, 6)], labels)
    weighted = _loss(_make_model(0.5), [(2, 6)], labels)
    assert torch.equal(baseline, weighted)


def test_continuation_loss_fires_on_later_windows():
    """8 token 的压缩区跨两个窗口 → 第二窗头部被监督，损失变大。"""
    labels = IDS.clone()
    labels[:, :8] = -100
    baseline = _loss(_make_model(0.0), [(2, 10)], labels)
    weighted = _loss(_make_model(0.5), [(2, 10)], labels)
    assert torch.isfinite(weighted)
    assert weighted.item() > baseline.item()


def test_continuation_gradients_reach_beacon_parameters():
    """labels 全 -100 时主 CE 为 0，梯度只能来自续写损失。"""
    model = _make_model(0.5).train()
    labels = torch.full_like(IDS, -100)
    loss = _loss(model, [(2, 10)], labels)

    assert loss.item() > 0
    loss.backward()
    for name in ("beacon_embed_tokens",):
        grad = getattr(model.model, name).weight.grad
        assert grad is not None and grad.abs().sum() > 0


def test_adjacent_single_window_documents_never_continue():
    """两个相邻子文档区各只有 1 个窗口 → 都不监督，文档边界处不产生伪续写。"""
    labels = IDS.clone()
    labels[:, :8] = -100
    baseline = _loss(_make_model(0.0), [(2, 6), (6, 10)], labels)
    weighted = _loss(_make_model(0.5), [(2, 6), (6, 10)], labels)
    assert torch.equal(baseline, weighted)


def _supervised_window_starts(model, ids, regions):
    """逐窗调用状态机，收集真正被续写监督的窗口起点。"""
    model._mem = _Qwen3BeaconMemory(model, model.beacon_config)
    model._mem.prepare(ids, ids.clone(), regions)
    starts = []
    while not model._mem.finish:
        start = model._mem._pos
        model._mem.step()
        if model._mem._step_cont_targets is not None:
            starts.append(start)
    return starts


def test_only_later_windows_within_each_document_are_supervised():
    """(2,10) 与 (10,18) 各 2 窗 → 监督 6 与 14，**不监督**文档边界 10。"""
    ids = torch.arange(10, 28, device=DEVICE).unsqueeze(0)
    starts = _supervised_window_starts(_make_model(0.5), ids, [(2, 10), (10, 18)])
    assert starts == [6, 14]


def _count_keep_windows(model, ids, regions):
    """逐窗走状态机，统计 keep 窗口数（不跑模型前向）。"""
    model._mem = _Qwen3BeaconMemory(model, model.beacon_config)
    model._mem.prepare(ids, None, regions)
    count = 0
    while not model._mem.finish:
        model._mem.step()
        if model._mem._store == "cache":
            count += 1
    return count


ALL_KEEP_IDS = torch.arange(10, 42, device=DEVICE).unsqueeze(0)   # 32 token，全 keep


def test_keep_window_default_keeps_historical_chunking():
    """None（默认）→ keep 仍按 beacon_window 切：32/4 = 8 个窗口。"""
    assert _count_keep_windows(_make_model(0.0), ALL_KEEP_IDS, []) == 8


def test_keep_window_zero_merges_whole_segment():
    model = _make_model(0.0)
    model.beacon_config.beacon_keep_window = 0
    assert _count_keep_windows(model, ALL_KEEP_IDS, []) == 1


def test_keep_window_positive_is_a_cap():
    model = _make_model(0.0)
    model.beacon_config.beacon_keep_window = 16
    assert _count_keep_windows(model, ALL_KEEP_IDS, []) == 2


def test_keep_window_does_not_change_result():
    """keep 切窗只是计算粒度：不切窗与历史切窗的 loss 在浮点误差内一致。"""
    labels = ALL_KEEP_IDS.clone()
    chunked = _make_model(0.0)(input_ids=ALL_KEEP_IDS, labels=labels, regions=[]).loss
    merged_model = _make_model(0.0)
    merged_model.beacon_config.beacon_keep_window = 0
    merged = merged_model(input_ids=ALL_KEEP_IDS, labels=labels, regions=[]).loss
    assert torch.allclose(chunked, merged, atol=1e-5, rtol=1e-4)


def test_keep_window_does_not_disturb_continuation_targets():
    """keep 不切窗不改变压缩段布局与续写监督点。"""
    ids = torch.arange(10, 28, device=DEVICE).unsqueeze(0)
    regions = [(2, 10), (10, 18)]
    default = _supervised_window_starts(_make_model(0.5), ids, regions)
    merged_model = _make_model(0.5)
    merged_model.beacon_config.beacon_keep_window = 0
    merged = _supervised_window_starts(merged_model, ids, regions)
    assert default == merged == [6, 14]


def test_describe_layout_reports_effective_settings():
    """评测启动打印的那一行必须反映实际生效的布局，而不是默认值。"""
    split = BeaconConfig(
        beacon_window=96, beacon_stride=96, beacon_ratio=16,
        beacon_doc_region_split=True, beacon_keep_window=1024,
    ).describe_layout()
    assert "子文档独立成段" in split
    assert "每窗 6 个 beacon" in split
    assert "keep_window=1024" in split

    chunked = BeaconConfig(beacon_window=512, beacon_stride=512, beacon_ratio=16).describe_layout()
    assert "整块成段" in chunked
    assert "每窗 32 个 beacon" in chunked
    assert "keep_window=跟随 window" in chunked

    merged = BeaconConfig(
        beacon_window=512, beacon_stride=512, beacon_ratio=16, beacon_keep_window=0,
    ).describe_layout()
    assert "keep_window=整段不切" in merged


def test_prefill_produces_no_continuation_loss():
    """推理路径 labels 为 None，续写目标不构造，损失列表保持为空。"""
    model = _make_model(0.5)
    memory = model.prefill_and_get_cache(IDS, regions=[(2, 10)])
    assert memory._cont_losses == []
    assert memory._shifted_ids is None
