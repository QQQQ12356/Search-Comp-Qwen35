from unittest.mock import patch

import torch
import torch.nn.functional as F
import pytest
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5

from search_comp.models.beacon_config import BeaconConfig
from search_comp.models.beacon_qwen3 import BeaconQwen3_5ForCausalLM


def _tiny_model(beacon_pos="append", layer_types=None):
    if layer_types is None:
        layer_types = ["linear_attention", "full_attention"]
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=layer_types,
        max_position_embeddings=256,
        eos_token_id=2,
        pad_token_id=0,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
        },
    )
    BeaconConfig(
        beacon_window=4,
        beacon_stride=4,
        beacon_ratio=2,
        beacon_linear_writer_rank=8,
        beacon_pos=beacon_pos,
    ).merge_into_config(config)
    with patch.multiple(
        modeling_qwen3_5,
        chunk_gated_delta_rule=None,
        fused_recurrent_gated_delta_rule=None,
        causal_conv1d_fn=None,
        FusedRMSNormGated=None,
    ):
        return BeaconQwen3_5ForCausalLM(config).eval()


def test_compressed_region_commits_only_beacon_linear_state():
    torch.manual_seed(7)
    model = _tiny_model()
    ids = torch.tensor([[10, 11, 20, 21, 22, 23, 30, 31]])
    memory = model.prefill_and_get_cache(ids, regions=[(2, 6)])

    assert set(memory._linear_recurrent) == {0}
    assert memory._linear_recurrent[0].shape == (1, 2, 8, 8)
    assert 0 in memory._linear_conv

    compressed_only = model.prefill_and_get_cache(ids[:, :6], regions=[(2, 6)])
    assert 0 not in compressed_only._linear_conv


@pytest.mark.parametrize("beacon_pos", ["append", "intersect"])
def test_linear_writer_receives_gradient_from_post_compression_tokens(beacon_pos):
    torch.manual_seed(7)
    model = _tiny_model(beacon_pos).train()
    ids = torch.tensor([[10, 11, 20, 21, 22, 23, 30, 31]])
    labels = ids.clone()
    output = model(input_ids=ids, labels=labels, compress_regions=[(2, 6)])
    output.loss.backward()

    writer = model.model.beacon_linear_writers["0"]
    assert writer.up.weight.grad is not None
    assert writer.up.weight.grad.abs().sum() > 0


def test_sparse_chunked_loss_matches_dense_cross_entropy():
    torch.manual_seed(7)
    model = _tiny_model().train()
    model.beacon_config.beacon_loss_chunk_size = 2
    hidden = torch.randn(1, 5, model.config.hidden_size, requires_grad=True)
    labels = torch.tensor([[10, -100, 20, 21, -100]])
    valid_num = (labels != -100).sum(-1)

    sparse = model._sparse_window_loss(hidden, labels, valid_num).mean()
    selected = labels[0] != -100
    dense = F.cross_entropy(
        model.lm_head(hidden[0, selected]).float(),
        labels[0, selected],
    )

    torch.testing.assert_close(sparse, dense, atol=1e-6, rtol=1e-5)
    sparse.backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_training_does_not_materialize_return_logits():
    torch.manual_seed(7)
    model = _tiny_model().train()
    ids = torch.tensor([[10, 11, 20, 21, 22, 23, 30, 31]])
    labels = ids.clone()
    output = model(input_ids=ids, labels=labels, compress_regions=[(2, 6)])

    assert output.logits is None
    assert torch.isfinite(output.loss)


def test_training_lm_head_never_exceeds_loss_chunk_size():
    torch.manual_seed(7)
    model = _tiny_model().train()
    model.beacon_config.beacon_loss_chunk_size = 2
    projected_lengths = []

    def record_shape(_module, inputs):
        projected_lengths.append(inputs[0].shape[-2])

    handle = model.lm_head.register_forward_pre_hook(record_shape)
    try:
        ids = torch.tensor([[10, 11, 20, 21, 22, 23, 30, 31]])
        labels = torch.tensor([[10, 11, -100, -100, -100, -100, 30, 31]])
        output = model(input_ids=ids, labels=labels, compress_regions=[(2, 6)])
        output.loss.backward()
    finally:
        handle.remove()

    assert projected_lengths
    assert max(projected_lengths) <= 2


def test_intersect_discards_reader_state_after_every_chunk(monkeypatch):
    model = _tiny_model("intersect")
    reader_inputs = []
    writer_outputs = []
    native_reader = model._linear_attention_forward

    def record_reader(module, hidden, previous_recurrent, previous_conv):
        reader_inputs.append((hidden.shape[1], previous_recurrent, previous_conv))
        return native_reader(module, hidden, previous_recurrent, previous_conv)

    monkeypatch.setattr(model, "_linear_attention_forward", record_reader)
    handle = model.model.beacon_linear_writers["0"].register_forward_hook(
        lambda module, inputs, output: writer_outputs.append(output)
    )
    try:
        ids = torch.tensor([[10, 11, 20, 21, 22, 23, 24]])
        memory = model.prefill_and_get_cache(ids, regions=[(2, 7)])
    finally:
        handle.remove()

    assert [entry[0] for entry in reader_inputs] == [2, 6, 2]
    assert len(writer_outputs) == 2
    torch.testing.assert_close(reader_inputs[2][1], writer_outputs[0])
    assert reader_inputs[2][1].data_ptr() != writer_outputs[0].data_ptr()
    assert reader_inputs[2][2] is None
    assert memory._linear_recurrent[0] is writer_outputs[-1]
    assert 0 not in memory._linear_conv
    assert memory._cache[1][0].shape[2] == 5


def test_intersect_first_beacon_is_independent_of_future_slice():
    torch.manual_seed(7)
    model = _tiny_model("intersect")
    first = model.prefill_and_get_cache(
        torch.tensor([[10, 11, 20, 21, 22, 23]]), regions=[(2, 6)]
    )
    second = model.prefill_and_get_cache(
        torch.tensor([[10, 11, 20, 21, 40, 41]]), regions=[(2, 6)]
    )
    for first_cache, second_cache in zip(first._cache[1], second._cache[1]):
        torch.testing.assert_close(first_cache[:, :, :3], second_cache[:, :, :3])
    logits = model.decode_step(torch.tensor([[30]]))
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("beacon_pos", ["append", "intersect"])
@torch.no_grad()
def test_compressed_reader_state_poison_cannot_escape_chunk(monkeypatch, beacon_pos):
    torch.manual_seed(7)
    model = _tiny_model(beacon_pos, ["linear_attention", "full_attention"] * 2)
    ids = torch.tensor([[10, 11, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31]])
    baseline, baseline_logits = model.prefill_and_get_cache(ids, regions=[(2, 11)], return_last_logits=True)
    baseline_decode = model.decode_step(torch.tensor([[32]]))
    native_reader = model._linear_attention_forward
    poisoned_layers = []

    def poison_reader(module, hidden, previous_recurrent, previous_conv):
        result, reader_recurrent, reader_conv = native_reader(module, hidden, previous_recurrent, previous_conv)
        if model._mem._store == "beacon":
            layer_idx = next(idx for idx, layer in enumerate(model.model.layers) if getattr(layer, "linear_attn", None) is module)
            persistent = model._mem._linear_recurrent[layer_idx]
            expected = persistent.clone()
            assert previous_recurrent.data_ptr() != persistent.data_ptr()
            previous_recurrent.fill_(float("nan"))
            torch.testing.assert_close(persistent, expected)
            if previous_conv is not None:
                persistent_conv = model._mem._linear_conv[layer_idx]
                expected_conv = persistent_conv.clone()
                assert previous_conv.data_ptr() != persistent_conv.data_ptr()
                previous_conv.fill_(float("nan"))
                torch.testing.assert_close(persistent_conv, expected_conv)
            poisoned_layers.append(layer_idx)
            return result, torch.full_like(reader_recurrent, float("nan")), torch.full_like(reader_conv, float("nan"))
        return result, reader_recurrent, reader_conv

    monkeypatch.setattr(model, "_linear_attention_forward", poison_reader)
    actual, actual_logits = model.prefill_and_get_cache(ids, regions=[(2, 11)], return_last_logits=True)
    torch.testing.assert_close(actual_logits, baseline_logits)
    torch.testing.assert_close(model.decode_step(torch.tensor([[32]])), baseline_decode)
    assert poisoned_layers == [0, 2] * 3
    for layer_idx in (0, 2):
        torch.testing.assert_close(actual._linear_recurrent[layer_idx], baseline._linear_recurrent[layer_idx])
        torch.testing.assert_close(actual._linear_conv[layer_idx], baseline._linear_conv[layer_idx])
    for layer_idx in (1, 3):
        for actual_cache, baseline_cache in zip(actual._cache[layer_idx], baseline._cache[layer_idx]):
            torch.testing.assert_close(actual_cache, baseline_cache)


@torch.no_grad()
def test_linear_persistent_state_depends_on_documents_only_through_beacons(monkeypatch):
    torch.manual_seed(7)
    model = _tiny_model("intersect", ["linear_attention", "full_attention"] * 2)
    for writer in model.model.beacon_linear_writers.values():
        native_writer = writer.forward

        def fixed_beacon_writer(beacons, previous, original=native_writer):
            assert beacons.shape[1] == 2
            return original(torch.zeros_like(beacons), previous)

        monkeypatch.setattr(writer, "forward", fixed_beacon_writer)

    first = model.prefill_and_get_cache(
        torch.tensor([[10, 11, 20, 21, 22, 23]]), regions=[(2, 6)]
    )
    second = model.prefill_and_get_cache(
        torch.tensor([[10, 11, 40, 41, 42, 43]]), regions=[(2, 6)]
    )
    for layer_idx in (0, 2):
        torch.testing.assert_close(first._linear_recurrent[layer_idx], second._linear_recurrent[layer_idx])
    assert first._linear_conv == second._linear_conv == {}


@pytest.mark.parametrize("beacon_pos", ["append", "intersect"])
def test_reader_state_copy_preserves_cross_chunk_writer_gradients(beacon_pos):
    torch.manual_seed(7)
    model = _tiny_model(beacon_pos).train()
    committed_states = []

    def record_state(module, inputs, output):
        output.retain_grad()
        committed_states.append(output)

    handle = model.model.beacon_linear_writers["0"].register_forward_hook(record_state)
    try:
        ids = torch.tensor([[10, 11, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31]])
        labels = torch.full_like(ids, -100)
        labels[:, 10:] = ids[:, 10:]
        output = model(input_ids=ids, labels=labels, compress_regions=[(2, 10)])
        output.loss.backward()
    finally:
        handle.remove()

    assert len(committed_states) == 2
    for state in committed_states:
        assert state.grad is not None
        assert torch.isfinite(state.grad).all()
        assert state.grad.abs().sum() > 0
