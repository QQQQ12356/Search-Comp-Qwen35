import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from search_comp.models.beacon_config import BeaconConfig
from search_comp.models.beacon_qwen3 import BeaconQwen3_5ForCausalLM


def _tiny_model():
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
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
    ).merge_into_config(config)
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


def test_linear_writer_receives_gradient_from_post_compression_tokens():
    torch.manual_seed(7)
    model = _tiny_model().train()
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
