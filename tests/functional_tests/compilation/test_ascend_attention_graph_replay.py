# Copyright (c) 2026 BAAI. All rights reserved.
"""Real NPU task capture/update/replay against the native eager attention path."""

import importlib
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.functional]


@pytest.mark.parametrize("grid", [(1, 14, 20), (2, 6, 10)])
def test_ascend_vision_position_interpolation(ascend_runtime, grid):
    _, _, device = ascend_runtime
    from vllm.model_executor.models.qwen3_vl import (
        pos_embed_interpolate_native,
        triton_pos_embed_interpolate,
    )

    weight = torch.randn(64, 1152).bfloat16()
    expected = pos_embed_interpolate_native(weight, *grid, 8, 2, weight.dtype)
    actual = triton_pos_embed_interpolate(weight.to(device), *grid, 8, 2, weight.dtype)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("head_size", [128, 256])
def test_ascend_attention_forward_keeps_engine_cache(ascend_runtime, head_size):
    attention, _, device = ascend_runtime
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm_fl.worker.model_runner import ModelRunnerFL

    block_size, kv_heads, heads = 128, 2, 4
    storage = torch.zeros(2, 2, block_size, kv_heads, head_size,
                          dtype=torch.bfloat16, device=device)
    group = SimpleNamespace(
        kv_cache_spec=FullAttentionSpec(block_size=128, num_kv_heads=kv_heads,
                                        head_size=head_size, dtype=storage.dtype),
        backend=attention.AscendAttentionBackend, kv_cache_group_id=0,
        layer_names=["cache_test"],
    )
    runner = SimpleNamespace(_kv_cache_spec_attn_group_iterator=lambda: iter([group]),
                             cache_config=SimpleNamespace(cache_dtype="auto"))
    ModelRunnerFL._update_hybrid_attention_mamba_layout(
        runner, {"cache_test":storage}, [block_size],
    )
    caches = [storage[0], storage[1]]
    assert all(cache.is_contiguous() for cache in caches)
    impl = object.__new__(attention.AscendAttentionBackendImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size = heads, kv_heads, head_size
    impl.scale, impl.sliding_window = head_size**-0.5, None
    impl.attn_type = attention.AttentionType.DECODER
    impl.key_cache = impl.value_cache = None
    layer = SimpleNamespace(_k_scale_float=1.0, _v_scale_float=1.0, layer_name="cache_test")
    q = torch.randn(4, heads, head_size).bfloat16()
    k, v = [torch.randn(4, kv_heads, head_size).bfloat16() for _ in range(2)]
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), 1).to(device)
    for start, end, state in [(0, 3, attention.AscendAttentionState.PrefillNoCache),
                              (3, 4, attention.AscendAttentionState.DecodeOnly)]:
        metadata = attention.AscendMetadata(
            attn_state=state, num_actual_tokens=end-start,
            actual_seq_lengths_q=[end-start], seq_lens_list=[end],
            seq_lens=torch.tensor([end], dtype=torch.int32), attn_mask=mask,
            slot_mapping=torch.arange(128+start, 128+end, dtype=torch.int64, device=device),
            block_tables=torch.tensor([[1]], dtype=torch.int32, device=device),
        )
        output = torch.empty(end-start, heads, head_size, dtype=q.dtype, device=device)
        impl.forward(layer, q[start:end].to(device), k[start:end].to(device),
                     v[start:end].to(device), caches, metadata, output)
        expected = torch.empty_like(q[start:end])
        for pos in range(start, end):
            keys = k[:pos+1].float().repeat_interleave(heads//kv_heads, dim=1)
            values = v[:pos+1].float().repeat_interleave(heads//kv_heads, dim=1)
            probabilities = (keys * q[pos].float()).sum(-1).T.mul(impl.scale).softmax(-1)
            expected[pos-start] = (probabilities.T[:, :, None] * values).sum(0)
        torch.testing.assert_close(output.cpu(), expected, rtol=0.03, atol=0.02)
        assert impl.key_cache.data_ptr() == caches[0].data_ptr()
        torch.testing.assert_close(caches[0][1, :end].cpu(), k[:end], rtol=0, atol=0)
        torch.testing.assert_close(caches[1][1, :end].cpu(), v[:end], rtol=0, atol=0)


@pytest.mark.parametrize("head_size", [72, 128])
@pytest.mark.parametrize("tokens", [17, 273])
def test_ascend_vision_attention_matches_reference(ascend_runtime, head_size, tokens):
    _, _, device = ascend_runtime
    from vllm_fl.dispatch.backends.vendor.ascend.impl.mm_encoder_attention import (
        AscendMMEncoderAttention,
    )

    impl = object.__new__(AscendMMEncoderAttention)
    impl.num_heads, impl.num_kv_heads, impl.head_size = 4, 2, head_size
    q = torch.randn(1, tokens, 4, head_size).bfloat16()
    k, v = [torch.randn(1, tokens, 2, head_size).bfloat16() for _ in range(2)]
    expected = torch.empty_like(q)
    for start, end in [(0, 5), (5, tokens)]:
        qs = q[:, start:end].float().transpose(1, 2)
        ks, vs = [t[:, start:end].float().repeat_interleave(2, dim=2).transpose(1, 2)
                  for t in (k, v)]
        probabilities = (qs @ ks.transpose(-1, -2) * head_size**-0.5).softmax(-1)
        expected[:, start:end] = (probabilities @ vs).transpose(1, 2)
    actual = impl.forward_oot(
        q.to(device), k.to(device), v.to(device),
        cu_seqlens=torch.tensor([0, 5, tokens], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.02)


def test_ascend_gdn_decode_matches_recurrence(ascend_runtime):
    _, _, device = ascend_runtime
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    batch, heads, value_heads, width = 3, 2, 4, 128
    mixed = torch.randn(batch, (2 * heads + value_heads) * width).bfloat16()
    a, b = [torch.randn(batch, value_heads).bfloat16() for _ in range(2)]
    a_log, dt_bias = [torch.randn(value_heads) for _ in range(2)]
    states = torch.randn(4, value_heads, width, width)
    expected_state = states.clone()
    expected = torch.zeros(batch, 1, value_heads, width).bfloat16()
    for row, index in enumerate([1, 3, 0]):
        if index == 0:
            continue
        q, k, v = torch.split(
            mixed[row].float(), [heads * width, heads * width, value_heads * width],
        )
        q, k = q.view(heads, width), k.view(heads, width)
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        q, k = [t.repeat_interleave(value_heads // heads, dim=0) for t in (q, k)]
        g = -a_log.exp() * torch.nn.functional.softplus(a[row].float() + dt_bias)
        beta = b[row].float().sigmoid().to(b.dtype).float()
        h = states[index] * g.exp()[:, None, None]
        residual = (v.view(value_heads, width) - (h * k[:, None]).sum(-1))
        h = h + (residual * beta[:, None])[:, :, None] * k[:, None, :]
        expected[row, 0] = (h * (q * width**-0.5)[:, None]).sum(-1)
        expected_state[index] = h
    actual_state = states.to(device)
    output = torch.empty_like(expected, device=device)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed.to(device), a=a.to(device), b=b.to(device),
        A_log=a_log.to(device), dt_bias=dt_bias.to(device), scale=width**-0.5,
        initial_state=actual_state, out=output,
        ssm_state_indices=torch.tensor([1, 3, 0], dtype=torch.int32, device=device),
        use_qk_l2norm_in_kernel=True,
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=0.02, atol=0.01)
    torch.testing.assert_close(actual_state.cpu(), expected_state, rtol=0.02, atol=0.01)


@pytest.mark.parametrize("seeded_state", [False, True])
@pytest.mark.parametrize("tokens", [17, 129])
def test_ascend_gdn_prefill_matches_recurrence(ascend_runtime, seeded_state, tokens):
    """Equal K/V sizes must not hide a transposed recurrent state."""
    _, _, device = ascend_runtime
    from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.chunk import (
        chunk_gated_delta_rule,
    )

    heads, value_heads, width = 2, 4, 128
    q = torch.nn.functional.normalize(torch.randn(1, tokens, heads, width), dim=-1)
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    q, k = q.bfloat16(), k.bfloat16()
    v = torch.randn(1, tokens, value_heads, width).bfloat16()
    g = -torch.rand(1, tokens, value_heads)
    beta = torch.rand_like(g)
    state = torch.randn(2, value_heads, width, width) if seeded_state else (
        torch.zeros(2, value_heads, width, width)
    )
    expected_state = state.clone()
    expected = torch.empty_like(v)
    expanded_q = q.float().repeat_interleave(value_heads // heads, dim=2)
    expanded_k = k.float().repeat_interleave(value_heads // heads, dim=2)
    for seq, (start, end) in enumerate([(0, 3), (3, tokens)]):
        h = state[seq].clone()
        for token in range(start, end):
            kt = expanded_k[0, token]
            h = h * g[0, token].exp()[:, None, None]
            residual = (v[0, token].float() - (h * kt[:, None]).sum(-1))
            residual = residual * beta[0, token, :, None]
            h = h + residual[:, :, None] * kt[:, None, :]
            expected[0, token] = (
                h * (expanded_q[0, token] * width**-0.5)[:, None]
            ).sum(-1)
        expected_state[seq] = h
    actual, final_state = chunk_gated_delta_rule(
        *[t.to(device) for t in (q, k, v, g, beta)],
        initial_state=state.to(device), output_final_state=True,
        cu_seqlens=torch.tensor([0, 3, tokens], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.02)
    torch.testing.assert_close(final_state.cpu(), expected_state, rtol=0.03, atol=0.02)


@pytest.mark.parametrize("tokens", [17, 129])
def test_ascend_conv_prefill_feeds_fused_post_conv(ascend_runtime, tokens):
    _, _, device = ascend_runtime
    from vllm_fl.dispatch.backends.vendor.ascend.impl.causal_conv1d import causal_conv1d_fn
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (
        fused_post_conv_prep,
    )

    heads, value_heads, width = 2, 4, 128
    channels = (2 * heads + value_heads) * width
    x = torch.randn(tokens, channels).bfloat16()
    weight = torch.randn(channels, 4).bfloat16()
    a, b = [torch.randn(tokens, value_heads).bfloat16() for _ in range(2)]
    a_log, dt_bias = [torch.randn(value_heads) for _ in range(2)]
    conv = torch.nn.functional.conv1d(
        x.T.unsqueeze(0), weight[:, None], padding=3, groups=channels,
    )[..., :tokens]
    conv = torch.nn.functional.silu(conv).squeeze(0).T
    q, k, v = torch.split(conv, [heads * width, heads * width, value_heads * width], -1)
    def normalize(tensor):
        tensor = tensor.view(tokens, heads, width).float()
        return (tensor / (tensor.square().sum(-1, keepdim=True) + 1e-6).sqrt()).bfloat16()
    expected = (
        normalize(q), normalize(k), v.view(tokens, value_heads, width),
        -a_log.exp() * torch.nn.functional.softplus(a.float() + dt_bias),
        b.float().sigmoid(),
    )
    conv_state = torch.zeros(2, channels, 3, dtype=x.dtype, device=device)
    actual_conv = causal_conv1d_fn(
        x.to(device).T, weight.to(device), None, conv_state,
        torch.tensor([0, tokens], dtype=torch.int32, device=device),
        cache_indices=torch.tensor([1], dtype=torch.int32, device=device),
        has_initial_state=torch.tensor([False], device=device),
    ).T
    actual = fused_post_conv_prep(
        actual_conv, *[t.to(device) for t in (a, b, a_log, dt_bias)],
        heads, width, width,
    )
    assert actual_conv.is_contiguous()
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result.cpu(), reference, rtol=0.03, atol=0.02)


def test_ascend_conv_prefill_vllm028_interface(ascend_runtime):
    """Warmup's positional cache arguments and null blocks must be supported."""
    _, _, device = ascend_runtime
    from vllm_fl.dispatch.backends.vendor.ascend.impl.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_ref,
    )

    x = torch.randn(16, 5, dtype=torch.bfloat16)
    weight = torch.randn(16, 4, dtype=x.dtype)
    states = torch.randn(3, 16, 3, dtype=x.dtype)
    expected_state = states.clone()
    first, _ = causal_conv1d_ref(
        x[:, :3], weight, initial_states=states[1],
        return_final_states=True, final_states_out=expected_state[1],
    )
    second, _ = causal_conv1d_ref(x[:, 3:], weight)
    actual_state = states.to(device)
    actual = causal_conv1d_fn(
        x.to(device), weight.to(device), None, actual_state,
        torch.tensor([0, 3, 5], dtype=torch.int32, device=device),
        cache_indices=torch.tensor([1, 0], dtype=torch.int32, device=device),
        has_initial_state=torch.tensor([True, True], device=device),
        activation="silu", null_block_id=0, validate_data=False,
    )
    assert actual.transpose(0, 1).is_contiguous()
    torch.testing.assert_close(actual.cpu(), torch.cat((first, second), dim=-1),
                               rtol=0.02, atol=0.02)
    torch.testing.assert_close(actual_state.cpu(), expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("vocab_size", [513, 248320])
def test_ascend_topk_topp_batch_eight(ascend_runtime, monkeypatch, vocab_size):
    _, _, device = ascend_runtime
    from vllm.v1.sample.ops import topk_topp_sampler as sampler
    from vllm_fl.dispatch.backends.vendor.ascend import patch

    monkeypatch.setattr(sampler, "apply_top_k_top_p", sampler.apply_top_k_top_p)
    patch.patch_topk_topp_sampler()
    logits = torch.randn(8, vocab_size)
    k = torch.tensor([1, 4, 8, 32, 64, 128, 256, vocab_size], dtype=torch.int32)
    p = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1])
    for top_k, top_p in [(k, p), (k, None), (None, p)]:
        expected = sampler.apply_top_k_top_p_pytorch(
            logits.clone(), top_k, top_p,
        )
        actual = sampler.apply_top_k_top_p(
            logits.to(device), top_k.to(device) if top_k is not None else None,
            top_p.to(device) if top_p is not None else None,
        ).cpu()
        kept = torch.isfinite(actual)
        torch.testing.assert_close(actual[kept], logits[kept], rtol=0, atol=0)
        different = kept != torch.isfinite(expected)
        if not different.any():
            continue
        # CPU/NPU softmax and cumsum can round on opposite sides of a top-p
        # boundary. Permit only boundary tokens, with unchanged finite logits.
        assert top_p is not None
        assert not different[top_p == 1].any()
        assert (different.sum(-1) <= 2).all()
        sorted_logits, order = logits.sort(dim=-1)
        if top_k is not None:
            cutoff = sorted_logits.gather(1, (vocab_size-top_k.long())[:, None])
            sorted_logits = sorted_logits.masked_fill(sorted_logits < cutoff, -torch.inf)
        cumulative = sorted_logits.double().softmax(-1).cumsum(-1)
        distance = (cumulative - (1-top_p.double())[:, None]).abs()
        assert (distance[different.gather(1, order)] < 2e-6).all()


@pytest.fixture
def ascend_runtime(backend, request, monkeypatch):
    # Do not import Ascend modules (or request an NPU device) on CPU/CUDA hosts.
    if backend != "ascend":
        pytest.skip("Ascend NPU graph replay requires the Ascend backend")
    importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    device = request.getfixturevalue("device")
    load_attention = request.getfixturevalue("ascend_attention_module")

    import vllm_fl.compilation.graph as graph_state

    monkeypatch.setattr(graph_state, "_ascend_graph_params", None)
    monkeypatch.setattr(graph_state, "_ascend_graph_capturing", False)
    try:
        attention = load_attention()
        with torch.npu.device(device):
            torch.npu.synchronize()
            yield attention, graph_state, device
            torch.npu.synchronize()
    finally:
        torch.npu.synchronize()
        graph_state.set_ascend_graph_capturing(False)
        graph_state._ascend_graph_params = None


@pytest.mark.parametrize("batch, live_batch", [(1, 1), (2, 2), (4, 3), (4, 1)])
@pytest.mark.parametrize("windows", [(8,), (None, 8, 64)], ids=["fia", "pa_fia"])
def test_ascend_attention_graph_replay(ascend_runtime, batch, live_batch, windows):
    attention, state, device = ascend_runtime
    impl_cls = attention.AscendAttentionBackendImpl
    state.set_ascend_graph_params([batch])
    # Keep all captured buffers alive until the final replay has synchronized.
    query = torch.randn(batch, 4, 128, dtype=torch.bfloat16, device=device)
    key = torch.randn(4, 128, 2, 128, dtype=query.dtype, device=device)
    value = torch.randn_like(key)
    mask = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), 1).to(device)
    layers = []
    for index, window in enumerate(windows):
        impl = object.__new__(impl_cls)
        impl.num_heads, impl.num_kv_heads, impl.head_size = 4, 2, 128
        impl.scale = 128**-0.5
        impl.sliding_window = window
        impl.key_cache, impl.value_cache = key, value
        layers.append((f"layer{index}", impl, torch.empty_like(query)))

    def metadata(lengths, swapped=False):
        valid = len(lengths)
        lengths = list(lengths) + [0] * (batch - valid)
        return attention.AscendMetadata(
            attn_state=attention.AscendAttentionState.DecodeOnly,
            attn_mask=mask,
            num_actual_tokens=batch,
            actual_seq_lengths_q=list(range(1, valid + 1)) + [valid] * (batch - valid),
            # PA requires CPU context lengths, even when query/cache are NPU tensors.
            seq_lens=torch.tensor(lengths, dtype=torch.int32, device="cpu"),
            seq_lens_list=list(lengths),
            block_tables=torch.tensor(
                [[2, 3] if swapped else [0, 1] for _ in range(valid)]
                + [[0, 0]] * (batch - valid),
                dtype=torch.int32,
                device=device,
            ),
        )

    captured = metadata([3] * batch)
    for name, impl, output in layers:
        impl.forward_impl(query, key, value, (), captured, output, name)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    stream, update_stream = torch.npu.Stream(), torch.npu.Stream()
    try:
        state.set_ascend_graph_capturing(True)
        try:
            with torch.npu.graph(graph, stream=stream):
                for name, impl, output in layers:
                    impl.forward_impl(query, key, value, (), captured, output, name)
        finally:
            state.set_ascend_graph_capturing(False)
        state.weak_ref_ascend_graph_workspaces()
        for step, length in enumerate((9, 129, 133)):
            live = metadata([length + i for i in range(live_batch)], swapped=bool(step % 2))
            expected = []
            for name, impl, output in layers:
                reference = torch.empty_like(output)
                impl.forward_impl(query, key, value, (), live, reference, name)
                expected.append(reference)
            torch.npu.synchronize()
            graph.replay()
            impl_cls.update_graph_params(
                update_stream,
                SimpleNamespace(attn_metadata={name: live for name, _, _ in layers}),
                batch,
            )
            torch.npu.synchronize()
            for (_, _, output), reference in zip(layers, expected):
                assert torch.isfinite(output[:live_batch]).all().item()
                torch.testing.assert_close(
                    output[:live_batch], reference[:live_batch], atol=0.02, rtol=0.02
                )
    finally:
        torch.npu.synchronize()
        graph.reset()


@pytest.mark.parametrize("top_k", [1, 8])
@pytest.mark.parametrize("weighted, with_bias", [(False, False), (True, False), (True, True)])
def test_ascend_moe_naive_graph_replay(ascend_runtime, top_k, weighted, with_bias):
    """Replay must use live experts/weights, including negative expert slots."""
    _, _, device = ascend_runtime
    from vllm_fl.dispatch.backends.vendor.ascend.impl import fused_moe_kernel as moe
    from vllm_fl.dispatch import CachedOp

    invoke_moe = CachedOp("invoke_fused_moe_triton_kernel")

    batch, experts, n, k = 4, 5, 16, 32
    dtype = torch.bfloat16
    inputs = torch.randn(batch, k, dtype=dtype).to(device)
    weights = torch.randn(experts, n, k, dtype=dtype).to(device)
    bias = torch.randn(experts, n, dtype=dtype).to(device) if with_bias else None
    route_weights = torch.rand(batch, top_k).to(device)
    ids = (torch.arange(batch * top_k, device=device) % experts).to(torch.int32)
    padded = torch.empty(1, dtype=torch.int32, device=device)
    output = torch.empty(batch, top_k, n, dtype=dtype, device=device)
    config = {"BLOCK_SIZE_M": 16}

    def run(target):
        invoke_moe(
            inputs, weights, target, None, None, route_weights, None, ids,
            padded, weighted, top_k, config, B_bias=bias,
        )

    run(output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    try:
        with torch.npu.graph(graph, stream=stream):
            run(output)
        for step in range(4):
            live_ids = (torch.arange(batch * top_k, device=device) + step + 1) % experts
            live_ids[step::3] = -1
            if step == 3:
                live_ids.fill_(-1)
            ids.copy_(live_ids)
            inputs.copy_(torch.randn(inputs.shape, dtype=dtype).to(device))
            route_weights.copy_(torch.rand(route_weights.shape).to(device))
            if bias is not None:
                bias.copy_(torch.randn(bias.shape, dtype=dtype).to(device))
            reference = torch.zeros_like(output)
            # Compare with the original per-expert eager implementation.
            moe._invoke_fused_moe_loop(
                inputs, weights, reference, None, None, route_weights, None, ids,
                padded, weighted, top_k, config, B_bias=bias,
            )
            output.fill_(float("nan"))
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            assert torch.isfinite(output).all().item()
            torch.testing.assert_close(output, reference, atol=0.03, rtol=0.02)
    finally:
        torch.npu.synchronize()
        graph.reset()


def test_ascend_moe_workspace_after_capture(ascend_runtime, monkeypatch):
    """A larger prefill must preserve the locked decode graph's workspace."""
    _, _, device = ascend_runtime
    import vllm.config
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import biased_moe_quant_config
    from vllm.v1.worker import workspace
    from vllm_fl.ops.fused_moe import fused_moe_utils as utils

    # Match Worker.init_device's NPU allocator compatibility patch.
    monkeypatch.setattr(torch.accelerator, "empty_cache", torch.npu.empty_cache)
    # No model construction config is available during weight postprocessing
    # or forward. Only the expert's scheduler capacity may be used here.
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: pytest.fail(
        "workspace allocation must not read the model construction config"
    ))
    manager = workspace.WorkspaceManager(device)
    monkeypatch.setattr(workspace, "_manager", manager)
    config = SimpleNamespace(
        max_num_tokens=256, swiglu_limit=None, swiglu_alpha=None,
        swiglu_beta=None, activation_situ_beta=None,
        activation_situ_linear_beta=None,
    )
    experts = utils.TritonExpertsFL(config, biased_moe_quant_config(None, None))
    kernel = object.__new__(mk.FusedMoEKernelModularImpl)
    kernel.fused_experts = experts

    def allocate(tokens):
        return kernel._allocate_buffers(
            torch.bfloat16, device, tokens, tokens, 32, 64, 8,
            16, 16, None, MoEActivation.SILU,
        )

    buffers = allocate(8)
    pointer = buffers[0].data_ptr()
    source = torch.ones(8, 64, dtype=torch.bfloat16, device=device)
    output = torch.empty_like(source)
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph, stream=torch.npu.Stream()):
            buffers[0].view(-1)[:source.numel()].copy_(source.view(-1))
            output.copy_(buffers[0].view(-1)[:source.numel()].view_as(source))
        manager.lock()
        for tokens in (65, 256):
            prefill = allocate(tokens)
            assert prefill[0].data_ptr() == pointer
            assert prefill[2].shape == (tokens, 64)
            prefill[0].zero_()
            source.fill_(tokens)
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output, source)
    finally:
        torch.npu.synchronize()
        graph.reset()


@pytest.mark.parametrize("transposed_conv_state", [False, True])
def test_ascend_gdn_graph_replay(ascend_runtime, transposed_conv_state, monkeypatch):
    """Replay must read live state indices and leave padding slots unchanged."""
    _, _, device = ascend_runtime
    from vllm_fl.dispatch.backends.vendor.ascend import patch

    # Undo the global kernel bindings installed by these patches after the test.
    for path in (
        "layers.mamba.ops.causal_conv1d", "models.qwen3_next",
        "layers.mamba.gdn.qwen_gdn_linear_attn",
    ):
        module = importlib.import_module("vllm.model_executor." + path)
        for name, value in list(vars(module).items()):
            if callable(value):
                monkeypatch.setattr(module, name, value)
    from vllm_fl.dispatch.backends.vendor.ascend.impl.fla import l2norm

    monkeypatch.setattr(l2norm, "l2norm_fwd", l2norm.l2norm_fwd)
    patch.patch_causal_conv1d()
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    dtype = torch.bfloat16
    batch, slots, heads, kv_heads, kd, vd = 4, 6, 4, 2, 4, 8
    channels = 2 * kv_heads * kd + heads * vd
    x = torch.randn(batch, channels, dtype=dtype, device=device)
    if transposed_conv_state:
        conv_state = torch.randn(slots, 4, channels, dtype=dtype, device=device).transpose(1, 2)
    else:
        conv_state = torch.randn(slots, channels, 4, dtype=dtype, device=device)
    state = torch.randn(slots, heads, vd, kd, dtype=dtype, device=device)
    indices = torch.arange(batch, dtype=torch.int32, device=device)
    weight = torch.randn(channels, 4, dtype=dtype, device=device)
    a, b = [torch.randn(batch, heads, dtype=dtype, device=device) for _ in range(2)]
    A_log, dt_bias = [torch.randn(heads, device=device) for _ in range(2)]
    out = torch.empty(batch, 1, heads, vd, dtype=dtype, device=device)

    def run(x, conv_state, state, out):
        qkv = gdn.causal_conv1d_update(
            x, conv_state, weight, activation="silu", conv_state_indices=indices
        )
        gdn.fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            scale=kd**-0.5, initial_state=state, out=out,
            ssm_state_indices=indices, use_qk_l2norm_in_kernel=True,
        )

    run(x.clone(), conv_state.clone(), state.clone(), out)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    try:
        with torch.npu.graph(graph, stream=stream):
            run(x, conv_state, state, out)
        for live_indices in ([3, -1, 0, 5], [-1, -1, -1, 0],
                             [5, 2, 4, -1], [-1, -1, -1, -1]):
            indices.copy_(torch.tensor(live_indices, dtype=indices.dtype, device=device))
            fresh = [torch.randn_like(t) for t in (x, conv_state, state, out)]
            fresh[2][3, 0, 0, 0] = float("nan")
            fresh[2][4, 0, 0, 0] = float("inf")
            expected = [t.clone() for t in fresh]
            run(*expected)
            for target, source in zip((x, conv_state, state, out), fresh):
                target.copy_(source)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            for name, actual, reference in zip(
                ("conv_output", "conv_state", "ssm_state", "output"),
                (x, conv_state, state, out), expected,
            ):
                torch.testing.assert_close(
                    actual, reference, atol=0, rtol=0, equal_nan=True,
                    msg=lambda message: f"{name}, indices={live_indices}: {message}",
                )
    finally:
        torch.npu.synchronize()
        graph.reset()
