# Copyright (c) 2026 BAAI. All rights reserved.
"""Real NPU task capture/update/replay against the native eager attention path."""

import importlib
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.functional]


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
        moe.invoke_fused_moe_torch(
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
    experts = utils.TritonExpertsFL(SimpleNamespace(max_num_tokens=256), None)
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
        "layers.mamba.gdn_linear_attn", "layers.fla.ops",
        "layers.fla.ops.fused_sigmoid_gating",
        "layers.fla.ops.fused_gdn_prefill_post_conv",
        "layers.fla.ops.fused_recurrent", "layers.fla.ops.chunk",
        "layers.fla.ops.l2norm",
    ):
        module = importlib.import_module("vllm.model_executor." + path)
        for name, value in list(vars(module).items()):
            if callable(value):
                monkeypatch.setattr(module, name, value)
    from vllm_fl.dispatch.backends.vendor.ascend.impl.fla import l2norm

    monkeypatch.setattr(l2norm, "l2norm_fwd", l2norm.l2norm_fwd)
    patch.patch_causal_conv1d()
    patch.patch_gdn_triton_ops()
    from vllm.model_executor.layers.mamba import gdn_linear_attn as gdn

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
            graph.replay()
            torch.npu.synchronize()
            for actual, reference in zip((x, conv_state, state, out), expected):
                torch.testing.assert_close(actual, reference, atol=0, rtol=0, equal_nan=True)
    finally:
        torch.npu.synchronize()
        graph.reset()
