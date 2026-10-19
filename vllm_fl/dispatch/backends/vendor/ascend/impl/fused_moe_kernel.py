# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend NPU pure-torch MoE kernels.

Replaces FlagGems Triton kernels that crash on Ascend NPU.
Unquantized naive decode keeps routing on device for NPUGraph replay;
aligned batches retain the CPU routing implementation.
"""

import torch
import numpy as np
from vllm.utils.math_utils import round_up


def moe_align_block_size_torch(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch moe_align_block_size for Ascend NPU (CPU-based)."""
    device = topk_ids.device
    num_tokens = topk_ids.numel()

    max_num_tokens_padded = num_tokens + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if num_tokens < num_experts:
        max_num_tokens_padded = min(num_tokens * block_size, max_num_tokens_padded)

    topk_ids_flat = topk_ids.view(-1).cpu()
    padding_value = num_tokens

    expert_counts = torch.bincount(topk_ids_flat.long(), minlength=num_experts)[:num_experts]

    sorted_ids_list = []
    expert_ids_list = []

    for e in range(num_experts):
        count = expert_counts[e].item()
        if count == 0 and ignore_invalid_experts:
            continue
        expert_tokens = (topk_ids_flat == e).nonzero(as_tuple=True)[0].to(torch.int32)
        padded_count = ((count + block_size - 1) // block_size) * block_size
        num_blocks = padded_count // block_size
        padded_tokens = torch.full((padded_count,), padding_value, dtype=torch.int32)
        padded_tokens[:count] = expert_tokens
        sorted_ids_list.append(padded_tokens)
        expert_ids_list.extend([e] * num_blocks)

    if not sorted_ids_list:
        sorted_ids = torch.full((max_num_tokens_padded,), padding_value, dtype=torch.int32)
        max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
        expert_ids_out = torch.full((max_num_m_blocks,), -1, dtype=torch.int32)
        num_tokens_post_pad = torch.zeros(1, dtype=torch.int32)
        return sorted_ids.to(device), expert_ids_out.to(device), num_tokens_post_pad.to(device)

    sorted_ids = torch.cat(sorted_ids_list)
    actual_len = sorted_ids.shape[0]

    if actual_len < max_num_tokens_padded:
        pad = torch.full((max_num_tokens_padded - actual_len,), padding_value, dtype=torch.int32)
        sorted_ids = torch.cat([sorted_ids, pad])
    else:
        sorted_ids = sorted_ids[:max_num_tokens_padded]

    expert_ids_tensor = torch.tensor(expert_ids_list, dtype=torch.int32)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    if expert_ids_tensor.shape[0] < max_num_m_blocks:
        pad = torch.full((max_num_m_blocks - expert_ids_tensor.shape[0],), -1, dtype=torch.int32)
        expert_ids_tensor = torch.cat([expert_ids_tensor, pad])

    num_tokens_post_pad = torch.tensor([actual_len], dtype=torch.int32)

    if expert_map is not None and not ignore_invalid_experts:
        expert_map_cpu = expert_map.cpu()
        valid = expert_ids_tensor >= 0
        expert_ids_tensor[valid] = expert_map_cpu[expert_ids_tensor[valid].long()]

    return sorted_ids.to(device), expert_ids_tensor.to(device), num_tokens_post_pad.to(device)


def invoke_fused_moe_torch(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    B_bias: torch.Tensor | None = None,
):
    """Ascend NPU MoE GEMM with capture-safe unquantized naive decode.

    npu_grouped_matmul is disabled — it crashes aicore on MoE models like
    Qwen3.6-35B-A3B (error 507015: aicore execution abnormal). The torch.mm
    loop is slightly slower but reliable.
    """
    if (
        sorted_token_ids is None and not use_fp8_w8a8 and not use_int8_w8a8
        and B.dtype in (torch.float32, torch.float16, torch.bfloat16)
    ):
        # One output row per token/expert pair. Routing must stay live on
        # device: CPU lists would freeze the experts selected during capture.
        _, n, k = B.shape
        output = C.view(-1, n)
        pairs = min(expert_ids.numel(), output.shape[0])
        ids = expert_ids[:pairs].long()
        valid = ids >= 0
        safe_ids = ids.clamp_min(0)
        # Keep the existing native mm path. The fixed pair count is unrolled
        # during capture; each replay gathers the currently routed weights.
        for pair in range(pairs):
            weight = torch.gather(
                B, 0, safe_ids[pair:pair+1].view(1, 1, 1).expand(1, n, k)
            ).squeeze(0)
            row = pair // max(top_k, 1)
            result = torch.mm(A[row:row+1], weight.t())
            if B_bias is not None:
                result = result + torch.gather(
                    B_bias, 0, safe_ids[pair:pair+1].view(1, 1).expand(1, n)
                )
            if mul_routed_weight and topk_weights is not None:
                result = result * topk_weights.view(-1)[pair:pair+1].to(result.dtype)
            output[pair:pair+1].copy_(
                torch.where(valid[pair:pair+1], result, 0).to(C.dtype)
            )
        return

    # Use per-expert torch.mm loop — more reliable on Ascend NPU than
    # npu_grouped_matmul which can crash aicore on certain model shapes.
    _invoke_fused_moe_loop(
        A, B, C, A_scale, B_scale, topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_routed_weight, top_k, config,
        use_fp8_w8a8=use_fp8_w8a8, use_int8_w8a8=use_int8_w8a8,
        B_bias=B_bias,
    )




def _invoke_fused_moe_loop(
    A, B, C, A_scale, B_scale, topk_weights,
    sorted_token_ids, expert_ids, num_tokens_post_padded,
    mul_routed_weight, top_k, config,
    use_fp8_w8a8=False, use_int8_w8a8=False, B_bias=None,
):
    """Fallback per-expert torch.mm loop."""
    N = B.shape[1]
    block_size = config["BLOCK_SIZE_M"]
    c_flat = C.view(-1, N)
    num_valid_tokens = c_flat.shape[0]

    if topk_weights is not None:
        topk_weights_flat = topk_weights.view(-1)
    else:
        topk_weights_flat = None

    device = A.device
    expert_indices = {}

    if sorted_token_ids is None:
        expert_ids_cpu = expert_ids.cpu().numpy()
        expert_batches = {}
        for pair_idx in range(len(expert_ids_cpu)):
            if pair_idx >= num_valid_tokens:
                break
            expert_id = int(expert_ids_cpu[pair_idx])
            if expert_id < 0:
                continue
            expert_batches.setdefault(expert_id, []).append(pair_idx)
        for expert_id, rows in expert_batches.items():
            valid_ids = torch.tensor(rows, dtype=torch.int64, device=device)
            a_idx = valid_ids // max(top_k, 1)
            expert_indices[expert_id] = (valid_ids, a_idx)
    else:
        sorted_ids_cpu = sorted_token_ids.cpu().numpy()
        expert_ids_cpu = expert_ids.cpu().numpy()
        total_padded = int(num_tokens_post_padded.cpu().item())

        expert_batches = {}
        num_blocks = len(expert_ids_cpu)
        for block_idx in range(num_blocks):
            expert_id = int(expert_ids_cpu[block_idx])
            if expert_id < 0:
                continue
            start = block_idx * block_size
            end = min(start + block_size, total_padded)
            if start >= end:
                break
            block_ids = sorted_ids_cpu[start:end]
            valid = block_ids[block_ids < num_valid_tokens]
            if len(valid) == 0:
                continue
            expert_batches.setdefault(expert_id, []).append(valid)

        for expert_id, id_arrays in expert_batches.items():
            all_valid = np.concatenate(id_arrays).astype(np.int64)
            valid_ids = torch.from_numpy(all_valid).to(device)
            a_idx = valid_ids // max(top_k, 1)
            expert_indices[expert_id] = (valid_ids, a_idx)

    for expert_id, (valid_ids, a_idx) in expert_indices.items():
        a_block = A[a_idx]
        out = torch.mm(a_block, B[expert_id].t())

        if B_bias is not None:
            out = out + B_bias[expert_id]

        if mul_routed_weight and topk_weights_flat is not None:
            w = topk_weights_flat[valid_ids].unsqueeze(-1)
            out = out * w.to(out.dtype)

        c_flat[valid_ids] = out.to(c_flat.dtype)

    del expert_indices
