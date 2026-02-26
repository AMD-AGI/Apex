"""
baseline_fused_moe.py — Standalone fused MoE Triton kernel (extracted from vLLM).

This is the baseline kernel for the optimization task. It performs:
  1. Top-k routing: each token selects top_k experts
  2. Token sorting: tokens are grouped by assigned expert
  3. Expert GEMM: batched matrix multiply per expert (token × expert_weight)
  4. Weighted reduction: multiply by routing weight and sum across experts

The kernel uses the "sorted token" approach: tokens are pre-sorted by expert
assignment, and expert_ids marks which expert handles each BLOCK_SIZE_M chunk.

Shapes (Mixtral-8x7B-like):
  - hidden_states: [num_tokens, hidden_dim]    e.g. [512, 4096]
  - w1 (gate+up):  [num_experts, 2*ffn_dim, hidden_dim]  e.g. [8, 28672, 4096]
  - w2 (down):     [num_experts, hidden_dim, ffn_dim]     e.g. [8, 4096, 14336]
  - top_k = 2

Target: AMD Instinct MI355X (gfx950, CDNA4)
"""

import torch
import triton
import triton.language as tl


# ── Triton kernel: fused expert GEMM ─────────────────────────────────────────

@triton.jit
def fused_moe_kernel(
    # Pointers
    a_ptr,          # input hidden states [num_tokens, K]
    b_ptr,          # expert weights [num_experts, N, K]
    c_ptr,          # output [num_tokens * top_k, N]
    topk_weights_ptr,   # routing weights [num_tokens * top_k]
    sorted_token_ids_ptr,  # sorted token indices [EM]
    expert_ids_ptr,     # expert id per block [EM // BLOCK_SIZE_M]
    num_tokens_post_padded_ptr,
    # Dimensions
    N,              # output dim
    K,              # input dim (hidden_dim)
    EM,             # total sorted entries (padded)
    num_valid_tokens,  # actual num_tokens * top_k
    # Strides
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Fused MoE GEMM kernel — one program computes a BLOCK_SIZE_M × BLOCK_SIZE_N tile."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ── Token sorting (runs on CPU/GPU, prepares inputs for the Triton kernel) ───

def moe_align_block_size(topk_ids, block_size, num_experts):
    """Sort tokens by expert and pad to block_size boundaries.

    Returns:
        sorted_token_ids: [EM] — token indices sorted by expert, padded
        expert_ids: [EM // block_size] — expert index per block
        num_tokens_post_padded: scalar tensor
    """
    num_tokens = topk_ids.numel()
    # Count tokens per expert
    tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    flat_ids = topk_ids.view(-1)
    for e in range(num_experts):
        tokens_per_expert[e] = (flat_ids == e).sum()

    # Pad each expert's count to block_size
    padded_per_expert = ((tokens_per_expert + block_size - 1) // block_size) * block_size
    total_padded = padded_per_expert.sum().item()

    sorted_token_ids = torch.full((total_padded,), num_tokens, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty(total_padded // block_size, dtype=torch.int32, device=topk_ids.device)

    # Fill sorted_token_ids and expert_ids
    offset = 0
    for e in range(num_experts):
        count = tokens_per_expert[e].item()
        padded = padded_per_expert[e].item()
        # Find token indices assigned to this expert
        mask = flat_ids == e
        indices = torch.nonzero(mask, as_tuple=True)[0].to(torch.int32)
        sorted_token_ids[offset:offset + count] = indices
        # Fill expert_ids for each block
        for b in range(padded // block_size):
            expert_ids[offset // block_size + b] = e
        offset += padded

    num_tokens_post_padded = torch.tensor([total_padded], dtype=torch.int32, device=topk_ids.device)
    return sorted_token_ids, expert_ids, num_tokens_post_padded


# ── High-level wrapper ───────────────────────────────────────────────────────

def fused_moe_forward(
    hidden_states: torch.Tensor,  # [num_tokens, hidden_dim]
    expert_weights: torch.Tensor, # [num_experts, out_dim, hidden_dim]
    topk_ids: torch.Tensor,       # [num_tokens, top_k]
    topk_weights: torch.Tensor,   # [num_tokens, top_k]
    mul_routed_weight: bool = True,
) -> torch.Tensor:
    """Run the fused MoE kernel: route tokens to experts, GEMM, reduce."""
    num_tokens, top_k = topk_ids.shape
    num_experts, N, K = expert_weights.shape
    assert hidden_states.shape == (num_tokens, K)

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    # Sort tokens by expert
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, BLOCK_SIZE_M, num_experts
    )

    num_valid_tokens = num_tokens * top_k
    EM = sorted_token_ids.shape[0]

    # Flatten topk_weights for indexed access
    topk_weights_flat = topk_weights.view(-1).contiguous()

    # Output buffer
    output = torch.zeros(num_valid_tokens, N, dtype=hidden_states.dtype, device=hidden_states.device)

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    compute_type = tl.float16 if hidden_states.dtype == torch.float16 else tl.bfloat16

    fused_moe_kernel[grid](
        hidden_states, expert_weights, output,
        topk_weights_flat,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_valid_tokens,
        hidden_states.stride(0), hidden_states.stride(1),
        # expert_weights is [E, N, K]; kernel indexes B as [E, K, N] (transposed)
        expert_weights.stride(0), expert_weights.stride(2), expert_weights.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
    )

    # Reduce across top_k: sum contributions from each expert
    output = output.view(num_tokens, top_k, N).sum(dim=1)
    return output


# ── PyTorch reference implementation ─────────────────────────────────────────

def moe_reference(
    hidden_states: torch.Tensor,
    expert_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch MoE reference (slow but correct)."""
    num_tokens, top_k = topk_ids.shape
    num_experts, N, K = expert_weights.shape
    output = torch.zeros(num_tokens, N, dtype=hidden_states.dtype, device=hidden_states.device)

    for t in range(num_tokens):
        for k in range(top_k):
            expert_id = topk_ids[t, k].item()
            weight = topk_weights[t, k].item()
            # GEMM: hidden_states[t] @ expert_weights[expert_id].T
            expert_out = hidden_states[t].float() @ expert_weights[expert_id].T.float()
            output[t] += (weight * expert_out).to(output.dtype)
    return output


if __name__ == "__main__":
    # Quick self-test
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Running on CPU (no GPU detected). Triton kernel won't work.")
        exit(0)

    num_tokens, hidden_dim, ffn_dim = 32, 512, 1024
    num_experts, top_k = 8, 2
    dtype = torch.float16

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
    expert_weights = torch.randn(num_experts, ffn_dim, hidden_dim, dtype=dtype, device=device) * 0.01
    # Simulate routing
    scores = torch.randn(num_tokens, num_experts, device=device)
    topk_weights, topk_ids = torch.topk(torch.softmax(scores, dim=-1), top_k, dim=-1)
    topk_ids = topk_ids.to(torch.int32)

    # Run reference
    ref_out = moe_reference(hidden_states, expert_weights, topk_ids, topk_weights)
    # Run Triton kernel
    tri_out = fused_moe_forward(hidden_states, expert_weights, topk_ids, topk_weights)

    diff = (ref_out.float() - tri_out.float()).abs().max().item()
    print(f"Max diff: {diff:.6f}")
    assert diff < 0.05, f"Correctness check failed: max diff {diff}"
    print("PASS: baseline kernel matches reference")
