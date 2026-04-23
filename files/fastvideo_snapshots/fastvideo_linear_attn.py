"""Standalone FastVideo sparse-linear attention Triton snapshot for Apex."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    qk_scale: tl.constexpr,
    topk: tl.constexpr,
    LUT,
    LSE,
    OS,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)
    qkv_offset = idx_bh * L * D
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    lse_offset = idx_bh * L
    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q_ptrs = Q + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    k_ptrs = K + qkv_offset + offs_n[None, :] * D + offs_d[:, None]
    v_ptrs = V + qkv_offset + offs_n[:, None] * D + offs_d[None, :]
    os_ptrs = OS + qkv_offset + offs_m[:, None] * D + offs_d[None, :]
    lut_ptr = LUT + lut_offset
    lse_ptrs = LSE + lse_offset + offs_m

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_s = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    q = tl.load(q_ptrs, mask=offs_m[:, None] < L)
    for block_idx in tl.static_range(topk):
        idx_n = tl.load(lut_ptr + block_idx)
        n_mask = offs_n < L - idx_n * BLOCK_N
        k = tl.load(k_ptrs + idx_n * BLOCK_N * D, mask=n_mask[None, :])
        qk = tl.dot(q, k) * (qk_scale * 1.4426950408889634)
        if L - idx_n * BLOCK_N < BLOCK_N:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))
        v = tl.load(v_ptrs + idx_n * BLOCK_N * D, mask=n_mask[:, None])
        local_m = tl.max(qk, 1)
        new_m = tl.maximum(m_i, local_m)
        p = tl.math.exp2(qk - new_m[:, None])
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - new_m)
        o_s = o_s * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + l_ij
        m_i = new_m

    tl.store(os_ptrs, (o_s / l_i[:, None]).to(OS.type.element_ty), mask=offs_m[:, None] < L)
    tl.store(lse_ptrs, m_i + tl.math.log2(l_i), mask=offs_m < L)


def sparse_linear_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lut: torch.Tensor,
    topk: int,
    block_m: int = 128,
    block_n: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, length, dim = q.shape
    m_blocks = triton.cdiv(length, block_m)
    out = torch.empty_like(v)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    _attn_fwd[(m_blocks, batch * heads)](
        q,
        k,
        v,
        dim**-0.5,
        topk,
        lut,
        lse,
        out,
        length,
        m_blocks,
        dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4 if dim == 64 else 8,
        num_stages=3,
    )
    return out, lse
