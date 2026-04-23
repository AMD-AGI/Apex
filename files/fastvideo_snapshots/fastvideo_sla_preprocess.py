"""Standalone FastVideo SLA preprocessing Triton snapshot for Apex."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def compress_kernel(
    X,
    XM,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)
    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)
    x_offset = idx_bh * L * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    x = tl.load(X + x_offset + offs_l[:, None] * D + offs_d[None, :], mask=offs_l[:, None] < L, other=0.0)
    nx = tl.minimum(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))


def mean_pool(x: torch.Tensor, block_l: int) -> torch.Tensor:
    b, h, l, d = x.shape
    l_blocks = (l + block_l - 1) // block_l
    x_mean = torch.empty((b, h, l_blocks, d), device=x.device, dtype=x.dtype)
    compress_kernel[(l_blocks, b * h)](x, x_mean, l, d, block_l)
    return x_mean


def get_block_map(
    q: torch.Tensor,
    k: torch.Tensor,
    topk_ratio: float,
    block_q: int = 64,
    block_k: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    pooled_q = mean_pool(q, block_q)
    pooled_k = mean_pool(k - torch.mean(k, dim=-2, keepdim=True), block_k)
    pooled_score = pooled_q @ pooled_k.transpose(-1, -2)
    k_blocks = pooled_score.shape[-1]
    topk = min(k_blocks, int(topk_ratio * k_blocks))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices
    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)
    return sparse_map, lut, topk
