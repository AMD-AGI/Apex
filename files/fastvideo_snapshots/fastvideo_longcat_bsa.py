"""Standalone FastVideo LongCat block-sparse attention Triton snapshot for Apex."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def create_mask_from_indices_kernel(
    block_indices,
    block_mask,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bs,
    stride_mz,
    stride_mh,
    stride_mm,
    stride_mn,
    H,
):
    i_zh, i_m, i_s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_z, i_h = i_zh // H, i_zh % H
    off_b = i_z.to(tl.int64) * stride_bz + i_h.to(tl.int64) * stride_bh + i_m.to(tl.int64) * stride_bm + i_s.to(tl.int64) * stride_bs
    b_i = tl.load(block_indices + off_b)
    off_m = i_z.to(tl.int64) * stride_mz + i_h.to(tl.int64) * stride_mh + i_m.to(tl.int64) * stride_mm + b_i.to(tl.int64) * stride_mn
    tl.store(block_mask + off_m, 1)


def create_mask_from_indices_triton(block_indices: torch.Tensor, num_cols: int) -> torch.Tensor:
    b, h, n_rows, s = block_indices.shape
    block_mask = torch.zeros((b, h, n_rows, num_cols), dtype=torch.bool, device=block_indices.device)
    create_mask_from_indices_kernel[(b * h, n_rows, s)](
        block_indices,
        block_mask,
        block_indices.stride(0),
        block_indices.stride(1),
        block_indices.stride(2),
        block_indices.stride(3),
        block_mask.stride(0),
        block_mask.stride(1),
        block_mask.stride(2),
        block_mask.stride(3),
        h,
    )
    return block_mask
