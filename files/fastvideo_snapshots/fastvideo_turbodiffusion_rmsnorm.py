"""Standalone FastVideo TurboDiffusion RMSNorm Triton snapshot for Apex."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd_fused(
    X,
    Y,
    W,
    Rstd,
    x_stride,
    y_stride,
    N: tl.constexpr,
    N2: tl.constexpr,
    eps,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, N2)
    mask = cols < N
    x_ptr = X + rows[:, None] * x_stride + cols[None, :]
    y_ptr = Y + rows[:, None] * y_stride + cols[None, :]
    x = tl.load(x_ptr, mask=mask[None, :], other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=1) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + rows, rstd)
    y = x * tl.reshape(rstd, (BLOCK_M, 1))
    w = tl.load(W + cols)
    tl.store(y_ptr, (y * w).to(Y.type.element_ty), mask=mask[None, :])


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float):
    assert x.is_contiguous()
    x2d = x.reshape(-1, x.shape[-1])
    rows, cols = x2d.shape
    out = torch.empty_like(x2d)
    rstd = torch.empty((rows,), dtype=torch.float32, device=x.device)
    n2 = triton.next_power_of_2(cols)
    block_m = 32 if cols <= 512 else 1
    _rms_norm_fwd_fused[(triton.cdiv(rows, block_m),)](
        x2d,
        out,
        w,
        rstd,
        x2d.stride(0),
        out.stride(0),
        cols,
        n2,
        eps,
        BLOCK_M=block_m,
        num_warps=8,
    )
    return out.reshape_as(x), rstd
