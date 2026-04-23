"""Standalone FastVideo TurboDiffusion LayerNorm Triton snapshot for Apex."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_param_fwd_fused(
    X,
    Y,
    W,
    B,
    Mean,
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
    mean = tl.sum(x, axis=1, keep_dims=True) / N
    var = tl.sum((x - mean) * (x - mean), axis=1, keep_dims=True) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + rows, tl.reshape(mean, (BLOCK_M,)))
    tl.store(Rstd + rows, tl.reshape(rstd, (BLOCK_M,)))
    w = tl.load(W + cols)
    b = tl.load(B + cols)
    y = (x - mean) * rstd * w + b
    tl.store(y_ptr, y.to(Y.type.element_ty), mask=mask[None, :])


@triton.jit
def _layer_norm_noparam_fwd_fused(
    X,
    Y,
    Mean,
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
    mean = tl.sum(x, axis=1, keep_dims=True) / N
    var = tl.sum((x - mean) * (x - mean), axis=1, keep_dims=True) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + rows, tl.reshape(mean, (BLOCK_M,)))
    tl.store(Rstd + rows, tl.reshape(rstd, (BLOCK_M,)))
    tl.store(y_ptr, ((x - mean) * rstd).to(Y.type.element_ty), mask=mask[None, :])


def layernorm(x: torch.Tensor, w: torch.Tensor | None, b: torch.Tensor | None, eps: float, elementwise_affine: bool = True):
    x2d = x.reshape(-1, x.shape[-1])
    rows, cols = x2d.shape
    out = torch.empty_like(x2d, dtype=torch.float32)
    mean = torch.empty((rows,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((rows,), dtype=torch.float32, device=x.device)
    n2 = triton.next_power_of_2(cols)
    block_m = 32 if cols <= 512 else 1
    if elementwise_affine:
        assert w is not None and b is not None
        _layer_norm_param_fwd_fused[(triton.cdiv(rows, block_m),)](
            x2d, out, w, b, mean, rstd, x2d.stride(0), out.stride(0), cols, n2, eps, BLOCK_M=block_m, num_warps=8
        )
    else:
        _layer_norm_noparam_fwd_fused[(triton.cdiv(rows, block_m),)](
            x2d, out, mean, rstd, x2d.stride(0), out.stride(0), cols, n2, eps, BLOCK_M=block_m, num_warps=8
        )
    return out.reshape_as(x), mean, rstd
