# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
triton_rms_norm_baseline.py — A small @triton.jit RMSNorm baseline used as a
target for `optimize-kernel --kernel-type triton --rewrite-as gluon`.

The kernel itself is intentionally simple (one-row-per-program with masked
loads/stores) so the "rewrite-as gluon" agent has clear room to improve it
with explicit Gluon BlockedLayouts, vectorized loads, and proper warp tiling
on AMD CDNA3 / CDNA4.

target: gfx942 (CDNA3) and gfx950 (CDNA4) — wavefront size 64
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    x_ptr, w_ptr, y_ptr,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(y_ptr + row * n_cols + cols, y.to(x.dtype), mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Solution entry point. Shape: x = (M, N), weight = (N,)."""
    assert x.is_cuda and weight.is_cuda
    assert x.dim() == 2 and weight.dim() == 1 and x.shape[1] == weight.shape[0]
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    _rms_norm_kernel[(M,)](x, weight, y, N, eps, BLOCK=BLOCK, num_warps=4)
    return y


# ── PyTorch reference (consumed by --correctness-mode pytorch) ────────────────


def baseline_fn(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference: standard RMSNorm in fp32 then cast back."""
    orig_dtype = x.dtype
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    y = x32 * torch.rsqrt(var + eps)
    return (y * weight.to(torch.float32)).to(orig_dtype)


def get_test_inputs() -> list[tuple]:
    """Representative shapes for the pytorch correctness harness."""
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cases: list[tuple] = []
    for (M, N) in [(64, 1024), (128, 4096), (32, 8192)]:
        x = torch.randn(M, N, dtype=torch.float16, device=dev)
        w = torch.randn(N, dtype=torch.float16, device=dev)
        cases.append((x, w))
    return cases


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA/HIP not available — Triton kernels require a GPU.")
        raise SystemExit(0)
    for x, w in get_test_inputs():
        out = rms_norm(x, w)
        ref = baseline_fn(x, w)
        ok = torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
        print(f"shape={tuple(x.shape)}  match={ok}")
