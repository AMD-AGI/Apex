# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
gluon_vector_add_baseline.py — A minimal Gluon (`@gluon.jit`) baseline kernel
for elementwise vector addition. Use this as a starting point or as the
baseline for `optimize-kernel --kernel-type gluon`.

target: gfx942 (CDNA3) and gfx950 (CDNA4) — wavefront size 64

Usage:
    python files/gluon_vector_add_baseline.py            # smoke test
    python workload_optimizer.py optimize-kernel \\
        --kernel files/gluon_vector_add_baseline.py \\
        --kernel-type gluon \\
        --kernel-name gluon_vector_add \\
        --correctness-mode pytorch \\
        --reference files/gluon_vector_add_baseline.py \\
        -r /tmp/gluon_va_results
"""
from __future__ import annotations

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


def _warp_size() -> int:
    """64 on AMD CDNA, 32 on NVIDIA."""
    if torch.cuda.is_available() and getattr(torch.version, "hip", None):
        return 64
    return 32


@gluon.jit
def _vector_add_kernel(
    a_ptr, b_ptr, c_ptr, n,
    BLOCK: gl.constexpr, layout: gl.constexpr,
):
    pid = gl.program_id(0)
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n
    a = gl.load(a_ptr + offs, mask=mask, other=0.0)
    b = gl.load(b_ptr + offs, mask=mask, other=0.0)
    gl.store(c_ptr + offs, a + b, mask=mask)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solution entry point — runs the Gluon kernel and returns c = a + b."""
    assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
    n = a.numel()
    c = torch.empty_like(a)
    warp = _warp_size()
    num_warps = 4
    BLOCK = max(1024, warp * num_warps)
    R = max(1, BLOCK // (warp * num_warps))
    layout = gl.BlockedLayout(
        size_per_thread=[R],
        threads_per_warp=[warp],
        warps_per_cta=[num_warps],
        order=[0],
    )
    grid = (triton.cdiv(n, BLOCK),)
    _vector_add_kernel[grid](a, b, c, n, BLOCK=BLOCK, layout=layout, num_warps=num_warps)
    return c


# ── PyTorch reference (consumed by `--correctness-mode pytorch`) ─────────────


def baseline_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference implementation — what the Gluon kernel must match."""
    return a + b


def get_test_inputs() -> list[tuple]:
    """Test shapes for the pytorch correctness harness."""
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cases: list[tuple] = []
    for n in (1024, 65537, 1_000_003):
        a = torch.randn(n, dtype=torch.float32, device=dev)
        b = torch.randn(n, dtype=torch.float32, device=dev)
        cases.append((a, b))
    return cases


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA/HIP not available — Gluon kernels require a GPU.")
        raise SystemExit(0)
    for a, b in get_test_inputs():
        out = vector_add(a, b)
        ref = baseline_fn(a, b)
        ok = torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
        print(f"n={a.numel():>9}  match={ok}")
