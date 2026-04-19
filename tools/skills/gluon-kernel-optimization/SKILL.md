---
name: gluon-kernel-optimization
description: This skill should be used when writing or tuning Gluon (`triton.experimental.gluon`) GPU kernels on AMD MI-series GPUs. Gluon is Triton's lower-level layout-aware DSL — use it when explicit control over `BlockedLayout`, `SliceLayout`, `DotOperandLayout`, MFMA paths, and shared-memory placement gives a measurable win over plain `@triton.jit`.
---

# Gluon Kernel Optimization (AMD MI300X / MI355X)

## Purpose

Write or rewrite GPU kernels in **Gluon** (`triton.experimental.gluon`), Triton's
lower-level dialect. Gluon exposes the layout system, MFMA / WMMA paths, and
shared-memory placement that `@triton.jit` hides. On AMD CDNA3 / CDNA4 this
matters most for:

- Memory-bound kernels where you want guaranteed `global_load_dwordx{2,4}`
- Reduction kernels (softmax, RMSNorm, online softmax) where the result layout
  must broadcast back without `convert_layout` spills
- GEMMs / attention where you want explicit control over the MFMA call
  (`gl.amd.cdna3.mfma`, `gl.amd.cdna4.mfma`) and operand layouts

If the baseline is already a well-tuned `@triton.jit` kernel that hits the
roofline, switching to Gluon will not magically gain speed — only commit a
Gluon rewrite when you can name the specific layout/coalescing/MFMA win.

## When to Use

- The user passes `--kernel-type gluon` or `--rewrite-as gluon` to
  `workload_optimizer.py optimize-kernel`.
- The bottleneck classifier categorizes the kernel as `gluon` (i.e. it
  matches `_GLUON_KERNEL_NAMES` in `pipeline/kernel_bottleneck.py`).
- A profile of a Triton baseline shows poor coalescing
  (`global_load_dword` instead of `global_load_dwordx4`), excessive
  `ds_read`/`ds_write` traffic from layout conversions, or sub-optimal MFMA
  scheduling — all of which Gluon lets you fix explicitly.

## Mental Model

Think of a Gluon program as **(work) + (layout)**:

| Concept | What you control | API |
|---|---|---|
| Work distribution across CTAs | program ID per axis | `gl.program_id(axis)` |
| Tile shape | Python `gl.constexpr` ints | `BLOCK_M`, `BLOCK_N`, ... |
| How tile elements map to threads/warps | `BlockedLayout` | `gl.BlockedLayout(...)` |
| Per-axis index for a 2-D tile | `SliceLayout` derived from a parent | `gl.SliceLayout(dim=D, parent=L)` |
| Index tensor with explicit layout | `gl.arange(0, N, layout=...)` |
| Memory load/store | `gl.load`, `gl.store` (mask, other supported) |
| MMA on AMD | `gl.amd.cdna3.mfma`, `gl.amd.cdna4.mfma` |
| LDS (shared mem) | `gl.shared_memory_descriptor`, async copy |

Layouts are the heart of Gluon. An incorrect layout will compile and run
correctly but burn 5×–10× of the bandwidth you could have used.

## Required Imports / Boilerplate

```python
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


def warp_size() -> int:
    """64 on AMD CDNA, 32 on NVIDIA. Use this for portable BlockedLayouts."""
    if torch.cuda.is_available() and getattr(torch.version, "hip", None):
        return 64
    return 32
```

The MI300X / MI355X **wavefront size is 64**. Hard-coding 32 will not error
but will leave half the lanes idle.

## Optimization Priority

### Phase 1: Foundation (correct, basic perf)

1. Use `gl.constexpr` for every compile-time int (block sizes, num_warps,
   `R = size_per_thread`).
2. Use `gl.arange(0, BLOCK, layout=layout)` — never bare `gl.arange` in
   Gluon (no implicit layout in 1-D tiles).
3. For 2-D tiles use `gl.SliceLayout(dim=D, parent=L)` for the per-axis
   indices, then broadcast back with `[:, None]` / `[None, :]`.
4. Apply masks with `gl.load(..., mask=..., other=0.0)` for tail handling.
5. Always pass `num_warps=N` at launch; Gluon does NOT autotune for you.

### Phase 2: Memory-bound layout tuning

6. **Coalesced loads:** make the inner-contiguous tensor dimension match the
   layout's most-threaded axis. For a `[M, N]` tile with `N` contiguous,
   the canonical layout is:
   ```python
   gl.BlockedLayout(
       size_per_thread=[1, R],          # R = vector width along N
       threads_per_warp=[1, warp_size()],
       warps_per_cta=[1, num_warps],
       order=[1, 0],                    # innermost dim varies fastest
   )
   ```
   With `R=4` and `warp_size=64`, each wavefront loads
   `64 lanes × 4 elems = 256` contiguous fp32 elements per
   instruction — that's `global_load_dwordx4`.
7. **Vectorize:** raise `R = size_per_thread[innermost]` from 1 → 2 → 4. Verify
   in ISA by looking for `global_load_dwordx{2,4}`.
8. **Mind LDS spills:** `gl.convert_layout(t, new_layout)` is a *real*
   operation that materializes through LDS if the layouts differ. Avoid by
   designing the layout once and using `SliceLayout`-derived broadcasts.

### Phase 3: Compute-bound MFMA path

9. For matmul-style kernels, accumulate in fp32 with explicit MFMA:
   ```python
   acc = gl.zeros((BM, BN), dtype=gl.float32, layout=mma_layout)
   for k in range(0, K, BK):
       a = gl.load(...)  # in DotOperandLayout for opIdx=0
       b = gl.load(...)  # in DotOperandLayout for opIdx=1
       acc = gl.amd.cdna3.mfma(a, b, acc)   # MI300X
       # gl.amd.cdna4.mfma(...)             # MI355X
   ```
10. Match `DotOperandLayout` to the MFMA instruction shape (16×16×16,
    32×32×8, 16×16×32, etc.) — see `gl.amd.cdna3` for the catalog.
11. Use `gl.async_copy_global_to_shared` / `gl.barrier_arrive` / `gl.barrier_wait`
    for software pipelining (overlap MFMA with the next K-tile's load).

### Phase 4: Reductions

12. After `m = gl.max(x, axis=0)` or `gl.sum`, `m`'s layout is
    `SliceLayout(dim=0, parent=<x's layout>)` — broadcast-compatible with `x`
    for free. Don't `convert_layout` it.
13. For numerically stable softmax: `e = gl.exp(x - m); z = gl.sum(e, axis=0); y = e / z`.
14. For online softmax (FlashAttention-style), keep the running `(m, l)` in
    fp32 and never materialize the full attention matrix.

### Phase 5: Architecture tuning

15. **MI300X (gfx942 / CDNA3):** wavefront 64, 8 XCDs, 256MB Infinity Cache;
    use `gl.amd.cdna3.mfma`. AccVGPR file is 256 regs × 64 lanes; large
    accumulator tiles are cheap.
16. **MI355X (gfx950 / CDNA4):** wavefront 64; use `gl.amd.cdna4.mfma`; some
    new low-precision (mxfp4/mxfp6) MFMA shapes available.
17. Set `num_warps` so each CTA holds 4–8 wavefronts (256–512 threads). On
    MI300X, this typically achieves 2–4 CTAs per CU — sufficient for
    bandwidth-bound kernels.

## Anti-patterns to avoid

- **Hardcoding `warp_size = 32`** in a `BlockedLayout` on AMD. Use the helper.
- **`gl.convert_layout` in the inner loop.** This is an LDS round-trip.
- **`gl.arange(0, N)` without a layout.** It will fail or pick a default
  layout you didn't want.
- **Mismatched `order` and stride.** If the inner-contiguous tensor dim is
  axis 1 but `order=[0, 1]`, you'll get strided `global_load_dword` instead
  of vectorized `global_load_dwordx4`.
- **Calling `@triton.jit` from `@gluon.jit`.** Use `@gluon.jit` for the
  callee too, or inline.
- **Letting `num_warps × threads_per_warp` exceed `BLOCK`.** Lanes have no
  work; the kernel still runs but wastes occupancy.

## Verification Workflow

After writing a Gluon solution, prove it's actually using the right
instructions by inspecting the AMD GCN ISA:

```python
compiled = my_kernel[grid](*args, BLOCK=1024, layout=layout, num_warps=4)
isa = compiled.asm.get("amdgcn", "")
print("loads:", [l for l in isa.splitlines() if "global_load" in l][:5])
print("mfma:",  [l for l in isa.splitlines() if "v_mfma"     in l][:5])
print("ds:",    [l for l in isa.splitlines() if l.lstrip().startswith(("ds_read","ds_write"))][:5])
```

Targets:

- For memory-bound: `global_load_dwordx4` (or x2). Any `global_load_dword`
  in the hot path means a layout mismatch.
- For matmul: `v_mfma_*` instructions and minimal `ds_*` traffic
  (LDS used only for staging, not layout fix-ups).
- High `ds_read` / `ds_write` count without matching MMA usage = wasted
  `convert_layout`.

## Reference Material

- Apex bundled tutorials: `tools/gluon_rag/` (curated Gluon kernels —
  rms_norm, memcpy, softmax, gemm).
- Local tutorials: `/home/sirafati/gluon-learning/` (guided walkthroughs
  with measured throughput on MI300X).
- Upstream tutorials:
  `code_combine/triton/python/tutorials/gluon/01-intro.py`,
  `code_combine/triton/python/tutorials/gluon/02-layouts.py`.
- Production examples in aiter: `tools/rocm/aiter/aiter/ops/triton/gluon/`
  (`gemm_a8w8.py`, `pa_decode_gluon.py`, `gemm_afp4wfp4.py`).

## Hand-off checklist

Before declaring a Gluon solution done:

- [ ] All `@gluon.jit` kernels use `gl.BlockedLayout` with `threads_per_warp[innermost] == 64` (on AMD).
- [ ] No `gl.convert_layout` in the inner loop.
- [ ] ISA shows `global_load_dwordx2` or `_dwordx4` for the hot loads.
- [ ] Reduction results use the natural `SliceLayout` (no manual `convert_layout`).
- [ ] Numerically stable formulation (softmax: subtract max; online: keep `(m, l)`).
- [ ] Speedup measured against the original baseline (not against itself).
- [ ] Solution still passes the same correctness harness as the Triton baseline.
