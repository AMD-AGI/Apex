---
name: gluon-kernel-reflection-prompts
description: Reflection / self-critique prompts for reviewing and fixing Gluon (`@gluon.jit`) kernels after generation or test failures on AMD MI300X / MI355X.
---

# Gluon Kernel Reflection Playbook

Use this skill **after** an iteration of a Gluon kernel optimization that
either failed correctness, failed compilation, regressed performance, or
produced a suspicious speedup. Each section gives you a focused prompt and
the key things to verify before the next attempt.

## When to Use

- Magpie reports `compiled=False` for a Gluon solution.
- Magpie reports `correct=False` (output mismatch) for a Gluon solution.
- Speedup is < 1.0× the Triton baseline despite being "more explicit".
- ISA inspection shows scalar `global_load_dword` in the hot path.
- ISA inspection shows excessive `ds_read` / `ds_write` traffic.
- The kernel runs but the hacking detector flags suspicious timing.

---

## R1. Compilation failed

> The Gluon kernel failed to compile. Read the error message and answer
> *concretely*:
>
> 1. Is the failure inside `@gluon.jit` or in host Python?
> 2. Did `gl.arange` get called WITHOUT `layout=`? In Gluon, every 1-D
>    tile needs an explicit layout — either a `BlockedLayout` (for the
>    primary tile) or a `SliceLayout(dim=D, parent=L)` (for an axis of a
>    2-D tile).
> 3. Are all `BLOCK_*` and `num_warps` parameters declared
>    `: gl.constexpr` in the kernel signature?
> 4. Is `threads_per_warp` the wavefront size? On MI300X / MI355X this
>    is **64**, not 32. (`gl.BlockedLayout(threads_per_warp=[64, ...])`)
> 5. Does the layout's `size_per_thread × threads_per_warp × warps_per_cta`
>    product **exactly** equal the corresponding tile dim? Off-by-one
>    here is a common compile error.
> 6. Are you using a Gluon API name that doesn't exist? Common mistakes:
>    `gl.thread_id` (use `gl.program_id` / lane indices via layout),
>    `gl.gridDim` (doesn't exist), `gl.amd.cdna3.mfma_*` (the actual
>    name is `gl.amd.cdna3.mfma`).

Action: Fix the import / layout / decorator. Recompile in isolation
before re-running the full grader.

---

## R2. Correctness failed (output mismatch)

> The kernel compiled and ran but the reference comparison failed.
> Diagnose:
>
> 1. **Tail handling.** Did you mask both the load AND the store?
>    `gl.load(ptr, mask=mask, other=0.0)` for the load, and
>    `gl.store(ptr, val, mask=mask)` for the store. Missing either
>    causes out-of-bounds writes or NaN propagation.
> 2. **Reduction layout.** After `m = gl.max(x, axis=0)`, `m` lives in
>    `SliceLayout(dim=0, parent=<x's layout>)`. If you applied a manual
>    `gl.convert_layout` you may have shuffled it incorrectly. Try
>    removing the explicit conversion.
> 3. **Numerical stability.** For softmax / log-sum-exp, did you
>    subtract the max before `exp`? `e = gl.exp(x - m)` is mandatory
>    for fp16 / bf16 inputs.
> 4. **Stride mistakes.** When laying out a 2-D tile, did you set
>    `order=[1, 0]` (so axis 1 varies fastest, matching row-major
>    PyTorch tensors)?
> 5. **Accumulator dtype.** For matmul / reductions, accumulate in
>    `gl.float32` and cast back to the input dtype only at the store.
>    Accumulating in fp16 produces large rounding errors.
> 6. **Per-program offsets.** `pid * BLOCK + indices` — verify `pid`
>    is the right axis (`gl.program_id(0)` for 1-D, `(0, 1)` for 2-D),
>    and `BLOCK` is the *Python* block size, not a Gluon tensor.

Action: Re-derive the offset / mask / accumulator dtypes from first
principles, write a 16-element CPU-side trace, and compare element by
element before re-grading.

---

## R3. Performance regressed (Gluon slower than Triton baseline)

> The Gluon rewrite is slower than the original `@triton.jit` baseline.
> Inspect the ISA via `compiled.asm.get("amdgcn")` and answer:
>
> 1. Are there `global_load_dwordx4` (or `x2`) instructions in the
>    inner loop, or only scalar `global_load_dword`? If scalar, your
>    layout's `size_per_thread[innermost]` is 1 — raise it to 2 or 4.
> 2. Is the inner-contiguous tensor dim the same as the layout's
>    most-threaded axis? If `tensor[m, n]` is row-major (n contiguous),
>    your layout's `order` MUST be `[1, 0]` and the bulk of
>    `threads_per_warp` MUST be on axis 1 (`[1, 64]`).
> 3. Count `ds_read` / `ds_write` instructions. If they vastly exceed
>    your intentional LDS staging, you have implicit `convert_layout`
>    calls. Find them by removing `convert_layout` calls one at a time
>    and seeing where the layout chain breaks.
> 4. What is your `num_warps`? Too few (< 4) and you have low occupancy.
>    Too many (> 8) and register pressure spills. Sweep 4 → 8 → 16.
> 5. For matmul: are you using `gl.amd.cdna3.mfma` (CDNA3) or
>    `gl.amd.cdna4.mfma` (CDNA4) with the right `DotOperandLayout`? The
>    layout's element shape MUST match the MFMA instruction shape.

Action: Tweak `BlockedLayout(size_per_thread=...)` and `order=...`
first — those usually move you onto / off the bandwidth roofline by 4×.

---

## R4. Suspicious / inflated speedup

> The reported speedup is much larger than the roofline allows. Sanity
> checks:
>
> 1. Is the solution actually computing the same shapes as the baseline,
>    or did you accidentally shrink the problem (e.g. only iterating
>    over a quarter of K)?
> 2. Did you skip a `gl.barrier` between the load and use of an LDS
>    buffer? You may be reading stale data and "completing" faster.
> 3. Did you cache output in a global pointer that wasn't synced — so
>    the next call sees the previous call's result?
> 4. Are you returning early under a mask that's always true? Verify
>    the mask covers the full work.
> 5. Did the magpie compare actually validate against the reference,
>    or did `compiled` and `correct` both pass on a degenerate shape?

Action: Re-run with a different shape from the test sweep. If the
speedup vanishes, you had a measurement artefact, not a real win.

---

## R5. Layout selection regret

> You picked a layout but the ISA / perf disagrees with your intent.
> Re-derive from scratch:
>
> 1. **What is the natural memory order of the tensor?**
>    For `x: [M, N]` row-major (default PyTorch), N is contiguous.
> 2. **Set `order = [N_axis, M_axis] = [1, 0]`** (innermost first).
> 3. **Place threads along the contiguous axis:**
>    `threads_per_warp = [1, warp_size()]`.
> 4. **Pick a vector width R ∈ {1, 2, 4, 8}** for `size_per_thread`.
>    Rule of thumb: `R = min(8 // sizeof(dtype), 4)`. fp32 → R=4, bf16 → R=8.
> 5. **Decide `num_warps`** so that
>    `BLOCK_N == size_per_thread[1] × threads_per_warp[1] × warps_per_cta[1]`.
>    For `BLOCK_N=1024`, `R=4`, `warp=64` → `warps_per_cta[1] = 4`.
> 6. **For 2-D, distribute `warps_per_cta` between axes** based on
>    which dim has more work per CTA.

Action: Recompute the layout from scratch using the formula above.
Run the ISA inspector after recompilation to confirm `dwordx{R}`.

---

## Reflection emission format

When emitting the reflection back to the next iteration, structure it as:

```
## Previous attempt: <one-line summary>
## What went wrong: <R1..R5 short tag>
## Concrete fix to try: <single, testable change>
## Verification step: <e.g. "check ISA for global_load_dwordx4">
```

Keep it under 200 words. The next iteration's agent has the previous
solution in context — don't paste it back; just point at the diff.
