---
name: amd-hip-kernel-optimization
description: Review and optimize AMD HIP C++ kernels without weakening their evaluator or making unsupported performance claims. Use for .hip/.cu sources, PyTorch HIP extensions, hip2hip tasks, CDNA-specific kernels, or questions about HIP memory access, reductions, LDS, wave operations, MFMA, launch geometry, and numerical correctness.
---

# AMD HIP Kernel Optimization

Use an evidence-first workflow. Treat the caller's harness and evaluator as immutable authority; this skill supplies optimization method, not execution, grading, reward, or winner authority.

## 1. Freeze the contract

Before editing, identify:

- editable source files and required exported symbols;
- shapes, dtypes, layouts, strides, aliasing, mutation, return structure, forward/backward coverage, and tolerances;
- exact GPU architecture, HIP/ROCm toolchain, current-device and current-stream behavior;
- reference, wrapper, compile command, correctness command, performance command, and case set actually executed.

Do not infer semantics from an operator name or dead reference code. Preserve tie behavior, NaN handling, reduction order requirements, empty and non-multiple tile cases, and any unusual task semantics.

## 2. Classify the bottleneck

Classify the kernel as elementwise, reduction/normalization, GEMM/fusion, transpose/layout, gather/scatter, point-cloud search, or attention/decode. Inspect the code and available diagnostic evidence to choose one causal hypothesis. Do not treat a diagnostic trace as a scoring measurement.

Read [HIP2HIP patterns and provenance](references/hip2hip-patterns.md) when choosing an operator-specific hypothesis or adapting an AgentKernelArena hip2hip task.

## 3. Make one evidence-preserving change

Prefer the smallest change that tests the hypothesis:

- Elementwise: coalesced grid-stride traversal, aligned vector prefix/bulk/tail, multiple elements per thread, and size-specific launch geometry.
- Reduction: stable math, wave shuffles plus a small LDS cross-wave reduction, and dispatch by reduction width.
- GEMM/fusion: remove legal intermediates, tile through LDS, vectorize loads, then consider architecture-supported MFMA.
- Transpose/layout: specialize fixed permutations, use 2D tiles, coalesce both sides where possible, and pad LDS against bank conflicts.
- Gather/scatter or point-cloud: cache query data, avoid unnecessary square roots, preserve exact tie/padding rules, and reduce atomic contention without changing ownership semantics.
- Attention/decode: online softmax, register-resident state, staged or swizzled LDS, and persistent scheduling only under explicit shape and architecture constraints.

Never assume CDNA wave64 on wave32 hardware. Never use a target-specific MFMA instruction, one-workgroup-per-CU schedule, fixed head shape, or relaxed tolerance outside the contract that justifies it. A library-call substitution is acceptable only when the task permits it and evaluator evidence proves that the intended implementation engaged.

## 4. Preserve runtime semantics

Use the caller's current HIP device and current stream. Keep launch and runtime error checks. Test a non-default stream when the kernel is used through PyTorch. Cover alignment boundaries, scalar tails, empty inputs, non-contiguous layouts where allowed, and out-of-bounds protection. Do not modify inputs unless mutation is part of the contract.

For correctness, generate each case once and give equivalent independent copies to reference and candidate. Use an independent oracle, exact comparison for discrete results, and justified dtype- and reduction-aware tolerances for floating point. Validate every path that performance measures, including backward paths.

## 5. Return to trusted evaluation

Validate in this order:

1. compile the frozen candidate in a fresh build/cache;
2. prove the loaded binary and target symbol correspond to the candidate;
3. run every correctness case;
4. run the fixed normal-performance measurement under matched GPU conditions;
5. report the source patch and evidence receipts, then let the trusted evaluator grade it.

Record raw per-case samples and measurement parameters. Do not replace device-event timing with Python, subprocess, allocation, transfer, or validation wall time. Agent-local timing is diagnostic only.

## Authority boundaries

- Do not edit the reference, wrapper, harness, cases, tolerances, scorer, or evaluator to make a candidate pass.
- Do not copy or invoke AgentKernelArena runners as an Apex evaluator.
- Do not claim reward, winner, acceptance, or speedup from agent text or task-local `PASS` output.
- Do not automatically apply a winning source bundle.
- Apex V1 still rejects arbitrary standalone HIP execution with `hip_execution_unavailable`; this instruction-only skill does not add a generic HIP compiler, loader, oracle, timer, or reward path.
- Apex formal kernel reward requires evaluator-owned invocation samples and p50/p99 grading; an upstream warmup plus averaged timing loop is not equivalent evidence.
