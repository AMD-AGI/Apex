# HIP2HIP patterns and provenance

## Reviewed source

This is a method-only synthesis from [AMD-AGI/AgentKernelArena](https://github.com/AMD-AGI/AgentKernelArena) `tasks/hip2hip` at commit `1292b4531fad8bed02c0ecc292704c44cb63c49a` (tree `4548155d92cb3c483eb61540feff36747db931f1`), licensed Apache-2.0. The reviewed tree contains 32 tracked task configurations. No kernel source, runner, scorer, validator, hidden test, or historical result is copied into this skill.

Representative evidence paths include:

- `gpumode/GELU`, `SoftmaxModule`, `layer_normalization`, `SimpleMatmulModule`, and `Transpose`;
- `others/ball_query`, `three_nn`, `furthest_point_sample`, `assign_score_withk`, and `matrix_multiplication`;
- fixed-shape `others/mla_decode` for architecture-specific decode attention.

The `campaign20` directory in the reviewed checkout had no tracked task source and is not evidence for this synthesis.

## Reusable operator patterns

| Family | First hypotheses | Contract risks |
|---|---|---|
| Elementwise | Grid-stride loops; aligned vector prefix/bulk/tail; elements per thread; launch dispatch by size | Exact transcendental semantics, alignment, scalar tails |
| Reduction / normalization | Stable formulation; wave shuffle; one partial per wave in LDS; tiny/small/large width dispatch | Wave width, accumulation precision, unbiased-vs-population statistics |
| GEMM / fusion | Eliminate legal intermediates; LDS tiles; vector loads; double buffering; supported MFMA | Layout, edge tiles, target architecture, library fallback |
| Transpose / layout | Fixed-permutation specialization; 2D tiles; coalesced access; LDS padding | Generic strides, metadata overhead, integer indexing |
| Point-cloud / top-k | Register query coordinates; squared distance; early exit; fixed-small-K register selection | Tie behavior, output padding, exact index semantics |
| Scatter / backward | Change ownership or locally reduce before atomics | Atomic ordering, collision behavior, missing backward validation |
| Decode attention | Online softmax; register state; staged/swizzled KV; MFMA; persistent grid | Fixed shape/head assumptions, gfx target, tolerance, occupancy |

## Correctness lessons

- Freeze the actual wrapper and executed reference path. A familiar operator name may hide unusual semantics; for example, a normalization task can require unbiased standard deviation rather than conventional LayerNorm variance.
- Generate one input set and deep-copy it for both sides. Fix RNG state and recursively compare tensor/list/tuple/dict results.
- For nearest-neighbor ties, different indices can be valid only when the oracle proves their distances are equivalent.
- Validate current-stream behavior. Several reviewed examples launch on stream 0 despite being PyTorch extensions; that is a bug pattern, not a template.
- Validate every measured path. A harness that checks forward but times forward-plus-backward cannot certify the candidate.
- Fail closed on missing cases and performance exceptions. Partial performance output must not participate in selection.

## Measurement lessons

The common gpumode loop uses 10 warmups and 100 iterations summarized by an average. It can help a developer form a hypothesis, but it does not provide Apex's required raw invocation samples, p50, p99, or formal reward.

Do not imitate a runner that wraps process startup, compilation, allocation, host/device copies, or validation in `perf_counter` and labels the result kernel execution time. Use matched device-event measurements on the correct stream, retain raw samples per case, and benchmark baseline and candidate under the same conditions.

## Authority boundary

AgentKernelArena remains authority for its own campaigns. Its task-local config and commands can describe an external intake contract, but Apex must not import its runner, central evaluator, case matcher, scorer, campaign selection, or postprocessing into this instruction skill. A task-local `PASS` is not Apex acceptance, and an agent-reported time is not evaluator evidence.

The reviewed AgentKernelArena Apex adapter explicitly keeps hip2hip outside formal integration until a trusted fixed build and verification recipe exists. Therefore this skill improves analysis and source proposals only; it does not change Apex's `hip_execution_unavailable` boundary for caller-authored HIP.
