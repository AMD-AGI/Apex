---
name: amd-kernel-optimization
description: Optimize or assess existing Python, Triton, CK, CKTile, AITER, ROCm, or AMD GPU kernels with Apex's evidence-first workflow. Use when a user asks to profile, tune, speed up, benchmark, prove performance, or produce a verified kernel bundle; do not use for ordinary non-kernel coding or unsupported standalone HIP execution.
---

# AMD Kernel Optimization

Keep agent proposals separate from evaluator evidence. Use Apex capabilities for
controlled reads and actions; never turn prose, a diagnostic trace, or a claimed
speedup into a grade.

## Choose the workflow

1. For explanation or a correctness-only edit, inspect the source and tests as a
   normal coding task. Do not start a GPU or formal campaign unless the user asks
   for optimization or a performance claim.
2. For an existing Magpie workload, call `workload.inspect` before proposing a
   benchmark. Preserve its config, dependency, corpus, image, and workload
   identities. Treat capability gaps as blockers, not defaults to invent.
3. For an existing diagnostic workspace, call `trace.analyze`, then use
   `hotspot.rank` with the returned artifact digest. Treat all trace output as
   reward-ineligible diagnostic evidence. Use `trace.compare` only when two
   diagnostics already have typed, CAS-bound artifact receipts.
4. For a standalone formal task, use the public Apex optimization campaign only
   when its trusted task/evaluation contract can be resolved. Standalone HIP is
   unavailable unless it is one of the exact reviewed template-bound campaigns.

## Hand off chat to formal optimization

When the user asks to optimize and prove a standalone Python or Triton kernel,
inspect the editable source plus an independent compile, correctness, and raw
measurement harness, then call `campaign.start` with those typed fields. This
creates an unverified draft only. Return its run locator and exact Evaluation
Contract draft digest, tell the user to exit the chat, and use the printed
`apex optimize kernel --campaign ...` command. Never call evaluator phases from
the still-running chat or treat your own digest echo as user confirmation.

## Form and test a hypothesis

1. Inspect shapes, dtypes, strides, launch geometry, source boundaries, runtime
   identity, and current tests before consulting static advice.
2. Write one independent bottleneck hypothesis.
3. Call `knowledge.search` only after forming that hypothesis. Keep returned cards
   attributed and advisory; current workload evidence wins.
4. Prefer changes with a clear causal link to the measured hotspot. Keep the
   frozen reference, harness, policy, image, and workload contract unchanged.

## Preserve the proof boundary

Require evaluator-owned compile and correctness evidence before normal timing.
Require at least 300 positive finite invocation samples per implementation and
case, with both p50 and p99 recomputed by the evaluator. The robust speedup is
the smaller of reference/candidate p50 and p99 ratios. Missing or insufficient
p99 means no reward.

Accept KEEP or REVERT only from the formal controller using current-anchor
evidence. Do not add attempt rewards to obtain the terminal task reward. A
diagnostic profile, agent message, configuration-only change, or sanitizer-like
claim is never scoring evidence.

Apex provides no sanitizer runtime or `kernel.sanitize` tool. If an independent
evaluator supplies external safety evidence, preserve its exact receipt and let
the configured policy decide whether evaluation may continue. Otherwise report
`safety_certified=false`; never say clean.

## Deliver

Return an immutable, unapplied source bundle only after trusted evaluation. Use
`bundle.verify` to verify an existing bundle without applying, rebuilding,
measuring, or grading it. Report unresolved dependency, source, runtime, image,
measurement, or authority gaps explicitly and avoid a winner claim.
