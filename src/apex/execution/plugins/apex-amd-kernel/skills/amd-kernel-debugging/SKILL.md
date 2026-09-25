---
name: amd-kernel-debugging
description: Diagnose failures in Python, Triton, CK, CKTile, AITER, ROCm, gfx950, or AMD GPU kernel workflows using Apex's typed evidence and capability boundaries. Use for intake, compile, correctness, launch, profiler, measurement, provenance, dependency, image, GPU-health, or resume failures; do not use to claim a speedup or replace formal grading.
---

# AMD Kernel Debugging

Diagnose the first failed trust boundary. Preserve exact error codes, receipts,
artifacts, source bytes, and runtime identity; avoid broad cleanup or speculative
performance conclusions.

## Classify the failure

1. **Intake:** verify language, editable scope, task identity, and trusted
   correctness/measurement contract. Unreviewed standalone HIP remains
   `hip_execution_unavailable`.
2. **Dependency or provenance:** verify exact commits, trees, locks, immutable
   image digests, loaded bytes, and build identity. Do not substitute a nearby
   checkout, mutable tag, or host `site-packages`.
3. **Compile or launch:** preserve fixed argv, bounded stdout/stderr, exit status,
   timeout, source engagement, target architecture, and runtime receipt.
4. **Correctness:** reproduce the smallest failing case while retaining dtype,
   shape, stride, aliasing, boundary, seed, and tolerance semantics. Never weaken
   the oracle to accept a candidate.
5. **Diagnostics:** for an existing Magpie workspace call `trace.analyze`; for its
   normalized artifact call `hotspot.rank` with the exact digest. Empty, partial,
   unresolved, or modeled trace fields remain diagnostic and reward-ineligible.
6. **Measurement:** check sample count, positivity, finiteness, timer/method
   identity, p50/p99 completeness, GPU health brackets, reference/candidate
   engagement, and matched conditions before interpreting a regression.
7. **Recovery:** use `campaign.status` for a scoped canonical run. Trust the
   verified event journal and CAS projection, not backend chat history or a stale
   snapshot.

## Keep operations scoped

Resolve visible devices and exact KFD process ownership before any live GPU work.
Do not kill by process name or pattern. A cooperative Apex lease does not
authorize terminating an unrelated process. Treat reset, health drift, stale
lease, PID reuse, truncated output, and incomplete cleanup as invalid evidence.

Use `workload.inspect` for a Magpie config rather than reimplementing its YAML
defaults. Use `bundle.verify` only for static integrity. Check capability
availability before calling a tool; report unavailable dependencies or authority
instead of falling back to arbitrary shell commands.

## Report the result

State the failed phase, stable reason code, evidence inspected, smallest supported
cause, and next safe check. Separate observations from hypotheses. Say explicitly
when compile, correctness, performance, reward, or safety is unproven.

Apex has no built-in sanitizer runtime, plugin, container, or MCP tool. External
safety evidence must retain its complete lineage and policy receipt; absent or
inconclusive evidence is not clean.
