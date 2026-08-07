# Safety evaluation

## Purpose

This package owns Apex's generic post-agent safety-verification contract. It
freezes the exact candidate and trusted verification plan, runs evaluator-owned
tool adapters, validates their reports and lineage, and produces a policy
decision before normal performance measurement. It does not implement an AKA
adapter, hidden task cases, scoring, or a ROCm sanitizer fleet. Concrete GPU
tools belong behind a Magpie verification backend.

## Public API

The supported API is listed in `apex.evaluation.safety.__all__` and consists of:

- task safety profiles and typed capability observations;
- immutable runtime identities, tool plans, verification plans, frozen-candidate
  fingerprints, and phase-isolation receipts;
- separate capability, execution, and finding states plus exact evidence
  receipts;
- evaluator-owned per-tool policy and the pure `decide_safety` truth table;
- `SafetyGate` and `SafetyGateRequest`.

Orchestrators depend on `apex.ports.SafetyVerificationPort`. Concrete verification
backends implement `SafetyToolRunner`; the direct local adapter is
`SubprocessSafetyToolRunner`.

## Invariants

Every tool keeps three state dimensions separate:

```text
capability = ready | adapter_required | unsupported | not_applicable |
             unavailable_runtime
execution  = not_run | completed | tool_error | timeout
finding    = not_evaluated | clean | found | inconclusive
```

`unsupported`, `not_applicable`, timeout, missing evidence, or a parser error can
never become `clean`. `not_applicable` is a coverage statement, not success.

The controller-owned transaction is:

```text
terminate agent process tree
  -> revoke credentials and agent-visible tool channels
  -> copy declared source to read-only controller storage
  -> compute canonical candidate digest
  -> freeze VerificationPlan and per-tool policy fingerprint
  -> create an empty evaluator-only report directory
  -> run direct argv with timeout and bounded stdout/stderr
  -> validate report, artifacts, cases, positive control, and lineage
```

The plan binds tool version, plugin digest, immutable image ID, helper and
correctness-dispatch digests, source/candidate/deployed digests, case set,
positive control, argv, environment, timeout, and output bound. Resume also
supplies current run, candidate, anchor, and deployed identity; stale plans are
rejected before tool execution.

Candidate/evidence paths are canonical workspace-relative POSIX paths. Absolute
paths, `..`, symlinks, hardlinks, writable frozen source, report-directory
overlap, unexpected reports, and artifact hash/size mismatches fail closed.
Candidate-authored reports cannot establish clean evidence.

Policy is evaluator-owned and separate from the profile and agent output:

```text
confirmed finding                     -> reject; do not measure
required applicable check not clean   -> reject; do not measure
advisory unsupported/inconclusive     -> continue; safety_certified=false
qualified applicable checks all clean -> safety_certified=true
only not_applicable / no enabled tool -> continue; safety_certified=false
```

`VerificationPolicy.no_tools()` is the explicit uncertified default, never an
implicit clean result. Safety is a promotion gate, not a hidden kernel-reward
term. Every safety artifact has `timing_eligible=false`, and
`SafetyGateResult.forbidden_timing_digests` identifies bytes that normal timing
must reject. Instrumentation overhead never enters p50, p99, speedup, or reward.

Tool reports use `apex.safety-tool-report/v1` at the exact
`APEX_SAFETY_REPORT_PATH`. They include completion, plan/runtime identity,
source/candidate/deployed lineage, positive-control status, the full planned case
set with dispatch digests, and checksummed evaluator-owned artifacts including
an `instrumented_artifact`. The gate derives the aggregate finding. A valid exact
finding remains a finding after nonzero tool exit; clean requires exit zero,
complete exact evidence, no timeout, and no truncated output.

## Dependencies

Domain contracts depend only on `apex.core` and the standard library. The port
definitions live in `apex.ports`; the local subprocess adapter reuses
`apex.execution.SubprocessSupervisor`. Image hardening, GPU locks/recovery,
sanitizer installation, evaluator-owned build/dispatch attestation, and gfx950
qualification belong to a concrete Magpie backend.

Files are split by responsibility: `profile.py` defines task/capability scope;
`plan.py` immutable plans and freeze receipts; `results.py` normalized evidence;
`policy.py` decisions; `gate.py` validation and orchestration; `runner.py` bounded
subprocess execution; `artifact_validation.py` isolated path/digest validation.

## Failure semantics

Preflight identity, isolation, path, or digest failure creates a fail-closed gate
result without invoking tools. Capability gaps remain `not_run`; runtime,
timeout, output, report, parser, positive-control, case-set, or lineage problems
become `inconclusive`, never clean. A confirmed exact-lineage finding rejects
regardless of advisory/required configuration. Required incomplete checks block
measurement; advisory incomplete checks may continue but cannot certify safety.
Until a task/tool/architecture pair has qualification evidence, configure it as
advisory; a clean run alone cannot make an unsupported claim.

## Tests

Run the offline truth-table, freeze, path-integrity, report-validation, timeout,
and integration coverage with:

```bash
pytest -q -p no:cacheprovider tests/unit/evaluation/test_safety.py \
  tests/integration/test_kernel_optimize_use_case.py
```

## Provenance

The trust-boundary design is a semantic port of useful ideas reviewed in
[AgentKernelArena PR #78](https://github.com/AMD-AGI/AgentKernelArena/pull/78),
research snapshot `844761ccb45b60661a8eb7933dd3ea888f093664`
(Apache-2.0). No AKA manager, worker, plugin, task adapter, or scoring code is
copied. That PR snapshot was open and stacked when reviewed, so it is a design
input rather than a production qualification claim.
