# Safety evaluation

## Purpose

This package defines Apex's caller-neutral safety evidence contract and pure
policy. The shipped product has no concrete sanitizer runtime, vendor adapter,
plugin, sidecar, container, or MCP/CLI sanitizer command. Its release defaults
are therefore:

```text
sanitizer_runtime=not_implemented
safety_certified=false
```

An independent trusted evaluator may provide a complete receipt bound to the
current candidate, cases, dispatches, runtime, and policy. Apex can validate
that receipt and apply the policy described here. That boundary is not an Apex
sanitizer implementation or a qualification claim. `VerificationPolicy.no_tools()`
is the explicit uncertified default; it never means clean.

## Public API

The supported names are the exact `apex.evaluation.safety.__all__` surface:

- profile and capability: `PROFILE_SCHEMA_VERSION`, `ArtifactKind`,
  `CapabilityCheck`, `CapabilityStatus`, `InstrumentationControl`,
  `KernelLanguage`, `TaskSafetyProfile`, and `ToolCapability`;
- frozen plans and lineage: `CANDIDATE_MANIFEST_SCHEMA_VERSION`,
  `ISOLATION_SCHEMA_VERSION`, `PLAN_SCHEMA_VERSION`, `FrozenCandidate`,
  `PhaseIsolationReceipt`, `ToolRuntimeIdentity`, `ToolVerificationPlan`,
  `VerificationPlan`, and `fingerprint_frozen_candidate`;
- normalized external evidence: `RESULT_SCHEMA_VERSION`,
  `TOOL_REPORT_SCHEMA_VERSION`, `EvidenceArtifactReceipt`, `ExecutionStatus`,
  `FindingStatus`, `LineageReceipt`, `SafetyGateResult`, and `ToolEvaluation`;
- policy: `POLICY_SCHEMA_VERSION`, `SafetyDecision`, `SafetyRequirement`,
  `ToolPolicy`, `VerificationPolicy`, and `decide_safety`;
- validation boundary: `SafetyGate` and `SafetyGateRequest`;
- low-level generic integration helper: `SubprocessSafetyToolRunner`.

`SubprocessSafetyToolRunner` is a generic library boundary for a caller that
already owns a trusted tool plan and evaluator authority. The production Apex
composition root does not bind it to a sanitizer binary. Its presence in the
API must not be reported as a built-in runtime, adapter, or qualification.

Orchestrators consume the separate `apex.ports.SafetyVerificationPort`; that
port validates externally supplied evidence and conveys policy results. It does
not imply that Apex has a runner.

## Invariants

### Current implementation status and non-goals

Apex does not install, launch, select, or parse a concrete sanitizer. It does
not build an instrumented candidate, capture a replay capsule, attest a vendor
plugin, or expose `kernel.sanitize`. Ordinary compile/correctness output, agent
commands, and process return codes cannot be converted into clean safety
evidence. The normal no-tool path remains usable but uncertified.

### Tool categories are not interchangeable

- A GPU AddressSanitizer-class tool checks device-memory errors only where the
  selected build and dispatch were instrumented.
- An FpSan-class tool compares floating-point expressions or candidate/reference
  semantics; it does not establish memory safety.
- A race detector or simulator observes synchronization and data races for
  particular dispatches; it does not establish whole-program race freedom.
- Ordinary correctness, held-out correctness, and sanitizer evidence are
  complementary signals. None substitutes for another.

Tool names in this document explain evidence categories only. They do not mean
that Apex integrates, supports, or has qualified any named product.

### Task profile dimensions

Applicability depends on `language`, artifact kind (Python/JIT, source AOT, or
precompiled HSACO), framework, source availability, instrumentation control,
GPU architecture, adapter, and case set. A tool understanding an ISA in theory
does not prove that the current candidate can be checked. `TaskSafetyProfile`
freezes these dimensions before an external result is accepted.

### Four-dimensional capability

Capability is not a single boolean:

- `engine` asks whether the underlying analysis engine can express the check;
- `adapter` asks whether the current framework/artifact can be translated and
  invoked with complete lineage;
- `runtime` asks whether the exact tool/plugin/image/runtime identity is present;
- `effective` asks whether all three apply to this candidate and case set.

Each dimension uses one of:

```text
ready | adapter_required | unsupported | not_applicable | unavailable_runtime
```

`ready` permits evaluation but does not assert a clean finding. `adapter_required`
means an integration is absent. `unsupported` means the engine cannot provide
the requested check. `not_applicable` is a scoped coverage statement.
`unavailable_runtime` means a theoretically supported path is not installed or
attested. No non-ready state may be collapsed into clean.

### Execution and finding are orthogonal

Execution and finding form a Cartesian product:

```text
execution = not_run | completed | tool_error | timeout
finding   = not_evaluated | clean | found | inconclusive
```

A confirmed finding may accompany a nonzero exit. Exit zero may still be
inconclusive when instrumentation, report lineage, cases, or dispatch coverage
is missing. `clean` is valid only for a completed, fully attested evaluation.

### Positive control is not candidate attestation

A seeded startup bug can show that a tool installation detected one known fault
on one lane. It cannot prove that the candidate was instrumented or replayed.
A clean result needs both runtime positive-control evidence and candidate-specific
build/dispatch engagement evidence.

### Instrumentation and dispatch engagement

A preload environment, an installed image, or a loadable HSACO is not coverage
proof. A complete receipt binds compiler flags/environment, source and artifact
digests, tool/plugin/runtime identity, expected kernel, case/dispatch digest,
and loaded artifact proof. The observed dispatch must derive from the frozen
candidate evaluated by the correctness phase.

### JIT, AOT, and replay-capsule boundary

An HSACO plus a kernel name is insufficient to replay a dispatch. A trustworthy
single-dispatch capsule also needs target architecture, producer/toolchain and
ABI, argument order, grid/block dimensions, allocation and input snapshots,
pointer relocation, scratch requirements, case identity, and code-object hashes.
The capsule must be bound to the frozen candidate's correctness dispatch. Apex
does not capture or launch such capsules in this release.

### Evaluator phase isolation

Before external safety evidence can influence a gate, the agent process tree
must be gone, credentials and agent-visible tool channels revoked, and candidate
source frozen read-only. The independent evaluator owns the GPU and artifact
directory, fixed argv, bounded logs, network policy, and private cases. An agent
skill, prompt, or self-run command can never be required-gate authority.

### Plan fingerprint and resume

The fingerprint binds policy, task profile, tool/plugin/runtime identity,
candidate/source/deployed digests, case set, positive-control policy, and
artifact/capsule digests. Resume rebuilds the current plan and rejects a stale
report. Finding an old result file is never sufficient.

### Policy truth table

| Evidence state | Measure performance? | `safety_certified` |
|---|---:|---:|
| Confirmed exact-lineage finding | no | false |
| Required applicable check is missing, non-clean, or incomplete | no | false |
| Advisory check is missing, unsupported, or inconclusive | yes | false |
| Every required/applicable external check is complete and clean | yes | true |
| All checks are not applicable, or no tool is configured | yes | false |

This policy is independent of correctness and reward shaping. A safety result
may reject measurement, but it is never a reward bonus.

### Performance isolation

Every instrumented or simulated artifact and timing is
`timing_eligible=false`. Normal scoring uses a fresh uninstrumented build,
cache, and process after safety evaluation. Sanitizer overhead must never enter
p50, p99, speedup, E2E QoS metrics, or reward.

### Coverage and held-out limitations

Clean means only that the attested cases and dispatches produced no finding
under the named tool and policy. It does not establish safety for all shapes,
strides, dtypes, schedules, or interleavings. Private correctness coverage and
private sanitizer coverage are reported separately.

### External receipt minimum fields

A usable external receipt contains schema and policy IDs, the full plan
fingerprint, task profile, all four capability dimensions, execution and finding
states, tool/plugin/runtime identity, positive-control outcome, candidate/source/
deployed/case/dispatch lineage, artifact digest and size, timeout/truncation
facts, and the evaluator decision. Missing required lineage makes the result
inconclusive rather than clean.

### Conceptual support matrix

| Candidate form | Why a distinct adapter/instrumentation path is needed | Apex implementation status |
|---|---|---|
| Triton JIT | Generated code and launch metadata exist only after specialization | not implemented / not qualified |
| HIP source | Compiler/link flags and the loaded code object must bind to source | not implemented / not qualified |
| FlyDSL JIT or AOT | Python staging and emitted artifacts require different lineage | not implemented / not qualified |
| Precompiled AITER/rocBLAS/RCCL | Source may be unavailable and library dispatch must be attributed | not implemented / not qualified |

The table is explanatory, not a product support matrix.

### Typed examples

These examples contain states only; they are not executable tool recipes.

```text
no-tools:
  capability=not_applicable execution=not_run finding=not_evaluated
  allow_measurement=true safety_certified=false

advisory-inconclusive:
  capability=ready execution=tool_error finding=inconclusive
  allow_measurement=true safety_certified=false

required-missing:
  capability=unavailable_runtime execution=not_run finding=not_evaluated
  allow_measurement=false safety_certified=false

exact-finding:
  capability=ready execution=completed finding=found
  allow_measurement=false safety_certified=false

external-clean:
  capability=ready execution=completed finding=clean
  exact_lineage=true policy_satisfied=true safety_certified=true
```

## Dependencies

The contracts depend on `apex.core`, standard-library immutable values, and the
port boundary. The generic validator can receive caller-owned plans and reports;
the stock bootstrap supplies `VerificationPolicy.no_tools()` and no concrete
tool. Vendor installation, image hardening, instrumentation, GPU scheduling,
private cases, and qualification remain responsibilities of an independent
external evaluator.

## Failure semantics

Malformed paths, stale fingerprints, digest drift, missing cases, incomplete
positive controls, tool errors, timeouts, truncated output, parser failures, and
lineage gaps remain explicit non-clean states. A confirmed exact-lineage finding
rejects performance. An incomplete required check fails closed. An incomplete
advisory check may continue only with `safety_certified=false`. No-tool and
not-applicable states are also uncertified.

## Tests

The release gate exercises only pure policy, plan/fingerprint validation, and
external-receipt parsing; it does not start Docker, a GPU, or a sanitizer binary:

```bash
pytest -q -p no:cacheprovider \
  tests/unit/evaluation/test_safety.py \
  tests/architecture/test_sanitizer_documentation.py
```

## Provenance

Safety decisions retain the external evaluator identity, policy/fingerprint,
candidate and dispatch lineage, case set, runtime/tool/plugin identities, and
content-addressed artifacts. This package contains Apex-native generic contract
code only. Third-party tool names above are category examples, not copied code,
integration claims, or qualification evidence.
