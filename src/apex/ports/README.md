# Ports

`apex.ports` contains dependency-inversion protocols and immutable request/result
objects for agent execution, Magpie benchmarking, TraceLens diagnostics, knowledge
retrieval, standalone kernel measurement, and external safety evidence. A port imports
only `apex.core` and the Python
standard library; it never imports a concrete adapter.

`CodingSessionRequest` is deliberately separate from `AgentRequest`. The former
delegates a user-owned native coding session and grants no evaluator authority;
the latter is a bounded, contained candidate turn inside a formal campaign.
`AgentRequest` therefore carries an explicit
`AgentExecutionAuthorityReceipt`. It binds one exact run/attempt/backend to one
sealed editable projection, parent authority digest, source anchor, requested
environment key names, stdin prompt policy, and backend-scoped credential
policy. Concrete adapters reject a missing or mismatched receipt before any
subprocess; serialized invocation evidence contains the receipt and credential
key name but no credential value.
Capability descriptors declare one backend-neutral JSON schema plus side effects,
authority, GPU need, timeout, artifact class, reward role, and a nonnegative
pre-dispatch cost estimate. Presentation adapters cannot upgrade those
declarations. Every active capability call carries a one-shot external
`CapabilityGrantReceipt` bound to the exact session, descriptor, arguments,
authority, effects, GPU-device set, timeout, artifact classes, reward role, and
cost ceiling. The MCP process cannot mint that receipt for itself. Its declared
ceiling must cover the descriptor estimate, the handler receives the same grant
for in-flight enforcement, and the registry rejects a result above the ceiling.
`MagpieExecutionAttestor` is the pre-execution observer boundary for published
Magpie main. Its `prepare` call happens before the benchmark subprocess and its
opaque session is completed with only the discovered official report and
process outcome. The benchmark adapter has an explicit unavailable default, so
absence of a trusted observer prevents GPU execution instead of becoming a
post-run evidence failure.
The request also carries a read-only GPU-lease snapshot plus resolved execution
mode, lifecycle, image, and config digest. Concrete observers consume this
authority; they do not create or extend the lease policy.
`WorkspaceRepositoryIdentity` keeps its absolute root for local containment, but
its serialized evaluator-contract view exposes only `root_sha256`; public
evidence never needs to disclose a caller's host checkout path.
`CodingSessionRequest.results_dir` optionally selects a capability artifact
root; it does not grant formal evaluator authority or turn the native session
into a campaign.

The safety boundary has two protocols. `SafetyToolRunner` is a low-level generic
shape available to an independent evaluator that already owns a fixed plan;
Apex's production composition root does not bind it to a sanitizer.
`SafetyVerificationPort` is the external-receipt validation boundary exposed to
an orchestrator, not evidence that Apex has a runner. Candidate code never
supplies a shell command, authority, trusted policy, or accepted receipt through
these ports. Fixed inputs, evaluator authority, and fail-closed lineage semantics
are defined in [the primary safety contract](../evaluation/safety/README.md).

Public API is the `__all__` list in `apex.ports`. Port values are safe to record
as canonical events after conversion to JSON-compatible dictionaries.

Terminal diagnostics use separate `TraceComparisonRequest` and
`TraceComparisonResult` values. Each request binds baseline/final raw traces,
benchmark reports, and analysis reports to their CAS receipts and producer-relative
logical paths. `PARTIAL` means the documented report comparison ran while full
attribution remained unavailable; `UNAVAILABLE` never implies either ran. Output
artifacts are explicit and these results can never be reward eligible.

Agent execution returns raw structured `AgentTranscriptEvent` objects alongside
provider-neutral `AgentSemanticEvent`, `AgentUsage`, and `AgentCost` values.
`AgentResult` also retains the controller-requested model and effort; unsupported
effort is rejected instead of being silently ignored.
`AgentCaptureStatus.CREDENTIAL_REDACTED` is a fail-closed outcome: exact backend
credential values were removed before transcript construction, and the result
is ineligible for candidate capture.
`AgentInvocationReceipt` declares `private_pid_namespace_init_pidfd_v1`, while
`AgentProcessContainmentReceipt` separately records the live namespace-init,
pidfd, private-procfs, wrapper/status-FD, and zero-member teardown proof. A
candidate cannot cross source freeze without a complete runtime receipt, whether
the CLI ended naturally or at an exact-turn boundary.
For matched external campaigns, the request and invocation receipt also carry
the controller's sealed backend-runtime-closure SHA-256. The value binds lineage
across controllers; it never substitutes for executable-byte or containment
evidence owned by the concrete execution layer.
Usage distinguishes input, cached input, cache creation, output, reasoning, total
tokens, turns, and tool calls, with exact source-event indexes. Cost keeps an
explicit provider amount as a canonical decimal string plus currency and source
field. Missing structured evidence remains `None`; implementations never derive
tokens or money from assistant prose, stderr, or heuristic text matching.

Knowledge queries include an independent evidence-derived hypothesis, dtype and
software-version scope, and a token budget. Advisory retrieval is never an
unscoped prelude that can replace live measurement.

`KernelMeasurementPort` is the only standalone boundary that may return raw
samples with grading authority. The controller gives it frozen source, harness,
method, and policy digests, a fixed runner argv/cwd/environment/timeout, plus a
fresh report path outside the candidate workspace. The adapter must keep that
output channel out of candidate code; its
result is still rehashed and wrapped in an evaluator-authored execution receipt.
Its immutable `measurement_method_sha256` must equal the frozen task method
before execution. The evaluator parent, not a candidate subprocess, writes the
report. An ordinary verifier command or candidate-created JSON never implements
this port.

Tests: `pytest tests/contract tests/architecture -q`.

## Purpose

Ports define narrow protocol boundaries between deterministic domain logic and
agents, benchmarks, diagnostics, knowledge, or safety infrastructure.

## Public API

Only protocol/request/result names exported from `apex.ports` are supported;
adapters implement them without leaking vendor SDK objects into use cases.

## Invariants

Requests and results are typed, caller-neutral, bounded, and serializable enough
to receipt. Agent usage/cost values validate nonnegative counts, finite explicit
amounts, and structured-event lineage. Ports contain contracts only and perform
no I/O at import time.

## Dependencies

Ports depend solely on core. Domain/application packages may depend on ports;
ports never depend on concrete adapters or optimization.

## Failure semantics

Infrastructure inability is represented explicitly by result status or a typed
error. Protocols never provide permissive defaults that certify missing evidence.
`QualificationAuthorityPort` is the release trust boundary: it must recompute a
qualification claim from evaluator-owned artifact manifests. Its path-free
receipt is accepted only as the direct return value of the composition-root
injected verifier; a caller-supplied JSON receipt is never release authority.

## Tests

Contract tests run multiple adapter/fake implementations against the same request
semantics; architecture tests protect dependency direction and public exports.

## Provenance

Port results carry backend/model/effort or tool identity, structured transcript lineage, and
artifact locators where applicable; the receiving domain assigns evidence class
and policy meaning.
