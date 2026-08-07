# Ports

`apex.ports` contains dependency-inversion protocols and immutable request/result
objects for agent execution, Magpie benchmarking, TraceLens diagnostics, knowledge
retrieval, and sanitizer evidence. A port imports only `apex.core` and the Python
standard library; it never imports a concrete adapter.

The safety boundary has two protocols. `SafetyToolRunner` receives only an
evaluator-owned argv vector, frozen paths, explicit environment, timeout, output
bound, and exact report location. `SafetyVerificationPort` exposes the complete
post-agent gate to an orchestrator. Candidate code never supplies a shell command,
trusted policy, or accepted receipt through these ports.

Public API is the `__all__` list in `apex.ports`. Port values are safe to record
as canonical events after conversion to JSON-compatible dictionaries.

Agent execution returns raw structured `AgentTranscriptEvent` objects alongside
provider-neutral `AgentSemanticEvent`, `AgentUsage`, and `AgentCost` values.
`AgentResult` also retains the controller-requested model and effort; unsupported
effort is rejected instead of being silently ignored.
Usage distinguishes input, cached input, cache creation, output, reasoning, total
tokens, turns, and tool calls, with exact source-event indexes. Cost keeps an
explicit provider amount as a canonical decimal string plus currency and source
field. Missing structured evidence remains `None`; implementations never derive
tokens or money from assistant prose, stderr, or heuristic text matching.

Knowledge queries include an independent evidence-derived hypothesis, dtype and
software-version scope, and a token budget. Advisory retrieval is never an
unscoped prelude that can replace live measurement.

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

## Tests

Contract tests run multiple adapter/fake implementations against the same request
semantics; architecture tests protect dependency direction and public exports.

## Provenance

Port results carry backend/model/effort or tool identity, structured transcript lineage, and
artifact locators where applicable; the receiving domain assigns evidence class
and policy meaning.
