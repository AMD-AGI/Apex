# Apex test architecture

## Purpose

The test tree proves domain contracts at the cheapest reliable layer and keeps
GPU/infrastructure campaigns separate from deterministic correctness checks.

## Public API

`unit/` owns pure component behavior, `contract/` owns replay and port
compatibility, `integration/` owns vertical slices, and `architecture/` owns
module boundaries. Future `gpu/` cases must be explicitly marked and opt-in.

## Invariants

CPU suites never require a GPU, network, API credential, Docker daemon, or
mutable sibling checkout. Tests use caller-neutral fixtures and assert typed
artifacts or domain events instead of parsing presentation text.

## Dependencies

Run tests in the project virtual environment. Magpie and TraceLens behavior is
represented through ports in deterministic tests; live dependency checkouts are
reserved for explicit integration or GPU campaigns. Linux CPU tests require
`bwrap`, PID namespaces, private procfs, and `pidfd_open`/`pidfd_send_signal`;
these are product prerequisites, so containment tests fail rather than skip when
the kernel boundary is unavailable.

## Failure semantics

No test silently skips a required contract. Infrastructure-dependent tests must
state their marker and precondition; deterministic suites fail closed on missing
fixtures, malformed receipts, nondeterminism, or architecture drift.

## Tests

Use the following CPU gate; narrow failures by running the owning directory or
file, and do not weaken a shared assertion to accommodate stale code.

The descriptor-free standalone journey is covered by native-session mounting,
draft-only MCP grants, exact-digest `--campaign` handoff, path tampering, and the
existing formal optimizer's 300/299-sample KEEP/REVERT/bundle fixtures. The E2E
V2 scope is covered both with synthetic preview/run/resume cases and the frozen
27-row corpus partition (21 Docker one-shot, 6 `e2e_docker_only`). Live GPU/model
receipts remain outside this CPU tree.

```bash
pytest -q -p no:cacheprovider \
  tests/unit tests/contract tests/integration tests/architecture
```

## Provenance

Fixtures should record schema and policy identifiers and use exact digests where
identity matters. Real benchmark evidence belongs in results directories, not
checked-in unit fixtures, unless it is minimized and explicitly attributed.
