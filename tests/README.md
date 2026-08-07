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
reserved for explicit integration or GPU campaigns.

## Failure semantics

No test silently skips a required contract. Infrastructure-dependent tests must
state their marker and precondition; deterministic suites fail closed on missing
fixtures, malformed receipts, nondeterminism, or architecture drift.

## Tests

Use the following CPU gate; narrow failures by running the owning directory or
file, and do not weaken a shared assertion to accommodate stale code.

```bash
pytest -q -p no:cacheprovider \
  tests/unit tests/contract tests/integration tests/architecture
```

## Provenance

Fixtures should record schema and policy identifiers and use exact digests where
identity matters. Real benchmark evidence belongs in results directories, not
checked-in unit fixtures, unless it is minimized and explicitly attributed.
