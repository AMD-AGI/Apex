# Core

`apex.core` owns standard-library-only primitives used at module boundaries:
typed errors, stable task status and validation enums, identifiers, and canonical
SHA-256 helpers. It performs no I/O at import time and depends on no other Apex
module.

Public API is the `__all__` list in `apex.core`. Domain objects belong to their
owning module rather than this package.

Tests: `pytest tests/unit/core tests/architecture -q`.

## Purpose

Core supplies stable errors, hashing, identifiers, and small enums shared across
all Apex domains without taking a dependency on any higher layer.

## Public API

Only names in `apex.core.__all__` are supported. Domain-specific records belong
to their owner and must not accumulate here as a generic utility bucket.

## Invariants

Canonical JSON is byte stable, digests are lowercase SHA-256, identifiers use a
bounded vocabulary, and every public error carries a stable reason code.

## Dependencies

Core is standard-library-only and sits at layer zero. It must never import an
Apex sibling package or an optional runtime dependency.

## Failure semantics

Invalid contracts fail at construction; integrity failures distinguish corrupted
or mismatched evidence from ordinary configuration errors.

## Tests

Core unit tests cover canonicalization, digests, identifiers, enums, and error
payloads; architecture tests enforce its dependency floor.

## Provenance

Core values originate locally and carry no copied performance knowledge. Hashes
computed here identify evidence but do not by themselves certify its trust class.
