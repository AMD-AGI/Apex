# Architecture tests

## Purpose

These tests make the clean-cut modular architecture executable rather than
leaving it as a convention that can silently decay.

## Public API

The suite exposes no runtime API. `test_source_architecture.py` is the single
policy entry point and reports exact files, imports, functions, or README
sections that violate the contract.

## Invariants

Every Python package is documented, imports point down an acyclic layer graph,
package exports are explicit, cross-package consumers avoid private names, and
source files/functions remain below the reviewability caps.

## Dependencies

The checks use only Python's standard library plus pytest. They inspect
`src/apex` without depending on a GPU, Magpie, TraceLens, or an agent backend.

## Failure semantics

Violations fail with a complete sorted list. There are no grandfathered files
or hidden allowlists: a refactor must repair the boundary before merging.

## Tests

Run `pytest -q -p no:cacheprovider tests/architecture`. The import-side-effect
check runs in an isolated subprocess with bytecode generation disabled.

## Provenance

The limits encode Apex's clean-cut refactor plan: 600 physical lines per module,
80 physical lines per function, explicit package APIs, and replay-safe imports.
