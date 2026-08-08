# Optimization

This package owns candidate lifecycles and kernel-only workload search. It does
not own agent processes, benchmark implementations, persistence, or delivery
I/O; those are reached through ports and explicit collaborators.

`kernel/` handles bounded standalone tasks. `e2e/` handles workload state,
dynamic kernel opportunity selection, live-anchor KEEP/REVERT, and replanning.

Tests: `pytest tests/unit/optimization tests/integration -q`.

## Purpose

Optimization owns the two application use cases while preserving one Apex state,
evidence, evaluation, and delivery architecture.

## Public API

Consumers select either `apex.optimization.kernel` for a standalone task or
`apex.optimization.e2e` for kernel-only workload search.

## Invariants

Agents propose frozen candidates; trusted evaluators decide. E2E search keeps a
live anchor, bounds attempts, and never changes model/workload semantics or config.

## Dependencies

As the application layer, optimization composes lower domain packages and ports.
No lower package may import it.

## Failure semantics

Agent, verification, measurement, deployment, and delivery failures become typed
events/results; missing formal proof never becomes a successful deliverable.
When search accepts no source patch, the terminal normal replay is retained only
as observed drift evidence. Its runtime variance cannot turn the unchanged source
identity into `verification_failed`: the result is `no_gain` (or the existing
unsupported result), `no_regression=true` is explicitly based on there being no
accepted or delivered source change, and formal-delivery claims remain false.

## Tests

Unit tests exercise local policies and workspaces; integration tests exercise
complete event chains with deterministic fake ports.

## Provenance

Each use case records context, agent, candidate, verification, measurement,
decision, cost, and delivery receipts for RL reconstruction.
