# Orchestration

This package is Apex's replayable workload state machine.

- `WorkloadState` and `ActionState` are immutable values.
- `reduce_event` is a pure transition function. It rejects stale anchors, branched
  journal parents, invalid action ordering, and transitions after a terminal state.
- `RunController` is the sole state writer. It validates a proposed transition,
  appends it to the journal, reduces the committed record, and only then refreshes
  the disposable snapshot.

An action progresses through queued, started, artifacts-ready, verified, and
committed states. A crash can leave any nonterminal action pending; recovery verifies
the journal, replays events after the last valid snapshot, and allows the caller to
continue or explicitly abort that action. Committing against an older anchor or
generation is rejected before it enters the journal.

The module depends on storage protocols rather than SQLite or filesystem adapters,
so orchestration policy remains independently testable.

For E2E runs, `WorkloadState.e2e` carries a compositional `SearchStage`, frozen
contract hashes, current opportunity queue, bounded budget, active candidate,
verification receipts, and decision ledger. Every worker result is checked against
the current anchor and state generation. KEEP alone advances the live anchor;
REVERT and rejected attempts remain in the ledger. No backend conversation is part
of recovery state. The E2E recovery projection joins that ledger to journal-bound
CAS receipts to rebuild the accepted source chain, live measurement anchor, active
derived configs, and any completed gate; the snapshot never supplies those values.

Each selected opportunity also creates one explicit, globally unique `attempt_id`.
That ID is carried through candidate, qualification, measurement, decision, and
reward events; it is never inferred from an action or substituted with a candidate
ID. Failed pre-measurement gates enter `DECIDING` without manufacturing a decision.
The evaluator then commits exactly one `e2e.candidate_decided` plus one
`reward_committed` in a single journal transaction. A transaction fault therefore
leaves both absent, never a half-committed training outcome. Source-free agent
outcomes retain a null candidate ID and an explicit REJECT.
`atomic.py` owns the short causal append/reduce transaction primitive; the
controller remains the only production caller and snapshot publisher.

## Purpose

Orchestration defines the replayable workload/E2E state machines and a controller
that rebuilds state from the append-only journal.

## Public API

Use the state/action/search enums and records, `reduce_event`, and `RunController`
interfaces exported by `apex.orchestration`.

## Invariants

Reducers are pure, generations are monotonic, stale worker evidence is rejected,
and snapshots are disposable accelerators rather than authoritative state.
State schema v2 is a clean cut: replay requires explicit attempt lineage and does
not reconstruct superseded source-free placeholders or legacy E2E decisions.
`performance_command_result` records untrusted command output; only evaluator-owned
`measurement_result` may carry reward or promotion authority. An accepted E2E patch
cannot enter `run.succeeded` until `final_clean_replay_verified` is true.
`experience.deferred` is a derived pending observation, not a measured outcome;
the reducer preserves it without granting reward or promotion authority.
`agent_message`, `tool_called`, `tool_result`, `usage_recorded`, and
`cost_recorded` are allowlisted evidence events but never mutate controller state.
Their attempt lineage remains replayable while usage/cost stays self-reported and
cannot masquerade as measured reward evidence.

## Dependencies

The package depends only on core primitives and protocol-shaped journal/snapshot
interfaces; it imports no concrete storage or optimization code.

## Failure semantics

Unknown, duplicate, out-of-order, stale-generation, or illegal phase events raise
typed transition/integrity failures without advancing the anchor.

## Tests

Unit and contract tests cover every legal transition, illegal edge, crash/replay,
snapshot corruption, idempotence, E2E search budget decisions, unique attempt
identity, and decision/reward transaction fault injection.

## Provenance

State retains causal event identifiers, anchor/state generations, policy hashes,
and verification receipts required to audit every KEEP or REVERT.
