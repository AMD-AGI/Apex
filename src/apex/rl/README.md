# RL

This package makes Apex's canonical journal useful for post-training without
creating another run writer. `EpisodeGraphMaterializer` verifies the SQLite
event chain and every declared `ArtifactReceipt`, then projects one standalone
task or E2E workload parent episode plus a child episode for every candidate
attempt. Rejected, non-best, tool-failed, and infrastructure-failed attempts
remain present.

## Observation and lineage contract

The policy observation is the exact `ContextPacket.canonical_bytes` stored in
CAS before an agent invocation. A `context_packet_created` event binds it using
the following payload shape:

```json
{
  "attempt_id": "attempt-1",
  "context_packet_id": "context-...",
  "state_generation": 3,
  "anchor_generation": 1,
  "artifacts": [
    {"role": "context_packet", "receipt": {"digest": "...", "size": 1,
      "media_type": "application/json", "relative_path": "sha256/../..."}}
  ]
}
```

All large prompt, message, tool I/O, candidate, source, harness, raw timing,
policy source, and delivery bytes use the same role-labelled receipt list. The
materializer never substitutes an end-of-run summary for a missing packet and
never parses stdout/stderr to recover a state transition. Passing
`WorkloadState` and the in-memory `ContextPacket` map enables an additional
journal-head and byte-identity check; neither object is written by this module.

A committed reward is trainable only when its event has `evidence_class` equal
to `measured`, a policy ID, and verified `source`, `harness|reference`,
`raw_measurement`, and `reward_policy` roles in the same child lineage. For
`kernel_robust_v1`, export recomputes the scalar from `compile`, `correctness`,
`integrity`, `anti_tampering`, `safety.finding`, and `kernel_srobust` in the
reward vector. Cost, variance, and safety uncertainty remain separate fields.

## Export

`DatasetExporter` consumes only an `EpisodeGraph` and `ArtifactStore`. It emits
canonical `dataset.json`, byte-stable child-transition `dataset.jsonl`, the
grouping `parent_episode.json`, real-candidate `sft.jsonl`,
`export_manifest.json`, and `validation_report.json`. Missing
artifacts and empty filters fail closed by default; `on_incomplete="skip"`
records an explicit skip reason. A train export always excludes private or
held-out episodes. Textual secrets fail export rather than being silently
rewritten, because rewriting would change the observation the policy saw.

There is no compatibility reader, migration path, remote view-specific store,
mutable reward side channel, or second event writer in this package.

## Files and tests

- `models.py`: immutable graph, episode, event, and artifact views.
- `episode_graph.py`: journal/CAS materializer and completeness validation.
- `exporter.py`: deterministic grouped RL and SFT export.
- `tests/unit/rl/`: crash/history, negative-attempt, artifact, reward replay,
  split, secret, and byte-stability tests.

Run `pytest tests/unit/rl -q`.

## Purpose

RL materializes canonical events into parent/candidate episode graphs and exports
validated RL/SFT datasets for post-training.

## Public API

Use `EpisodeGraphMaterializer`, immutable episode/artifact models, and
`DatasetExporter` with an explicit `DatasetExportConfig`.

## Invariants

Transitions come only from canonical events, observations point to exact context
receipts, reward is independently replayed, and incomplete episodes are explicit.

## Dependencies

RL depends downward on core, context contracts, evaluation policy, orchestration
state, and read-only storage artifacts; it never invokes an agent or GPU.

## Failure semantics

Missing artifacts/context, mixed unpartitioned policies, reward mismatch, secrets,
or incomplete trainability fail export unless an explicit skip policy allows it.

## Tests

Run the hermetic unit suite for materialization, crash histories, negative
attempts, policy partitioning, reward replay, secret rejection, and byte stability.

## Provenance

Datasets embed graph/run/event IDs, high-water marks, policy IDs, artifact hashes,
split/visibility, and an output manifest of every generated file.
