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
never parses stdout/stderr to recover a state transition. Attempt membership is
accepted only from an explicit `attempt_id`; `candidate_id` and `action_id` are
never treated as compatibility aliases. Conflicting candidate identities within
an attempt, or reuse of one candidate identity by two attempts, fails closed.
Passing `WorkloadState` and the in-memory `ContextPacket` map enables additional
checks; neither object is written by this module. The state is accepted only
when every field equals a pure replay of its verified canonical journal prefix,
not merely when its sequence and head ID match. Context packets must retain
byte identity with CAS.

Both standalone and E2E attempts preserve provider-neutral agent activity as
ordered canonical `agent_message`, `tool_called`, `tool_result`,
`usage_recorded`, and `cost_recorded` events. Each references the same v3
transcript CAS artifact; the exporter therefore places messages in `actions`,
tool calls/results in `tools`, and explicit usage/cost in `costs` without
parsing aggregate prose or stdout.

Profiler-on terminal diagnostics are parent-episode observations. Their raw
trace/report and typed TraceLens comparison receipts remain role-labelled CAS
artifacts on `tool_result` events with `evidence_class=diagnostic` and
`reward_eligible=false`; they are preserved in `parent_episode.json` but never
enter candidate reward replay or measured promotion evidence.

For E2E attempts, `opportunity_id` is part of the child identity rather than
optional reporting metadata. It must agree across the context packet target,
attempt events, decision document, atomic decision event, and reward event.
Mixing evidence from two opportunities fails materialization or export.

An E2E terminal candidate's receipt-bound `experience.measured` event is an
`outcome` in the exported transition. `knowledge_outcome_linked` remains an
observation: it associates the cards read for that attempt with the same decision
receipt, but is `inconclusive` unless a future frozen card-to-action binding can
support stronger attribution. These are projections from the canonical journal,
not a second experience or knowledge writer.

A standalone candidate awaiting an external evaluator is represented by
`experience.deferred` as a derived observation. The child is explicitly
truncated with `external_evaluation_pending`, so it cannot be exported as a
rewarded transition or SFT success before external evaluation is bound.

A standalone `kernel_robust_v1` reward is trainable only when its event has
`evidence_class=measured`, a policy ID, and verified `source`,
`harness|reference`, `raw_measurement`, `measurement_execution`, and
`reward_policy` roles in the same child lineage. The execution receipt binds the
trusted adapter, measurement phase/timeline, frozen harness/method/policy, and
raw report; without it an otherwise plausible scalar is truncated. Export
recomputes its scalar from `compile`, `correctness`,
`integrity`, `anti_tampering`, `safety.finding`, and `kernel_srobust`.

An `e2e_kernel_candidate_v1` reward is evaluator-derived rather than a timing
event, so it must use `evidence_class=derived` and bind canonical
`reward_policy`, `e2e_grade`, `decision_evidence`, and `candidate_manifest`
artifacts. KEEP and REVERT additionally require the clean-cut measured evidence
roles: `benchmark_config`, `normalized_benchmark`, `benchmark_report`,
`quality_evidence`, `quality_result`, `primary_delivery`, and all three
delivery configs. The superseded E2E `raw_measurement` role is rejected; it
remains valid only for standalone kernel grading. A terminal,
non-infrastructure E2E attempt is trainable only with exactly one canonical
`e2e.candidate_decided` and one reward. Their canonical `transaction_id` values
must match, and that SQLite transaction must contain exactly those two event
records. Independent appends, split transactions, or an extra transaction
member make the episode truncated. A source-free REJECT keeps its manifest,
decision, grade, and policy proof but intentionally has no `candidate_id` or
`candidate_source`; it is exported as an RL transition and never as an SFT pair.
For candidates that do exist, manifest source receipts must exactly match all
`candidate_source` artifacts. Cost, variance, and safety uncertainty remain
separate fields.

Measured E2E export independently rebuilds the anchor and candidate
`E2EMeasurement` values. It verifies raw normal-lane Magpie report and quality
files against their normalized documents, rejects diagnostic scoring, and
requires receipt fields to name the exact bound CAS objects. The benchmark
config digest must equal the deployed measurement-config digest and the serving
runtime input digest; requested and resolved runtime images must equal the
engaged immutable primary-delivery image. Decision receipts must point to those
same micro, safety, delivery, and normalized benchmark artifacts.

The acceptance gates are never inferred from current code defaults. Export
reads the unique canonical `run_request` artifact, checks
`sha256_json(spec.goal)` against `e2e.initialized.objective_policy_hash`, and
constructs `E2EAcceptancePolicy` from the frozen gates. It then reruns current-
anchor acceptance and evaluator reward grading and requires exact equality with
the decision document and reward event. This makes non-default gates replayable
and prevents policy drift from manufacturing training rewards.

## Export

`DatasetExporter` consumes only an `EpisodeGraph` and `ArtifactStore`. It emits
canonical `dataset.json`, byte-stable child-transition `dataset.jsonl`, the
grouping `parent_episode.json`, real-candidate `sft.jsonl`,
`export_manifest.json`, and `validation_report.json`. Missing
artifacts and empty filters fail closed by default; `on_incomplete="skip"`
records an explicit skip reason. A train export always excludes private or
held-out episodes. Textual secrets fail export rather than being silently
rewritten, because rewriting would change the observation the policy saw.
E2E export independently replays `e2e_kernel_candidate_v1`, compares both the
embedded and event scalars, and byte-validates the canonical policy, grade,
decision, manifest, and source lineage before writing any record.

There is no compatibility reader, migration path, remote view-specific store,
mutable reward side channel, or second event writer in this package.

## Files and tests

- `models.py`: immutable graph, episode, event, and artifact views.
- `episode_graph.py`: journal/CAS materializer and completeness validation.
- `kernel_measurement_validation.py`: offline standalone writer/phase/report receipt validation.
- `e2e_validation.py`: explicit lineage and replayable E2E proof validation.
- `e2e_measurement_validation.py`: offline image/config/measurement acceptance replay.
- `e2e_quality_validation.py`: raw evaluator quality receipt and metric validation.
- `projection_validation.py`: fail-closed identity and generation merges.
- `state_validation.py`: full canonical replay check for supplied workload state.
- `exporter.py`: deterministic grouped RL and SFT export.
- `tests/unit/rl/`: explicit-lineage, candidate-conflict, source-free E2E,
  standalone semantic agent export, crash/history, raw-CAS E2E replay,
  image/config/receipt tampering, non-default
  gates, diagnostic rejection, artifact, reward replay, split, secret, and
  byte-stability tests.

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
