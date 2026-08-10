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

Local persistent-server measurements additionally carry a CAS-backed
`local_server_lineage` reference. `e2e_server_lineage_validation.py`
independently reconstructs the active and retired generations from raw
execution attestations, client configs, GPU leases, owners, anchors, and
decisions. A cleanup observation must close the same active generation with
verified quiescence and no attempt identity. Its artifacts are rejected if any
reward event references them, so lifecycle success cannot be learned as a
performance reward. Missing, duplicated, reordered, drifted, stale-lease, or
retired-generation evidence makes the episode invalid.

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
Compile, correctness, and normal-performance command artifacts are accepted as
terminal gate evidence only when `argv[0]` equals the recorded absolute
executable path, the executable identity includes size/SHA-256 and filesystem
identity, and `executable_identity_reverified=true`. Older containment-only
command artifacts are not a compatibility input to trainable reward replay.

An `e2e_throughput_qos_v1` reward is evaluator-derived rather than a timing
event, so it must use `evidence_class=derived` and bind canonical
`reward_policy`, `e2e_reward_vector`, `decision_evidence`, and `candidate_manifest`
artifacts. KEEP and REVERT additionally require the clean-cut measured evidence
roles: `benchmark_config`, `normalized_benchmark`, `benchmark_report`,
`quality_evidence`, `quality_result`, `primary_delivery`, and all three
delivery configs. Their reward lineage also binds the one canonical
`matched_promotion_pair`; a decision binds its digest as
`promotion_pair_receipt`. The superseded E2E `raw_measurement` role and legacy
decision `benchmark_receipt` are rejected; the former remains valid only for
standalone kernel grading. A terminal,
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

The one measured-E2E exception to a complete four-leg promotion is the trusted
quality hard stop. When `performance_skipped=quality_gate`,
`e2e_hard_gate_validation.py` requires `REVERT`, reason
`quality_gate_failed`, the policy-defined scalar `20`, and the exact normalized
benchmark, unchanged official report, Apex execution attestation, quality
result/sample, candidate manifest, decision, policy, grade, and deployed-runtime
receipts. It independently validates that the candidate runtime was engaged and
that quality failed before performance. It rejects any manufactured paired
promotion, summary-only result, private field in the official report, or missing
raw evaluator artifact.

Safety findings or clean outcomes enter an episode only when the external
receipt has complete candidate, case, dispatch, policy, tool/runtime, and
artifact lineage. Missing, advisory, unsupported, and inconclusive evidence is
never trained as clean or safety-positive evidence. When the frozen policy still
allows correctness/performance scoring, the independent reward may remain
trainable, but the episode must retain `safety_certified=false`. The authoritative
truth table is in [the primary safety contract](../evaluation/safety/README.md).

Measured E2E export accepts only `apex.e2e-matched-promotion/v2`. It locates one
aggregate pair event and exactly four action-derived measurement events in
`anchor, candidate, candidate, anchor` order. All four legs must precede the
aggregate event and have exact attempt, opportunity, candidate, and anchor-
generation lineage. Every aggregate `promotion_{position}_{side}_{kind}` binding
must name the same normalized, quality, and config receipt as its leg. There is
no single-candidate-benchmark compatibility path.

Each leg is independently rebuilt as an `E2EObservation`. Export verifies the
unchanged public Magpie report, an independently stored
`benchmark_execution_attestation`, and raw quality files against normalized
documents. Lane, reward role, process, dependency, runtime, GPU, and quality
authority come only from the Apex side attestation; finding any Apex-private
field in the official report is a hard evidence mismatch. Export rejects
diagnostic scoring and requires receipt fields to name the exact bound CAS
objects. Both candidate legs must use the deployed measurement config and
engaged immutable image; each side must retain one config and requested/resolved
image identity across its two legs. The pair's canonical GPU lease is rebuilt
through the public runtime receipt contracts, including selector, clean HSA,
KFD/RSMI join, selected devices, process ownership, physical locks, and empty
foreign-owner evidence.

Export recomputes AB and BA current-anchor comparisons from the frozen gates.
The shared `conservative_e2e_reward_v1` selector chooses the worse comparison;
the pair comparison list, policy descriptor, selected index, verdict, aggregate
event, decision, grade, and reward must all agree exactly. Decision receipts
must also point to the same micro, safety, delivery, and matched-pair artifacts.

The acceptance gates are never inferred from current code defaults. Export
reads the unique path-free `e2e_reward_contract` artifact and checks its metric,
full acceptance/estimator policy, protocol hash, and objective hash against
`e2e.initialized`. The private recovery `run_request` is not reward authority.
Export then reruns current-
anchor acceptance and evaluator reward grading and requires exact equality with
the decision document and reward event. This makes non-default gates replayable
and prevents policy drift from manufacturing training rewards.

## Export

`DatasetExporter` consumes only an `EpisodeGraph` and `ArtifactStore`. It emits
canonical `dataset.json`, byte-stable child-transition `dataset.jsonl`, the
grouping `parent_episode.json`, real-candidate `sft.jsonl`,
`export_manifest.json`, and `validation_report.json`. Missing
artifacts and empty filters fail closed by default; `on_incomplete="skip"`
records an explicit skip reason. Every dataset export is a public projection:
the presence of a `private`/`heldout_private` episode or held-out split aborts
the whole export regardless of the requested split or skip policy. Textual
credentials likewise fail closed. Private host absolute paths are the one
deterministic transformation: `host_absolute_path_redaction_v1` replaces them
in parent/events/context/artifact projections while retaining the original CAS
receipt plus an exported-byte digest, so the transformation is explicit rather
than mistaken for evaluator evidence.
E2E export independently replays `e2e_throughput_qos_v1`, compares both the
embedded and event scalars, and byte-validates the canonical policy, grade,
decision, manifest, and source lineage before writing any record.

The standalone parent is likewise scored only by its unique
`scope=task_terminal` reward. Materialization checks the frozen
EvaluationContract and terminal source, replays compile/correctness gate evidence
for `0`/`20`, or reparses the raw ABBA invocation report and recomputes the
selected grade. A measured no-op is independently fixed at `120`; missing or
invalid measurement authority remains `null + untrainable_reason`. Child rewards
are retained for credit assignment but never summed or substituted.

`ReferenceDatasetLoader` is the trainer-agnostic reference consumer. It accepts
only the exact six-file export inventory, verifies the manifest hashes,
canonical JSON/JSONL encoding, and cross-file parent/transition/count lineage,
then provides three JSON-native views: a terminal scalar-reward episode,
ordered attempt transitions with either an explicitly centered or zero-baseline
reference advantage, and semantic tool/decision supervision. Null evaluator
rewards stay null and are marked `performance_trainable=false`; the consumer
does not turn infrastructure failures into negative performance samples or
collapse the trajectory into one prompt/response pair. Framework-specific
discounting, normalization, GAE, batching, and tensor conversion remain trainer
policy rather than Apex evidence.

`apex.rl_dataset_manifest/v2` is self-describing. It binds a public-only
visibility policy, deterministic redaction policy, source-terms/no-relicense
license policy, immutable export/source-CAS retention policy, and a recomputed
summary of record/split/visibility/artifact/redaction counts. The reference
loader verifies these policy IDs and recomputes the summary; rewriting the
manifest after changing a transition does not create a valid public export.

The E2E parent is scored only by the unique `scope=task_terminal` reward, never by
an attempt reward or their aggregate. Materialization replays the terminal paired
measurement under the frozen acceptance policy, then reads every bound clean-
replay benchmark and quality artifact from CAS and recomputes throughput,
completed requests, TTFT-p99, TPOT-p99, and accuracy. A missing, reordered,
modified, or summary-only terminal proof makes the parent untrainable or fails
integrity validation; it cannot silently fall back to the best child.

There is no compatibility reader, migration path, remote view-specific store,
mutable reward side channel, or second event writer in this package.
Parent kinds are the product task identities `single_kernel` and
`e2e_kernel_only`; the superseded presentation labels `standalone_task` and
`workload` are not emitted.

## Files and tests

- `models.py`: immutable graph, episode, event, and artifact views.
- `episode_semantics.py`: pure event-role, evidence-class, status, and decision projections.
- `episode_graph.py`: journal/CAS materializer and completeness validation.
- `graph_loader.py`: strict typed decoder for an exported graph projection.
- `graph_validation.py`: contiguous event-chain validation and parent raw-reward replay.
- `kernel_measurement_validation.py`: offline standalone writer/phase/report and GPU pre/post-bracket validation.
- `kernel_parent_reward.py`: standalone terminal source/gate/raw-measurement replay.
- `e2e_validation.py`: explicit lineage and replayable E2E proof validation.
- `e2e_measurement_validation.py`: composition of frozen-policy matched-promotion replay.
- `e2e_benchmark_validation.py`: raw Magpie bundle and candidate-delivery reconstruction.
- `e2e_gpu_lease_validation.py`: semantic runtime GPU-lease and per-action heartbeat-bracket reconstruction.
- `e2e_promotion_validation.py`: exact four-leg ABBA lineage and conservative selection replay.
- `e2e_quality_validation.py`: raw evaluator quality receipt and metric validation.
- `e2e_hard_gate_validation.py`: raw quality-stopped REVERT and runtime-only reward replay.
- `parent_reward.py`: task-terminal parent reward and reward-null projection.
- `terminal_raw_validation.py`: second-clean-replay raw CAS reconstruction.
- `projection_validation.py`: fail-closed identity and generation merges.
- `state_validation.py`: full canonical replay check for supplied workload state.
- `exporter.py`: deterministic grouped RL and SFT export.
- `export_sanitization.py`: deterministic public host-path transformation and
  manifest summary policy.
- `consumer.py`: verified trainer-neutral terminal, transition/advantage, and
  tool/decision supervision views.
- `backend_qualification.py` and `backend_qualification_agent.py`: strict raw-CAS
  and typed agent-receipt replay for Codex/Claude/Cursor gfx950 release gates.
- `tests/unit/rl/`: explicit-lineage, candidate-conflict, source-free E2E,
  standalone semantic agent export, crash/history, four-leg raw-CAS E2E replay,
  conservative comparison selection, ABBA order/completeness, pair/GPU/image/
  config/receipt tampering, quality-hard-stop reconstruction, non-default gates,
  diagnostic rejection, artifact,
  reward replay, split, secret, and byte-stability tests.

Run `pytest tests/unit/rl -q`.

## Purpose

RL materializes canonical events into parent/candidate episode graphs and exports
validated RL/SFT datasets for post-training.

## Public API

Use `EpisodeGraphMaterializer`, immutable episode/artifact models, and
`DatasetExporter` with an explicit `DatasetExportConfig`. Load a completed export
with `ReferenceDatasetLoader`; choose the reference advantage mode explicitly
when consuming attempt transitions. `BackendLiveQualificationArtifactVerifier`
and `backend_live_qualification_verifiers` provide the shared production replay
for the three gfx950 backend release gates.

## Invariants

Transitions come only from canonical events, observations point to exact context
receipts, reward is independently replayed, and incomplete episodes are explicit.
Backend qualification additionally requires a real contained agent invocation,
a clean exact Apex tree, the canonical measurement policy, at least 300 positive
finite samples per implementation and case, and a selected evaluator-owned
terminal reward with compile and correctness evidence.

## Dependencies

RL depends downward on core, context contracts, evaluation policy, orchestration
state, runtime read-only qualification receipts, and storage artifacts; it never
invokes an agent or GPU.

## Failure semantics

Missing artifacts/context, mixed unpartitioned policies, reward mismatch, secrets,
or incomplete trainability fail export unless an explicit skip policy allows it.
Backend verification always fails closed; manifest-only and summary claims have
no qualification authority.

## Tests

Run the hermetic unit suite for materialization, crash histories, negative
attempts, policy partitioning, measured and quality-hard-stop reward replay,
secret rejection, byte stability, and backend receipt/identity/sample tampering.

## Provenance

Datasets embed graph/run/event IDs, high-water marks, policy IDs, artifact hashes,
split/visibility, and an output manifest of every generated file. Backend release
evidence binds the raw manifest, coding and terminal-result receipts plus a
versioned verifier identity digest.
Quality-hard-stop transitions preserve the public report, private execution
attestation, raw quality files, and deployed-runtime receipts as separate
provenance; neither the decision text nor scalar reward can replace them.
