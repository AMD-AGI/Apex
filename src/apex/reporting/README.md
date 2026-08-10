# Reporting

Reporting is a disposable projection over `EpisodeGraph`; it is never a source
of state, reward, or acceptance decisions. Deleting `report.md` or the
replication guide and rebuilding them from the same journal/CAS produces the
same bytes.

`build_report` separates evaluator-owned `measured` outcomes from derived,
estimated, and agent self-reported observations. Its headline section therefore
cannot promote an agent-claimed speedup. It lists every attempt, KEEP/REVERT,
reward policy, cost/failure record, real ContextPacket receipt, RL completeness,
and all content-addressed artifact receipts. Secret-shaped keys and values are
redacted in both JSON and Markdown views.

For both `single_kernel` and E2E runs the parent headline uses only the
independently replayed terminal `task_reward`, vector, policy, and raw terminal
lineage. Child attempt rewards
remain visible as local current-anchor outcomes but are never summed, averaged,
or substituted when terminal evidence is null.

Safety presentation keeps capability, execution, finding, policy satisfaction,
and `safety_certified` as separate fields. A no-tool, not-applicable,
unavailable, or inconclusive state must never render as clean. Reports only
project the external receipt or explicit uncertified default described by the
[primary safety contract](../evaluation/safety/README.md); they do not infer
coverage.

`build_replication_guide` renders only a unique typed
`apex.replication-declaration/v1` committed by a canonical event. It never
guesses a shell command from stdout or a
human log. Exact dependency receipts, 40-hex source commits, parent image digest,
and argv are required. A run with a kept patch additionally needs a derived
image digest plus named `apply_bundle`, `build_image`, and `clean_replay`
commands. `single_kernel` binds exact Git commit/tree, evaluation authority,
verified bundle receipts, and verify/apply/compile/correctness/performance argv;
it does not invent E2E images. `e2e_kernel_only` additionally requires parent
and derived images, config receipts, fixed build argv, and clean replay. Missing
evidence produces `reproducible=false` and stable reason
codes, not a plausible-looking guide.

`write_run_projections` atomically publishes `report.json`, `report.md`,
`replication_guide.json`, and `replication_guide.md`. This writer writes views
only; it has no access to `EventJournal.append` and cannot advance
`WorkloadState`.

`ShowcaseExporter` is another read-only projection over the same verified
EpisodeGraph and CAS. It emits byte-stable `showcase.json`, terminal reward,
result, full parent/child episode, artifact manifest, included small text/JSON
artifacts, report, reproduction declaration, and checksums. Its fixed inventory
also contains `README.md`, `template/raw_config_snapshot.json`,
`winner/winner.diff`, and independent dependency/source/image/GPU receipt
projections under `receipts/`. A missing configuration, diff, or receipt class
is written as an explicit missing projection and becomes a qualification
blocker; a partial file tree can never be labelled `published`. Private/held-out
episodes, secret-bearing artifacts, symlinks, CAS drift, or unknown output files
fail closed. Private host paths inside otherwise portable text evidence are
deterministically replaced under `host_path_redaction_v1`; the manifest keeps
both the original receipt identity and the exported-byte digest. Binary or
oversized artifacts retain a typed source-run CAS locator and make the export
nonportable.

An export is `published` only when the replay-validated parent has a trainable
terminal reward above 120, a KEEP, policy/raw-measurement lineage, a CAS-backed
winner bundle that passes the official loader again, complete reproduction
evidence, and no nonportable artifacts. Otherwise all evidence is retained with
status `pending` and explicit
qualification blockers. The exporter never runs optimization, chooses a best
attempt, or computes a replacement reward. Role names alone are not proof:
`verify_showcase` reconstructs every bound bundle file and invokes the official
kernel or E2E loader with the recorded digest. It also checks the complete
file inventory and every byte checksum, loads the sanitized episode through the
typed graph schema, reconstructs the event chain and artifact manifest, replays
the evaluator-owned terminal reward from exported raw CAS evidence, rebuilds the
replication declaration, and recomputes every `pending`/`published` blocker.
It independently rebuilds the required config/diff/receipt inventory and README
from the sanitized graph/CAS and requires byte equality. Changing all
projections and regenerating checksums therefore cannot manufacture a reward,
fill a missing evidence class, or promote an incomplete run.

Successful verification also returns a path-free,
self-digested `apex.showcase-verification/v2` receipt. It binds the checksums,
episode, artifact manifest, reward, result, and reproduction file digests plus
event/artifact counts and the replay, bundle, and reproduction verdicts. Release
evidence consumes this exact receipt; a caller-supplied set of success booleans is
not a showcase qualification.

Run `pytest tests/unit/rl/test_reporting.py -q`.

## Purpose

Reporting projects canonical EpisodeGraph evidence into deterministic human and
machine reports and an exact replication guide.

## Public API

Use `build_report`, `build_replication_guide`, `write_run_projections`,
`ShowcaseExporter`, and `verify_showcase` plus their immutable result types.
`RunEvidenceSource`, `resolve_run_source`, and `materialize_run_graph` provide the
single read-only run-layout resolver shared by CLI reporting and MCP campaign
status; callers do not reimplement run-ID or journal/CAS path rules.

## Invariants

Reports are rebuildable, evaluator-owned measured outcomes alone become headline
metrics, secrets are redacted, and report writers cannot append events.

## Dependencies

Reporting depends only on core, the read-only RL graph API, and CAS reads. It
imports no backend, benchmark runner, optimizer, or event writer.

## Failure semantics

Missing replication identity, unsupported commands, secret content,
unredactable/noncanonical evidence, artifact drift, checksum mismatch, or an
unsafe output tree fails projection instead of producing a misleading
guide/showcase.

## Tests

Reporting tests assert deterministic bytes, redaction, artifact indexing,
measured-evidence filtering, replication completeness, showcase qualification,
repeat export, secret/private rejection, and offline tamper detection.

## Provenance

Every projection identifies its EpisodeGraph, journal high-water mark, policies,
artifact receipts, dependency commits, image digests, and source locks.
