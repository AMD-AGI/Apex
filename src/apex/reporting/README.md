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

`build_replication_guide` renders only explicit argv committed under a parent
event's `replication` payload. It never guesses a shell command from stdout or a
human log. Exact dependency receipts, 40-hex source commits, parent image digest,
and argv are required. A run with a kept patch additionally needs a derived
image digest plus named `apply_bundle`, `build_image`, and `clean_replay`
commands. Missing evidence produces `reproducible=false` and stable reason
codes, not a plausible-looking guide.

`write_run_projections` atomically publishes `report.json`, `report.md`,
`replication_guide.json`, and `replication_guide.md`. This writer writes views
only; it has no access to `EventJournal.append` and cannot advance
`WorkloadState`.

Run `pytest tests/unit/rl/test_reporting.py -q`.

## Purpose

Reporting projects canonical EpisodeGraph evidence into deterministic human and
machine reports and an exact replication guide.

## Public API

Use `build_report`, `build_replication_guide`, and `write_run_projections` plus
their immutable projection result types.

## Invariants

Reports are rebuildable, evaluator-owned measured outcomes alone become headline
metrics, secrets are redacted, and report writers cannot append events.

## Dependencies

Reporting depends only on core and the read-only RL graph API. It imports no
backend, benchmark runner, optimizer, or mutable storage adapter.

## Failure semantics

Missing replication identity, unsupported commands, secret-bearing content, or
noncanonical evidence fails projection instead of producing a misleading guide.

## Tests

Reporting tests assert deterministic bytes, redaction, artifact indexing,
measured-evidence filtering, and replication completeness.

## Provenance

Every projection identifies its EpisodeGraph, journal high-water mark, policies,
artifact receipts, dependency commits, image digests, and source locks.
