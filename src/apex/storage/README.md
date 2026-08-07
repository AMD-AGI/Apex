# Storage

This package owns durable evidence, not optimization policy.

- `EventJournal` is the canonical append-only SQLite history. Events have a global
  sequence, explicit parent, content checksum, transaction receipt, and run-scoped
  idempotency key.
- `ArtifactStore` publishes SHA-256-addressed bytes with temp-file, `fsync`, atomic
  rename, parent-directory `fsync`, and receipt verification on reads.
- `SnapshotStore` caches state projections. Snapshots are checksummed but disposable;
  controllers must be able to delete or rebuild them from the journal.

Storage types are immutable at their public boundary. SQLite triggers reject updates
and deletes; application state must advance by appending a new event.

## Purpose

Storage provides append-only event journals, content-addressed artifacts, and
disposable checksummed snapshots for replayable optimization and RL evidence.

## Public API

Use `ArtifactStore`, `EventJournal`, `SnapshotStore`, and their immutable receipt,
input, record, transaction, and snapshot types exported by `apex.storage`.

## Invariants

Artifact identity is content-derived, event identity is deterministic, journal
rows cannot update/delete, and authoritative state is always replayable.

## Dependencies

Storage depends only on core hashing/errors and standard-library filesystem/SQLite
support; it imports no domain reducer or application use case.

## Failure semantics

Digest mismatch, symlink/path escape, duplicate conflict, torn transaction, or
snapshot corruption raises integrity failure; corrupt snapshots may be rebuilt.

## Tests

Hermetic tests cover atomic CAS writes, concurrent/deduplicated appends, transaction
boundaries, crash recovery, snapshot validation, and replay ordering.

## Provenance

Receipts record digest, media type, byte count, relative locator, and schema;
events retain causal IDs and monotonic journal sequence numbers.
