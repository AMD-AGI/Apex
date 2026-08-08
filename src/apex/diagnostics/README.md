# Diagnostics

This module is Apex's control-plane boundary for profiler evidence. Magpie owns
trace acquisition and TraceLens owns trace analysis. Apex validates their
artifacts, preserves loss accounting, and normalizes them into immutable
`TraceEvidence` records used by planning, context construction, and RL export.

`targeted_trace.py` is an independent consumer-side validator for Magpie's
`magpie.targeted-kernel-trace@1.0.0` contract. It does not import Magpie. Before
any event is exposed it validates the report's diagnostic-only marker, manifest
schema, manifest/summary coverage, exact shard set, file SHA-256 and byte count,
per-envelope sequence and hash chain, header identity, typed event payload,
rank/PID/run consistency, end sentinel, and receipt counters. Shards are read in
a checked streaming pass; a second checked pass feeds the normalizer only after
the complete artifact set has passed. When launch source is resolvable, its
semantic SHA-256 is also checked before the candidate becomes patchable.

Coverage follows the producer's loss-accounting equations exactly:

```text
seen = written + dropped
sampled = written + dropped - dropped_by_reason["sampling"]
sum(dropped_by_reason) = dropped
```

Sampling, caps, serialization, and I/O drops remain named observations. They are
never collapsed into a generic success bit.

`evidence.py` contains the versioned evidence contract and report normalizer.
Targeted semantic/runtime records are joined to aggregate profiler and gap/source
rows only when `runtime.gpu_symbol` equals the aggregate symbol exactly. A failed
join preserves both observations with `match_confidence=unknown`; name fragments,
counts, and trace order are never used as substitutes for correlation. Same-name
launches with different phase, rank, shape, dtype, stride, graph mode, grid,
constexpr, scalar, or meta values retain different candidate identities. When an
aggregate symbol covers multiple observed signatures, measured calls/time/share
are partitioned by sampled duration (or event count when duration is incomplete)
and explicitly warned.

Every normalized row carries typed, workspace-relative artifact receipts with
kind, media type, byte count, and SHA-256 for the benchmark report, gap CSV,
targeted manifest, summary, and shards. These receipts are relocatable inputs to
CAS publication and make normalization replayable without trusting stdout.

`ranking.py` produces two separate rankings:
measured GPU share and expected recoverable gain. Missing or low-confidence
roofline data is never converted to zero and never silently overrides measured
share. `adapter.py` implements `DiagnosticsPort` without launching a second
profiler.

The normalizer requires TargetedKernelTrace and has no aggregate-only fallback.
Planning/reprofile passes publish only receipt-bound diagnostic files. When the
controller explicitly requests terminal preservation, `adapter.py` additionally
validates and publishes only the Magpie report's declared rank-0 raw trace and
TraceLens output files; it never recursively captures the disposable InferenceX
checkout or unrelated workspace files. `comparison.py` binds the baseline and
terminal raw trace, benchmark report, and report sheets to verified CAS receipts,
then invokes the documented report-comparison API from the exact locked TraceLens
tree. Common stage/sheet groups produce immutable CSV/XLSX outputs which are
republished to Apex CAS. This is a typed `PARTIAL` result: report comparison ran,
but the pin still lacks a stable full-attribution contract and MI355X profile.
That unavailable capability is never inferred from the report diff.
Diagnostic metrics are observations only; they must never be used as the formal
E2E reward measurement.

## Purpose

Diagnostics validates targeted acquisition independently, normalizes exact-symbol
kernel evidence, and ranks observations for kernel-only search.

## Public API

Consume the evidence contracts, `TargetedTraceValidator`,
`TraceEvidenceNormalizer`, `MagpieTraceEvidenceAdapter`,
`PinnedTraceLensComparisonAdapter`, and ranking functions
exported by `apex.diagnostics`.

## Invariants

Every shard, sentinel, counter, receipt, and hash chain validates before events
reach planning; joins use exact runtime symbols and preserve unmatched evidence.

## Dependencies

The package depends on core contracts and diagnostics ports. Acquisition
validation deliberately does not import Magpie. The comparison adapter loads only
the hash-checked documented function from the configured locked TraceLens tree at
call time.

## Failure semantics

Incomplete traces, ambiguous symbols, checksum drift, path escape, source-digest
disagreement, or missing targeted evidence fail closed.

## Tests

Run `pytest -q -p no:cacheprovider tests/unit/test_targeted_trace_ingestion.py
tests/unit/test_trace_comparison.py tests/unit/test_trace_evidence.py
tests/unit/test_e2e_kernel_lane.py`.

## Provenance

Artifact receipts identify the Magpie report, gap CSV, manifest, summary, and
JSONL shards; source confidence remains explicit rather than guessed.
