# Knowledge

`apex.knowledge` provides two read-only inputs to optimization: attributed
static advisory cards and measured experience projected from canonical events.
Neither input can select a winner, mutate controller state, execute source text,
or bypass Magpie verification.

## Trust model

The runtime order is live evidence (L0), receipt-backed Apex experience (L1),
normalized upstream cards (L2), and raw attributed source (L3). L0 is the
judge. L1 and L2 only add or order hypotheses. A card remains advisory even if
its prose says “MANDATED”, “DEPRECATED”, or contains shell/Python instructions.

`KnowledgeCard` binds kind, evidence status, structured scope, claim, application,
verification advice, caution, exact upstream commit/path/content digest/license,
and explicit conflict links. `validate_catalog` rejects missing, asymmetric, or
scope-disjoint conflicts and supersession cycles. Runtime code has no card
promotion or write API.

`load_knowledge_catalog()` additionally requires canonical JSON, the exact
content-derived snapshot digest, complete source-manifest provenance, and a
fully valid card relationship graph before any generated card reaches runtime
retrieval.

## Source and generated boundary

The audited source pin is
`GEAK@6fa40c36b68bad9d543ae551b95bd3d169865744`, Apache-2.0. The pin covers:

| Estate | Files | Bytes | Card policy |
|---|---:|---:|---|
| `perf_knowledge/` | 689 | 4,207,132 | Markdown becomes inert advisory cards |
| `kernel_workflow/knowledge/` | 8 | 63,988 | Markdown becomes inert advisory cards |
| `e2e_workflow/knowledge/` | 36 | 249,043 | Markdown except archive becomes cards |

`geak_source_pin.json` is the reviewable copy of the typed pin in `sources.py`.
`scripts/build_knowledge_cards.py` uses `git archive` at that exact revision,
validates the root license and aggregate estate sizes, hashes every source, and
generates `source_manifest.json`, `cards.json`, `capability_index.json`, and
`ATTRIBUTION.md`, plus an exact `LICENSE.upstream` copy. It does not read
uncommitted bytes or access the network.

Executable `.py`/`.sh`, registries, archived documents, source templates, and
nested expert/analysis-skill bundles stay in the source manifest with an
exclusion reason. They are not cards. Ordinary source prose is preserved
byte-for-byte under `tools/perf_knowledge/upstream/`; unresolved nested skill
bundles and executable templates remain manifest-only pending separate source
and license review. Nothing in the raw layer is imported or executed at runtime.

Build from an already pinned checkout:

```bash
python scripts/build_knowledge_cards.py \
  --geak-root /home/viouyang/GEAK \
  --output-dir tools/perf_knowledge \
  --package-catalog src/apex/knowledge/data/cards.json
```

Repeat with `--check` to prove both the attributed source estate and the
wheel-shipped catalog, exact Apache-2.0 text, and package notice are byte stable.
A custom `--pin` is only intended for hermetic fixtures and dependency review.

## Retrieval contract

`KnowledgeRetriever` requires an independent live-evidence hypothesis before it
accepts a query. It filters on operator/GPU/dtype/regime/language/framework and
version scope, then selects zero or 2–4 cards under a token budget. Selection is
deterministic and tries to include a complementary anti-pattern, stale card, or
contradiction. A mismatch such as `gfx942` versus `gfx950` is excluded rather
than generalized. Disabled or insufficient knowledge returns a typed empty
selection and never falls back to old storage.

The caller records `KnowledgeSelection.to_dict()` as `knowledge_read`. Later
measurement may append `knowledge_outcome_linked` with supported, contradicted,
or inconclusive. Those events do not rewrite static cards.

## Measured experience

`ExperienceView.from_events()` accepts committed event-like records and includes
only `experience.measured` payloads marked `evidence_class=measured`, not dry
runs or agent self-reports. Every attempt must bind exact task/operator/GPU,
versions, shape/source/harness/policy hashes, and evidence receipts. Success,
failure, no-gain, regression, and infrastructure failure remain distinct.
Deleting a view loses no truth: replaying the journal reconstructs it byte for
byte.
The MCP projection uses the same `ExperienceView` over a checksum-verified
read-only canonical journal and requires the complete `ExperienceIdentity`; it
does not add a second experience store or fuzzy cross-workload retrieval policy.

## File map

- `cards.py`: card, scope, provenance, and catalog validation.
- `catalog.py`: strict generated-card loader and snapshot receipt.
- `sources.py`: pinned offline archive and per-file source manifest.
- `build.py`: deterministic Markdown normalization and capability index.
- `retrieval.py`: scoped complementary retrieval and read receipt.
- `experience.py`: event-derived measured experience projection.
- `geak_source_pin.json`: human-reviewable audited GEAK pin.

## Non-goals and tests

This package is not GEAK execution, a mutable RAG server, a graph/remote store,
or a decision engine. It has no mutable flat-store compatibility reader, old
writer, mtime/pickle cache, or global experiment-history file.

Run `pytest tests/unit/knowledge -q`. Tests use temporary local Git repositories;
they require no network, API key, GPU, Magpie, TraceLens, or GEAK checkout.

## Purpose

Knowledge supplies inert, attributed optimization cards and measured experience
views without letting copied advice override live task evidence.

## Public API

Call the catalog loader/retriever and immutable card, scope, provenance,
selection, and experience types exported by `apex.knowledge`.

## Invariants

Cards are immutable snapshots, exact operator matches outrank generic hints,
retrieval is bounded/deterministic, and measured outcomes stay event-derived.

## Dependencies

The runtime package depends only on core. The offline builder reads pinned source
manifests but generated cards execute no upstream code.

## Failure semantics

Digest mismatch, unknown schema, missing attribution, source drift, or malformed
scope rejects the catalog; an absent default catalog yields an explicit disabled view.

## Tests

Knowledge tests cover pinning, archive safety, deterministic card generation,
catalog validation, scoped ranking, and experience identities.

## Provenance

GEAK-derived material records repository, exact commit, path, license, source
manifest, snapshot, and card digests in the copied knowledge estate.
