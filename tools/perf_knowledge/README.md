# Performance knowledge source estate

## Responsibility

This subtree preserves attributed, inert performance-engineering source material
and the deterministic Apex cards generated from it. Runtime optimization reads
only `cards/cards.json`; it never executes, edits, or imports upstream content.
Live evaluator-owned measurements remain authoritative.

## Provenance and license

The source layer under `upstream/geak/` is imported from
`AMD-AGI/GEAK@6fa40c36b68bad9d543ae551b95bd3d169865744` under Apache-2.0.
Exact source paths, content hashes, card eligibility, and exclusions are recorded
in `cards/source_manifest.json`; `LICENSE.upstream` is the pinned upstream license.
This separately licensed subtree is not represented as native Apex MIT material.

The complete 733-file upstream estate is represented in the manifest. Ordinary
`perf_knowledge`, kernel-workflow, and selected E2E prose is preserved byte-for-byte
under the commit directory. Nested expert/analysis skill bundles and executable
`.py`/`.sh` sources remain manifest-only while their nested provenance and licenses
are audited. They are not release content and are never executable capabilities.

## Generated boundary and consumer

- `upstream/geak/<sha>/`: immutable released source bytes; never hand-edit.
- `cards/`: generated canonical cards, source manifest, attribution, and index.
- `capability_index.yaml`: reviewable generated scope index.
- `UPSTREAM.json` and `MANIFEST.sha256`: snapshot and released-file receipts.

`apex.knowledge.load_knowledge_catalog` verifies canonical serialization, snapshot
hashes, card identities, conflicts, and provenance before retrieval. The context
compiler injects only 2–4 scoped, advisory cards after an independent hypothesis.

## Rebuild and sync

From the repository root:

```bash
python scripts/build_knowledge_cards.py \
  --geak-root /home/viouyang/GEAK \
  --output-dir tools/perf_knowledge \
  --package-catalog src/apex/knowledge/data/cards.json
python scripts/build_knowledge_cards.py \
  --geak-root /home/viouyang/GEAK \
  --output-dir tools/perf_knowledge \
  --package-catalog src/apex/knowledge/data/cards.json --check
```

A source revision change requires a reviewed pin update, a fresh source/license
audit, regenerated manifests/cards, and its own dependency-update commit.

## Non-goals

This directory is not a mutable RAG database, agent skill bundle, executable tool
registry, winner cache, or substitute for compile/correctness/performance evidence.
