# Generated Apex knowledge cards

## Responsibility

This directory contains byte-stable outputs of
`scripts/build_knowledge_cards.py`. `cards.json` is the sole static knowledge
catalog consumed by Apex; `capability_index.json` supports deterministic scope
inspection; `source_manifest.json` retains every upstream source disposition.

## Contract

Cards are inert `fact`, `procedure`, `experience`, or `anti_pattern` advisories.
Each card includes scope, status, caution, exact repository/commit/path/content
hash/license, and conflict metadata. Imported cards remain
`imported_unverified`; they cannot select or accept a candidate, alter controller
state, execute embedded commands, or become reward evidence.

Do not hand-edit generated files. Rebuild from the pinned local GEAK checkout and
run the builder again with `--check`. Runtime validation fails closed on malformed,
non-canonical, duplicate, hash-drifted, or provenance-incomplete catalogs.

## Tests

Run `pytest tests/unit/knowledge -q` from the Apex repository root.
