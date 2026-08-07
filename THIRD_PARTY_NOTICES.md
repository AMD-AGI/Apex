# Third-party notices

## GEAK performance knowledge

Selected immutable source prose and normalized excerpts under
`tools/perf_knowledge/` originate from
[AMD-AGI/GEAK commit `6fa40c3`](https://github.com/AMD-AGI/GEAK/tree/6fa40c36b68bad9d543ae551b95bd3d169865744),
licensed under Apache-2.0.
The exact license is retained at `tools/perf_knowledge/LICENSE.upstream`; per-file
hashes and exclusions are recorded in `tools/perf_knowledge/cards/source_manifest.json`.
Normalized cards are modified excerpts for advisory retrieval only.

Nested expert/analysis skill bundles and executable source files are excluded
from the released raw snapshot pending separate source and license review.

## EleutherAI lm-evaluation-harness runtime

Serving quality evaluation uses
[EleutherAI/lm-evaluation-harness commit `b315ef3`](https://github.com/EleutherAI/lm-evaluation-harness/tree/b315ef3b05176acc9732bb7fdec116abe1ecc476),
licensed under the MIT License (Copyright 2020 EleutherAI). Apex does not vendor
its source. The runtime producer downloads the exact reviewed archive, verifies
its Git tree, and builds the locked wheel into a content-addressed local runtime;
the upstream license is retained inside that wheel's distribution metadata.
