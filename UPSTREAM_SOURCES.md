# Upstream source ledger

| Apex capability | Upstream source | Pinned revision | Adoption |
|---|---|---|---|
| Static performance advisories | [AMD-AGI/GEAK](https://github.com/AMD-AGI/GEAK/tree/6fa40c36b68bad9d543ae551b95bd3d169865744) | `6fa40c36b68bad9d543ae551b95bd3d169865744` | Inert raw prose plus generated, attributed cards under `tools/perf_knowledge/` |
| Workload benchmark and targeted trace | [AMD-AGI/Magpie](https://github.com/AMD-AGI/Magpie/tree/12896a49a731ad72c791b7a23abcef7a0d6c4487) | `12896a49a731ad72c791b7a23abcef7a0d6c4487` | Published `main` runtime dependency through the typed adapter; not vendored into Apex |
| Trace analysis | [AMD-AGI/TraceLens](https://github.com/AMD-AGI/TraceLens/tree/4f25c1a6f03441e710a97d71a5de9cc5c2fc1555) | `4f25c1a6f03441e710a97d71a5de9cc5c2fc1555` | Runtime dependency through the typed adapter; not vendored into Apex |
| Serving benchmark harness | [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX/tree/23f04b8baca7774f9c0bbcb7a31e9ad551a3b84b) | `23f04b8baca7774f9c0bbcb7a31e9ad551a3b84b` | Repository-only runtime dependency selected through resolved Magpie views; not vendored into Apex |
| Serving quality evaluator | [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/b315ef3b05176acc9732bb7fdec116abe1ecc476) | commit `b315ef3b05176acc9732bb7fdec116abe1ecc476`, tree `6574cdae47205fcee11b76510fd09c5ae60a34c9`, version `0.4.9.2` | Deterministically source-built into a local read-only CAS from hash-locked artifacts; not vendored into Apex |
| Formal vLLM patch baseline | [vllm-project/vllm](https://github.com/vllm-project/vllm/tree/b1388b1fbf5aaef47937fabe98931211684666a6) | commit `b1388b1fbf5aaef47937fabe98931211684666a6`, tree `33b782e425e42d42851a33f7876e97a8deeabb29` | Exact clean managed source checkout used for evidence-bound patch capture and source rebuild; not vendored into Apex |
| Formal AITER patch baseline | [ROCm/aiter](https://github.com/ROCm/aiter/tree/c3708fb7445899c14cdc6e8055953ee02ed78ddf) | commit `c3708fb7445899c14cdc6e8055953ee02ed78ddf`, tree `a30409ac03524781f175cbb03e82eefcafd52af1` | Exact clean managed source checkout used for evidence-bound patch capture and source rebuild; not vendored into Apex |

All upstream claims remain subordinate to current evaluator-owned evidence.
Dependency installation and verification use `scripts/bootstrap_dependencies.py`.
