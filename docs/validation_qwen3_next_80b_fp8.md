# Historical Qwen3-Next 80B FP8 v19 diagnostic record

This report preserves a 2026-08-09 live control-flow exercise under the older v19
contract. It is historical diagnostic provenance: the controller diagnosed,
proposed, isolated, measured, rejected, and reported candidates for the named
workload. It is **not current Plan V2 live qualification**, an optimized winner,
formal source-rebuild delivery, or release evidence.

The current contract additionally requires the Apex config-resolution receipt
derived from published Magpie main, independent semantic trace accounting and
planning coverage, paired
scalar attempt/terminal reward evidence, release-candidate baseline identity,
and the current recovery/delivery gates. This run predates those requirements;
its 11.3% targeted-trace coverage would fail the current 90% planning-coverage
policy before an agent. Reusing its hashes or re-labeling its projections cannot
upgrade it. A new positive-control and agent campaign must start from the current
release candidate.

## Frozen workload

| Field | Value |
|---|---|
| Magpie config | `/home/viouyang/Magpie/examples/benchmarks/benchmark_vllm_qwen3_next_80b_fp8.yaml` |
| Apex run ID | `e2e-f6021fce3cc048a6ad7fc4de5cfd8faa` |
| Results directory | `/data/viouyang/apex/results/e2e_qwen3_next_80b_fp8_acceptance_v19` |
| Apex code exercised | `4e6e5eddaf73a4bbfa752a9721f0ee16458dc5a5` |
| Magpie revision | `210513b31b2f3607920be4000d37fc51f14c5711` |
| GPU | MI355X (`gfx950`), physical UUID `GPU-5b46125e1dcf53d3` |
| Model revision | `c5f5f263bdd5cc134092897864e8905d8fe7b928` |
| Baseline image ID | `sha256:b599932816fe09f9ea2541655f5388457ac2494b87b551cefdbf2a207b0ed3a9` |

The exact Magpie config remained unchanged. Candidate deployments changed only the
derived image identity, and the terminal replay returned to the baseline source and
image identity.

## Outcome

The terminal status was `no_gain` with reason `insufficient_throughput_gain`.
No candidate was accepted, no patch was delivered, and the final source identity
was unchanged. The terminal comparison therefore establishes no regression under a
same-source replay; it cannot be counted as an optimization speedup.

| Measurement | Throughput | TTFT p99 (ms) | TPOT p99 (ms) | GSM8K accuracy |
|---|---:|---:|---:|---:|
| Initial baseline | 2458.173883 | 445.489428 | 13.055343 | 0.954511 |
| Final unchanged-source replay | 2531.164661 | 442.661081 | 12.729220 | 0.957544 |
| Observed replay delta | +2.969% | improved | improved | improved |

The throughput delta is repeat-run variance because both measurements identify the
same implementation. Apex intentionally reports `validation_level=none`,
`formal_delivery_verified=false`, `delivery_attempted=false`, and an empty accepted
patch list. A future formal success still requires a winning candidate, exact source
build, loaded-byte/build-ID engagement proof, and a second fresh clean replay.

Three isolated candidate episodes completed and were reverted:

1. A paged-attention proposal failed the strict quality gate in its matched window.
2. A fused-recurrent launch-configuration proposal failed the strict quality and
   TTFT gates in its matched window.
3. A reshape/cache specialization produced no admissible matched improvement.

These are evaluator observations under noisy live measurements, not causal claims
that the proposals intrinsically reduce model accuracy. Post-run hardening made
matched-window reward selection conservative and order-independent: any failing
comparison dominates a passing one, then the lowest scalar and deterministic
tie-break fields select the recorded verdict.

## Diagnostics and tracing

The diagnostic phase completed with reward disabled. Magpie produced the raw trace
and targeted trace shards; TraceLens produced a typed partial comparison. The
targeted capture reached its configured cap, so semantic coverage was partial and
the comparison was used only for candidate context—not grading or reward.

| Artifact | Evidence |
|---|---|
| Raw trace | CAS receipt `0844ea0144561db2df0cfd9041aae34902963971362421e2d64bb1cf3834f16d`, 159,520,658 bytes |
| Targeted shards | CAS receipts `82fd0e678687f9913e9c1c6a7aa38946643d43db4b2dcc7ed277881835dd5667` and `a969c27d339e2f764d19bd457bdb5b8a16cbe54bdcbc978423b41cfc9e4c9644`, 100,000 rows each |
| Capture accounting | 1,766,865 rows seen; 200,000 written; 1,566,865 cap drops |
| Semantic coverage | 0.113195; incomplete |
| TraceLens comparison | CAS receipt `9255aa584c8abfded251e74787a7aab4e1ddaafc64a4b35c15d53d4dc65d61be`; partial and reward-ineligible |
| Safety | Apex no-tool policy; no external sanitizer finding evidence; `safety_certified=false` |

## Integrity audit

The terminal run was audited independently of report projections:

- 411 contiguous canonical events in 408 transactions;
- 226 of 226 CAS objects rehashed successfully (1,198,846,855 bytes total);
- all 16 supervised action receipts were committed, successful, and exact-bound;
- SQLite integrity, foreign keys, event parents, attempt graph, and snapshot replay
  all passed;
- regenerated report and replication-guide projections were byte-identical.

Key terminal hashes are:

| Object | SHA-256 |
|---|---|
| `report.json` | `39718af525d445cc789154384c00bc3d94e682e2c3f5e7f9e0f833160255b4cf` |
| `report.md` | `880d5d5f16e39f46aa2b3319dafcdba6bbaa5957b6755b52f52aa8dccd7cdb05` |
| replication JSON | `c61b650dca2c394fd0d44a0fa12148b44632016394097e904a7b87ee2111e569` |
| replication Markdown | `c4d63eaf631df6013552509b6a3e8180cda56320013f63c649d0cd6efb759980` |
| `run.db` | `9b093b54617c3ccf03c4d70b5940a0b09b96b4fcf2c7b917a9acd75a2e8c4f5b` |
| snapshot | `f952e799d2a3beef8be47b43826b7995201339094ab09168254f4eb769296c00` |
| `result.json` | `a23e6dd66c214c1e122f8bc353e328f737491b3f391969bf05b650e5d6da3720` |

Intake provenance was deliberately permissive for this historical run and remained
partial: the baseline did not supply runtime-loaded-byte proof. Candidate overlays
did prove their loaded bytes. This is why the run may prove control-flow and
no-regression behavior under its old contract while making no current
qualification or formal delivery claim.
