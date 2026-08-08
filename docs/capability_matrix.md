# Apex capability matrix

This matrix separates implemented contracts from live qualification. A CPU or
synthetic test proves schema and control-flow behavior; it is not evidence that a
GPU workload, model backend, or external evaluator passed. Live campaign receipts
replace the `pending` cells only after the named run completes.

| Surface | Implemented contract | Current qualification | Deliberate boundary |
|---|---|---|---|
| Natural-language kernel CLI | Resolves prose through a trusted task descriptor; unresolved input returns atomic `needs_input` | CPU unit/contract tests | Prose cannot expand editable files or invent evaluator commands |
| Machine kernel CLI | Caller-neutral `TaskSpec`, isolated fresh attempts, default Codex with Claude/Cursor overrides | CPU multi-attempt and backend contract tests; live campaign pending | No legacy CLI/state/result reader |
| Kernel correctness | Fixed-argv compile and correctness checks bound to frozen source bytes | CPU adversarial fixtures; representative GPU pilot pending | Agent stdout and self-reported speedup are never verdicts |
| Kernel performance | Invocation-level raw samples, seeded ABBA blocks, GPU-health brackets, median/p99 robust grade | Deterministic statistical fixtures; GPU timer qualification pending | Missing/invalid p99 has null reward; aggregate/batched means cannot impersonate p99 |
| Safety | Typed capability/execution/finding, immutable plan and candidate evidence, required/advisory policy | CPU truth-table and isolation tests | Unsupported or inconclusive never means clean; AKA remains its own authority |
| Kernel delivery | Content-digested source patch bundle; verify and explicit exact-clean-baseline apply with rollback | CPU integrity/apply tests | Default optimization never mutates the caller repository |
| E2E intake | Exact Magpie config, kernel-only scope, frozen throughput/accuracy/TTFT/TPOT gates | CPU config-invariant tests | Config-only winners are forbidden |
| E2E control plane | Typed state machine, fresh bounded contexts, current-anchor KEEP/REVERT and replayable journal | CPU transition/resume tests | Conversation history is not canonical state |
| E2E diagnostics | Magpie owns acquisition, TraceLens owns analysis, Apex validates typed evidence and ranks opportunities | Qwen3-Next 80B FP8 live flow passed with capped partial TraceLens evidence; [qualification report](validation_qwen3_next_80b_fp8.md) | Apex has no second low-level tracing engine |
| E2E source delivery | Exact source locks, cumulative patches, immutable derived image, loaded-byte/build evidence, second clean replay | CPU build/replay adversarial tests; live Qwen run ended `no_gain`, so formal winner replay remains unqualified | Overlay success is only `runtime_overlay_verified`; formal success requires rebuild and clean replay |
| Supported source changes | Standalone Python/Triton; E2E changes remain bound to their exact-lock source-build contract | CPU source/bundle tests; live Python/Triton path pending | Standalone HIP fails with `hip_execution_unavailable` in V1; monolithic extension modules and system ROCm libraries are unsupported |
| Knowledge | Attributed GEAK raw snapshot, normalized advisory cards, deterministic scoped retrieval, event-derived experience view | Corpus validation and retrieval tests | Knowledge cannot alter trusted policy, correctness, or measured reward |
| RL environment | Canonical event/CAS trajectory, parent/attempt episode graph, deterministic RL/SFT export with split/visibility | CPU replay/export tests | Missing artifacts and held-out-private leakage fail closed |
| Dependencies | One-command exact-pin Magpie/TraceLens/InferenceX install and receipt verification | Bootstrap tests and local pin verification | Runtime auto-clone of moving branches is forbidden |
| AgentKernelArena | AKA-only Apex launcher translates Arena tasks into `TaskSpec`; AKA centrally regrades bundles | AKA CPU tests; matched 10-task Apex/Codex GPU campaign pending | Apex repository contains no AKA adapter, tasks, hidden checks, or score logic |

The clean-cut deletion ledger is
[`deletion_inventory.yaml`](../deletion_inventory.yaml). It assigns every removed
production, test, documentation, config, and data file to a replacement owner and
is enforced by the architecture zero-reference gate.
