# Apex

Apex is an evidence-driven RL environment and optimization agent for AMD GPU
kernels. It has two user-facing modes built on the same state, evidence, safety,
and delivery contracts:

- **Kernel optimization** accepts a trusted task descriptor plus a natural-language
  objective (or a complete `TaskSpec`) and delivers an unapplied source bundle.
- **End-to-end optimization** accepts a workload spec pointing at a Magpie benchmark
  configuration, diagnoses the live workload, improves kernel source, and validates
  the resulting workload without changing model or benchmark semantics.

Codex is the default agent backend. Claude and Cursor are selectable alternatives.
Agents propose code; evaluator-owned compile, correctness, safety, raw timing, and
workload evidence decide whether a candidate is accepted.

## Design guarantees

Apex is a clean-cut, modular implementation. There is no compatibility layer for
prior command, state, result, or dataset formats.

- The append-only event journal and content-addressed artifact store are canonical;
  snapshots and reports are rebuildable views.
- Every agent invocation starts from a bounded deterministic `ContextPacket`.
  Long E2E searches resume from typed state and receipts, not conversation memory.
- E2E scope is kernel source only. Config-only winners are forbidden.
- Apex emits bundles by default and does not modify the caller's repository.
- Normal benchmark measurements are distinct from instrumented diagnostic traces.
- Missing proof stays missing. Agent output, command success, or an image tag cannot
  manufacture correctness, reward, provenance, or a higher validation level.

The validation levels are intentionally explicit:

| Level | Meaning |
|---|---|
| `none` | No independently verified deployed implementation. |
| `runtime_overlay_verified` | An immutable derived image loaded the measured overlay bytes, but no clean source rebuild was proven. |
| `source_rebuild_verified` | Exact source locks and patches rebuilt successfully and a second fresh clean replay passed quality, latency, engagement, and objective gates. |

The default Docker E2E adapter is honest about its boundary: an overlay can reach
`runtime_overlay_verified`; it cannot be presented as a formal source-rebuild
delivery.

## Install

Prerequisites are Linux, Python 3.10 or newer, Git, Docker for container workloads,
and a supported AMD ROCm environment for real measurements. Install and verify the
exact reviewed Magpie, TraceLens, and InferenceX revisions plus the vLLM and AITER
source locks used for formal E2E delivery with one command:

```bash
./setup.sh
source .venv/bin/activate
apex dependencies prepare-runtime --json
apex dependencies verify --json
```

`setup.sh` is a small wrapper around `scripts/bootstrap_dependencies.py`. It installs
Apex into `.venv`, resolves clean locked sibling checkouts when possible, and falls
back to managed exact-commit checkouts. For vLLM and AITER it publishes a separate
managed checkout under `~/.cache/apex/source-locks` unless an explicit exact root is
selected. Local siblings may supply Git objects offline but are never reset or
switched; origin, commit, tree, and cleanliness are verified before use.
`prepare-runtime` builds the hash-locked lm-eval quality evaluator once, using the
pinned vLLM image with networking disabled during wheel build, install, and smoke
validation; subsequent verification rehashes its read-only CAS tree.
See [docs/dependencies.md](docs/dependencies.md) for offline and path overrides.

Install only Apex development dependencies when external runtime adapters are not
needed:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Authenticate any agent CLI you intend to use. Apex does not bundle model access or
credentials.

## Kernel optimization CLI

A natural-language request selects a checked-in trusted task descriptor. The
descriptor—not the prose—owns editable paths and fixed-argv compile, correctness,
performance, and optional safety/measurement contracts.

Production raw-measurement descriptors name
`adapter_id: apex-structured-kernel-v1`, freeze the protected harness and method,
and provide a structured `measurement.runner` that emits at least 300 reference
and optimized invocation samples per case. A successful performance command
alone is never a measured candidate. Formal controllers such as AKA may instead
declare a trusted `external_evaluator` recipe; that explicit path receives a
source bundle with no Apex reward for central scoring.

V1 executes standalone Python and Triton tasks. A standalone task whose language is
`hip` fails intake with `hip_execution_unavailable`, even if it carries a fixed HIP
recipe: the current kernel loop does not yet bind trusted build, deploy, and loaded-byte
engagement phases, so accepting that descriptor would overstate its evidence.

```bash
apex optimize kernel \
  "Optimize rms_norm in kernels/rms_norm.py for gfx950" \
  --workspace /absolute/path/to/kernel-repo \
  --results /absolute/path/to/run \
  --agent-backend codex
```

Descriptors may be named `apex-task.yaml`, `task_spec.yaml`, or placed under
`.apex/tasks/`. Machine callers can bypass natural-language selection with a complete
caller-neutral JSON or YAML contract:

```bash
apex optimize kernel \
  --task-spec /absolute/path/to/task.yaml \
  --result-json /absolute/path/to/run/result.json
```

Use `--agent-backend claude` or `--agent-backend cursor` to change the backend;
omitting it selects Codex. `--dry-run` validates and receipts intake without invoking
an agent. A successful candidate produces a source-only bundle and a stable machine
result. Verify a kernel bundle independently with:

```bash
apex bundle verify --bundle /absolute/path/to/bundle --json
```

Formal E2E bundles require a fresh evidence directory. This command checks out
the exact locked sources, rebuilds the immutable image, proves runtime engagement,
and repeats the unchanged quality/performance replay before returning success:

```bash
apex bundle verify \
  --bundle /absolute/path/to/e2e-candidate-bundle \
  --results /absolute/path/to/fresh-verification \
  --json
```

### Kernel grade

Reward uses robust median and tail speedup:

```text
S50     = Tref,p50 / Topt,p50
S99     = Tref,p99 / Topt,p99
Srobust = min(S50, S99)

Reward = 20 * Icompile
       + Icorrect * (100 + 200 * clip(Srobust - 1, -0.25, 1.00))
```

The evaluator recomputes p50 and nearest-rank p99 from at least 300 valid raw kernel
invocation samples for each implementation in every case. Missing or insufficient
p99 yields no reward. `Icorrect` represents correctness, integrity, and
anti-tampering—not an agent assertion. Safety is a separate promotion gate: a
confirmed finding rejects the candidate and suppresses performance reward without
silently changing the public formula.

## End-to-end optimization CLI

An E2E spec wraps an unchanged Magpie benchmark config and freezes search budgets
and no-regression gates:

```yaml
schema_version: 1
config_path: /absolute/path/to/Magpie/examples/benchmarks/benchmark.yaml
results_dir: /absolute/path/to/apex-run
agent_backend: codex
agent_model: gpt-5.5
agent_effort: xhigh
scope: kernels
gpu_arch: gfx950
goal:
  primary: throughput
  direction: maximize
  gates:
    accuracy_regression_pct: 0
    ttft_p99_regression_pct: 5
    tpot_p99_regression_pct: 2
max_iterations: 3
max_kernels: 10
max_turns: 25
agent_timeout_seconds: 3600
```

Run it with:

```bash
apex optimize e2e --spec /absolute/path/to/e2e.yaml
```

The controller benchmarks a clean baseline, collects targeted Magpie/TraceLens
evidence, ranks dynamic kernel opportunities, asks fresh bounded agent sessions for
source candidates, freezes and checks each candidate, deploys eligible candidates,
and measures against the current live anchor. It records `KEEP` or `REVERT`, then
reprofiles before the next decision. The primary target is throughput; accuracy may
not regress, TTFT p99 may regress by at most 5%, and TPOT p99 by at most 2%.

The deliverable is an immutable patch bundle tied to exact source repositories and
the baseline container identity. Formal success additionally requires a trusted
source build, runtime loaded-byte/build-ID proof, and a second fresh clean replay.
The benchmark config is not the deliverable and changing its non-image semantics is
not an optimization.

## RL post-training surface

The exact context packet is the policy observation. Candidate source, tool results,
raw measurements, policies, safety findings, decisions, and delivery receipts are
stored by digest and linked through canonical events. The RL materializer produces a
parent workload episode plus every candidate child attempt—including rejected,
failed, and no-gain attempts. Dataset export replays measured rewards and fails
closed on missing evidence or secret-bearing content.

Reporting and RL data are projections, not alternate state writers. See
[src/apex/rl/README.md](src/apex/rl/README.md),
[src/apex/storage/README.md](src/apex/storage/README.md), and
[src/apex/reporting/README.md](src/apex/reporting/README.md).

Rebuild reports or export a dataset from an existing canonical run without changing
its journal or artifact store:

```bash
apex report \
  --run-root /absolute/path/to/run \
  --output /absolute/path/to/report-view \
  --json

apex export-rl \
  --run-root /absolute/path/to/run \
  --output /absolute/path/to/dataset \
  --split train \
  --on-incomplete fail \
  --json
```

Use `--run-id` when it cannot be derived from the run result or directory name.
`export-rl` also accepts `--policy-id`, `--on-incomplete skip`, and `--no-sft`.
Projection output is rejected if it overlaps canonical `events/` or `artifacts/`.

## Architecture

`main.py` and the installed `apex` command both enter the same thin CLI. The sole
composition root is `apex.bootstrap`.

```text
main.py
src/apex/
├── cli/             parsing and dispatch only
├── intake/          trusted kernel and E2E contracts
├── optimization/    standalone and E2E use cases
├── orchestration/   replayable state machines
├── context/         bounded backend-neutral observations
├── execution/       Codex, Claude, Cursor, and supervised argv adapters
├── benchmark/       Magpie config and measurement adapters
├── diagnostics/     targeted trace evidence and ranking
├── evaluation/      robust kernel reward and E2E gates
├── evaluation/safety/ sanitizer execution and policy
├── delivery/        immutable kernel and E2E patch bundles
├── storage/         event journal, CAS, and snapshots
├── rl/              episode materialization and dataset export
├── reporting/       deterministic reports and replication guides
├── knowledge/       attributed static cards and measured experience views
├── runtime/         pinned dependencies, provenance, repositories, GPU leases
├── ports/           dependency-inversion protocols
└── core/            shared immutable primitives and errors
```

Every package has its own README describing purpose, public API, invariants,
dependencies, failure semantics, provenance, and focused tests. Architecture tests
enforce layer direction, file/function size limits, import purity, and public surface.
The [capability matrix](docs/capability_matrix.md) distinguishes implemented CPU
contracts from pending live GPU qualification; the clean-cut migration is tracked in
[`deletion_inventory.yaml`](deletion_inventory.yaml).

## Test

The default gate is CPU-only and hermetic:

```bash
source .venv/bin/activate
pytest -q -p no:cacheprovider --import-mode=importlib \
  tests/unit tests/contract tests/integration tests/architecture \
  tests/test_bootstrap_dependencies.py
```

GPU and live agent/workload campaigns must be invoked explicitly and write to a
caller-selected results directory. They must retain dependency, image, source,
measurement, and GPU-lease receipts. See [tests/README.md](tests/README.md).

## Knowledge and upstream provenance

Apex loads inert optimization cards from `tools/perf_knowledge`; it does not execute
copied upstream code. The estate is derived from
[AMD-AGI/GEAK at `6fa40c3`](https://github.com/AMD-AGI/GEAK/tree/6fa40c36b68bad9d543ae551b95bd3d169865744)
and retains per-file hashes, exclusions, license, and attribution. Obsolete mutable
knowledge storage is not read.

Benchmarking and targeted trace collection use
[Magpie at `210513b`](https://github.com/AMD-AGI/Magpie/tree/210513b31b2f3607920be4000d37fc51f14c5711);
trace analysis uses
[TraceLens at `4f25c1a`](https://github.com/AMD-AGI/TraceLens/tree/4f25c1a6f03441e710a97d71a5de9cc5c2fc1555).
Serving benchmark execution uses
[InferenceX at `23f04b8`](https://github.com/SemiAnalysisAI/InferenceX/tree/23f04b8baca7774f9c0bbcb7a31e9ad551a3b84b).
Quality evaluation uses a source-built, hash-locked
[EleutherAI lm-evaluation-harness at `b315ef3`](https://github.com/EleutherAI/lm-evaluation-harness/tree/b315ef3b05176acc9732bb7fdec116abe1ecc476).
All are exact runtime dependencies rather than vendored copies. Review
[UPSTREAM_SOURCES.md](UPSTREAM_SOURCES.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistributing the
knowledge snapshot.

## License and security

Apex is licensed under the [MIT License](LICENSE). Third-party material retains its
own notice and license. Report vulnerabilities through the private channels in
[SECURITY.md](SECURITY.md); do not publish credentials, generated exploit kernels,
or private workload data in an issue.
