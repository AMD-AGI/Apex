# AGENTS.md

This repository contains Apex, an evidence-driven RL environment and agent for
optimizing AMD GPU kernels. Work from the repository root and preserve the clean-cut
architecture under `src/apex`; do not add compatibility paths for superseded APIs or
artifacts.

## Setup

```bash
cd /home/viouyang/Apex
./setup.sh
source .venv/bin/activate
apex dependencies verify --json
```

The setup command installs Apex plus the exact Magpie/TraceLens/InferenceX revisions
in `scripts/dependencies.lock.json` and materializes the exact vLLM/AITER trees from
`scripts/e2e_source_locks.json`. Never choose an arbitrary dependency checkout,
silently switch/reset a sibling repository, or add another import root. For Apex-only
CPU development, `python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'` is
sufficient.

## Product boundaries

Apex supports two application use cases:

1. `apex optimize kernel` optimizes an existing Python or Triton kernel and emits
   an unapplied source bundle. Standalone HIP execution is unavailable in V1 and
   fails intake with `hip_execution_unavailable`.
2. `apex optimize e2e` performs kernel-only optimization inside an unchanged Magpie
   workload contract and emits evidence-bound patches.

Codex is the default backend; Claude and Cursor are explicit alternatives. Agent text
is never evaluator evidence. The E2E controller preserves long-running memory through
typed state, canonical events, CAS artifacts, and fresh bounded context packets—not
through one unbounded agent conversation.

Do not introduce config-only E2E winners, host `site-packages` mutation, implicit
shell commands, mutable image tags as provenance, or automatic application of a
delivery bundle. Agent proposals must return to trusted evaluation.

## Module rules

- `src/apex/bootstrap.py` is the only production composition root.
- `core` and `ports` are at the bottom of the dependency graph.
- Domain packages do not import concrete adapters.
- `cli` parses and dispatches; it owns no optimization, grading, or persistence policy.
- `storage` is the sole canonical event/CAS implementation. Snapshots, reports, RL
  datasets, and leaderboards are rebuildable projections.
- `orchestration` transitions are pure and reject stale anchors or generations.
- `evaluation` is the only owner of measured grades and rewards.
- `delivery` produces immutable, unapplied bundles and cannot infer missing proof.
- `knowledge` contains attributed inert advice plus event-derived measured experience;
  static cards never override current workload evidence.
- Every package keeps a substantive `README.md` covering purpose, public API,
  invariants, dependencies, failure semantics, provenance, and focused tests.

Prefer immutable dataclasses and explicit ports. Keep implementation files at most
600 lines and functions at most 80 lines; the architecture tests enforce these
limits and dependency direction. Do not add a second state writer or duplicate
backend-specific optimization loop.

## Evidence and grading invariants

Standalone timing reports use `apex.kernel-measurement/v1` and policy
`kernel_invocation_nearest_rank_v1`. The evaluator recomputes statistics from at
least 300 positive finite raw invocation samples per implementation in every case:

```text
S50     = Tref,p50 / Topt,p50
S99     = Tref,p99 / Topt,p99
Srobust = min(S50, S99)

Reward = 20 * Icompile
       + Icorrect * (100 + 200 * clip(Srobust - 1, -0.25, 1.00))
```

Missing or insufficient p99 means no reward. Keep these event meanings distinct:

- `performance_command_result`: a normal-runtime command exited;
- `measurement_result`: the evaluator validated and recomputed raw measurements;
- `reward_committed`: an eligible evaluator-owned reward was committed.

Standalone tasks and E2E tasks with a strict micro harness use freeze, compile,
correctness, safety, then normal performance. The reviewed Qwen E2E binding has no
trusted raw-sample micro harness: its typed deferred stage proves frozen-source
integrity only, makes no compile/correctness/timing/reward claim, then applies the
evaluator-owned safety policy before an isolated immutable image build and unchanged
Magpie quality plus normal-performance measurement. A required or finding-level
safety failure skips deployment/performance. Advisory gaps may continue only with
`safety_certified=false`.

E2E acceptance requires no accuracy regression, no more than 5% TTFT-p99 regression,
no more than 2% TPOT-p99 regression, and a throughput improvement over the current
live anchor. Only `KEEP` advances that anchor; `REVERT` remains part of the episode.
Formal `source_rebuild_verified` delivery requires exact source locks, a trusted
build, loaded-byte/build-ID engagement proof, and a second fresh clean replay.

## Development workflow

Inspect the worktree before editing and preserve unrelated user changes. Use
`apply_patch` for source/document edits. Add or update focused tests with behavior
changes; do not weaken contracts to make a test pass. Avoid import-time I/O and
network-dependent CPU tests.

Run the complete CPU gate with importlib mode because independently scoped test
modules may share filenames:

```bash
source .venv/bin/activate
pytest -q -p no:cacheprovider --import-mode=importlib \
  tests/unit tests/contract tests/integration tests/architecture \
  tests/test_bootstrap_dependencies.py
```

Useful focused commands are documented by each module. Before handing off, also run:

```bash
python -m compileall -q src/apex main.py scripts
rg -n 'shell=True|os\.system' src/apex
```

`shell=True` and caller-supplied shell strings are prohibited. Fixed commands are
argv vectors supervised with timeout and output bounds.

## GPU and live campaigns

CPU tests are the default. Real GPU, agent, Docker, and model runs must be explicit,
must use a caller-selected results directory, and must retain dependency, source,
image, measurement, safety, and GPU-lease receipts.

Before launching, resolve visible devices and inspect exact KFD process ownership.
Do not use broad process-name killing in product code. Apex's run-scoped GPU lease is
cooperative metadata, not authority to terminate an unrelated process. Benchmark
baseline and candidate under matched conditions and never reuse a diagnostic trace
as a scoring measurement.

## Provenance

The static cards under `tools/perf_knowledge` are attributed GEAK-derived material,
not executable source. Preserve `THIRD_PARTY_NOTICES.md`, `UPSTREAM_SOURCES.md`, the
upstream license, source manifest, and all content hashes when updating them. Rebuild
cards only with `scripts/build_knowledge_cards.py` from the exact reviewed source pin.

Do not hand-edit generated cards, claim upstream advice as measured evidence, or
copy executable upstream bundles without a separate license and source review.
