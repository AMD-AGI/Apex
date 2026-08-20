# Apex

Apex is a general coding-agent launcher with evidence-driven AMD GPU kernel
capabilities. Ordinary coding stays an ordinary backend-native session. When a
user enters formal optimization, Apex adds state, evidence, safety, grading, and
delivery contracts for exactly two optimization task kinds:

- **Single-kernel optimization** freezes a trusted evaluation contract and
  delivers an unapplied source bundle.
- **End-to-end optimization** accepts a workload spec pointing at a Magpie benchmark
  configuration, diagnoses the live workload, improves kernel source, and validates
  the resulting workload without changing model or benchmark semantics.

Codex is the default agent backend. Claude and Cursor are selectable alternatives.
Agents propose code; evaluator-owned compile, correctness, raw timing, workload
evidence, and any independently supplied safety receipt decide whether a
candidate is accepted. Apex does not ship a sanitizer runtime in this release.

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
  manufacture correctness, reward, provenance, safety certification, or a higher
  validation level.
- The default safety state is `sanitizer_runtime=not_implemented` and
  `safety_certified=false`. Apex can validate a complete receipt from an
  independent trusted evaluator; it does not install or launch a sanitizer.

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
The same dependency lock requires `scripts/magpie_corpus_manifest.json`;
verification recomputes the `examples/benchmarks` Git tree plus every YAML path
and SHA-256, so a changed or newly added benchmark cannot bypass compatibility
review. It also requires the generated
`scripts/magpie_compatibility_ledger.json`. Its 27 Apex-owned projections bind
the frozen config hashes and reward fields, but are not live release evidence.
Release collection loads every row through the published Magpie `main` public
configuration model, then binds Apex-owned plan and capability-receipt digests.
The exact pin is `main@12896a49`; no unpublished Magpie branch or resolver API is
required. Even a complete configuration-resolution pass proves compatibility only; its zero
workflow/formal-delivery counts cannot be presented as live qualification.
`prepare-runtime` builds the hash-locked lm-eval quality evaluator once, using the
pinned vLLM image with networking disabled during wheel build, install, and smoke
validation; subsequent verification rehashes its read-only CAS tree. Formal
Docker composition also includes an exact-image lm-eval sidecar authority. It
locks an offline dataset and private InferenceX task projection, starts a bounded
Unix handoff to the observed Magpie listener, and executes one no-network/no-GPU,
read-only-root evaluator container whose result, sample, runtime, lifecycle, and
cleanup receipts are independently bound. This is implemented composition and
CPU-tested contract evidence only: no live Docker/GPU/model campaign has yet
qualified the sidecar, so it cannot clear a workflow, quality, reward, showcase,
or release gate.
See [docs/dependencies.md](docs/dependencies.md) for offline and path overrides.

Install only Apex development dependencies when external runtime adapters are not
needed:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Authenticate any agent CLI you intend to use. Apex does not bundle model access or
credentials.

## General coding-agent CLI

Run `apex` for the selected backend's native interactive experience, or provide
an initial request directly. No Apex descriptor, results directory, GPU probe,
campaign, measurement, or reward is created for this ordinary path:

```bash
apex
apex "Refactor the request parser" --workspace /absolute/path/to/repo
apex "Explain this Triton kernel on gfx950" --kernel
apex "Fix the tests" --backend claude --print
apex doctor --backend codex --json
apex doctor gpu --gpu-devices 0 --json
```

Codex is the default. `--plain` disables kernel augmentation, while `--kernel`
forces lazy mounting of the available Apex capability façade and the packaged
`amd-kernel-optimization`, `amd-kernel-debugging`, and
`amd-hip-kernel-optimization` skills. Auto mode mounts them
only for kernel-related requests. The skills are instruction-only, have no
scripts, and cannot create evaluator or sanitizer authority. A native kernel
session exposes attributed read-only knowledge plus one narrow `campaign.start`
handoff. That handoff accepts agent-discovered typed kernel scope/commands,
records an unverified Evaluation Contract draft without a descriptor file, and
returns its locator and exact digest. It does not run an agent, acquire a GPU,
or grant evaluator authority. Active Magpie/TraceLens and formal evaluator
capabilities remain hidden from the live chat because its MCP channel cannot
mint trusted evaluation authority.
Codex receives a session-local tool allowlist and auto-approval only for this
unverified draft mutation; global shell/file approvals and sandbox policy remain
native, and no evaluator capability is auto-approved.
`campaign.stop` closes a standalone formal campaign without selecting a
candidate. It records any missing REVERT/terminal decision, derives exactly one
task-terminal reward from evidence already sealed by the evaluator (including
the measured baseline no-op), and otherwise records an explicit null reward.
Repeated stop requests are journal-idempotent.
All capability artifacts stay under `--results` (or the lazy hidden sibling
default outside the source workspace).
`apex capabilities --json` shows exact schemas, authority, side effects, GPU
need, timeout, artifact classes, availability, and reward roles. Planned but
unimplemented surfaces are explicit `available=false` entries and are not
projected as MCP tools. No sanitizer tool is registered. Cursor currently
mounts the same skills and emits an explicit MCP-bridge-unavailable notice when
kernel augmentation was requested; it does not pretend the missing tools exist.
`apex doctor --backend codex|claude|cursor --json` checks the selected CLI's
exact entrypoint/version, credential-redacted authentication state, and native
interactive/headless/resume/effort/MCP surfaces without starting a session or
probing a GPU. A missing CLI, missing login, or capability gap is a typed,
nonzero preflight result.
`apex doctor gpu --gpu-devices 0,1 --json` is a read-only ownership probe. It
does not acquire a lease or terminate a process: it freezes the visible
HSA/KFD/DRM/RSMI mapping, selected physical UUIDs, and race-checked KFD
PID/UID/start-time identities in a digest-bound receipt. It then race-checks
procfs cgroup, container, namespace, and Slurm identity for each owner and scans
the process table for exact NHC/`rocminfo`/`rocm-smi`/`amd-smi` activity without
retaining command arguments. The same ownership-bound library is queried for
temperature, current system clock, busy percentage, and VRAM usage. The command
is `ready` only when those fixed APIs succeed, ownership is clean, scheduler
identity is consistent, and no health/diagnostic process is active. An
unavailable health API is `incomplete`; other conflicts are `blocked`.

Use `--print` for headless text and `--json` for the backend's structured stream.
`--resume ID` and `--continue` pass through native session persistence. These
sessions may edit the user-authorized workspace according to backend-native
approval and sandbox behavior, but their text and tool output are not evaluator
evidence. `apex optimize ...` is the separate formal boundary.

For a descriptor-free Python/Triton optimization, let the kernel session create
the unverified draft, exit the chat, then confirm exactly those bytes:

```bash
apex "Optimize this kernel and prove it is faster" --kernel \
  --workspace /absolute/path/to/repo --results /absolute/path/to/results

apex optimize kernel \
  --campaign /absolute/path/to/results/campaigns/<campaign-id> \
  --workspace /absolute/path/to/repo --results /absolute/path/to/results \
  --evaluation-contract-draft-digest <digest-from-campaign.start> \
  --release-candidate-receipt /absolute/path/to/campaign-baseline.json
```

The second command revalidates the exact draft and clean repository, then uses
the same bounded formal optimizer as descriptor-backed tasks. The chat process
must have exited first; agent text or a tool call cannot confirm the digest.
`campaign.start` returns this complete command as
`formal_continuation.argv_template`; clients should render it without guessing
or dropping arguments. A `ready=false` template names the release-baseline
blocker that must be resolved first.

## Kernel optimization CLI

A formal natural-language request currently selects a checked-in trusted task descriptor. The
descriptor—not the prose—owns editable paths and fixed-argv compile, correctness,
performance, and optional safety/measurement contracts.

Production raw-measurement descriptors name
`adapter_id: apex-structured-kernel-v1`, freeze the protected harness and method,
and provide a structured `measurement.runner` that emits at least 300 reference
and optimized invocation samples per case. A successful performance command
alone is never a measured candidate. Formal controllers such as AKA may instead
declare a trusted `external_evaluator` recipe; that explicit path receives a
source bundle with no Apex reward for central scoring.

V1 executes ordinary standalone Python and Triton tasks. Caller-authored `hip`
fails intake with `hip_execution_unavailable`, even if it claims a fixed recipe.
The only planned exception is the exact packaged template-bound image-kernel
lane: registry admission, immutable image/source identities, Apex-owned evaluator,
and materialization authority are all mandatory. The three attributed examples
are currently `pending`, so none launches Docker or a GPU and no HIP capability
or performance result is claimed.

The pending input snapshots and their byte/license provenance are under
`examples/optimization_showcases/`. The command shape is already fixed:

```bash
apex optimize kernel \
  "Optimize the declared kernel on MI355X/gfx950" \
  --template examples/optimization_showcases/kernel_triton_paged_attention_2d \
  --results /absolute/path/to/run \
  --release-candidate-receipt /absolute/path/to/campaign-baseline.json \
  --backend codex
```

Until its blockers are resolved this returns `template_not_materializable`
before agent, container, GPU, or evaluator execution.

```bash
apex optimize kernel \
  "Optimize rms_norm in kernels/rms_norm.py for gfx950" \
  --workspace /absolute/path/to/kernel-repo \
  --results /absolute/path/to/run \
  --release-candidate-receipt /absolute/path/to/campaign-baseline.json \
  --backend codex
```

Descriptors may be named `apex-task.yaml`, `task_spec.yaml`, or placed under
`.apex/tasks/`. Machine callers can bypass natural-language selection with a complete
caller-neutral JSON or YAML contract:

```bash
apex optimize kernel \
  --task-spec /absolute/path/to/task.yaml \
  --release-candidate-receipt /absolute/path/to/campaign-baseline.json \
  --result-json /absolute/path/to/run/result.json
```

Use `--backend claude` or `--backend cursor` to change the backend;
omitting it selects Codex. `--dry-run` validates and receipts intake without invoking
an agent and emits an Evaluation Contract draft digest. Formal local execution
requires repeating that exact digest with
`--evaluation-contract-draft-digest`; repository, source, harness, command, or
policy drift fails before a GPU is acquired. A successful candidate produces a source-only bundle and a stable machine
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
anti-tampering—not an agent assertion. Safety is a separate promotion gate only
when an independent trusted evaluator supplies an exact-lineage receipt. A
confirmed finding rejects the candidate and suppresses performance reward without
silently changing the public formula. With no external authority, optimization
may continue under the no-tool policy but remains `safety_certified=false`.

## End-to-end optimization CLI

The raw Magpie benchmark config is the only workload document. Apex freezes its
exact bytes and constructs the internal kernel-only budget/backend request from
CLI flags; users do not copy model, image, metric, or shape fields into another
spec.
This V2 entry supports Docker one-shot configs only. Local, Ray, reuse, and
cleanup return `e2e_docker_only` before provenance, GPU, or agent work.

```bash
apex optimize e2e \
  --config /absolute/path/to/Magpie/examples/benchmarks/benchmark.yaml \
  --results /absolute/path/to/apex-run \
  --backend codex \
  --release-candidate-receipt /absolute/path/to/campaign-baseline.json \
  --gpu-arch gfx950
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

Export a deterministic, sanitized showcase from the same canonical evidence:

```bash
apex showcase export \
  --run-root /absolute/path/to/run \
  --run-id run-id-if-needed \
  --id kernel-example \
  --output /absolute/path/to/showcase

apex showcase verify --path /absolute/path/to/showcase
```

`showcase show` and `showcase list --root <dir>` verify before rendering. Runs
without a replay-valid reward above 120, KEEP, a CAS-backed winner bundle that
survives reconstruction and the official bundle loader,
portable artifacts, and reproduction evidence remain `pending`; export never
turns a positive control, old score, or hand-edited summary into a winner.
Offline verification reconstructs the typed parent/child event chain and CAS
manifest, recomputes single-kernel or E2E terminal reward from raw evidence, and
re-derives the qualification blockers even if every checksum was regenerated.

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
├── evaluation/safety/ external safety-receipt contract and pure policy
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
contracts, real Docker lifecycle smokes, live GPU terminal results, and still
pending model-quality/reward qualification. The
[Qwen3-Next 80B FP8 report](docs/validation_qwen3_next_80b_fp8.md) records the
completed E2E no-gain/no-regression campaign without overstating it as a formal
source-delivery winner. The clean-cut migration is tracked in
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
The operator sequence and typed receipt ownership are documented in
[docs/live_qualification_runbook.md](docs/live_qualification_runbook.md).

Release/live baselines additionally use `apex.release-candidate-receipt/v2`.
`scripts/build_release_candidate_receipt.py` recomputes current Apex Git, lock,
Magpie 27-config corpus/ledger, lm-eval, and attributed-template identity, then
joins only explicit fresh-fetch, dependency, CPU-gate, CLI, immutable-image,
live-qualification, and four-showcase evidence. It performs no live work and
currently reports `blocked` until those gates genuinely exist. A modified status
plus a recomputed self-hash is rejected because verification reconstructs the
entire receipt from current bytes and typed evidence.

On a clean candidate commit, generate the locally provable subset first:

```bash
.venv/bin/apex release collect-local \
  --apex-root "$PWD" \
  --output /absolute/operator-selected/local-release-evidence.json
```

This command runs the complete fixed CPU gate, exact dependency/runtime verifier,
fresh installed-CLI import probe, and the Apex config projection over the frozen
Magpie corpus using published `main` APIs. It binds the result to unchanged source
bytes before and after execution. All 27 pinned configs resolve for identity;
the V2 live scope is the derived 21-row Docker one-shot slice, while the six
Local/Ray/reuse/cleanup rows are required rejection tests. The command performs
no fetch, GPU, image, agent, Magpie live
campaign, or showcase work, so those release blockers cannot be cleared by it.

After real runs, `apex release collect-showcase` converts an official offline
showcase-verifier v2 receipt into path-free release evidence on the exact clean
Apex tree. `apex release join-evidence` combines only already validated
qualification/showcase fragments into a new evidence file; it never manufactures
or upgrades a live claim.

The same document has a narrower, non-circular `baseline_status`: clean reviewed
source, exact dependency/runtime verification, the full CPU/static gate, and CLI
identity can authorize live qualification through `--require-baseline`; images,
live results, and published showcases are required only by final
`--require-ready`.

## Knowledge and upstream provenance

Apex loads inert optimization cards from `tools/perf_knowledge`; it does not execute
copied upstream code. The estate is derived from
[AMD-AGI/GEAK at `6fa40c3`](https://github.com/AMD-AGI/GEAK/tree/6fa40c36b68bad9d543ae551b95bd3d169865744)
and retains per-file hashes, exclusions, license, and attribution. Obsolete mutable
knowledge storage is not read.

Benchmarking and targeted trace collection use
[Magpie at `12896a4`](https://github.com/AMD-AGI/Magpie/tree/12896a49a731ad72c791b7a23abcef7a0d6c4487);
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
