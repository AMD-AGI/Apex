# CLI

`apex.cli` is a thin parser and dispatcher. It does not own state, validation,
agent execution, or benchmark logic. Both the installed `apex` console script
and repository `main.py` call the same `apex.cli:main` function.

The root command is a backend-native coding session: `apex`,
`apex "<request>"`, `--print`, `--json`, `--resume`, and `--continue` do not
create a formal task, require a descriptor, probe a GPU, or compute reward.
Codex is the default; `--backend` selects Claude or Cursor. `--plain` disables
kernel augmentation and `--kernel` requests it explicitly. Auto mode mounts the
three packaged AMD kernel skills, including instruction-only HIP guidance, plus
the run-scoped MCP façade only for
kernel-related prompts. Codex uses session-local `skills.config`; Claude and
Cursor use the same local plugin path. Cursor lacks the MCP bridge and emits a
typed notice while retaining the instruction-only skills.
`--results` selects the artifact root for scoped capabilities; absent an
override, the launcher uses a stable hidden sibling of the workspace and creates it
only when an artifact-producing tool is called.

Public command groups are `optimize`, `run`, `bundle`, `dependencies`, `doctor`,
`capabilities`, `release`, `showcase`, `report`, and `export-rl`; `mcp-server` is an internal stdio
transport. Formal Codex is the task default; `--backend` can explicitly select
Claude or Cursor with the same spelling as an ordinary session. Machine kernel
callers use `--task-spec` and `--result-json`.
Kernel and E2E commands expose explicit model, effort, iteration, turn, and
timeout overrides; E2E also exposes `--max-kernels`. Overrides replace and
revalidate the immutable intake contract before any process starts.
Formal natural-language ambiguity or a missing trusted oracle does not enter the
evaluator. In the default interactive mode, `apex optimize kernel "..."
--workspace ... --results ...` opens a kernel-enhanced native discovery session
to help the user identify one trusted descriptor and target. That session is
composed without the formal optimizer and explicitly carries no evaluation
authorization. `--non-interactive`, `--json`, `--dry-run`, and `--result-json`
instead return the atomic typed `needs_input` result immediately; callers can
add an explicit source path/target symbol and retry.

An explicit root `apex "..." --kernel` session may call only the host-authorized
`campaign.start` mutation in addition to inert knowledge tools. It can create an
unverified descriptor-free task draft, but Magpie acquisition and formal
compile/correctness/measurement/grade tools stay hidden. After the backend exits,
the user continues with:

```bash
apex optimize kernel --campaign <run> --workspace <repo> --results <root> \
  --evaluation-contract-draft-digest <digest> \
  --release-candidate-receipt <receipt>
```

The CLI reloads canonical draft bytes,
checks that the campaign is results-scoped, recomputes the current contract, and
delegates once to `KernelOptimizeUseCase`. Drift fails before GPU acquisition.

`apex optimize kernel ... --dry-run` emits the complete unverified Evaluation
Contract draft and digest without a GPU or agent. A local user can confirm
exactly those previewed bytes with `--evaluation-contract-draft-digest`; any
drift returns `evaluation_authority_mismatch`. This explicit local confirmation
is visibly different from reviewed-template or external-evaluator authority.

`apex optimize kernel "..." --template <showcase-dir> --results <dir>` is the
single entry for attributed image-kernel templates. It accepts neither a
second TaskSpec nor `--workspace`. The current checked-in examples are honestly
`pending`: unresolved immutable image/source identity and missing Apex-owned
evaluators produce `template_not_materializable` before an agent, Docker, GPU,
or scoring action. Upstream AgentKernelArena YAML/runners do not become trusted
by being copied into the repository.

Apex provides no sanitizer command, MCP tool, built-in runtime, automatic tool
selection, or vendor adapter in this release. CLI runs use the explicit
uncertified no-tool policy unless the caller's independently trusted evaluator
supplies a complete external receipt; the default remains
`sanitizer_runtime=not_implemented` and `safety_certified=false`. See the
[primary safety contract](../evaluation/safety/README.md).

`apex doctor --backend codex|claude|cursor --json` is a GPU-free native-backend
preflight. It reports exact CLI identity and a credential-redacted authentication
state only when the fixed native status command returns recognized affirmative or
negative evidence; an unknown command or opaque output is
`authentication_probe_failed`, not “logged out.” Feature entries are explicitly
marked `launcher_contract_only`: they describe syntax represented by Apex after
CLI/auth prerequisites pass, not a live protocol, tool-registration, approval,
resume, or cleanup qualification. It does not log in, start a coding session,
mutate backend config, or claim that a later model/GPU campaign is qualified.
Non-ready state returns a nonzero exit for automation.
`apex doctor gpu [--gpu-devices 0,1] --json` performs only the existing
race-checked ownership inspection and emits its full digest-bound receipt. It
does not acquire a cooperative lease, run a health command, or clean up any
process. The receipt also binds owner cgroup/container/namespace/Slurm identity,
the supervisor's scheduler identity, and a bounded exact-name process-table scan
for NHC and ROCm diagnostic activity. Fixed ctypes calls against the exact
ownership-bound RSMI library capture selected-device temperature, current system
clock, busy percentage, and VRAM usage. Only a complete clean receipt is `ready`
and exits zero; an unavailable required health API is `incomplete`, while
foreign ownership, scheduler mismatch, or active health activity is `blocked`.
`apex capabilities --json` resolves scoped availability against the current
workspace and its sibling capability-results root; use `--workspace` and
`--results` to inspect another caller-selected scope. Inventory inspection does
not create the results directory, verify Magpie, launch a process, or probe a
GPU. It includes unavailable planned contracts as well as implemented tools.
Unavailable entries are machine-readable roadmap/capability debt, not callable
MCP tools; the local MCP server exposes only entries backed by a handler. In
particular, no `kernel.sanitize` descriptor exists.

`apex dependencies prepare-runtime` builds or revalidates the exact lm-eval
quality runtime in the local read-only CAS and materializes missing formal vLLM/AITER
source pins. `install` also materializes those pins; `verify` and `verify-runtime`
only accept their exact source receipt and never clone them. `verify-runtime` always
requires lm-eval; plain `verify` includes it when present. `--lm-eval-lock`,
`--lm-eval-runtime`, `APEX_LM_EVAL_RUNTIME`, `--artifact-cache`, and `--offline`
are explicit locator/cache controls; none permits dependency or digest drift.

`apex release check --apex-root <checkout>` is read-only and GPU/network-free.
It either rebuilds `apex.release-candidate-receipt/v2` from current bytes plus an
optional typed `--evidence` document, or independently reconstructs an existing
`--receipt`. The output separates `baseline_status` from final release `status`.
`--require-baseline` is the non-circular precondition for starting live
qualification; `--require-ready` additionally requires images, live receipts,
and all four published showcases. Symlinked roots/evidence and edited self-hashed
receipts fail closed.
Live receipts use kind-specific v2 schemas: three backend/GFX950 receipts,
two-task crash/resume fault coverage, three-arm matched knowledge ablation,
independently validated AKA matched evaluation, and the exact Magpie 27-row
resolution subject with 21 Docker live rows plus six typed early rejections. A
qualification name, count, or arbitrary digest is insufficient.
`--qualification-artifact-root <existing-absolute-formal-root>` asks the
composition root for a read-only authority over that root. It never creates the
root, rejects source/dependency overlap and symlink traversal, and still leaves
each claim blocked when its dedicated campaign verifier is unavailable or its
CAS lineage is invalid.

`apex release collect-local --apex-root <clean-checkout> --output <absolute-json>`
is the explicit executable half of that workflow. It runs only the fixed complete
CPU/static gate, exact dependency/runtime verification, and installed CLI/import
probe plus the official all-corpus Magpie resolver, then creates one typed evidence
file without overwriting. Dirty or changing source, missing `.venv`, timeout,
truncated output, an unresolved scan/resolver, or CLI byte drift fails without
evidence. It deliberately cannot fill fetched/ancestry-reviewed baseline,
GPU/image, live qualification, or showcase fields.
`apex release collect-showcase --apex-root <clean-checkout> --path <export>
--output <absolute-json>` runs the official offline verifier and creates one
path-free showcase v2 fragment. `apex release join-evidence --base <json>
--qualification <fragment> --showcase <fragment> --output <new-json>` reparses,
sorts, and joins typed fragments without overwriting or inventing claims. It is
not a producer for recovery, ablation, backend, Magpie, or AKA live facts; those
remain owned by their trusted campaign harnesses. See the
[live qualification runbook](../../../docs/live_qualification_runbook.md).
`apex release collect-qualifications --artifact-root <formal-root> --output
<absolute-json>` writes a path-free inspection report with one explicit
`verified|unavailable|invalid` entry per required kind. The report is diagnostic,
not a qualification fragment or authority receipt, and the resolver performs no
writes to the formal root.

Non-dry-run `apex optimize kernel` and `apex optimize e2e` therefore require
`--release-candidate-receipt <json>`. The CLI reconstructs it against the Apex
source that supplies the installed entrypoint before composition or GPU lease,
accepts `baseline_status=ready` even while final release gates are pending, and
passes the verified bytes into the use case for canonical CAS/event recording.
E2E resume requires the same flag and byte-identical original receipt; dependency
or source drift cannot be hidden by supplying a newer unrelated baseline.

Public API: `main`.

Tests: `pytest tests/unit/cli tests/integration -q`.
`apex optimize e2e --config <magpie.yaml> --results <dir>` verifies the pinned
Magpie/TraceLens/InferenceX receipt before constructing the E2E use case. The
raw Magpie config is the sole workload document; Apex constructs its internal
budget/backend spec from CLI flags and never asks users to duplicate workload
fields in an Apex-specific wrapper.
V2 accepts Docker one-shot configs only. Local, Ray, reuse, and cleanup inputs
return `e2e_docker_only` before provenance resolution, GPU lease, or agent work.
Add `--dry-run` to write `preflight.json` without creating a canonical run or
requesting a GPU lease. The receipt separates config compatibility from source-
optimization and formal-delivery readiness, so Atom/SGLang/local/Ray gaps remain
visible instead of entering a Qwen-specific fallback.

`apex report --run-root <run> --output <dir>` rebuilds `report.{json,md}` and
`replication_guide.{json,md}` from the verified journal/CAS. `apex export-rl`
uses the same source and the sole `DatasetExporter`; `--split`, `--policy-id`,
`--on-incomplete`, and `--no-sft` expose its fail-closed selection policy.
Both commands require an explicit output directory, never append events, and
refuse to place disposable projections inside `events/` or `artifacts/`.
`apex showcase export --run-root ... --id ... --output ...` uses that same
verified graph/CAS source and never executes or re-grades a task. `show`, `list`,
and `verify` are read-only operations that validate checksums, reconstruct the
typed trajectory/artifact graph, replay terminal reward from raw evidence, and
recompute qualification blockers. `verify` emits a path-free v2 receipt binding
all critical artifact digests; release evidence consumes that receipt rather than
unbound booleans. An incomplete or
non-winning run is retained as `pending` with blockers; CLI never promotes it to
a published winner.
`apex bundle verify` detects both standalone kernel bundles and formal E2E
patch bundles. Kernel verification preserves its static content/baseline
contract. E2E verification additionally requires a new absolute `--results`
directory and invokes the composition-root verifier: exact clean source
materialization, trusted source rebuild, loaded-byte/build-ID engagement, and a
fresh unchanged-workload replay all contribute to the typed terminal outcome.
Static E2E tree loading is never presented as terminal verification.
`apex bundle apply --bundle <dir> --workspace <clean-git-root>` is the only
opt-in caller-workspace mutation. It accepts kernel bundles only, repeats digest,
patch-target, exact baseline and candidate-byte checks, refuses a dirty tree,
and restores baseline bytes if post-apply validation fails. E2E bundles remain
owned by their independent clean-replay verifier.

## Purpose

The CLI translates user intent into intake contracts and invokes one bootstrap
composition root; it contains no optimization or grading policy.

## Public API

`apex.cli:main` is the sole parser and dispatcher. The repository-level
`main.py` and installed `apex` command both delegate to it.

## Invariants

Codex is the default backend, Claude and Cursor require an explicit selection,
and commands never recover state or measurements from agent stdout.
Ordinary root sessions preserve backend-native project instructions, approvals,
sandboxing, and persistence. They never call the formal optimizer or manufacture
an evaluation contract. Capability schemas and availability come from the one
composition-root registry rather than CLI-owned policy.

`apex run resume --run /absolute/run/root` verifies the persisted resolved
request, dependency/source/oracle policy identities, benchmark views, event
journal, CAS receipts, and current GPU lease before continuing. V1 resumes
baseline and diagnostic boundaries (including an interrupted `diagnostic-0`)
and fails closed for a partially executed candidate without a frozen candidate
checkpoint. Re-running resume on a terminal run reads the journal-bound CAS
result receipt and requires `result.json` to match those canonical bytes. The
terminal value also binds its run ID, provenance, benchmark views, root-bound
journal/artifact paths, and CAS-verified diagnostic evidence; any drift is
rejected.

## Dependencies

The CLI may depend on bootstrap, intake, delivery verification, and stable domain
APIs. Lower layers never import the CLI.

## Failure semantics

Invalid arguments and typed Apex failures produce deterministic nonzero exits;
unexpected exceptions are not relabeled as successful task results.
Missing or corrupt run evidence, ambiguous run IDs, empty dataset filters, and
artifact validation failures remain typed errors rather than synthetic output.

## Tests

CLI parser/dispatch behavior belongs in CPU unit and integration suites. Live
backend authentication and GPU execution remain explicit campaign preconditions.

## Provenance

Every mutating run receives a results locator and obtains dependency/provenance
receipts through bootstrap before the use case starts.
