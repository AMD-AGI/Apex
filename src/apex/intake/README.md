# Intake

## Responsibility

`apex.intake` turns either a natural-language request or caller-owned JSON/YAML
into an immutable task contract before an agent or GPU process starts. Machine
callers submit `TaskSpec`; human callers submit `NaturalLanguageRequest` plus a
workspace. Direct formal intake takes editable files, fixed commands, recipes,
and scope only from a trusted checked-in descriptor. A kernel-enhanced native
chat may instead discover those fields and submit the exact typed
`campaign.start` schema, but that creates only an unverified journal/CAS draft.
The host injects workspace/results roots, reloads and recomputes the draft after
the backend exits, and requires exact user digest confirmation before the single
formal optimizer receives authority. Chat text never becomes a trusted oracle.

Descriptors are discovered at the workspace root (`apex-task.*` or
`task_spec.*`), under `.apex/tasks/`, or beside an explicitly named source as
`<source>.apex.yaml|json`. Missing or ambiguous trusted oracles fail before an
agent or GPU process starts.

Both direct `TaskSpec.from_file` imports and natural-language discovery use the
same strict descriptor loader. It accepts only a nonempty, bounded, regular file
with one link, verifies descriptor identity and exact bytes before and after
parsing, and rejects replacement or mutation races. JSON and YAML both reject
duplicate or non-string keys, excessive values and nesting; YAML additionally
rejects every alias and excessive parser events.

## Contract and public API

The public API is the exact `apex.intake.__all__` list. `TaskScope` carries
trusted dtype, regime, framework, and version facts for strict advisory
retrieval; unknown dimensions stay empty and never get guessed from prose.
`AgentOptions` and `TaskBudget` freeze backend/model/effort/turn/iteration/time
controls for fair external comparisons. An optional
`runtime_closure_sha256` additionally binds a formal external campaign's sealed
backend runtime closure; it is provenance supplied by that controller, not an
Apex measurement claim. Codex is the default backend and bundle delivery never
modifies the caller workspace.
`E2EOptimizeSpec` is an internal run request built from the caller's raw Magpie
config locator plus CLI budgets; it is not a second user-authored workload
document and does not duplicate model/image/shape semantics. It carries the same
explicit `agent_model` and `agent_effort` identity; an empty value is invalid
rather than an implicit backend-specific guess.
`dataset_split` and `data_visibility` are frozen before execution and copied to
every candidate event. Defaults are `train/public`; `heldout_private` is valid
only with the `heldout` split, and the RL exporter excludes private episodes
from train exports.

`load_kernel_template` verifies an attributed `apex.kernel-template/v1`
manifest, its self-digest, exact upstream Git/file receipts, license/notice
snapshots, immutable runtime identity, in-image source identity, and protected
evaluator contract. A template may be recorded as `pending` with explicit
blockers, but `require_materializable()` then fails before any agent, container,
GPU, or measurement execution. The checked-in AgentKernelArena-derived inputs
are currently pending; their YAML files are provenance snapshots, never
TaskSpecs or evaluator authority.

`KernelMeasurementSpec` optionally names a trusted evaluator adapter, the exact
protected harness files, a frozen measurement-method SHA-256, a fixed-argv
`runner` command, and the
`equal_case` or `workload_weighted` aggregation. Harness files must be regular,
workspace-confined, and disjoint from `editable_files`; `TaskResolver` records
their individual hashes and aggregate harness digest. Raw report paths are not
caller or candidate vocabulary. The controller allocates a fresh output outside
the candidate workspace, and only the named measurement port may populate it.
Natural-language instructions and the ordinary performance command cannot
redefine this authority contract. The runner emits one strict
`apex.kernel-measurement/v1` JSON document to stdout; it never receives the
controller's private report path.
The same spec freezes minimum sample/tail counts, warmup, strict KEEP threshold,
confidence and worst-case floors, maximum CV, bootstrap confidence level, seed,
repetitions, and minimum paired-unit count. The report may declare its timing
protocol but cannot weaken those trusted acceptance fields.
Task descriptors may choose stricter values; validation rejects values weaker
than the canonical `1.05` point threshold, `1.0` confidence/worst-case floors,
`0.10` maximum CV, or 95% confidence.

## Boundaries and failures

This package owns no mutable run state and does not execute commands, invoke
agents, or write workspaces. It rejects shell strings, path escape,
symlink/hardlink editable sources, and config-only E2E scope using stable reason
codes. Standalone HIP is deliberately fail-closed in V1 with
`hip_execution_unavailable`, even when a descriptor includes a complete trusted
fixed recipe. Caller-authored `TaskRecipe(kind="fixed_hip")` remains inert parser
vocabulary; only the internal reviewed-template materializer can bind it to
immutable image/source and evaluator authority.
The only intended exception is an exact reviewed template-bound image-kernel
contract. Unknown HIP/C++ workspaces never acquire that authority from a copied
manifest or a mutable image tag.

## Tests

Run `pytest tests/unit/intake tests/contract -q`. New task modes require a trusted
recipe/oracle contract plus path, ambiguity, and round-trip tests.

## Purpose

Intake turns benchmark YAML or natural-language kernel requests into immutable,
caller-neutral task contracts before any agent or command executes.

## Public API

Use the task/E2E spec dataclasses, `NaturalLanguageRequest`, `TaskResolver`, and
the fail-closed `load_kernel_template` interface exported from `apex.intake`.

## Invariants

Workspace paths are relative and confined, executable commands are fixed argv,
editable paths are explicit, and measurement harness files are frozen and never
agent-editable. Candidate workspaces contain no reward-bearing report destination.

## Dependencies

Intake depends only on core primitives. It has no filesystem mutation, backend,
benchmark, storage, or optimization dependency.

## Failure semantics

Ambiguous language, unsupported modes, unavailable HIP execution, unsafe paths,
invalid budgets, or an incomplete correctness/measurement recipe fail before
execution. Missing adapter identity, harness files, method digest, structured
runner, an unsafe harness, or overlap with editable source has a typed intake failure and cannot
fall back to self-reported timing.

## Provenance

Resolved specs retain the original request plus normalized scope, framework,
version, oracle, budget, delivery, and raw-measurement contract.

`E2EOptimizeSpec` contains only workload and search intent; release-candidate
receipts are explicitly rejected as a superseded optimization input. Production
composition automatically records `apex.execution-identity/v1`, while intake
continues to preserve workload, image, source, and evaluation contracts without
granting release authority.
