# Intake

## Responsibility

`apex.intake` turns either a natural-language request or caller-owned JSON/YAML
into an immutable task contract before an agent or GPU process starts. Machine
callers submit `TaskSpec`; human callers submit `NaturalLanguageRequest` plus a
workspace. Natural language may select a task and refine its objective, but
editable files, compile/correctness/performance commands, recipes, and scope facts
only come from a trusted checked-in descriptor.

Descriptors are discovered at the workspace root (`apex-task.*` or
`task_spec.*`), under `.apex/tasks/`, or beside an explicitly named source as
`<source>.apex.yaml|json`. Missing or ambiguous trusted oracles fail before an
agent or GPU process starts.

## Contract and public API

The public API is the exact `apex.intake.__all__` list. `TaskScope` carries
trusted dtype, regime, framework, and version facts for strict advisory
retrieval; unknown dimensions stay empty and never get guessed from prose.
`AgentOptions` and `TaskBudget` freeze backend/model/effort/turn/iteration/time
controls for fair external comparisons. Codex is the default backend and bundle
delivery never modifies the caller workspace.
The E2E spec carries the same explicit `agent_model` and `agent_effort` identity;
an empty value is invalid rather than an implicit backend-specific guess.
`dataset_split` and `data_visibility` are frozen before execution and copied to
every candidate event. Defaults are `train/public`; `heldout_private` is valid
only with the `heldout` split, and the RL exporter excludes private episodes
from train exports.

`KernelMeasurementSpec` optionally declares the trusted workspace-relative
destination of a fresh `apex.kernel-measurement/v1` report and its
`equal_case` or `workload_weighted` aggregation. The report path is output, not
editable source: it cannot appear in `editable_files`, cannot already exist at
resolution time, and must have a confined non-symlink parent. The trusted
performance argv writes health-bracketed, seeded paired ABBA blocks of raw
reference and optimized invocation samples there; natural-language instructions
cannot redefine this contract.
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
fixed recipe. `TaskRecipe(kind="fixed_hip")` remains reserved parser vocabulary;
it does not authorize execution until the kernel use case binds and verifies its
build, deploy, and loaded-byte engagement phases.

## Tests

Run `pytest tests/unit/intake tests/contract -q`. New task modes require a trusted
recipe/oracle contract plus path, ambiguity, and round-trip tests.

## Purpose

Intake turns benchmark YAML or natural-language kernel requests into immutable,
caller-neutral task contracts before any agent or command executes.

## Public API

Use the task/E2E spec dataclasses, `NaturalLanguageRequest`, and `TaskResolver`
interfaces exported from `apex.intake`.

## Invariants

Workspace paths are relative and confined, executable commands are fixed argv,
editable paths are explicit, and measurement reports are never agent-editable.
Resolving a task rejects a stale report before copying the workspace, preventing
a prior run or agent-created file from being mistaken for fresh evaluator-owned
timing evidence.

## Dependencies

Intake depends only on core primitives. It has no filesystem mutation, backend,
benchmark, storage, or optimization dependency.

## Failure semantics

Ambiguous language, unsupported modes, unavailable HIP execution, unsafe paths,
invalid budgets, or an incomplete correctness/measurement recipe fail before execution. Measurement
path escape, an unsafe parent, a pre-existing report, or overlap with editable
source has a typed intake failure and cannot fall back to self-reported timing.

## Provenance

Resolved specs retain the original request plus normalized scope, framework,
version, oracle, budget, delivery, and raw-measurement contract.
