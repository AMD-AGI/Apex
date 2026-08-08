# CLI

`apex.cli` is a thin parser and dispatcher. It does not own state, validation,
agent execution, or benchmark logic. Both the installed `apex` console script
and repository `main.py` call the same `apex.cli:main` function.

Current public command groups are `optimize`, `run`, `bundle`, `dependencies`,
`report`, and `export-rl`.
Codex is the task default; `--agent-backend` can explicitly select Claude or
Cursor. Machine callers use `--task-spec` and `--result-json`.
Kernel and E2E commands expose explicit model, effort, iteration, turn, and
timeout overrides; E2E also exposes `--max-kernels`. Overrides replace and
revalidate the immutable intake contract before any process starts.
Natural-language ambiguity or a missing trusted oracle is an atomic
`needs_input` result, not a generic crash. `--non-interactive` guarantees that
the CLI returns that machine-readable result immediately; callers can add an
explicit source path/target symbol and retry.

`apex dependencies prepare-runtime` builds or revalidates the exact lm-eval
quality runtime in the local read-only CAS and materializes missing formal vLLM/AITER
source pins. `install` also materializes those pins; `verify` and `verify-runtime`
only accept their exact source receipt and never clone them. `verify-runtime` always
requires lm-eval; plain `verify` includes it when present. `--lm-eval-lock`,
`--lm-eval-runtime`, `APEX_LM_EVAL_RUNTIME`, `--artifact-cache`, and `--offline`
are explicit locator/cache controls; none permits dependency or digest drift.

Public API: `main`.

Tests: `pytest tests/unit/cli tests/integration -q`.
`apex optimize e2e --spec <yaml> [--results <dir>]` verifies the pinned
Magpie/TraceLens/InferenceX receipt before constructing the E2E use case. The spec remains
the workload contract; `--results` is only a locator override.

`apex report --run-root <run> --output <dir>` rebuilds `report.{json,md}` and
`replication_guide.{json,md}` from the verified journal/CAS. `apex export-rl`
uses the same source and the sole `DatasetExporter`; `--split`, `--policy-id`,
`--on-incomplete`, and `--no-sft` expose its fail-closed selection policy.
Both commands require an explicit output directory, never append events, and
refuse to place disposable projections inside `events/` or `artifacts/`.
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
