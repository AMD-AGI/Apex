# Kernel optimization

## Purpose

This package implements the caller-neutral, standalone single-kernel use case.
It executes exactly the configured `TaskBudget.max_iterations` unless a typed
terminal policy (currently an agent timeout) makes another attempt dishonest.
Every iteration copies the canonical baseline into a new workspace, compiles a
fresh bounded `ContextPacket`, invokes a fresh stateless backend process, freezes
the declared source bytes, and independently verifies that candidate. The
original workspace is never modified.

The bundle is the standalone deliverable for AKA or another external evaluator.
Apex's local command evidence is provisional and never grants itself an external
score. V1 executes Python and Triton tasks only. HIP descriptors fail at intake
with `hip_execution_unavailable`, including descriptors with a fixed recipe,
because this use case does not yet execute or evidence the recipe's build, deploy,
and loaded-byte engagement phases.

## Public API

The supported API is the set exported by `apex.optimization.kernel.__all__`:
`KernelOptimizeRequest`, `KernelOptimizeUseCase`, `CandidateVerifier`,
`CommandEvidence`, `candidate_source_digest`, `CandidateWorkspace`, and the
context/run-record contracts. External callers should normally use only the
request and use-case types.

## Invariants

The evaluator-owned phase order is fixed:

```text
agent exits
  -> candidate source freeze
  -> compile
  -> correctness
  -> safety
  -> normal, uninstrumented performance qualification
  -> trusted evaluator measurement port
     OR explicit external-evaluator recipe deferral
  -> typed measured outcome in canonical history
     OR non-measured pending-external-evaluator observation
  -> next fresh attempt (while budget remains)
  -> deterministic best-candidate selection
  -> one source-only bundle
```

“Agent exits” includes a controlled exact-turn-boundary checkpoint. Apex stops
the structured stream exactly at `max_turns`; if the invocation receipt names
`structured_agent_turn_checkpoint_v2`, the observed count equals the requested
count, `private_pid_namespace_init_pidfd_v1` proves the agent namespace empty,
output is complete, and containment cleanup is verified, changed source may proceed to
the same freeze and evaluator-owned gates as a natural exit-zero
completion. This is not a grading shortcut: unchanged source is `no_gain`, and
turn overrun, invalid stream, timeout, truncation, or cleanup failure is rejected
before freeze. The canonical `agent_completed` event and v3 transcript retain
the exact termination/capture evidence.

The agent-mutated workspace is never used as an evaluator cwd. After containment
and allowlist validation, `CandidateWorkspace` rechecks the pristine anchor,
copies it into a new evaluator-owned projection, and overlays only content whose
editable-file digest was frozen. Interpreter/test caches, `.pyc`/`.pyo`, and all
other ignored agent artifacts are absent. Compile, correctness, safety, normal
performance, and delivery use this projection; source is rehashed while copied
to reject a freeze race.

- Attempts never inherit mutable source, build output, timing reports, backend
  process state, or evaluator workspaces from an earlier attempt. Later agents
  see only bounded typed outcomes rebuilt from the append-only event journal.
- Compile, correctness, safety, and performance run in that order on every
  attempt. An earlier gate failure is recorded and may inform a later attempt;
  it does not silently consume the rest of the search.
- Candidate selection is not last-write-wins. Candidates with trusted grades
  are ordered by `Srobust`, then reward, then stable attempt order. If the caller
  (for example AKA) explicitly supplies a trusted recipe whose provenance is
  `external_evaluator`, Apex does not infer speed from stdout: equally eligible
  deferred candidates use the earliest stable attempt as the deterministic
  fallback. Command success alone cannot activate this path.
- A deferred external-evaluator candidate emits `experience.deferred` with
  `evidence_class=derived` and `status=pending_external_evaluator`. It has no
  outcome field, is never projected into measured experience, and cannot become
  a trainable RL/SFT success until a future evaluator-owned result is bound.
- Each normally closed attempt has exactly one semantic
  KEEP/REVERT/REJECT/NEEDS_MORE_MEASUREMENT decision and at most one
  evaluator-owned reward. Integrity-fatal attempts terminate as typed failures.
  Final selection never recomputes or duplicates an attempt reward.
- A compile or correctness failure ends the attempt before safety or performance.
- A confirmed safety finding, or an incomplete required safety check, ends the
  attempt before normal performance.
- Advisory incomplete safety may continue, but the result is explicitly
  `safety_certified=false`.
- The default `VerificationPolicy.no_tools()` keeps compile, correctness, and
  performance-command behavior available while returning `not_configured`, not
  a fabricated clean result.
- Candidate source digests are checked before and after every verifier command.
  Symlinks, hardlinks, undeclared edits, and source mutation fail integrity.
- Every compile, correctness, and normal-performance command runs in a private
  PID namespace. Apex requires the namespace-init/pidfd receipt to prove zero
  surviving members before the next phase; a candidate cannot leave a detached
  writer racing measurement freeze.
- Verifier commands receive only an explicit host runtime allowlist (paths,
  locale, GPU visibility/ROCm, and non-secret Hugging Face cache controls) plus
  safe `CommandSpec.env` entries. Shell/Python/loader injection and credentials
  are rejected before the command starts; agent credentials never cross into
  evaluator-owned compile, correctness, or performance processes.
- Safety artifacts are diagnostic-only. Their instrumented bytes and timing are
  forbidden inputs to normal performance evidence.
- The standalone `performance` command is only a normal-runtime qualification
  gate. Its stdout, workspace files, and self-reported scores are untrusted. A
  candidate cannot create a robust grade by writing a timing-report file. A
  robust grade is created only through the frozen `KernelMeasurementPort` named
  by the task, writing into a fresh controller-owned directory outside the
  candidate workspace. The controller authors an execution receipt binding the
  adapter writer, measurement phase/timeline, source, harness, method, policy,
  and exact report digest. Only then does it parse `apex.kernel-measurement/v1`:
  a production adapter's evaluator parent writes the report, and no candidate
  subprocess may receive or discover the report path. The production structured
  adapter runs the frozen `measurement.runner` in a private PID namespace,
  accepts one strict JSON stdout document, proves teardown, and only then has
  the parent process publish the report.
  seeded paired ABBA blocks, `inner_repeats=1`, explicit timer resolution and
  measurement-method hash, healthy GPU snapshots around every block, and at
  least 300 raw invocations per implementation and case. Apex uses true-median
  p50, `nearest_rank_v1` p99, `Srobust=min(S50,S99)`, and the sole
  `kernel_robust_v1` reward formula.
- A valid point grade still receives that unchanged scalar reward. It becomes a
  KEEP candidate only when `Srobust > 1.05`, the seeded paired/block-bootstrap
  confidence lower bound is above `1.0`, every case/implementation population
  CV is at most `0.10`, and `worst_case_srobust >= 1.0`. Exact `1.05`, noisy or
  inconclusive gains, and aggregate wins that regress one case remain `no_gain`
  with their typed reason and point reward retained for training.
- The event chain distinguishes `performance_command_result` (the command
  completed) from evaluator-owned `measurement_result` (raw report parsed and
  grade recomputed). Both measurement and reward events bind the raw report,
  execution receipt, and protected harness. Only a valid measured grade emits
  `reward_committed`.
  Missing or insufficient p99 cannot be promoted to a speedup or reward.
- Before invocation, `KernelContextBuilder` records source/harness receipts,
  bounded knowledge selection (including typed unavailability), prompt, and
context packet in CAS and the append-only event chain.
- A caller-supplied sealed backend-runtime-closure digest flows unchanged from
  `TaskSpec` to the agent invocation receipt. This supports matched external
  campaigns without letting agent text assert runtime identity.
- Every normalized agent transcript is stored as one canonical JSON CAS
  artifact in addition to raw stdout/stderr. The shared backend-neutral recorder
  emits each `agent_message`, `tool_called`, and `tool_result`, followed by
  explicit `usage_recorded` and `cost_recorded` events, all bound to that CAS
  receipt before `agent_completed|agent_failed`. Turn, tool, usage, and cost
  metadata is projected only from structured backend events; human text is never
  parsed as accounting evidence.

## Dependencies

The package depends inward on `apex.core`, `apex.context`, `apex.intake`,
`apex.knowledge`, orchestration/storage primitives, agent and safety ports, and
the delivery bundle contract. It does not import AKA, an E2E optimizer, or a
concrete sanitizer implementation. Concrete agents and safety runtimes are
injected through ports. `safety_bridge.py` owns the translation from standalone
kernel state to generic safety contracts; it does not execute a concrete tool.

## Failure semantics

Invalid input or a forged/stale safety result returns `invalid_request` or
`rejected` with a stable reason code. Agent timeout is terminal; an ordinary
backend, compile, correctness, safety, or measurement failure is retained as a
typed outcome while later budgeted attempts continue. No source change is
`no_gain`. A missing measurement contract without an explicit trusted
`external_evaluator` recipe, an unsuccessful normal performance command,
invalid report, or fewer than 300 valid samples per implementation/case is
`no_measurement` with no
reward. A valid but non-improving robust grade is `no_gain` while retaining its
evaluator-owned training reward. After search, Apex writes exactly one final
machine result and at most one immutable source-only bundle. A selected candidate
is `candidate_ready`, `applied=false`, and
`external_verification_required=true`; AKA or another caller retains authority
for external scoring and host application. An explicit external-evaluator
candidate uses `candidate_deferred_to_external_evaluator`, carries no Apex
reward, and remains subject to the caller's central
compile/correctness/performance score.
Standalone HIP fails before a run, GPU lease, agent invocation, or evaluator
command with the stable `hip_execution_unavailable` reason.
The result binds its run ID, baseline resolution/file hashes, internal verdict
event, verification-summary receipts, event-journal head, artifact-store
receipts, and any typed terminal error; these references do not alter the bundle
manifest or grant Apex external scoring authority.

## Tests

Run the offline unit and integration coverage with:

```bash
pytest -q -p no:cacheprovider tests/unit/optimization \
  tests/integration/test_kernel_optimize_use_case.py
```

The integration suite asserts strict phase order, default no-tool behavior,
safety-finding performance skip, advisory-incomplete continuation, source-only
delivery, raw measurement/reward ownership, candidate-forged report rejection,
writer/harness/method receipt mismatch, 299-versus-300 sample boundary,
three-attempt robust best selection, compile-failure retry, exact iteration
bounds, fresh-workspace/report isolation, canonical typed context history,
canonical agent transcripts, original-workspace immutability, and verifier
environment isolation. Intake coverage additionally proves HIP fails closed both
with and without a complete fixed recipe.

## Provenance

This implementation is Apex-native. The generic safety boundary it calls is
documented in `apex.evaluation.safety`, including the exact upstream design input
and license notice. No AKA task adapter, scorer, manager, worker, or sanitizer
runtime is copied into this package.
