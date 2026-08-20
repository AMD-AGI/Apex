# Kernel optimization

## Purpose

This package implements the caller-neutral, standalone single-kernel use case.
It executes exactly the configured `TaskBudget.max_iterations` unless a typed
terminal policy (currently an agent timeout) makes another attempt dishonest.
Every iteration copies the canonical baseline into a new workspace, compiles a
fresh bounded `ContextPacket`, invokes a fresh stateless backend process, freezes
the declared source bytes, and independently verifies that candidate. The
original workspace is never modified.

Before a GPU lease, the use case resolves Git repository identity and freezes
an evaluator-owned Evaluation Contract. Missing or mismatched authority returns
an unverified machine result without invoking an agent. The
`preview_evaluation_contract` method performs the same discovery without
execution and supports the CLI's exact digest-confirmation handshake.
`KernelCampaignDraftUseCase` is the descriptor-free capability bridge: it
accepts agent-discovered typed task fields, injects the caller-fixed workspace
and results roots, resolves and hashes the baseline, and records an explicitly
unverified contract in canonical journal/CAS. It performs no agent, GPU,
evaluation, or reward work. The later formal command must re-resolve the same
draft digest under explicit user or reviewed-template authority.
The supported host-owned continuation is `apex optimize kernel --campaign ...`.
It runs only after the discovery backend has exited, reloads the CAS-bound draft,
requires the exact digest and release baseline from the CLI caller, recomputes
the contract, and delegates to the existing `KernelOptimizeUseCase`; it does not
introduce another optimizer or state writer.
`FormalKernelCampaign` and `KernelFormalEvaluator` implement that later
chat-to-formal bridge without starting a second optimization loop. They recover
the draft from canonical events/CAS, require its original Git identity to have
been and remain clean, keep agent edits in a persistent results-scoped candidate
projection, and reconstruct every evaluator phase from frozen baseline, harness, and
candidate receipts. `kernel.measure` records raw capture only; `kernel.grade`
alone recomputes and commits reward, and missing authority/evidence is an
explicit unverified result rather than a failed coding session.
`stop_formal_campaign` is the standalone user-stop boundary. It closes pending
work and derives the terminal grade only from evidence already in the journal;
it cannot select or deliver a candidate.

The bundle is the standalone deliverable for AKA or another external evaluator.
Apex's local command evidence is provisional and never grants itself an external
score. Ordinary V1 tasks execute Python and Triton only. Unknown HIP descriptors
fail with `hip_execution_unavailable`, including descriptors with a claimed
fixed recipe. The narrowly scoped template-bound lane can create an internal HIP
TaskSpec only after packaged-registry admission, immutable image/source
materialization, an Apex-owned evaluator recipe, and a non-replayable authority
receipt. No checked-in template currently satisfies those gates.

## Public API

The supported API is the set exported by `apex.optimization.kernel.__all__`:
`KernelOptimizeRequest`, `KernelOptimizeUseCase`, `KernelCampaignDraftUseCase`,
`KernelCampaignDraft`, `FormalKernelCampaign`, `KernelFormalEvaluator`,
`KernelFormalCapabilityUseCase`, `FormalEvaluationAuthorityProvider`,
`OneShotEvaluationAuthorityProvider`, `FormalEvaluatorResult`, `FormalStopResult`,
`stop_formal_campaign`, `CandidateVerifier`,
`CommandEvidence`, `ExecutableIdentity`, `candidate_source_digest`,
`CandidateWorkspace`, and the
context/run-record contracts. External callers should normally use only the
request and use-case types.

`KernelOptimizeRequest.campaign_baseline` carries the receipt already rebuilt by
the formal CLI. When present, the use case records it through the shared CAS/event
writer before any kernel attempt. CPU/direct callers may omit it for isolated
contract tests, but the public non-dry-run CLI does not. A blocked or self-digest-
tampered document is rejected and cannot become run provenance.

## Invariants

The evaluator-owned phase order is fixed:

```text
agent exits
  -> candidate source freeze
  -> compile
  -> correctness
  -> optional external safety-receipt validation
  -> normal, uninstrumented performance qualification
  -> trusted evaluator measurement port
     OR explicit external-evaluator recipe deferral
  -> typed measured outcome in canonical history
     OR non-measured pending-external-evaluator observation
  -> next fresh attempt (while budget remains)
  -> deterministic best-candidate selection
  -> one source-only bundle
```

For a chat-started formal attempt the same order is split across typed tools;
each invocation replays canonical state and rematerializes the candidate. The
local MCP process supplies evaluator execution authority, while a trusted local
composition boundary supplies one non-replayable receipt bound to the exact run
and draft. `confirmed_draft_digest` only detects mismatch and cannot mint
authority. The frozen release baseline is rebuilt before the first evaluator or
GPU action. None of these authorities can be inferred from agent text.
`KernelFormalCapabilityUseCase` records the
typed request and response as CAS-backed tool events around the evaluator
operation; these diagnostic events never grant reward authority.

Standalone scoring runs normal performance and raw capture inside one typed GPU
measurement bracket. Apex commits the bracket before any measurement grade,
reward, verified-attempt decision, or delivery. The bracket rechecks the same
lease holder and physical inventory before and after timing; an expired lease,
missing lifecycle implementation, PID reuse, or device/owner drift leaves the
attempt without measured reward or delivery. Offline RL validation reconstructs
this bracket before treating a kernel reward as trainable.

The phased MCP evaluator for a still-running chat cannot attest termination of the
external chat agent's process tree, credential revocation, tool-channel
revocation, or concealment of the evaluator report directory. It records those
four isolation facts as false and records only the locally verified read-only
candidate freeze as true. The safety preflight therefore returns
`phase_isolation_incomplete`; `kernel.measure` stops before its measurement GPU
lease, normal performance command, raw timing capture, reward, or delivery.
Compile and correctness receipts produced before that boundary remain
provisional. A future trusted isolation authority must supply real evidence for
all required facts before this formal lane can measure; agent text or a no-tool
safety policy cannot supply it. The host-owned `--campaign` continuation avoids
that lane by ending discovery first and entering the normal bounded optimizer.

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

Before each formal backend call, `agent_request.py` requires the verified
Evaluation Contract and issues an `apex.agent-execution-authority/v1` receipt
bound to its exact digest, source anchor, attempt, backend, candidate workspace,
and editable files. All three formal adapters carry prompts over stdin and only
their own credential through the controlled environment. Missing/mismatched
authority or a credential echo is rejected before candidate freeze and reward.

The agent-mutated workspace is never used as an evaluator cwd. After containment
and allowlist validation, `CandidateWorkspace` rechecks the pristine anchor,
copies it into a new evaluator-owned projection, and overlays only content whose
editable-file digest was frozen. Interpreter/test caches, `.pyc`/`.pyo`, and all
other ignored agent artifacts are absent. Compile, correctness, safety, normal
performance, and delivery use this projection; source is rehashed while copied
to reject a freeze race.

For chat-started formal work, that mutable tree is the persistent
`formal-work/candidate-projection/editable` directory under the run results, not
the source checkout. Its pristine anchor and edits survive handler restarts;
each evaluator phase receives a separate persistent frozen projection.

- Attempts never inherit mutable source, build output, timing reports, backend
  process state, or evaluator workspaces from an earlier attempt. Later agents
  see only bounded typed outcomes rebuilt from the append-only event journal.
- Compile, correctness, optional external safety validation, and performance run
  in that order on every attempt. An earlier gate failure is recorded and may inform a later attempt;
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
  Final selection never aggregates attempt rewards; task finalization records a
  separate, uniquely scoped terminal reward.
- A trusted compile or correctness failure ends the attempt before safety or
  performance and records the formula-defined attempt reward (`0` or `20`).
- A confirmed exact-lineage external finding, or an incomplete required external safety check, ends the
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
- Before each verifier phase, Apex resolves `argv[0]` through the evaluator
  `PATH` (or the phase `cwd` for a relative path), replaces it with one canonical
  absolute regular-file path, and freezes its path, size, SHA-256, device,
  inode, mode, and timestamps. `CommandSpec.env` cannot override `PATH`.
  Successful command evidence records that identity and proves it was hashed
  and revalidated after process-tree teardown; byte or filesystem-identity
  drift fails the phase. Generic executables do not have a reliable common
  version command, so Apex does not execute an untrusted `--version` probe or
  substitute version text for the byte identity.
- Externally supplied safety artifacts are diagnostic-only. Their instrumented bytes and timing are
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
  execution receipt, and protected harness. A valid timing grade emits a
  measurement-stage `reward_committed`; trusted compile/correctness failures emit
  gate-stage rewards without pretending timing evidence exists.
  Missing or insufficient p99 cannot be promoted to a speedup or reward.
- Formal task finalization emits one `scope=task_terminal` reward independent of
  the number of attempts. A selected candidate reuses its raw-replayed grade; a
  measured no-op is `Srobust=1` and reward `120`; trusted compile/correctness
  terminal failures are `0`/`20`. Missing or invalid measurement authority is
  `task_reward=null` with an explicit `untrainable_reason`. The terminal event
  binds the frozen EvaluationContract, source identity, commands, harness,
  execution receipt, raw report, attempt policy, and recomputed grade.
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
concrete sanitizer implementation. Concrete agents are injected through ports;
safety input is either the explicit no-tool default or a receipt from an
independent trusted evaluator. `safety_bridge.py` translates standalone kernel
state to generic safety contracts and does not execute a tool. Normal
measurement always uses fresh uninstrumented runtime bytes. See the
[primary safety contract](../../evaluation/safety/README.md).

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
Unknown or caller-authored HIP fails before a run, GPU lease, agent invocation,
or evaluator command with the stable `hip_execution_unavailable` reason. A
copied/recomputed manifest cannot bypass this: only an exact packaged reviewed
registry entry can enter the fixed template-bound image-kernel lane.
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
environment isolation. Intake coverage additionally proves caller-authored HIP
fails closed both with and without a complete fixed recipe; template coverage
proves pending/unregistered manifests never reach Docker.
The MCP formal-evaluator suite additionally proves that missing chat-agent phase
isolation blocks performance, raw capture, reward, and delivery; non-editable
workspace drift and missing evaluation authority also fail closed before GPU
evaluation.

## Provenance

This implementation is Apex-native. The generic external-receipt safety boundary
is documented in `apex.evaluation.safety`. No AKA task adapter, scorer, manager,
worker, or sanitizer runtime is copied into this package.
