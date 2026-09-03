# Execution

`apex.execution` implements the `AgentBackend` port for Codex, Claude Code, and
Cursor Agent, plus the production `StructuredKernelMeasurementAdapter` and
`MagpieKernelDiagnosticsAdapter`. Codex
is the registry default; callers can explicitly select the other two without
preflighting unrelated credentials.

All adapters execute argv with `shell=False` through `SubprocessSupervisor`,
drain stdout and stderr concurrently, bound captured output, and run both CLI
identity discovery and candidate production inside an authoritative Linux PID
namespace. They return normalized transcript events and never decide whether a
candidate is correct or fast.

Public API: `AgentRegistry`, `build_default_registry`, `SubprocessSupervisor`,
`ProcessResult`, `build_subprocess_environment`,
`NativeBackendDoctor`,
`StructuredKernelMeasurementAdapter`, `KernelTemplateMaterializer`,
`DockerTemplateImageSourceRuntime`, and their immutable receipts/identities.

## Reviewed template materialization

`KernelTemplateMaterializer` is the only image-kernel template entry. It accepts
an exact manifest already admitted by the packaged intake registry, requires
`reviewed` status with no blockers, inspects a digest-pinned Docker image, and
copies the declared in-image source from a stopped container. It does not start
the image or acquire a GPU. The copied tree rejects symlinks/hardlinks and must
match the manifest's full byte/mode digest before protected Apex evaluator files
are added.

The materializer creates an isolated Git baseline with a synthetic
`templates.apex.invalid` origin, writes an external
`apex.kernel-template-materialization/v1` receipt, and constructs an internal
authority-bound TaskSpec. Serialized authority cannot be parsed back through
the user TaskSpec path. A partial copy, image/source mismatch, evaluator overlap,
or stale output is removed before returning a typed failure. Current checked-in
templates are `pending`, so this code path stops before Docker.

## Native coding sessions

`NativeCodingSessionLauncher` is the non-formal path used by bare `apex`. It
delegates interactive, text, JSONL, and resume behavior to Codex, Claude, or
Cursor without the formal candidate sandbox, source freeze, turn checkpoint, or
reward pipeline. Backend-native project instructions, approvals, and persistence
remain active. Cross-backend credentials and language/dynamic-loader injection
are still excluded by the common environment builder.
The interactive natural-language `apex optimize kernel` intake also uses this
boundary when a trusted descriptor or target cannot be selected. It forces
kernel enhancement, labels the session as discovery-only, and composes no formal
optimizer or evaluation authorizer. Machine/headless intake never opens it.

Kernel-related Codex and Claude sessions receive an ephemeral command-line MCP
configuration pointing to the local Apex server; user configuration is not
mutated. Non-kernel and `--plain` sessions do not start that server. Cursor has no
equivalent run-scoped MCP flag in the supported surface, so the launcher reports
that typed difference.
The scoped server receives a host grant authority that supports only
`campaign.start`: it may freeze an unverified chat-discovered draft under the
selected results root, but it cannot expose GPU acquisition or evaluator tools.
Kernel prompts explicitly activate the packaged optimization skill, preserve the
original user request verbatim as a labeled suffix, and direct the backend to the
tool's declared schema instead of backend session history. The `campaign.start`
schema names every required TaskSpec field, fixed-argv command, and optional
structured measurement contract; `workspace` and `results_dir` are deliberately
absent because the host injects those scoped roots.
The result also includes a host-rendered `formal_continuation.argv_template`
with absolute campaign/workspace/results paths, the exact draft digest, and the
release receipt argument. The backend returns it unchanged and does not search
the workspace or session history for CLI syntax. A missing or invalid baseline
keeps this template `ready=false` with a typed blocker.
Codex receives a session-local MCP allowlist for exactly `campaign.start` and
the two inert knowledge tools. Only `campaign.start` has tool-specific
`approval_mode="approve"`, so headless draft creation cannot be canceled by an
impossible prompt; all other MCP tools default to prompt, and shell/file approval
plus the normal sandbox are unchanged. Apex never sets global approval to
`never` or uses the dangerous approval-and-sandbox bypass.
Claude receives the equivalent exact `--allowedTools` list for these three Apex
tools; this does not pre-approve shell, edit, or any other native tool. The host
grant itself is consumed after one `campaign.start`, so a backend cannot create
multiple drafts in one enhanced MCP server process even if it retries the
auto-approved call. A later native CLI resume starts a new scoped server and is
therefore a new grant boundary.
Measured work resumes through the host-owned `apex optimize kernel --campaign`
CLI after the native backend exits and the user confirms the exact draft digest.
Kernel-related sessions also receive the same integrity-checked, instruction-only
`amd-kernel-optimization`, `amd-kernel-debugging`, and
`amd-hip-kernel-optimization` skills. The HIP-specific method is an attributed
synthesis of reviewed AgentKernelArena hip2hip tasks; it copies no runner or
evaluator and does not make generic HIP execution available. Codex receives
session-local `skills.config` paths; Claude and Cursor receive the packaged local
plugin through `--plugin-dir`. The skills guide typed capability selection and
evidence boundaries, contain no executable scripts, and cannot award reward or
provide a sanitizer runtime. Non-kernel and `--plain` sessions mount neither the
skills nor MCP. Cursor therefore retains kernel methodology while explicitly
lacking the MCP tool bridge.
The ephemeral command also fixes the current workspace and either the caller's
`--results` root or the stable hidden sibling
`.WORKSPACE_NAME.apex-capability-results`; tools cannot replace
those roots through their input schema.

`NativeBackendDoctor` performs a separate read-only preflight. For exactly one
selected backend it resolves and hashes the CLI entrypoint and runs fixed bounded
`--version` and native authentication-status argv. Authentication requires
recognized backend-specific evidence; exit status alone cannot turn an unsupported
status command into “authentication required.” Feature entries are a
`launcher_contract_only` inventory after CLI/auth prerequisites, not proof that a
live interactive/headless/resume/tool/approval/cleanup probe succeeded. It never
records status-command output or a credential value. Missing identity, ambiguous
auth evidence, or Cursor MCP/effort gaps remain explicit machine states.

## Kernel measurement adapter

The production adapter runs the descriptor's protected measurement runner in a
private PID namespace with a bounded, fail-closed runtime environment. Stdout
must be exactly one strict `apex.kernel-measurement/v1` JSON object with the
frozen method digest. After the namespace is proven empty, the parent process
canonicalizes that object and exclusively publishes it into the
controller-owned private output directory. The runner receives no report path;
malformed, duplicate-key, nonfinite, truncated, timed-out, nonzero-exit, or
incompletely contained output fails closed. Statistics and reward remain owned
by `apex.evaluation`, not this adapter.

## Magpie kernel diagnostics adapter

Formal kernel CLI runs compose the exact dependency-receipt Magpie revision and
invoke its public `compare` command only after Apex has committed the protected
raw measurement and reward. Magpie receives disposable baseline/candidate
projections, fixed compile/correctness/profile argv, and an evaluator-private
output root. Apex seals Magpie's unchanged report, generated config, dependency
lock/commit, and process-containment receipt as `diagnostic` CAS evidence with
`reward_eligible=false`. The adapter also supports one-candidate `analyze` for
scoped callers, while the optimization loop uses paired `compare` evidence.

This reuses Magpie's structured correctness result, comparative ranking,
rocprof-compute/Metrix metrics, and GPU-monitoring fields without making them a
second grade. Magpie hardware control, GPU selection, Ray scheduling, and its
score/winner do not replace Apex GPU leases, robust ABBA timing, promotion
gates, or `kernel_robust_v1` reward authority. Accordo can be admitted later
through a reviewed correctness contract when standalone HIP execution is
available; it is not inferred from a caller command today.
Its V1 identity is `apex-structured-kernel-v1`; the matching immutable method
digest is
`4bb99ecf991a6d28448f46c071bc3c09fbe91aba2cf5f5194e3c1928d96990c1`.

## Structured transcripts

Each backend requests a JSON stream and preserves its JSON objects as bounded raw
events. The parser separately emits provider-neutral `agent_message`,
`tool_called`, and `tool_result` entries. Token totals come only from an explicit
structured `usage` mapping; the last top-level summary wins over intermediate
message usage to avoid double counting. Turn and tool counts use explicit fields
when present and otherwise count structured turn/message/tool events. Cost exists
only when a JSON field such as `total_cost_usd`, `cost_usd`, or a typed
`cost.amount/currency` object is present. Non-JSON stdout, malformed lines,
assistant claims about tokens, and stderr cannot create usage or cost.

`agent_transcript_document` produces the canonical
`apex.agent-transcript/v3` projection containing the source JSON events,
normalized semantic events, requested model/effort, usage, and cost. Raw stdout/stderr remain separate
diagnostic artifacts.

Every production result also embeds an `apex.agent-invocation/v4` receipt. It
records the discovered and resolved CLI entrypoint, SHA-256 of those exact
entrypoint bytes, the CLI's bounded `--version` output, actual argv, prompt
transport, requested editable files, turn policy, and explicit isolation modes.
When a formal caller supplies a sealed backend-runtime-closure digest, the exact
digest is copied from the immutable request into this receipt so an outer
evaluator can reject cross-runtime lineage drift.
It also binds `private_pid_namespace_init_pidfd_v1` as the non-configurable
process-containment policy. Runtime proof is separate from invocation intent and
uses `apex.agent-process-containment/v1`.
The exact receipt is stored in the transcript CAS artifact and projected into
the run's `agent_completed`/`agent_failed` event. `allowed_files_enforced_by_cli=false` is
intentional: current CLIs only provide workspace-level isolation; Apex freezes
and rejects undeclared changes after exit instead of claiming a path allowlist
that the CLI cannot enforce.

Codex and Claude receive their explicit effort through supported CLI options.
Cursor has no independent effort switch in its CLI, so a non-null Cursor effort
fails preflight with `agent_effort_unsupported`; it is never recorded as if applied.

## CLI isolation and turn budget

Codex runs with `workspace-write`, strict `approval_policy="never"`, ignored user
config and exec-policy rules, and an ephemeral session. Claude runs in `--bare`
and `--safe-mode`, disables slash commands, loads only an explicit empty strict
MCP configuration, uses noninteractive `dontAsk` permissions, and does not
persist the session. Cursor runs with its explicit sandbox enabled and without
`--force` or automatic MCP approval. Cursor still needs explicit headless
workspace trust and exposes no flag that disables every project configuration;
the receipt therefore says `config_sources=backend_default_may_load` rather
than asserting stronger isolation.

`max_turns` is enforced while stdout JSONL is drained under
`structured_agent_turn_checkpoint_v2`. One complete structured assistant
decision consumes one turn; a standalone structured tool request consumes one
when the backend does not wrap it in an assistant message. The observer kills
the PID namespace init through its pidfd as soon as turn `max_turns` is observed,
including a final assistant message, so a valid run never starts turn
`max_turns + 1`. Explicit
provider summaries above the limit are typed `turn_overrun` and rejected.

`AgentResult` separates termination from capture. Termination is one of
`completed`, `exact_turn_boundary`, `timeout`, `invalid_stream`,
`turn_overrun`, or `process_failed`; capture is `complete`,
`output_truncated`, or `cleanup_failed`. The supervisor drains both pipes and
requires a verified empty PID namespace before it returns agent results. Only a
normal exit-zero completion or an exact-boundary result whose invocation uses
the containment policy, whose observed count equals the requested count, and
whose capture is complete may cross the source-freeze boundary. A malformed
JSON object, missing structured evidence, timeout, overrun, truncated output,
or unverified cleanup fails closed. Non-JSON diagnostic lines may coexist with
valid events but cannot satisfy the turn proof.

The boundary line itself is included in formal stdout. Any stdout already
buffered after that line is still drained so the child cannot block, but it is
excluded from parsing and the formal transcript. Apex records the discarded
tail's line count, byte count, and SHA-256 in termination evidence; later
buffered events therefore cannot appear as a hidden turn 51 in an exact-50
transcript.

Before the agent command is released, bubblewrap reports its namespace PID 1
and namespace inodes, and the supervisor binds that identity to a start time and
pidfd. The JSON status can precede bubblewrap's mount setup, so it is not itself
a readiness signal. Within one bounded launch deadline Apex repeatedly checks
the pidfd, start time, parent/inner PID mapping, and PID/mount/IPC/user namespace
identity. It opens the target's actually visible `/proc` and uses that file
descriptor's `mnt_id` to select the topmost mountinfo record; inherited or shared
host procfs, propagation-enabled mounts, and proc superblocks matching the
supervisor's `/proc` are rejected. The gate is released only after a second
unchanged identity snapshot and a private `/proc/1` view.

Exact-turn and timeout paths send `SIGKILL` to that exact pidfd; Linux
then kills every namespace member, including `setsid`, double-forked, and
environment-cleared descendants. Natural exit uses the same kernel semantics.
Both paths require pidfd readiness, wrapper/status-FD completion, a complete
namespace-membership scan, and zero live members before source capture. A
process-group scan is not accepted as formal proof. Apex retains a read-only
directory descriptor for the verified private procfs while the namespace is
alive and enumerates that same procfs after teardown. A permission or I/O error
makes `namespace_membership_scan_complete=false`; it can never be reported as a
successful empty scan. If the stdout observer reaches its exact boundary while
the wrapper is exiting, Apex drains the stdout decision before freezing the
termination reason. The reason remains `stdout_budget_boundary`, while
`teardown_mode` independently records whether pidfd `SIGKILL` was sent or the
namespace had already exited naturally.

The containment mount creates a private `/proc`, unshares user and IPC
namespaces, and rebuilds Docker's masked/read-only system paths. Therefore an
AgentKernelArena outer wrapper for the Apex arm must not pre-install `/proc`
submount masks: Apex owns the immediate private procfs and masks, after which a
backend managed sandbox may safely fall back to that procfs. Missing bubblewrap,
pidfd support, private procfs, or identity evidence fails closed before candidate
execution.

The v3 transcript records the typed termination kind/reason, capture status,
derived `candidate_capture_allowed` decision, exact observed/requested turns,
and policy identity. A controlled exact-boundary capture is recorded as
`agent_completed` with its nonzero process exit and boundary reason still
visible; this means the candidate-production phase completed with freezeable
bytes, not that agent text became trusted evidence.

The invocation receipt also states
`response_token_limit=not_supported_context_advisory_only`. None of the three
locally supported CLI surfaces provides a portable output-token limit; the
ContextPacket's response allocation is therefore not represented as an
execution cap.

Every formal backend invocation uses stdin for the complete prompt. The
`apex.agent-invocation/v4` receipt binds an
`apex.agent-execution-authority/v1` permission receipt to the exact run,
attempt, backend, writable projection, editable files, requested environment
key names, and parent evaluation/controller receipt. Missing or mismatched
authority fails before the CLI version probe or agent process. The receipt names
the backend credential environment key and redaction policy, but never its
value. Ordinary user-owned interactive sessions remain a separate native
backend surface and may use the backend's native argv UX.

## Process environments

Subprocesses never receive a copy of `os.environ`. The shared builder inherits a
small named set for executable discovery, user/config/cache locations, locale,
certificate/proxy routing, and explicitly selected GPU, Hugging Face, or Docker
runtime fields. Normal caller-provided variables are accepted explicitly and
bounded by count and size. `BASH_ENV`, `ENV`, dynamic-loader injection, language
startup injection (including `PYTHON*`), and credential-shaped variables fail
closed. `PYTHONNOUSERSITE=1` is adapter-owned.

Agent adapters opt in to exactly one ambient credential: Codex receives only
`OPENAI_API_KEY`, Claude only `ANTHROPIC_API_KEY`, and Cursor only
`CURSOR_API_KEY`. Cross-backend credentials and unrelated host secrets are not
inherited. An explicit request may override only that same backend credential.
Formal stdout/stderr and structured events are exact-value scrubbed before an
`AgentResult` exists. A credential echo produces
`capture_status=credential_redacted`, preserves only the replacement count,
and cannot yield a candidate or reward. Credential text in a CLI identity probe
fails provenance closed, so neither invocation receipts nor CAS artifacts can
retain it.

Tests: `pytest tests/unit/execution tests/contract -q`.

## Purpose

Execution contains stateless Codex, Claude, Cursor, structured kernel
measurement, and digest-pinned template materialization adapters plus one
bounded subprocess supervisor for trusted fixed-argv commands.

## Public API

Consumers use `AgentRegistry`, `build_default_registry`, `SubprocessSupervisor`,
`ProcessResult`, the canonical transcript projection, and the fail-closed
environment builder; vendor implementations remain adapter details.

## Invariants

Requests are immutable, argv is never shell text, time/output limits are explicit,
backend output cannot directly mutate domain state, and subprocess environment
inheritance is explicit rather than ambient.

## Dependencies

Execution depends only on core, intake contracts, and ports. It never imports
optimization, evaluation, storage, benchmark, or CLI packages.

## Failure semantics

Missing executable, timeout, invalid stream, exact-boundary stop, turn overrun,
nonzero exit, truncated output, cleanup failure, or invalid candidate is
returned as typed process/agent evidence rather than inferred success.

## Tests

Unit tests use fake executables to cover argv, timeout, limits, transcript
capture, backend selection, environment injection rejection, credential
isolation, Codex/Claude-shaped structured streams, summary de-duplication, and
deterministic error mapping. CPU process tests prove exact-turn and natural-exit
teardown defeat a `setsid` + double-fork + `clearenv` delayed writer. Deterministic
containment tests cover the status/mount readiness race, visible topmost procfs
selection, identity changes, incomplete membership scans, and the observer/wrapper
exit race.
Template tests use a fake image runtime plus real local Git to prove source-tree
binding, baseline cleanliness, non-replayable authority, cleanup, and that a
pending manifest never calls Docker.
Doctor tests use bounded fake probes to cover exact argv, entrypoint identity,
credential/output redaction, authentication-required state, and backend feature
differences without contacting a model service.

## Provenance

Results record backend/model/effort identity, entrypoint-byte/argv/isolation
receipts, bounded structured transcripts, and source-indexed usage/cost;
environment credentials are never copied into transcript artifacts.
