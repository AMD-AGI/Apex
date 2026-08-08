# Execution

`apex.execution` implements the `AgentBackend` port for Codex, Claude Code, and
Cursor Agent. Codex is the registry default; callers can explicitly select the
other two without preflighting unrelated credentials.

All adapters execute argv with `shell=False` through `SubprocessSupervisor`,
drain stdout and stderr concurrently, bound captured output, and run both CLI
identity discovery and candidate production inside an authoritative Linux PID
namespace. They return normalized transcript events and never decide whether a
candidate is correct or fast.

Public API: `AgentRegistry`, `build_default_registry`, `SubprocessSupervisor`,
`ProcessResult`, and `build_subprocess_environment`.

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

Every production result also embeds an `apex.agent-invocation/v3` receipt. It
records the discovered and resolved CLI entrypoint, SHA-256 of those exact
entrypoint bytes, the CLI's bounded `--version` output, actual argv, prompt
transport, requested editable files, turn policy, and explicit isolation modes.
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
pidfd. Exact-turn and timeout paths send `SIGKILL` to that exact pidfd; Linux
then kills every namespace member, including `setsid`, double-forked, and
environment-cleared descendants. Natural exit uses the same kernel semantics.
Both paths require pidfd readiness, wrapper/status-FD completion, a complete
namespace-membership scan, and zero live members before source capture. A
process-group scan is not accepted as formal proof.

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

Tests: `pytest tests/unit/execution tests/contract -q`.

## Purpose

Execution contains stateless Codex, Claude, and Cursor adapters plus one bounded
subprocess supervisor for trusted fixed-argv commands.

## Public API

Consumers use `AgentRegistry`, `build_default_registry`, `SubprocessSupervisor`,
`ProcessResult`, the canonical transcript projection, and the fail-closed
environment builder; vendor implementations remain adapter details.

## Invariants

Requests are immutable, argv is never shell text, time/output limits are explicit,
backend output cannot directly mutate domain state, and subprocess environment
inheritance is explicit rather than ambient.

## Dependencies

Execution depends only on core and agent ports. It never imports optimization,
evaluation, storage, benchmark, or CLI packages.

## Failure semantics

Missing executable, timeout, invalid stream, exact-boundary stop, turn overrun,
nonzero exit, truncated output, cleanup failure, or invalid candidate is
returned as typed process/agent evidence rather than inferred success.

## Tests

Unit tests use fake executables to cover argv, timeout, limits, transcript
capture, backend selection, environment injection rejection, credential
isolation, Codex/Claude-shaped structured streams, summary de-duplication, and
deterministic error mapping. CPU process tests prove exact-turn and natural-exit
teardown defeat a `setsid` + double-fork + `clearenv` delayed writer.

## Provenance

Results record backend/model/effort identity, entrypoint-byte/argv/isolation
receipts, bounded structured transcripts, and source-indexed usage/cost;
environment credentials are never copied into transcript artifacts.
