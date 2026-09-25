# MCP façade

## Purpose

`apex.mcp` projects the canonical run-scoped capability registry as a local
stdio MCP server. It is a presentation adapter, not a second optimizer.

## Public API

Use `CapabilityRegistry`, `CapabilityScope`, the knowledge and diagnostics
descriptors/handlers, `WorkloadInspectHandler`, `BundleVerifyHandler`,
`CampaignStatusHandler`, `CampaignCheckpointHandler`,
`KernelEvaluatorHandler`, `MagpieAcquisitionHandler`, `TraceCompareHandler`,
`build_low_level_server`, and
`run_stdio_server` through `apex.mcp`. `planned_capability_descriptors()` adds
the typed, model-neutral contract inventory for skills, benchmark/profile,
analysis, evaluator, campaign, and delivery surfaces. Entries without an
executable handler remain honest unavailable inventory rows.

## Invariants

One descriptor defines each tool schema, side effects, authority, GPU need,
timeout, artifact class, and reward role. Only available tools are presented.
Instruction-only skills are available registry presentations mounted by the
native launcher and are deliberately excluded from MCP `tools/list`; trying to
invoke one through the registry fails with `capability_not_callable`.
Planned descriptors remain visible through `apex capabilities` with
`available=false` and `capability_not_implemented`; they are omitted from MCP
`list_tools`, so an agent cannot call a placeholder. `kernel.sanitize` is absent
from both inventories by design because this release has no sanitizer runtime.
Tool output cannot acquire evaluator or reward authority through MCP. The server
never infers authority from a registered handler and never grants authority to
itself. Every call with a side effect, authority, GPU need, or evidence/reward
role consumes an externally minted, one-shot `apex.capability-grant/v1` receipt.
That receipt binds the MCP session, exact descriptor and arguments, authority,
side effects, allowed GPU devices, timeout, artifact classes, reward role, and
cost ceiling. A replay, drifted argument, insufficient pre-dispatch cost ceiling,
or result cost above that ceiling fails closed. Without an injected grant
authority, or when the injected authority does not support a descriptor's exact
role, the production stdio server omits grant-requiring entries from
`tools/list`; the canonical CLI inventory still reports their implementation
availability, while MCP exposes only inert, authority-free, GPU-free,
reward-ineligible tools. An echoed draft digest
is only an equality check: formal execution also consumes its draft-bound
evaluator receipt outside the coding backend channel.
Artifact-producing handlers receive one fixed workspace and results root from
the launcher. `trace.analyze` accepts only workspace-relative, non-symlink
inputs, creates one immutable run output, hashes every returned artifact, and
is always reward-ineligible. It normalizes an existing Magpie diagnostic
workspace; it never starts a profiler or GPU command.
`benchmark.run` and `profile.capture` are the explicit acquisition exceptions.
They accept only a workspace-scoped raw Magpie config, lazily verify the pinned
dependency set, acquire a physical GPU lease, and run exactly one immutable
normal or config-declared diagnostic pass. Normal acquisition is evidence-only;
diagnostic acquisition is reward-ineligible. Both return scope-relative hashes,
the complete ownership/lease receipt, and no grade, KEEP/REVERT decision, or
performance claim. Inventory construction never verifies dependencies or probes
a GPU.
`workload.inspect` accepts one non-symlink, workspace-relative raw Magpie YAML,
verifies pinned dependencies only when invoked, and builds an Apex-owned
projection through published Magpie main APIs. Compatible configs emit immutable
original, measurement, diagnostic, and replay views based on the Apex scoring
policy. A capability
upgrade emits only the source-config receipt and exact blockers, with
`view_status=capability_upgrade_required` and no fabricated executable or reward
view. It reports unresolved live-runtime receipts without starting a benchmark
or GPU.
`experience.retrieve` accepts a results-relative canonical run plus the complete
task/operator/GPU/framework/version/shape/source/harness/policy identity. It opens
the existing event journal read-only, verifies checksums/transactions, projects
`experience.measured`, and returns exact-identity matches only. It cannot fuzzy
match, append state, read backend chat history, or turn self-reported/dry-run
events into measured experience.
`hotspot.rank` is a thin projection over the canonical `trace_evidence.json`
already emitted by `trace.analyze`. The caller supplies its exact SHA-256; the
handler requires canonical JSON and the typed measured/recoverable ranking
shape, then applies only an output limit. It never recomputes a grade, reads a
raw profiler file, or treats modeled headroom as measured speedup.
`trace.compare` accepts two explicit results-scoped CAS roots plus their typed
diagnostic artifact receipts, verifies every input in the pinned TraceLens
adapter, and writes bounded comparison tables under one immutable capability
output. Dependency verification is lazy. The projection removes the host
dependency path, retains the pinned commit/API hash and output receipts, and is
always reward-ineligible; full attribution remains explicitly unavailable until
TraceLens exposes that stable contract.
`bundle.verify` resolves exactly one non-symlink bundle inside the launcher-owned
workspace/results scope, requires workspace-user authority, and invokes the
official kernel or E2E static loader. Its declared side effects are read-only.
It returns digest, kind, and typed bundle identity only; it does not apply,
rebuild, measure, write a result, or award reward. `campaign.status` likewise
declares only `read_results`: it opens one results-scoped canonical run, verifies
the event journal and CAS layout, replays workload state, materializes the parent
plus all attempts, and returns status/reward/trainability plus current phase,
anchor/generation, pending action, and bounded E2E stage/budget projections. It
never appends an event and does not treat backend-native session history as
campaign memory. `campaign.start`, `campaign.stop`, `campaign.checkpoint`, and
`campaign.resume` do not share one blanket status. `campaign.checkpoint` is implemented: it
verifies and replays the canonical journal, rebuilds only the disposable state
snapshot, then proves journal events remained byte-for-byte unchanged.
`campaign.start` is also implemented for standalone kernels: an agent submits
its discovered editable scope and fixed command/measurement draft directly as
typed data, so no YAML file is required. Its MCP JSON Schema enumerates the
Python/Triton language boundary, required source scope, three fixed-argv command
phases, and optional canonical raw-measurement runner. It rejects invented
nested targets, shell-string commands, caller-supplied workspace/results roots,
and standalone HIP before the handler runs. The optimization use case resolves and
hashes the workspace, records an unverified Evaluation Contract plus the observed
Apex execution identity in a new journal/CAS, and returns the exact draft digest
plus a results-scoped candidate projection. It also returns a host-generated `formal_continuation.argv_template`
containing every required CLI argument. Agents must relay that argv unchanged
after the chat exits instead of searching files or inventing CLI syntax. It never invokes an agent,
acquires a GPU, trusts self-declared evaluator authority, or awards reward;
trusted local confirmation and the formal composition boundary still own execution. E2E remains
the explicit long-running `apex optimize e2e` workflow. An explicit
kernel-enhanced native session receives a one-shot authority that supports only
`campaign.start`; it cannot expose acquisition, compile, correctness,
measurement, grade, stop, resume, or bundle mutations. The supported measured
continuation is the host CLI's `apex optimize kernel --campaign ...` path after
the native backend exits. A second start attempt in the same MCP server is
rejected as `capability_grant_replayed`, even when the backend requests a fresh
argument set. `campaign.stop` applies
only to these standalone formal campaigns. It closes pending work, records
REVERT for an unselected verified candidate, and derives one evidence-bound
terminal reward or an explicit untrainable/null result; replaying it adds no
events. `campaign.resume` is a
thin route back into that same E2E `resume()` use case: it accepts only a
results-scoped run, recomputes the current Apex execution identity, lazily verifies
dependencies, then preserves the
same provenance and GPU preflight as the formal CLI. It does not implement a
second recovery loop or reuse backend chat state.

The standalone `kernel.compile`, `kernel.correctness`, `kernel.measure`,
`kernel.grade`, and `bundle.build` surfaces are implemented over that draft.
Compile verifies that the current Apex execution identity matches the draft,
requires the original target Git checkout to remain clean, consumes injected one-shot authority,
freezes allowlisted edits from the persistent results-scoped candidate tree into
CAS, and runs in a fresh evaluator projection under a GPU lease. Correctness and
measurement reconstruct those same bytes from CAS.
Measurement first records the explicit `no_tools/not_configured` safety result,
runs normal performance, and seals raw timing plus execution receipts without a
grade or reward. Grade independently reloads the raw report and is the only tool
that can append a measurement reward; absent authority/evidence returns
`unverified|no_measurement`. Bundle build accepts only a verified improving
attempt and emits an immutable unapplied source bundle plus terminal task reward.
Every mutating formal capability also seals canonical typed argument/result JSON
as paired `tool_called`/`tool_result` events. Identical completed calls are
replayed after terminalization, an interrupted open call is resumed, and a new
active retry receives a distinct call ID. Read-only status remains non-mutating.

## Dependencies

The façade depends on lower-level ports, knowledge, reporting, orchestration,
and core contracts. The
official MCP SDK is imported only when a server is built or run. It imports no
CLI, bootstrap, or backend implementation. Formal draft persistence is delegated
to an injected kernel optimization use case; the façade does not reproduce that
policy. Its only direct storage write is rebuilding a disposable snapshot from
verified canonical events.
Workload inspection additionally consumes the public benchmark view builder and
a dependency-receipt provider injected by the composition root. The provider is
not called during application or MCP inventory construction.

## Failure semantics

Unknown, unavailable, unauthorized, ungranted, replayed, over-budget, ambiguous-path, schema-invalid,
identity-mismatched, or
reward-role-mismatched calls fail closed with typed contract errors. A disabled
knowledge catalog remains an honest unavailable result rather than a fallback.
Missing exact dependencies, unsafe config/cache/source paths, incompatible
derived images for local or Ray mode, and immutable output reuse fail before an
inspection receipt is returned.

## Tests

CPU tests cover registry uniqueness, authority, availability, knowledge
attribution, bounded retrieval, scoped path traversal/symlink/output-reuse
failures, diagnostic artifact receipts, and exact MCP schema projection without
opening a socket, GPU, or backend process.
Workload tests also cover lazy dependency verification, local/Ray image
semantics, corpus membership, phase-view hashes, and reward ineligibility.
Delivery tests cover authority, results/workspace ambiguity, official-loader
dispatch, and byte tampering.
Campaign tests cover typed descriptor-free draft creation, missing authority,
absence of GPU/agent/reward work, scoped resume/baseline delegation, scoped
lookup, authority, journal replay, complete attempt projection, snapshot
reconstruction, and canonical-event non-mutation.
Formal evaluator tests cover the explicit confirmation handshake, non-editable
drift rejection, phase recovery, capture-before-grade reward separation,
evaluator-owned robust reward, verified delivery, and absence of a sanitizer
runtime.
Acquisition tests use fake Magpie/GPU ports to cover explicit pass selection,
path scope, lease binding, receipt hashes, and reward ineligibility without
opening a GPU or backend process.
Experience tests cover exact identity, mismatch exclusion, authority/path scope,
read-only journal behavior, and absence of journal-byte mutation.

## Provenance

Knowledge results retain card IDs, source/content hashes, selection policy, and
the `advisory_only` marker. MCP is never evaluator evidence by itself.
Diagnostic receipts retain role, scope-relative locator, byte count, and
SHA-256; they remain observations rather than grading evidence.
Workload receipts bind the dependency lock, Magpie commit/corpus identity, raw
config, phase views, opaque model identity, and workload-semantics digest. They
prove inspection and freezing only, never benchmark success or reward.
