# End-to-end kernel optimization

## Purpose

This package optimizes source kernels inside an unchanged Magpie workload. It owns
the full closed loop: profiler-off baseline, profiler-on diagnosis, dynamic source
opportunity selection, one fresh bounded agent invocation per candidate,
independent qualification, source deployment, current-anchor E2E A/B, KEEP or
REVERT, and reprofile/replan. It does not optimize serving flags, benchmark fields,
model parameters, or workload shape.

## Public API

`E2EOptimizeUseCase` is the composition entry. `KernelOpportunityPlan` is the
auditable dynamic plan, and `E2EOptimizationResult` is the terminal machine result.
Concrete adapters implement the typed ports in `services.py` and the
`CandidateWorker` protocol in `candidate.py`.

Production composition uses `AgentCandidateWorker` with the default Codex-first
registry, `E2EDeferredMicroQualifier` when no trusted raw-sample micro harness is
available, and `DockerOverlayDeployment` for runtime-only vLLM/AITER experiments.
The latter builds from the inspected parent image ID, changes one installed
Python file, proves the loaded bytes in a clean container, and derives Magpie
configs whose sole workload change is `benchmark.docker_image`.

Formal delivery is an explicit reviewed capability, never an inference from a
runtime overlay. `SourceRebuildFinalDelivery` accepts only a
controller-owned `FormalSourceDeliveryProfile` for the exact immutable parent
image and exact changed repository set. A profile binds vLLM/AITER source URLs,
editable allowlists, dependency order, licensing metadata, and a fixed-argv
`BuildRecipeLock`. `FormalDeliveryBinding` also supplies two independent trusted
boundaries: a `PrimarySourceBuildPort` for the first clean source build,
loaded-byte/quality/performance validation and SBOM, and an `E2EBundleVerifier`
whose separate backends perform the second clone, rebuild, engagement probe, and
unchanged replay. The production default selects this binding only for the exact
reviewed Qwen3-Next 80B FP8 Magpie acceptance config, parent image, model revision,
and vLLM/AITER source locks; every identity drift fails closed. Other workloads
remain ineligible until they receive their own reviewed binding.
For that one config, composition injects the reviewed model revision and local
source-lock paths before provenance resolution; conflicting user hints are
rejected rather than silently replaced.

`deployment_hints.hf_cache_path` may bind an existing absolute Hugging Face
cache. Set `deployment_hints.hf_offline: true` as a YAML boolean to freeze
`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, and `HF_DATASETS_OFFLINE` in every
benchmark view; string or numeric coercions are rejected, and offline mode is
invalid without the verified cache path.

Internal responsibilities are intentionally narrow:

- `benchmarking.py` binds measurements and TraceLens diagnostics to journal/CAS.
- `kernel_lane.py` turns measured evidence into source-only opportunities.
- `oracles.py` resolves version-locked source paths to reviewed correctness tests.
- `context.py` compiles a fresh bounded `ContextPacket` from durable state.
- `candidate.py` materializes and freezes an isolated source checkout.
- `deferred.py` represents the no-micro-harness truth boundary without reward.
- `overlay_runtime.py` owns fixed-argv Docker inspection, build, and byte probes.
- `overlay_config.py` derives immutable image-only benchmark views.
- `docker_overlay.py` binds source locks, runtime engagement, and overlay receipts.
- `source_delivery_models.py` defines trusted repository/profile and primary-build ports.
- `source_delivery_provenance.py` rejects incomplete model, agent, policy, or image identity.
- `source_delivery.py` accumulates accepted bytes and drives bundle plus independent replay.
- `source_image_runtime.py` builds a reproducible source-baked image, SPDX SBOM, and byte probes.
- `source_image_sbom.py` emits the deterministic file-level SPDX 2.3 inventory.
- `source_delivery_adapters.py` owns primary measurement and independent rebuild/replay adapters.
- `qwen_profile.py` binds the one reviewed Qwen acceptance config and immutable source/image locks.
- `search.py` consumes the bounded queue and performs KEEP/REVERT decisions.
- `finalization.py` requires source delivery and a second clean replay.
- `run_record.py` is the only E2E evidence/journal facade.
- `result.py` owns the terminal result schema and atomic write.

## Invariants

Only regular Python/Triton source inside a resolved root and backed by an
independent correctness oracle enters the candidate lane. Config-only proposals
cannot be constructed. Each agent process is stateless and sees only one bounded
packet; durable decisions, receipts, dead ends, and the current anchor replace
conversation memory.

Correctness-oracle routing never preselects profiler symbols. The diagnostic rank
remains dynamic; after a source has been resolved into an exact source-lock root,
`CorrectnessOracleRegistry` matches only its repository-relative source path to a
reviewed in-tree test. The policy and individual binding SHA-256 values are bound
into the run accuracy contract and protected harness context. A root mismatch,
missing test, symlink, hard link, path escape, or conflicting Magpie test mapping
fails closed. For the reviewed Qwen profile this makes paged attention, recurrent
GDN decode, causal-conv update, prefix-prefill, and reshape/cache sources eligible
without hardcoding their dynamic rank or runtime symbol.

After every candidate worker exits, `run_record.py` writes one canonical
`apex.agent-transcript/v2` CAS artifact and projects its normalized actions into
attempt-scoped `agent_message`, `tool_called`, and `tool_result` journal events.
Structured usage and explicit provider cost become separate `usage_recorded` and
`cost_recorded` events before `agent_completed`/`agent_failed`; all are marked
`self_reported` and carry source-event indexes plus the transcript receipt. They
are training/cost provenance, never evaluator correctness or performance proof.

Candidate production uses `structured_agent_turn_checkpoint_v2`. A source-changing
candidate stopped exactly at the requested turn count may be frozen only when the
typed capture is complete, `sigstop_process_group_snapshot_v1` verifies group
quiescence, and same-process-group cleanup is verified. Its
`agent_completed` event still records `termination_kind=exact_turn_boundary` and
the controlled process exit. A count below the limit is valid only as a natural
completed process; a count above it, timeout, invalid stream, truncated output,
or cleanup failure is rejected. Every admitted checkpoint then follows the same
micro/deferred qualification, safety, deployment, and E2E acceptance path.
The E2E workspace purge removes interpreter and test caches before source
fingerprinting; ignored artifacts are neither candidate bytes nor evaluator input.

Strict micro qualification accepts one canonical `KernelGrade`; it has no second
statistics or threshold implementation. KEEP therefore requires compile,
correctness and integrity plus valid p50/p99 evidence, at least 300 raw samples,
strict `Srobust > 1.05`, a passing seeded paired-bootstrap confidence bound, the
frozen CV limit, and no worst-case regression. Exact `1.05` is not KEEP. Where no
trusted micro harness exists, the typed `e2e_quality_deferred` mode carries no
grade and may assert only frozen source integrity—no compile, correctness, timing,
or kernel reward—while later unchanged Magpie quality/E2E evidence remains
authoritative. Its exact order is freeze/integrity, evaluator safety policy,
isolated immutable deployment with loaded-byte proof, and unchanged Magpie
quality plus normal performance. This is deliberately not presented as strict
micro ordering or as a kernel-level correctness/reward result.

KEEP is evaluated against the current live anchor, never the original baseline.
REVERT rolls back the candidate deployment. A KEEP forces a fresh diagnostic pass
before another opportunity is selected. Formal success is impossible until a
source-rebuilt bundle passes engagement verification and a second clean replay.

The cumulative source patch is recreated from exact clean Git locks, in KEEP
order, in disposable clones. Repeated edits to one file use the last accepted
full source bytes, matching immutable image overlay semantics; edits to distinct
files/repositories accumulate. The caller checkout and host site-packages are
never modified. A config-only result cannot enter bundle construction.

The provenance lock uses actual, consistent `AgentResult.backend/model` values,
the exact frozen quality/regression policy hash, the exact
`E2EAcceptancePolicy` digest, and the configured safety-policy fingerprint. A
backend default whose actual model is unknown, or a custom safety adapter without
a policy fingerprint, is intentionally ineligible for formal success.

The GPU lease spans the entire run. It is cooperative and fail-fast; this package
does not discover or kill unrelated processes.

## Dependencies

Benchmark execution and diagnostic analysis enter through ports. Context assembly
depends on `apex.context` and the curated `apex.knowledge` retrieval API. Control
state is owned by `apex.orchestration`; evidence bytes are owned by `apex.storage`.
Source materialization uses Git without initializing or traversing submodules.
Safety and final delivery remain evaluator-owned boundaries.
Docker overlays require one exact clean vLLM/AITER source lock whose repository
bytes match the installed parent image. They do not mount or modify host
site-packages, and rollback merely selects the previous immutable config/image.

## Failure semantics

Baseline or quality failure terminates as `baseline_invalid`. Missing source,
micro, or deployment capability is reported as unsupported; compile, correctness,
safety, build, engagement, or E2E failures reject/revert only the active candidate
until its bounded retry budget is exhausted. Infrastructure exceptions fail the
run without promoting an anchor. Missing source provenance or second-clean-replay
proof can retain primary evidence but cannot produce formal success.

Missing exact source/model/image identity is `provenance_unresolved`. An exact
source stack without a matching trusted fixed recipe, attestor, engagement
backend, or replay backend is `verification_failed`. Primary output must contain
an immutable derived image ID, matching SBOM, build/engagement/benchmark/safety
receipts, normal-runtime gates, and overlay-to-rebuild parity. Any false or
missing gate stops before a final bundle. Only the independently finalized bundle
returns `succeeded/source_rebuild_verified`.

The terminal result labels the initial observation as
`intake_provenance_status` and preserves `intake_missing_evidence`; those facts
are not retroactively rewritten when later loaded-byte checks succeed. The
separate `formal_delivery_verified` field becomes true only for
`succeeded/source_rebuild_verified`, so a truthful partial intake cannot be
mistaken for an unverified final delivery.

Safe in-root symlinks retain their identity. Escaping symlinks, hard links,
unmerged Git paths, edits outside the declared source, materialized gitlink
content, changed file modes, stale generations, and workload-semantic changes fail
closed.

## Tests

CPU-only tests cover dynamic eligibility, config exclusion, safe symlink/gitlink
handling, 300-sample enforcement, retry and fresh-context history, safety blocking,
deployment failure, current-overlay chaining, KEEP/REVERT rollback, GPU lease
scope/contention, attempt-scoped message/tool/usage/cost lineage, final provenance
failure, crash recovery, and the mandatory second clean replay.

Run the focused suite with:

```bash
pytest -q -p no:cacheprovider \
  tests/unit/test_e2e_kernel_lane.py \
  tests/unit/optimization/test_e2e_candidate_workspace.py \
  tests/unit/optimization/test_e2e_production_adapters.py \
  tests/unit/optimization/test_e2e_source_delivery.py \
  tests/unit/orchestration/test_e2e_state_machine.py \
  tests/unit/runtime/test_gpu_lease.py \
  tests/integration/test_e2e_optimize_use_case.py \
  tests/integration/test_e2e_candidate_loop.py
```

## Provenance

Every baseline, diagnostic, context packet, prompt, agent transcript, frozen source,
micro/safety/delivery result, E2E verdict, and final bundle result is content-addressed
and linked from the append-only run journal. State binds provenance and policy
hashes, anchor/state generations, candidate source digests, opportunity identity,
benchmark semantics, and agent transcript events bind their exact structured
source indexes. Runtime-overlay evidence is explicitly weaker than
`SOURCE_REBUILD_VERIFIED` and is never relabeled as formal delivery.

No implementation in this package was copied from GEAK, HyperLoom, Magpie, or
TraceLens. Their externally integrated evidence/knowledge remains identified by
its own receipts and upstream provenance metadata.

The reviewed Qwen binding builds a canonical source layer under `/opt/apex` on
the immutable parent image. A deterministic import finder makes those exact
vLLM/AITER Python bytes authoritative while retaining the parent's generated
metadata and compiled extensions. The primary and independent builders use the
same network-disabled fixed-argv recipe but separate worktrees and output roots;
formal success requires identical image and SBOM digests, actual imported-byte
receipts, and a second unchanged Magpie replay. No host checkout or site-packages
tree is mutated. Apex may still report runtime-overlay evidence for unreviewed
images, but it cannot relabel that evidence as formal source-rebuild success.
