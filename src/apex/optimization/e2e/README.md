# End-to-end kernel optimization

## Purpose

This package optimizes source kernels inside an unchanged Magpie workload. It owns
the full closed loop: profiler-off baseline, profiler-on diagnosis, dynamic source
opportunity selection, one fresh bounded agent invocation per candidate,
independent qualification, source deployment, matched current-live-anchor E2E
promotion, KEEP or REVERT, and reprofile/replan. It does not optimize serving
flags, benchmark fields, model parameters, or workload shape.

## Public API

`E2EOptimizeUseCase` is the composition entry. `KernelOpportunityPlan` is the
auditable dynamic plan, and `E2EOptimizationResult` is the terminal machine result.
Concrete adapters implement the typed ports in `services.py` and the
`CandidateWorker` protocol in `candidate.py`.

Production composition uses `AgentCandidateWorker` with the default Codex-first
registry, `E2EDeferredMicroQualifier` when no trusted raw-sample micro harness is
available, and `DockerOverlayDeployment` for runtime-only vLLM/AITER experiments.
The reviewed Qwen profile uses `QwenCompositeMicroQualifier`: vLLM routes to
`DockerOracleMicroQualifier`, which executes a small source-relative subset of
the exact locked vLLM tests in an immutable candidate-overlay image, while AITER
routes to frozen-source-only `E2EDeferredMicroQualifier` because no equivalent
reviewed micro oracle exists. Both lanes still continue through evaluator safety
and unchanged Magpie quality/performance gates before promotion.
This is a fail-closed preflight, not a canonical kernel grade; compile,
correctness, timing, and reward remain explicitly unmeasured.
The latter binds the inspected parent image ID to the unique
provenance-approved immutable `repo@sha256` locator from the same inspection,
re-inspects that locator immediately before build, changes one installed Python
file, proves the loaded bytes in a clean container, and derives Magpie configs
whose sole workload change is `benchmark.docker_image`. The measurement,
trace-only diagnostic, and replay views are validated together against one
formal workload hash before any derived YAML is written. For hash comparison
only, the diagnostic view restores the measurement view's `RUN_EVAL=true` and
receipt-pinned lm-eval runtime; its emitted config remains `RUN_EVAL=false` and
contains no evaluator runtime or quality claim. The first layer always uses the
unique provenance-approved `repo@sha256` locator for Dockerfile `FROM`; a bare
initial image ID cannot bypass that rule. After KEEP is atomically committed, the
exact derived image ID may parent the next layer, but only when the request carries
the complete accepted stack and its hashed build receipts prove an unbroken chain
back to that initial locator. Docker receives a content-derived local alias because
BuildKit cannot use a bare local image ID as `FROM`; Apex assigns that alias from
the exact ID and re-inspects it both before and after the build. It never falls back
to a caller-supplied mutable tag.

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
- `benchmark_document.py` serializes the complete normalized benchmark evidence,
  including the verified Magpie serving-runtime receipt.
- `benchmark_artifacts.py` stores the exact input config, normalized result,
  evaluator-owned quality document, raw report, quality results/samples, and side
  evidence as distinct CAS artifacts; no local Magpie workspace is recovery truth.
- `kernel_lane.py` turns measured evidence into source-only opportunities.
- `oracles.py` resolves version-locked source paths to reviewed correctness tests.
- `oracle_preflight.py` owns Qwen tests-only policy, source locks, and qualification.
- `oracle_container.py` owns the immutable overlay and same-process test runner.
- `context.py` compiles a fresh bounded `ContextPacket` from durable state and
  identity-compatible measured experience projected from the same canonical journal.
- `candidate.py` materializes the isolated checkout and owns agent outcome routing.
- `candidate_fingerprint.py` owns bounded tree/Git traversal, source fingerprints,
  and pre-hash entry, depth, file-size, and changed-byte budgets.
- `candidate_snapshot.py` owns bounded source identity, immutable byte capture,
  validation, and evaluator-only read-only materialization without reopening the
  agent workspace.
- `candidate_record.py` persists frozen candidate bytes and their CAS manifest.
- `deferred.py` represents the no-micro-harness truth boundary without reward.
- `overlay_runtime.py` owns fixed-argv Docker inspection, build, and byte probes.
- `overlay_config.py` derives immutable image-only benchmark views.
- `overlay_lineage.py` validates the committed KEEP ancestry and hashed per-layer
  build receipts before an exact derived image ID can become a Docker parent.
- `docker_overlay.py` binds source locks, runtime engagement, and overlay receipts.
- `deployment_artifacts.py` records the three derived measurement, diagnostic, and
  replay configs in CAS against the deployment's typed SHA-256 identities.
- `source_delivery_models.py` defines trusted repository/profile and primary-build ports.
- `source_delivery_provenance.py` rejects incomplete model, agent, policy, or image identity.
- `source_delivery.py` accumulates accepted bytes and drives bundle plus independent replay.
- `source_image_runtime.py` builds a reproducible source-baked image, SPDX SBOM, and byte probes.
- `source_image_sbom.py` emits the deterministic file-level SPDX 2.3 inventory.
- `source_delivery_adapters.py` owns primary measurement and independent rebuild/replay adapters.
- `qwen_profile.py` binds the one reviewed Qwen acceptance config and immutable source/image locks.
- `qwen_qualification.py` routes vLLM and AITER to distinct reviewed truth boundaries.
- `search.py` consumes the bounded queue and closes every non-infrastructure
  attempt with one evaluator decision/reward outcome.
- `promotion.py` runs the counterbalanced `A, B, B, A` promotion window under one
  GPU lease, where A is the current live anchor and B is the candidate, and stores
  one content-addressed matched-pair receipt.
- `promotion_recovery.py` recomputes that receipt from canonical observation
  evidence and rejects role, order, config, image, anchor, or lease substitution.
- `search_support.py` owns immutable attempt records and deployment/runtime
  identity validation; `outcomes.py` owns outcome grading and atomic commit.
- `learning.py` appends the post-decision measured experience and associates each
  selected knowledge card with the decision evidence. Until Apex records a frozen,
  verifiable card-to-action binding, every such card association is
  `inconclusive`, including KEEP and REVERT outcomes.
- `search_recovery.py` reconciles an interrupted active gate or KEEP/reprofile
  boundary without duplicating search policy or an evaluator reward.
- `finalization.py` requires source delivery and a second clean replay.
- `run_record.py` is the only E2E evidence/journal facade and prepares immutable
  decision, grade, policy, and measurement bindings for atomic commit.
- `recovery.py` binds persisted requests, action receipts, and diagnosis history to
  journal/CAS evidence. `recovery_artifacts.py` rebuilds typed candidate, gate,
  deployment, config, quality, and measurement values without agent prose.
  `recovery_bindings.py` cross-checks config/quality/measurement event-to-CAS
  joins. `recovery_search.py` projects the accepted chain, live anchor, active
  configs, current diagnosis, and in-flight attempt from canonical events and CAS.
- `result.py` owns the terminal result schema and atomic write.

## Invariants

Only regular Python/Triton source inside a resolved root and backed by an
independent correctness oracle enters the candidate lane. Config-only proposals
cannot be constructed. The run root contains a CAS-bound `run.request.json`,
per-action completion receipts, and a CAS-bound full opportunity plan. `apex run
resume --run ...` replays the canonical journal (never the disposable snapshot),
rebuilds every accepted candidate and the current anchor from immutable evidence,
then reconciles BASELINING, DIAGNOSING, agent generation, micro, safety, delivery,
matched promotion, DECIDING, KEEP reprofile, UPDATING, or FINALIZING. A process
lost before a frozen candidate receipt becomes an explicit source-free REJECT;
partial agent text is never reused. A complete promotion receipt is replayed after
all observations and joins have been revalidated. An interrupted promotion window
is never completed from old observations: resume starts a fresh four-observation
window under the newly acquired lease. Recorded evaluator results are verified and
continued, while an incomplete external action gets a fresh action ID. The oracle
policy digest and every dynamically ranked opportunity remain part of the recovery
lineage. Each agent process is stateless and sees only one bounded packet; durable
decisions, receipts, dead ends, and the current anchor replace conversation memory.
Every terminal source candidate also emits one `experience.measured` event bound
to its decision receipt. The next fresh context may reuse it only when task,
operator, GPU, framework, shape, source, harness, and policy identity all match.

Each side of a promotion window is config-bound as well as image-bound. The
candidate deployment's typed measurement-config digest, its CAS
`delivery_measurement_config` receipt, the benchmark's CAS `benchmark_config`, and
Magpie's `serving_runtime.input_config_sha256` must all be equal; the anchor side is
bound to the current anchor config and image at the same generation. Requested and
resolved runtime images must equal the side selected by the canonical A/B order.
The pair receipt additionally binds its window ID, observation order, four action
receipts, lease and physical GPU ownership evidence, and both A/B comparisons.
This prevents a historical result, config-only winner, different image, or swapped
anchor/candidate receipt from entering a KEEP decision.

Correctness-oracle routing never preselects profiler symbols. The diagnostic rank
remains dynamic; after a source has been resolved into an exact source-lock root,
`CorrectnessOracleRegistry` matches only its repository-relative source path to a
reviewed, target-filtered in-tree test. The policy and individual binding SHA-256 values are bound
into the run accuracy contract and protected harness context. A root mismatch,
missing test, symlink, hard link, path escape, or conflicting Magpie test mapping
fails closed. For the reviewed Qwen profile this makes paged attention, recurrent
GDN decode, causal-conv update, prefix-prefill, and reshape/cache sources eligible
without hardcoding their dynamic rank or runtime symbol.

For generic or unbound workloads, that registry is routing metadata rather than
executed correctness evidence. Those workloads compose
`E2EDeferredMicroQualifier`; its receipt records `executed=false`, and unchanged
Magpie quality remains the correctness authority. The reviewed Qwen profile is
the explicit exception: `DockerOracleMicroQualifier` imports the installed
candidate vLLM, mounts only the locked evaluator test subset, executes the exact
pinned pytest node IDs, and proves candidate loaded bytes in the same process.
That preflight remains a candidate-rejection boundary, not a canonical kernel
grade or reward. Mounting the whole source checkout would shadow the installed
candidate and is prohibited.

After every candidate worker exits, `run_record.py` writes one canonical
`apex.agent-transcript/v3` CAS artifact and projects its normalized actions into
attempt-scoped `agent_message`, `tool_called`, and `tool_result` journal events.
Structured usage and explicit provider cost become separate `usage_recorded` and
`cost_recorded` events before `agent_completed`/`agent_failed`; all are marked
`self_reported` and carry source-event indexes plus the transcript receipt. They
are training/cost provenance, never evaluator correctness or performance proof.

Candidate production uses `structured_agent_turn_checkpoint_v2`. A source-changing
candidate stopped exactly at the requested turn count may be frozen only when the
typed capture is complete, `private_pid_namespace_init_pidfd_v1` verifies the
agent PID namespace is empty, and authoritative containment cleanup is verified. Its
`agent_completed` event still records `termination_kind=exact_turn_boundary` and
the controlled process exit. A count below the limit is valid only as a natural
completed process; a count above it, timeout, invalid stream, truncated output,
or cleanup failure is rejected. Unverified containment or cleanup is an
infrastructure failure, and no post-agent workspace traversal is permitted.
Infrastructure failures stop without a decision or reward. A typed, contained
agent result that produces no source instead persists its transcript,
termination, capture, containment, and candidate-manifest receipts, then records
an explicit source-free execution rejection and evaluator outcome. The exact
`agent_made_no_source_change` outcome earns the smaller no-source penalty; other
non-infrastructure rejected captures earn the general candidate-rejection
penalty. Neither path invents a candidate ID or reads unapproved agent paths.
Every admitted checkpoint then follows the same
micro/deferred qualification, safety, deployment, and E2E acceptance path.
The bounded workspace walker counts entries incrementally, prunes interpreter,
test, and Git-control directories without enumerating their children, and excludes
ignored artifacts from source fingerprints. They remain untrusted workspace bytes
and are never candidate snapshot bytes or evaluator input.

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

The reviewed Qwen preflight adds a candidate-rejection check between freeze and
safety. Passing means only that the exact pinned pytest node IDs executed with
the expected JUnit case count; it does not upgrade the deferred claim. The
preflight starts from the exact parent digest, verifies the image ID, builds a
one-file overlay, and proves the imported candidate bytes in the test process. It
copies only reviewed tests, explicit helpers, and an evaluator-owned runner into a
read-only `/opt/apex-oracle` mount, so the host `vllm/` checkout cannot shadow the
installed candidate module. Before calling `pytest.main`, that same Python process
imports the exact module and verifies its resolved `__file__` and SHA-256; after
pytest it verifies the same module object, path, and bytes again. The runner then
atomically publishes a read-only receipt which the host revalidates. Exact node
IDs avoid collecting the 388-, 164-, and 1850-case source files wholesale.

The receipt binds source commit/tree, parent and derived image IDs, baseline and
candidate bytes, same-process before/after loaded-byte proof, evaluator-runner and
test/helper hashes, exact argv, dependency
versions, GPU scope, bounded stdout/stderr, exit status, and JUnit digest/counts.
The pinned parent lacks `tblib`; `--confcutdir=/opt/apex-oracle/tests/kernels`
keeps the repository-wide `tests/conftest.py` that imports it outside collection.
The dependencies actually used by selected tests (`pytest==9.0.2` and
`einops==0.8.2`) are checked in-image first. Missing/drifted dependencies,
skips, truncation, timeout, nonzero exit, source/test/image drift, or a loaded-byte
mismatch reject the preflight.

KEEP is evaluated only from a same-window matched comparison against the current
live anchor, never the original baseline or a historical candidate measurement.
The counterbalanced order is `A(current), B(candidate), B(candidate), A(current)`;
both independently recomputed A/B comparisons must pass quality, tail-latency, and
throughput policy. The lower of their throughput gains is the promotion grade, so
one favorable ordering cannot hide a regression in the other. REVERT rolls back
the candidate deployment. A KEEP forces a fresh diagnostic pass before another
opportunity is selected. Formal success is impossible until a source-rebuilt
bundle passes engagement verification and a second clean replay.
Every selected opportunity has one explicit attempt child. Candidate E2E
measurements carry that attempt, candidate, and opportunity lineage; they do not
reuse action IDs as episode identity. Compile/correctness/safety/delivery rejects,
invalid candidate measurement, and measured KEEP/REVERT outcomes each close with
exactly one decision and one `e2e_kernel_candidate_v1` reward in the same journal
transaction. Retrying a used attempt ID fails closed.

A successful `CandidateDeployment` names one immutable `deployed_image_id` and
must carry the same ID in its derived-image evidence. Before any candidate E2E
grade, Apex requires Magpie's serving-runtime receipt to prove both the requested
and actually resolved container image equal that exact deployment ID. Missing,
mutable, or drifted identity rolls back the candidate and terminates as
infrastructure failure without a decision or reward; agent text and derived YAML
are not runtime-engagement proof.

Overlay parent authorization has two non-interchangeable paths. Generation zero
must pass the original container-provenance and unique repo-digest checks. A later
generation must identify the last committed accepted deployment exactly. Every
prior deployment's `overlay_build_receipt` is content-hashed and binds its
generation, ordered candidate/image ancestry, parent KEEP decision receipt,
parent/derived image IDs, Dockerfile, frozen candidate file, and clean-container
loaded bytes. Missing, reordered, duplicated, or modified ancestry fails before
the package probe or build; a rejected or merely measured image is never a parent.

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

The GPU lease spans the entire run and its full receipt is bound to every promotion
pair. The initial run request freezes the physical device scope; resume must acquire
the same scope and fails before journal mutation if it changed. Ownership evidence
with a foreign process is not promotion evidence. Leases remain cooperative and
fail-fast; this package does not discover or kill unrelated processes.

## Dependencies

Benchmark execution and diagnostic analysis enter through ports. Context assembly
depends on `apex.context` and the curated `apex.knowledge` retrieval API. Control
state is owned by `apex.orchestration`; evidence bytes are owned by `apex.storage`.
Source materialization uses Git without initializing or traversing submodules.
Safety and final delivery remain evaluator-owned boundaries.
Docker overlays require one exact clean vLLM/AITER source lock whose repository
bytes match the installed parent image. They do not mount or modify host
site-packages, and rollback merely selects the previous immutable config/image.
The overlay adapter retries one transient Docker build failure once against the
same immutable candidate context. The agent process has already terminated and
the safety boundary has made the frozen candidate read-only; Apex additionally
makes both context files read-only and rechecks their SHA-256 values before each
attempt. Every failed command receipt retains stage, redacted argv, cwd,
exit/timeout/cleanup state, and bounded redacted stdout/stderr with hashes;
environment variables and credential values are not serialized.

## Failure semantics

Baseline or quality failure terminates as `baseline_invalid`. Missing source,
micro, or deployment capability is reported as unsupported. Candidate-caused
compile, correctness, safety, or E2E failures reject/revert only the active
candidate until its bounded search budget is exhausted. After those evaluator
verdict boundaries pass, every immutable-overlay deployment failure is
infrastructure: frozen-source, source-mapping, safety-state, Docker, image
identity, provenance, and adapter failures all persist the delivery
failure receipt and terminate as `infrastructure_error`; they do not commit a
candidate reject, rotate to another opportunity, or run a final baseline replay.
Missing source provenance or second-clean-replay proof can retain primary
evidence but cannot produce formal success.

If the search accepts no source candidate, Apex still runs and records the final
normal measurement so baseline-versus-replay drift remains observable. That
measurement is not a candidate no-regression gate because no source change was
accepted or delivered: the unchanged source identity returns `no_gain` with
`no_regression=true`, while unsupported capability reasons remain unsupported.
`details.observed_replay_verdict` and its CAS-bound final lineage preserve any
measured runtime drift without relabeling it as a candidate regression, while
`details.final_replay_basis` states that no patch was accepted and delivery was
not attempted. Once a patch is accepted, the cumulative replay against the live
anchor is likewise a hard gate.

After the final profiler-off measurement, Apex runs one profiler-on terminal
diagnostic against the same live source/image state. Its Magpie benchmark,
declared raw rank-0 trace, TraceLens reports, normalized evidence, and typed
comparison receipt are stored in CAS and journaled with
`evidence_class=diagnostic` and `reward_eligible=false`. The pinned TraceLens
revision's documented report-diff API receives receipt-verified report sheets
from both observations and its CSV/XLSX outputs are republished to CAS. Successful
report comparison is recorded as `PARTIAL`, because the pin has no stable
full-attribution contract or MI355X analysis profile. The result explicitly makes
no attribution, grade, or reward claim. Missing inputs or API capability remain
typed failures/unavailability. The terminal result links the comparison receipt
under `details.terminal_diagnostics`.

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
Terminal result bytes are written to CAS and linked from a `delivery_result`
journal event before the run transitions to a terminal phase. `result.json` is
only a byte-checked projection; resume never trusts modified metrics or details.

Safe in-root symlinks retain their identity. Escaping symlinks, hard links,
unmerged Git paths, edits outside the declared source, materialized gitlink
content, changed file modes, stale generations, and workload-semantic changes fail
closed.

## Tests

CPU-only tests cover dynamic eligibility, config exclusion, safe symlink/gitlink
handling, 300-sample enforcement, retry and fresh-context history, safety blocking,
immutable parent binding for tag, image-ID, and repo-digest inputs, bounded Docker
retry/failure evidence, candidate-versus-infrastructure deployment failure,
current-overlay chaining, KEEP/REVERT rollback, atomic decision/reward lineage,
explicit no-source reward, GPU lease scope/contention,
attempt-scoped message/tool/usage/cost lineage, final provenance failure, crash
recovery injected after every candidate gate and after one, two, or three promotion
observations, complete-pair reuse, physical-scope drift before journal mutation,
malicious anchor/candidate receipt swaps, asymmetric AB/BA outcomes, atomic KEEP,
reprofile plan/commit, update, and final-measurement launch, plus the mandatory
second clean replay.

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
