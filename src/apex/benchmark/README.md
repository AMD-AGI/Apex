# Benchmark

This module is Apex's only E2E benchmark execution boundary. It delegates
execution to the exact Magpie checkout recorded by `DependencyReceipt`; it does
not import an in-tree fallback, manipulate kernels, or implement another
profiler.

## Published-main configuration resolution

`MagpieMainConfigAdapter` uses `Magpie.main.load_benchmark_config` and
`BenchmarkConfig.from_dict/to_dict` from the exact verified published `main`
checkout. Apex owns the content-bound plan, scoring/evaluator policy, capability
receipt, result expectation, reward contract, redaction, and compatibility
decision. Unknown semantic fields produce `capability_upgrade_required`; strict
YAML, import-origin, file-identity, and cross-document drift checks fail closed.

The exact pin reports all 27 configs compatible at the published Magpie model
boundary. Apex V2 derives 21 Docker one-shot product rows and rejects the six
Local/Ray/reuse/cleanup rows with `e2e_docker_only` before this execution
boundary. Legacy local/Ray adapters remain typed implementation inventory for a
future product scope; their existence is not V2 support or release evidence.
Configuration resolution does not itself run a workload and never proves GPU
execution, quality, performance, or formal delivery.

## Phase views

`build_config_views` writes four immutable artifacts:

Before adding any Apex-owned runtime binding, it reconstructs the benchmark
mapping from the Apex-owned scoring projection of Magpie's effective config and verifies
that view's exact digest. Apex restores only values that Magpie explicitly
redacted, using the frozen raw input at the same path. It does not parse Magpie
defaults, infer framework/run-mode semantics, or materialize views for a
`capability_upgrade_required` receipt. Every materialized view embeds the plan,
capability, phase-view, scoring-view, and resolver-invocation digests.

- `benchmark.original.yaml` is a byte-for-byte copy of user input.
- `benchmark.measurement.resolved.yaml` forces `run_kind=measurement`, disables
  Torch, system, TraceLens, GPU monitoring, and gap analysis. Its metrics may be
  used by E2E policy.
- `benchmark.diagnostic.resolved.yaml` forces `run_kind=diagnostic`, enables
  Torch profiling, TraceLens at the
  receipt's exact root, deterministic TargetedKernelTrace acquisition, GPU
  monitoring, and gap analysis. For serving workloads it explicitly sets
  `RUN_EVAL=false`: the diagnostic lane exercises the profiled workload but
  does not repeat lm-eval. Its performance numbers are observations only,
  never reward or quality truth.
- `benchmark.replay.yaml` has measurement instrumentation and may differ from
  measurement only in `docker_image`.

Every executable view binds the exact InferenceX receipt, model revision, cache
root, and physical GPU selection. Formal serving measurement and replay also
freeze the verified `benchmark.lm_eval_runtime` path, digest, and full identity.
The serving diagnostic intentionally omits that field, so Magpie does not
validate, snapshot, mount, import, or invoke the evaluator runtime. The runtime
is never installed into or inferred from the workload image. For lm-eval quality
contracts, measurement and replay consume the official scoring view's explicit
`RUN_EVAL=true`, `MAGPIE_EVAL_TASKS` (Magpie currently defaults it to `gsm8k`),
and batch policy. Apex does not identify this lane from a framework-name list.
The diagnostic view keeps that task
and evaluator-policy identity as an inert reference while setting
`quality_contract.kind=trace_only`, `required=false`, and `RUN_EVAL=false`.
An input that explicitly disables formal evaluation is still rejected before an
agent or GPU starts. The workload digest is computed from the formal measurement
contract; diagnostic validation restores the receipt-pinned runtime and
normalizes `RUN_EVAL=false` only for this comparison before proving every other
workload input is identical.

`validate_phase_set_contract` is the shared, receipt-independent guard used
before the initial views are written and again before and after an immutable
image overlay is serialized. It verifies exact phase roles, common provenance
metadata, formal/replay identity, quality and evaluator-policy consistency,
profiler isolation, and the normalized workload digest across all three views.
This self-consistency check complements `validate_resolved_view`, whose trusted
dependency receipt proves that embedded paths, revisions, and runtime identities
match the live environment.

## Result contract

`MagpieBenchmarkAdapter` invokes an argv array through `SubprocessSupervisor`
with the receipt's Python interpreter and Magpie root. It uses an explicit
environment allowlist rather than copying the host environment. GPU visibility,
ROCm locations, and non-secret Hugging Face cache/offline fields may be inherited;
named Docker daemon/context/config/TLS fields are inherited so Magpie can reach
the operator-selected engine. The single Magpie-owned host control
`MAGPIE_PROTECT_BENCHMARK_CONTAINER` is also inherited exactly: operators may
opt into shared-host stop protection without broadening the environment
boundary. A Hugging Face token may be supplied only as an explicit request
override.
Hugging Face model/dataset offline switches are allowlisted so a prewarmed,
revision-audited cache can be reused without network access. When
`hf_offline=true`, the view builder requires an explicit existing cache and
freezes `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`HF_DATASETS_OFFLINE=1` into the workload semantics shared by all three views.
`PYTHON*`, shell startup hooks, dynamic-loader injection, unrelated API keys, and
`DOCKER_AUTH_CONFIG` are excluded. The verified Magpie and TraceLens roots plus
`PYTHONNOUSERSITE=1` are adapter-owned and cannot be overridden. Every run gets a
new immutable output directory.

`parse_benchmark_report` normalizes throughput plus TTFT/TPOT/ITL/E2EL
mean/median/p99 distributions. Serving quality is read from exactly one
lm-eval `results*.json` inside the Magpie workspace; scriptable quality comes
from the Apex evaluator attestation's `quality_gate` mapping. The official
Magpie report is never consulted for a quality verdict. The official Magpie
scoring view declares the
immutable evaluator policy before execution: policy ID, task set, primary
metric, maximum sequence length, and maximum generated tokens. Apex consumes
those fields without model-name, filename, or config-hash routing and includes
the policy digest in workload and accuracy semantics; the original benchmark
YAML is copied byte-for-byte. Apex independently rehashes the results and raw
sample receipts and requires Magpie's outcome and sample-set digests to match.

Required measurement/replay quality evidence that is missing, ambiguous,
skipped, or empty makes the run fail closed even if Magpie reports process
success. The requested lane and reward eligibility are Apex evaluator facts;
they are not read from the official report. Measurement/replay require a
verified measurement attestation and profiling off, while diagnostics are
always reward-ineligible. Candidate-vs-baseline regression policy belongs to
the E2E optimization module, not this adapter.

The formal Docker result contract requires the evaluator attestation's runtime
mapping to contain a validated
`apex.magpie-serving-runtime-observation/v3` value. This Apex-owned observation
is never read from the official report. Its `container_spec_sha256` is the
canonical digest of Docker inspect's final `Path` and `Args`; it is deliberately
not described as the unobserved Docker launch argv. The receipt keeps the
frozen view's `input_image` and resolved immutable `input_image_id` distinct
from the image reference and ID actually passed to Docker. Measurement and
replay accept only a direct identity binding. A diagnostic may execute a
TraceLens-derived vLLM image only when its resolved view explicitly enables
inference auto-patching and Magpie proves the complete base-to-derived lineage:
base locator and ID, derived reference and ID, pinned TraceLens commit and tree,
runtime schema, patch version/path/digest, dependency-wheel manifest digest,
and successful runtime validation. Apex checks that proof against its verified
dependency receipt and the actual serving receipt. Missing, legacy, unpinned,
or internally inconsistent lineage fails closed; changing the requested image
field alone cannot bypass the frozen input identity.

Published Magpie `main@12896a49` does not emit this Apex schema and does not
echo Apex lane fields or protected runtime receipts. The
resolved plan now records `apex.magpie-main-result-contract/v1`: it keeps the
official `benchmark_report.json` distinct from an evaluator-owned
`apex.magpie-execution-attestation/v1` side artifact outside Magpie's writable
workspace. Its `official_report_path`, size, and SHA-256 bind the unchanged
report; its config SHA-256, run ID, pass, process, dependency, runtime, GPU,
and quality mappings bind evaluator facts. The sidecar must be a regular,
non-hardlinked file below an `evaluator` root that is a sibling of the Magpie
workspace. Any private lane, reward, runtime-receipt, or quality field in the
official report is rejected rather than used as fallback. Apex still treats
its 27-config resolver pass as configuration compatibility only. Report success
alone is never a substitute for the Apex sidecar.

`MagpieBenchmarkAdapter` therefore defaults to an explicit unavailable
`MagpieExecutionAttestor`. It validates the resolved view on CPU, then returns
`magpie_execution_attestor_unavailable` before creating the run directory or
calling the subprocess supervisor. A trusted observer must be injected and its
`prepare` hook must succeed before Magpie can start; its `complete` hook supplies
the external sidecar path after observing process completion. Merely arranging
for Magpie or a test supervisor to write a similarly named file does not satisfy
this preflight contract.

### Exact-image lm-eval authority

Formal Docker measurement is not delegated to published Magpie's evaluator
helper. Before Magpie starts, `LmEvalExecutionPreparer` rehashes the checked
offline evaluator dataset CAS, materializes a private task definition from the
exact locked InferenceX YAML, and freezes one
`apex.lm-eval-execution-contract/v2`. That contract binds the measurement config,
policy and policy-lock digests, source/effective task digests, dataset revision
and file receipt, immutable lm-eval runtime, launcher bytes, exact evaluator
image repo digest and image ID, model endpoint, generation limits, concurrency,
timeout, argv, environment, and output limits. Preparation creates a new
run-scoped `authority/lm_eval` tree and never edits the Magpie or InferenceX
checkouts.

The dependency runtime and offline dataset may live on a root-squashed home
filesystem that the Docker daemon cannot bind. Preparation therefore copies the
already verified dataset, runtime, and launcher into a private
`authority/lm_eval/sidecar-inputs` projection on the caller-selected results
filesystem, rehashes the copied bytes and runtime manifest, seals the projection
read-only, and records `apex.evaluator-sidecar-input-projection/v1`. The
projected runtime receipt differs only in its run-local root. The projection
digest is part of the sidecar-spec digest; source-cache paths are never handed
to Docker as an implicit fallback. A results filesystem that the selected
Docker daemon cannot bind still fails closed during container creation.

The preparer also creates a private InferenceX projection. It copies only the
locked source plus the reviewed Magpie launch scripts, installs the bounded
Unix-socket handoff overlay, records complete source/projection manifests, and
derives a launch-only Magpie config that replaces the canonical InferenceX root
with that projection. The original and resolved benchmark views remain
unchanged. The launch config and projection receipts bind both exact dependency
commits/trees and are revalidated after the workload exits. A private projection
is execution plumbing, not proof that those bytes were used; engagement becomes
evidence only after the handoff and runtime publication checks succeed.

`EvaluatorHandoffBarrier` listens on the private projection's Unix socket before
the workload launch. At Magpie's quality boundary the overlay sends one bounded,
schema-checked request and blocks for the evaluator exit code. The evaluator
authority independently observes the already-running Magpie container and its
verified TCP listener. `EvaluatorServingBroker` then exposes only that one
listener over a run-scoped Unix socket; the evaluator container itself never
joins the workload network namespace.

Serving images commonly run as root, so an unprivileged Apex process may be
unable to read their host `/proc/<pid>/cwd` and file descriptors. The frozen
host process identity permits an unavailable cwd only for containment use;
local argv/cwd matching still rejects it. Listener ownership is obtained by a
fixed, bounded `docker container exec` probe inside the exact observed
container, then mapped back to exactly one member of the pre-probe host process
closure using PID-reuse-resistant start time and command-line digest. Missing,
foreign, ambiguous, or unbounded owners fail closed.

`DockerEvaluatorSidecarAuthority` creates exactly one immutable-image sidecar
with `--network none`, a read-only root, no GPU devices, all capabilities
dropped, `no-new-privileges`, a PID limit, fixed uid/gid, bounded tmpfs, and only
the seven contract-declared mounts. Dataset, task, runtime, launcher, contract,
and broker mounts are read-only; only the private authority output is writable.
The Docker adapter inspects and digest-binds both the created and exited states,
requires a clean zero exit with untruncated bounded output, re-observes the
unchanged Magpie listener, and removes the sidecar. Stop/remove or broker cleanup
failure is itself a failed authority result rather than ignorable cleanup noise.

After execution, publication accepts a bounded regular-file tree only: no links,
path escape, duplicate destination, pre-existing result, more than 256 files, or
more than 128 MiB total. Result and sample artifacts are independently classified,
hashed, exclusively copied into the official workspace, and sealed read-only.
Runtime publication reopens the sidecar probe, immutable manifest, and installed
`lm_eval` module and binds them to the verified runtime receipt. Private
InferenceX publication separately rehashes the source/projection/handoff/launch
artifacts and writes `apex.inferencex-runtime-receipt/v2`; it does not pretend
that a copied checkout or ordinary Magpie success is private-materialization
evidence. The final `apex.lm-eval-execution-receipt/v3` joins every contract,
container, listener, broker, cleanup, runtime, result, and sample digest before
the quality gate may be marked verified.

These paths are implemented and covered by CPU contract tests and production
composition. A real exact-image smoke has verified root-owned listener binding
and sidecar `create/inspect/remove` with every bind source under the run-scoped
results tree. A representative Qwen workload produced real baseline performance
and completed the full 1,319-sample GSM8K sidecar with a verified quality gate.
A fresh run after the bounded 20-second observer-drain fix completed formal
baseline quality and performance without observer errors. Its diagnostic action
then failed before container start because Docker could not bind the original
InferenceX checkout from the root-squashed source filesystem. Diagnostic input
therefore still needs a plain, run-scoped InferenceX projection under the
Docker-visible results root, with a diagnostic-only receipt and post-run drift
check. It must not reuse the evaluator projection's lm-eval handoff overlay.
Until that path is implemented and replayed, there is no new full trace,
candidate, reward, winner, or release evidence. A lifecycle smoke, isolated
quality receipt, or prepared contract is not reward, winner, or release
evidence.

Production composition injects `DockerOneShotMagpieExecutionAttestor` for the
published Docker one-shot path. Before Magpie starts, it freezes the config,
dependency worktrees, requested image ID, and GPU-lease authority, and confirms
that no matching container predates the observation. A bounded background
observer identifies exactly one live `magpie-benchmark-*` container by its
unique `/workspace` bind below the run root. It repeatedly checks the immutable
image ID, exact InferenceX bind, `/dev/kfd` and `/dev/dri` exposure, and uses
RSMI plus procfs cgroups to prove that KFD processes belong to that container
and cover exactly the leased physical devices. Completion rechecks image,
dependencies, config, report location and digest. It also queries the exact
observed container ID across all Docker states and requires it to be absent;
a still-running, stopped, paused, dead, or otherwise retained container fails
one-shot cleanup. Only then does it write the immutable
sibling `evaluator/execution_attestation.json`; it never edits the official
report. Missed/multiple containers, image or dependency drift, missing GPU
engagement, and observer errors are recorded fail closed. For measurement, the
same attestor owns the prepared exact-image evaluator handoff described above;
reward eligibility additionally requires its independently published lm-eval,
InferenceX, result/sample, and quality receipts. The model-revision field remains
an independent provenance input and is never inferred from container success.

`LocalMagpieExecutionAttestor` is the corresponding host-runtime observer for
published Magpie's built-in vLLM, SGLang, and Atom scripts. It accepts only an
exact pinned InferenceX checkout and an active Apex GPU lease. It binds the exact
Magpie process to the lease owner by parent PID, process group, session, cgroup,
and start time; records the observed listener and descendant process identities;
and requires one-shot/cleanup GPU quiescence. Reuse binds Magpie's PID and
metadata files to the exact listener and process identity; cleanup must start
from that attested state and remove it. The immutable sidecar stores these facts
as `apex.magpie-local-runtime-observation/v2` in
`serving_runtime_receipt`. It deliberately leaves
`inferencex_runtime_receipt` empty because published Magpie does not create a
run-scoped private InferenceX materialization for local mode.

The local receipt keeps the workload config SHA separate from
`server_source_generation_sha256`. The latter includes only exact dependency
source plus Magpie's server-compatibility fields, so the published reuse and
cleanup configs may differ in client-only ISL/OSL and cleanup flags without
rewriting the server's source identity. Server generation separately binds the
PID/start time, stable compatibility digest, source generation, and port.
`LocalRuntimeEvidence` independently validates every key, digest, source,
process, listener, cgroup, lifecycle, quiescence, and generation relationship.
This does not make quality-required local measurements reward-eligible.
`formal_measurement` is a separate typed capability from
`benchmark_execution`: the latter means that Apex can observe a Magpie process,
not that the quality evaluator is usable. At the pinned dependency receipt the
six published serving scripts pass `--concurrent-requests`, while the pinned
InferenceX `run_lm_eval` parser rejects that argument before Python starts.
The remote helper is not a substitute: it omits required sample artifacts and
policy limits, and its built-in GSM8K task definition differs from InferenceX's
task YAML. The local observer's formal support receipt therefore reports
`magpie_local_quality_execution_unavailable` with typed argument, interpreter,
sample, policy, and task-contract blockers. Production does not register that
observer yet, so its earlier preflight reason remains
`magpie_execution_attestor_unavailable`. No local runtime or lm-eval receipt is
inferred from a successful official report, and formal optimization fails
before a GPU lease until an independent exact-image evaluator authority closes
these facts.

Published Magpie Ray runs place the worker workspace below the configured
shared-storage `results/<task-id>/` tree rather than below the driver's
`--output-dir`. `RayOneShotMagpieExecutionAttestor` therefore resolves the Ray
address and shared-storage root from each frozen config, observes the exact
driver job and `Magpie.remote.tasks.run_task`, and delegates worker procfs,
dependency, KFD, runtime, and artifact claims to an injected node-side
`RayNodeEvidenceAuthority`. The worker may not attest itself, and the default
authority is deliberately unavailable. Authority-declared regular files are
copied into a new local workspace with no-follow opens, size bounds, two digest
passes, and source identity checks. The unchanged official report retains its
remote `workspace_dir`; the execution sidecar binds that origin to the imported
manifest and the evaluator rehashes every local copy. Missing node coverage,
ambiguous tasks, shared-path drift, links, races, or an absent authority keep
the Ray lane unavailable and cannot become reward evidence.

## Purpose

The package is Apex's Magpie boundary: it freezes phase-specific workload views
and normalizes evaluator-owned performance and quality evidence.

## Public API

Use `MagpieMainConfigAdapter`, `validate_apex_magpie_config_documents`,
`build_config_views`, `validate_phase_set_contract`, `validate_resolved_view`,
`MagpieBenchmarkAdapter`, `DockerOneShotMagpieExecutionAttestor`,
`MagpieExecutionAttestorRegistry`, and the immutable result types exported by
`apex.benchmark`. Exact-image sidecar preparation/execution helpers are internal
attestor collaborators, not an alternate public benchmark API.

## Invariants

Measurement views disable profiling and require quality; serving diagnostic
views cannot supply rewards or quality claims; replay changes only the allowed
image locator; derived runtime images are diagnostic-only and evidence-bound;
host environment state cannot silently change the imported evaluator. Formal
Docker quality uses one policy-locked offline dataset and one no-network/no-GPU
exact-image sidecar; Magpie cannot self-attest that authority.

## Dependencies

Benchmark code depends downward on core contracts, benchmark ports, supervised
execution, and pinned runtime receipts; it never imports optimization policy.

## Failure semantics

Malformed YAML/JSON, unavailable published Magpie APIs, dependency drift, phase
leakage, workspace escape, dataset/task/runtime drift, an unsafe handoff or
sidecar observation, ambiguous quality files, or missing required quality
evidence fail closed with reason codes. Partial sidecar outputs and cleanup
failures never become a successful quality receipt.

## Tests

Run `pytest -q -p no:cacheprovider tests/unit/benchmark`; fixtures are CPU-only
and use temporary immutable views, fake Docker/Unix authorities, offline dataset
trees, and reports. The evaluator-focused files cover policy/contract creation,
dataset CAS/materialization, task and InferenceX projection, handoff, broker,
sidecar spec/lifecycle, output/runtime publication, and tamper/cleanup failures.

## Provenance

Resolved views record original/config semantic hashes plus exact Magpie,
TraceLens, Python, evaluator-policy/dataset, runtime/image, and dependency-lock
identities. The execution receipt additionally records the exact sidecar create
contract and observed lifecycle rather than treating requested Docker argv as
observed execution.

When requested, formal results fail closed unless Apex can independently re-read
the workspace `model_revision_receipt.json` and private-materialization
`inferencex_runtime_receipt.json`. The latter must bind the configured clean
InferenceX checkout, exact commit and tree, unchanged empty Git status, and the
run-scoped private-index materialization method; a copied or dirty source tree
cannot become reward evidence. Local execution has no such Magpie artifact and
uses the distinct Apex-owned local runtime evidence described above.
The evaluator attestation's `lm_eval_runtime_receipt` evidence is independently rehashed
against the snapshotted runtime manifest and execution receipt, then compared
with Apex's verified runtime identity. Both artifacts are retained in the
normalized result and canonical E2E run record; missing, writable-mounted, or
tampered evaluator evidence cannot become reward truth.
