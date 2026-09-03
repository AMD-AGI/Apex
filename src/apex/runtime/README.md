# Runtime

`apex.runtime` owns external dependency resolution and environment-facing
bootstrap operations. Domain and orchestration modules consume immutable
dependency receipts; they do not search `PATH`, inspect sibling repositories, or
select a Magpie checkout independently.

## Public API

| Symbol | Owner | Purpose |
|---|---|---|
| `DependencyLock`, `LockedDependency`, `load_lock` | `dependencies.py` | Strict lock schema and content digest |
| `DependencyBootstrapper`, `PythonEnvironment`, `PythonProbe` | `dependencies.py` | Idempotent install, import verification, and receipt generation |
| `RepositoryResolver`, `RepositoryState`, `ResolvedRepository` | `repositories.py` | Exact Git source selection and local/remote materialization |
| `WorkspaceGitIdentityResolver` | `workspace_identity.py` | Read-only formal workspace origin/commit/tree/dirty identity before GPU work |
| `BootstrapError` | `repositories.py` | Deterministic bootstrap failure boundary |
| `canonical_repository`, `inspect_repository` | `repositories.py` | Repository identity helpers |
| `probe_errors`, `version_matches` | `dependencies.py` | Package version/import-root verification helpers |
| `MagpieCorpusManifest`, `build_magpie_corpus_manifest`, `load_magpie_corpus_manifest`, `verify_magpie_corpus_manifest` | `magpie_corpus.py` | Frozen benchmark subtree/path/hash inventory and exact-checkout verification |
| `MagpieCompatibilityLedger`, `load_magpie_compatibility_ledger`, `verify_magpie_compatibility_ledger` | `magpie_compatibility.py` | Self-digested 27-config phase-view/capability/reward-contract compatibility gate |
| `MagpieMainPublicApi`, `MagpieMainConfigAdapter`, `MagpieConfigContract`, `validate_apex_magpie_config_documents` | `magpie_main.py`, `magpie_config.py` | Exact-origin public Magpie loader plus Apex-owned content projection over its effective configuration model |
| `QualificationEvidence`, `build_qualification_evidence` | `release_qualification.py` | Self-digested, kind-specific backend/recovery/ablation/AKA/Magpie live claims; never authority by themselves |
| `EvaluatorQualificationArtifactAuthority`, `QualificationArtifactCollection` | `qualification_artifacts.py` | Read-only strict-index/CAS resolver that delegates semantic recomputation to installed kind-specific verifiers and emits path-free unavailable/invalid/verified outcomes |
| `ReadOnlyQualificationArtifactStore` | `qualification_artifacts.py` | Receipt-only, race-checked reader used by production qualification replay; it exposes no caller-selected artifact paths and never writes the formal root |
| `ShowcaseEvidence`, `build_showcase_evidence` | `release_showcase.py` | Path-free binding to an official showcase-verifier v2 receipt and all critical projection digests |
| `ReleaseEvidence` and other typed evidence records | `release_evidence.py` | Path-free fetch, Magpie config resolution, environment, CPU-gate, CLI, image, showcase, and live-qualification inputs |
| `LocalReleaseEvidenceCollector`, `collect_local_release_evidence` | `release_collection.py` | Clean-tree producer for exact dependency, all-corpus config resolution, fixed CPU/static, and installed-CLI evidence |
| `ReleaseCandidateReceipt`, `inspect_release_candidate`, `verify_release_candidate_receipt`, `freeze_campaign_baseline`, `freeze_release_candidate` | `release_candidate.py` | Rebuilt qualification/release identity gates; not an optimization intake artifact |
| `ApexExecutionIdentity`, `collect_apex_execution_identity` | `execution_identity.py` | Automatic path-free observation of executed Apex package bytes, Git state, and dependency lock |
| dependency CLI parser/composition | `dependency_cli.py` | Executable setup/verify surface kept out of the lock domain module |
| `DependencyReceipt`, `verify_runtime_dependencies` | `receipt.py` | One verified identity consumed by all runtime adapters |
| `LmEvalRuntimeLock`, `load_lm_eval_runtime_lock` | `lm_eval_lock.py` | Strict source, wheel, base-image, ABI, installed-tree, and evaluator-identity lock |
| `EvaluatorPolicyLock`, `EvaluatorDatasetLockFile`, `load_evaluator_policy_lock` | `evaluator_lock.py` | Exact task definition, metric, sample requirement, offline dataset revision, and file inventory lock |
| `LmEvalRuntimePreparer` | `lm_eval_prepare.py` | Networkless exact wheel build/install/smoke producer and atomic CAS publisher |
| `LmEvalRuntimeReceipt`, `verify_lm_eval_runtime` | `lm_eval_runtime.py` | Independent byte/mode/tree verification for the immutable evaluator CAS |
| `SourceLockSet`, `SourceLockSpec`, `load_source_lock` | `source_locks.py` | Strict formal E2E source lock and checked-in content digest |
| `SourceLockManager`, `SourceLockReceipt` | `source_locks.py` | Managed exact-checkout materialization, read-only verification, and receipts |
| `RunProvenance`, `ProvenanceResolver`, `ComponentSourceLockSet`, `RepositoryLock` | `provenance.py` | Best-effort image observation and per-run active-component source locks |
| `GpuLeaseManager`, `LocalGpuLeaseManager`, `GpuLeaseReceipt` | `gpu.py` | Run-scoped physical-GPU lock bound to ownership evidence |
| `GpuLeaseHeartbeatReceipt`, `GpuMeasurementBracketReceipt`, `GpuMeasurementGuard`, `require_gpu_lease_heartbeat`, `require_gpu_measurement_guard` | `gpu_lifecycle.py` | Expiring-owner revalidation and pre/post measurement bracket contracts |
| `GpuDeviceIdentity`, `GpuSelectorRequest`, `RsmiDeviceIdentity` | `gpu_topology.py` | Ordered HSA selector composition and HSA/KFD/DRM/RSMI identity join |
| Internal bounded RSMI adapter | `gpu_rsmi.py` | Fixed-signature ctypes calls for monitor identities and KFD process maps |
| `HsaInventoryEvidence`, `CleanHsaInventoryProvider` | `hsa_inventory.py` | Hash-bound, unfiltered HSA agent enumeration in a fresh helper process |
| `GpuOwnershipReceipt`, `RocmSmiGpuOwnershipInspector` | `gpu_ownership.py` | Race-checked physical identity and RSMI PID-to-GPU preflight |
| `GpuDoctorReceipt`, `LinuxGpuDoctorInspector` | `gpu_doctor.py` | Read-only owner cgroup/namespace/Slurm and health-activity preflight |
| `load_gpu_doctor_receipt` | `gpu_doctor_load.py` | Strict typed reload of a persisted doctor receipt |
| `RocmHealthReceipt`, `CtypesRocmHealthInspector` | `gpu_health.py` | Ownership-bound selected-device temperature/clock/busy/VRAM snapshot |
| `FormalResultsRootValidator`, `formal_results_validator` | `formal_results.py` | No-write validation of an external evaluator-owned result root and overlap policy |

The package `__all__` is authoritative. Public symbols are loaded lazily so
`python -m apex.runtime.dependencies` can serve as the bootstrap subprocess
without import-order warnings or import-time I/O.

## Dependency direction and invariants

`dependencies.py` may depend on `repositories.py` and `apex.core`; the repository
module is standard-library-only and does not import other Apex modules. Neither
module imports adapters, orchestration, CLI, or `main.py`.

The resolver applies this precedence: explicit CLI root, explicit environment
root, exact sibling, exact managed checkout, then exact clone. Explicit roots
fail closed. Existing sibling repositories are never reset or switched. Every
successful receipt binds the lock digest, repository URL and commit, worktree
state, distribution/version policy, actual import file, and selected Python.

`receipt.py` materializes that single receipt for benchmark, diagnostics,
grader, and agent-tool adapters. A mismatched checkout or Python import fails
before a workload starts.

Formal E2E sources have their own source-only lock. By default, vLLM and AITER are
materialized beneath `~/.cache/apex/source-locks`; a matching local sibling is a
clone source, never the selected mutable workspace. Publication verifies expected
origin, commit, `HEAD^{tree}`, and clean state before an atomic rename. The read-only
verifier performs no clone or fetch. Production composition receives verified roots
from `DependencyReceipt.source_locks`, so a guessed cache path cannot become formal
delivery evidence.

`ComponentSourceLockSet` is the per-run projection of those available pins. It
names the exact framework/runtime components actually required by the resolved
workload, admits at most one clean lock per component, rejects locks for inactive
components, and reports missing exact components without model-name routing.
Adding a new reviewed SGLang, Atom, Triton, or FlyDSL pin extends the dependency
lock/adapter registry; it does not create another E2E controller. The current
checked-in source lock still contains only vLLM and AITER, so other components
remain explicit capability debt.

## Deterministic lm-eval runtime

Serving quality evaluation never installs into the benchmark image and never
resolves packages at run time. `apex dependencies prepare-runtime` verifies the
exact lm-eval source archive and Git tree, builds the four source-only wheels in
the pinned vLLM parent image, verifies every downloaded/built wheel hash, then
installs 24 non-base wheels with `--no-index --no-deps --no-compile --target`.
Build, install, and smoke containers use `--network=none`. Packages already
provided by the parent image—including Torch, Transformers, Datasets, NumPy,
SciPy, and their locked versions—cannot be shadowed by target wheels.

The producer runs wheel construction as container root only because the locked
wheel bytes depend on setuptools' deterministic source-mode handling. A fixed
cleanup pass in the same pinned, networkless image restores every temporary
mount to the host uid/gid; cleanup failure fails the operation. Final files are
0444, directories are 0555, links/hardlinks/special files are forbidden, and
publication is a same-parent atomic rename into
`.cache/apex-runtime/lm-eval/sha256/<runtime_sha256>`.

The reviewed derived lock values are installed-tree
`23dc17079da4619a4cb37100f66f015dd9dd818df46e9f0ea16b541deaf27f60`
and runtime
`ca744a9e0ab994eba275a0fc0b01b762247f76f9cd0129b31b5dc2969b23732e`.
They replace an unreproducible intermediate digest; no source, wheel,
base-image, ABI, or InferenceX pin changed. Two empty-directory builds produce
byte-identical manifests, and both Apex and Magpie independently accept them.

The separate `scripts/evaluator_policy.lock.json` is part of dependency and
release identity. `load_evaluator_policy_lock` requires the exact v2 fields,
sample logging, primary metric, one InferenceX-owned task definition, immutable
dataset repository/path/name/revision, sorted split set, and sorted path/size/
SHA-256 file inventory. When an InferenceX root is available it rehashes the task
YAML and verifies that its task and dataset declarations equal the lock. The
runtime receipt exposes this typed lock to the benchmark boundary; it does not
download or trust a dataset merely because its revision string matches.

The offline dataset CAS is materialized and reverified by the benchmark
evaluator authority under its own size, link, path, and race checks. Runtime
therefore owns static policy identity, while benchmark owns run-scoped dataset
bytes, private task projection, and execution evidence. Neither layer treats a
Hugging Face cache, network resolution, or InferenceX checkout as evaluator
output evidence.

`provenance.py` consumes Magpie's validated resolved contract rather than reading
benchmark YAML. Framework, model, run mode, requested image, and the ordered
active source components all come from that versioned contract; Apex adds only
environment observation and exact source-lock evidence. This prevents a second
framework/env/default parser from silently diverging from Magpie.

`provenance.py` intentionally separates permissive intake from formal delivery.
A Docker tag can enter diagnosis after its local image ID is observed, but missing
model revision, clean source locks, build recipe, or loaded-byte attestation remains
explicit evidence debt and cannot produce a source-rebuild-verified result.
Its Docker/Git subprocesses use the shared minimal environment policy. Docker may
inherit only named daemon/context/config/TLS fields; inline
`DOCKER_AUTH_CONFIG`, API keys, shell startup hooks, loader injection, and
`PYTHON*` are not forwarded. Registry authentication remains Docker CLI-owned via
the selected `DOCKER_CONFIG` directory. Git inspection disables system/global
configuration and prompting so provenance reads cannot execute user-configured
helpers or silently change repository identity.

## GPU ownership boundary

AMD exposes several unrelated number spaces: an HSA/ROCR GPU ordinal, a KFD node,
a DRM render minor, and a ROCm SMI monitoring index. Apex never assumes those
indices have the same order. A fixed-argv fresh Python helper removes
`ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, and
`GPU_DEVICE_ORDINAL`, then calls `libhsa-runtime64` directly to enumerate the
unfiltered HSA GPU order with UUID, KFD driver node, PCI domain, and BDF. The
parent hashes the helper and HSA library before execution and again afterward.
It joins each HSA agent bijectively to KFD sysfs, DRM render identity, and RSMI
using UUID, KFD node, PCI identity, and render minor. KFD topology sysfs can be a
host-global view even when a container receives only one render node, so Apex
reads and validates exactly the KFD driver-node IDs returned by the clean HSA
inventory and ignores unrelated host nodes. Every HSA-visible device must still
map to exactly one device in RSMI's monitor-global inventory, with no joined
device sharing an RSMI index. RSMI-only monitor entries are retained explicitly
in the ownership receipt but are not selectable execution devices. Missing APIs,
byte changes, duplicate identities, disagreement, or a partial HSA-device join
fail closed; there is no fallback to KFD directory order or an RSMI-index guess.

Numeric requested and ROCR selectors address that clean HSA order; `GPU-<uuid>`
selectors address the same inventory directly. Selector order is retained.
ROCR filtering is applied first, then HIP-level ordinal filters are applied to
the resulting ordered view. HIP aliases (`HIP_VISIBLE_DEVICES`,
`CUDA_VISIBLE_DEVICES`, and `GPU_DEVICE_ORDINAL`) must all resolve to the same
ordered physical sequence. The explicit request, when present, must resolve to
that same sequence. Unsupported UUIDs at the HIP layer, out-of-range ordinals,
or any ambiguity fail closed. Magpie's normal contract—ROCR set to the requested
ordered devices and HIP set to `0..N-1`—is therefore represented exactly.

The V3 lease receipt deliberately separates `execution_scope` from
`physical_scope`. `execution_scope` is an ordered, UUID-normalized
`amd-gpu-set=` value used to construct ROCR/HIP execution environments and keep
rank mapping stable. `physical_scope` is a sorted
`amd-gpu-unique-id-set=` used only for locks, recovery checks, and evidence.
Conflating them is invalid. The ownership receipt retains the clean HSA helper
evidence, full cross-namespace inventory, original requested/ambient selector
inputs, and selected devices.

After mapping, Apex queries the RSMI process inventory twice and maps every PID
using the explicit `rsmi_index` field. Any API failure, inventory race,
unreadable process identity, or foreign owner fails closed. PID receipts bind
UID, Linux start time, command-line digest, and RSMI device indices without
persisting command arguments or credentials. The cooperative lease acquires
one lock for every selected physical UUID in canonical sorted order. Therefore
overlapping sets contend on the same per-device inode. A partial acquisition or
post-lock verification failure releases every lock acquired by that attempt.
RSMI can include KFD processes whose process-to-GPU query succeeds with an empty
device set (for example parent or compiler workers before they own a queue).
Those entries remain part of both race-checked snapshots but are not attributed
as owners of any selected physical GPU; a later non-empty mapping changes the
snapshot and invalidates the preflight.

The product never terminates a foreign process. `gpu_foreign_owner` includes the
typed ownership receipt so an authorized operator can resolve the exact PID and
start-time tuple outside Apex and retry. A successful lease records the second
post-lock ownership observation in the canonical run artifacts. V3 embeds the
complete post-lock GPU doctor receipt; a lease cannot be constructed unless
ownership, process/scheduler context, activity scan, and RSMI health are all
formally ready.

The acquisition receipt is not a timeless permission to measure. Each active
lease has a finite TTL (10,800 seconds by default). A heartbeat can renew it
only before expiry and only after rereading the Apex holder's PID/UID/Linux
start-time/cmdline identity plus the complete doctor observation for the same
physical inventory. The fixed RSMI health snapshot uses the junction temperature
sensor supported by MI-series datacenter GPUs together with system clock, busy
percentage, and VRAM usage. Clock regression, expiry, PID reuse, owner drift, foreign
KFD ownership, health loss, or device/visibility drift fails closed. Every
formal measurement is enclosed by typed `measurement_pre` and
`measurement_post` heartbeats; the resulting V1 bracket binds the action, lease
digest, owner, devices, and time interval. No bracket is produced when the post
check fails. Lock metadata is cooperative evidence only: Apex never uses TTL or
heartbeat state to signal or terminate another process.

`apex doctor gpu` is a read-only CLI projection of this ownership inspector. It
never acquires a lock or lease and therefore cannot reserve a device. Its V1
output additionally race-checks procfs cgroup/container, PID/mount/user
namespace, and Slurm identity for every mapped owner and the supervisor. A
bounded exact-name scan records active NHC/`rocminfo`/`rocm-smi`/`amd-smi`
process contexts without retaining command arguments. ROCm health is bound to
the same library hash and ownership receipt: fixed APIs capture edge
Junction temperature, current system clock, busy percentage, and VRAM used/total for each
selected physical UUID. Missing APIs remain an explicit `incomplete` result;
only a clean complete receipt reports formal measurement readiness.

`PythonEnvironment` removes `PYTHONPATH` and disables user-site packages for
package probes, preventing an old editable Magpie checkout from silently winning
import resolution. TraceLens uses a locked Git commit plus a base-version prefix
because its upstream editable version includes build date and commit metadata.
InferenceX is a repository-only dependency: its exact clean Git identity is
verified and receipted, but the bootstrapper does not invent a Python package
installation for it. Resolved benchmark views carry that exact root so Magpie's
moving-branch auto-clone path is never used.

The dependency lock also names one mandatory Magpie corpus manifest. Verification
binds the Magpie repository commit/tree, the `examples/benchmarks` subtree, and
the sorted path/SHA-256 inventory of every YAML config. Missing, added, renamed,
or byte-modified configs fail dependency verification before an E2E controller is
constructed. Regenerate the checked file only from the exact reviewed checkout
with `scripts/build_magpie_corpus_manifest.py`; the manifest is evidence about the
locked corpus, not a claim that every config has completed live qualification.
The same lock names a generated compatibility ledger. The checked-in file is an
Apex projection: each row is joined to the corpus by path and config
SHA-256 and freezes the required `e2e_throughput_qos_v1` metrics, but the file is
explicitly marked non-authoritative for live release claims. `collect-local`
loads every frozen config through published Magpie main APIs and binds each
Apex-owned plan/capability receipt plus one self-digested corpus result. Missing
rows, digest drift, an unavailable public API, or any
`capability_upgrade_required` blocks release qualification. The exact pin resolves
all 27 configs for identity, while the V2 live product scope is the exact 21
Docker one-shot entries; the other six remain audited inputs and must return
`e2e_docker_only` before GPU or agent work. Regenerate a future ledger only with
`scripts/build_magpie_compatibility_ledger.py` from the exact locked checkout.
No config-resolution or ledger claim is GPU, workflow,
winner, or formal-delivery evidence.

## Release-candidate identity

`inspect_release_candidate` combines locally recomputed Git/lock/corpus/template
identity with explicit typed evidence. It performs no fetch, image build, GPU
work, agent launch, benchmark, or sanitizer execution. A release remains
`blocked` until Apex and Magpie have fresh reviewed remote-main audits, the Apex
checkout is clean, dependencies/imports and lm-eval have fresh verification,
the complete CPU/static gate passes on that tree, the installed CLI matches, all
required images are immutable, all named live qualifications pass, and all four
canonical showcases pass official offline bundle/reward/reproduction checks.

Live gates use `apex.release-qualification/v2`, not a name plus an arbitrary
digest. Backend receipts bind agent identity, ordinary coding, formal kernel,
measurement policy, GFX950 coverage, and delivery. Recovery binds both task kinds,
all eight before/after fault boundaries, reference/recovered manifests, duplicate
prevention, partial-window discard, and GPU-identity rejection. Knowledge
ablation binds all three arms and matched dimensions to evaluator-owned episodes;
AKA binds the independent validator, central regrade, and matched cohort of at
least ten tasks. Magpie V2 live evidence keeps the full 27-row resolution
manifest as its subject but binds live coverage to the derived Docker one-shot
slice, plus workflow, quality, reward, framework, lifecycle, source-adapter, and
formal-delivery receipts. The release gate anchors every resolved row's
run-mode/lifecycle to the checked product scope, recomputes the exact 21-row
slice and six-row rejection-complement digests, requires early-rejection
receipts, and binds each delivery representative to a concrete selected config,
plan, and capability receipt. A self-consistent live receipt therefore cannot
reclassify, omit, or substitute rows.
Missing or weakened typed fields fail before release status is assessed.
Typed shape and a self-digest still do not prove that any campaign ran. Release
candidate v2 therefore requires every `qualified` record to be reverified by an
injected `QualificationAuthorityPort` over evaluator-owned artifact manifests.
The resulting authority receipts are persisted in the release candidate and
recomputed on verification. An explicit external formal result root is resolved
through `FormalResultsRootValidator`; the index is re-read around verification,
every CAS file is no-follow, single-link, size/digest checked, and only an
installed kind-specific verifier may recompute evidence. The current production
registry installs the three backend verifiers but not the Magpie, recovery,
ablation, or AKA verifiers. Those claims therefore remain typed `unavailable`
even with a valid CAS index; joined JSON fragments cannot manufacture release
readiness.

Published showcases similarly require the path-free
`apex.showcase-verification/v2` receipt produced by offline verification. Its
self-digest binds checksums, episode, artifact manifest, reward, result, and
reproduction bytes plus replay/bundle/reproduction verdicts. The obsolete
boolean-only release showcase shape is rejected.

The checked-in 27-row Magpie ledger is historical identity input, not release
authority. Fresh baseline evidence must cover the exact corpus through the Apex
config adapter and bind its resolution-manifest digest. Separate
`magpie-corpus-live` evidence must bind that full digest, cover the exact derived
Docker one-shot V2 slice, and include formal-delivery coverage.
The three attributed template locks
similarly remain visibly pending until immutable in-image source identities,
Apex-owned evaluation, and real published showcases exist.

The receipt is not trusted merely because its SHA-256 is internally consistent.
`verify_release_candidate_receipt` parses every evidence record, rereads current
static source/lock bytes, reconstructs the whole document, and requires
byte-equivalent output. Editing `status`, removing a blocker, changing a lock,
adding a config, or reusing evidence from another Apex tree fails. Only
`freeze_release_candidate` returns a final release-authorizing receipt.

The document deliberately has two non-circular gates. `baseline_status` and
`baseline_blockers` require a clean official Apex checkout at the exact fetched,
reviewed tip of either remote `main` or a remote `codex/*` qualification ref,
plus fresh exact dependency/runtime verification (including the clean locked
Magpie tree), the full CPU/static gate, config resolution, and installed CLI.
`freeze_campaign_baseline` accepts that state so a reviewed PR commit's experimental
evidence can be admitted to qualification before merge. Optimization does not
consume this receipt; it records `apex.execution-identity/v1` and qualification
must match that recorded tree. Top-level release `status`/`blockers`
still require exact remote `main` audits for both Apex and Magpie, immutable
showcase images, all live/independent qualifications, and four published
showcases. A `codex/*` ref can never authorize final release. Experimental runs
may use dirty or unreviewed Apex bytes only with that state recorded explicitly;
such runs cannot pass release qualification.

Build a truthful snapshot, or verify an existing one:

```bash
.venv/bin/apex release collect-local \
  --apex-root /absolute/clean/Apex \
  --output /absolute/operator-selected/local-release-evidence.json

.venv/bin/python scripts/build_release_candidate_receipt.py \
  --evidence /absolute/operator-selected/local-release-evidence.json \
  --output /absolute/operator-selected/release-candidate.json

.venv/bin/python scripts/build_release_candidate_receipt.py \
  --verify /absolute/operator-selected/release-candidate.json
```

Pass `--evidence evidence.json` with `ReleaseEvidence.to_dict()` and
`--require-baseline` for a live-campaign preflight or `--require-ready` for the
final release gate. Output paths are operator-selected and never overwritten.
`release collect-local` is the only built-in producer for the local subset. It
refuses a dirty checkout, runs the exact full pytest/compileall/forbidden-source
argv with bounded output and timeouts, reuses the single pinned dependency
verifier, proves the installed console script imports this checkout, and checks
the complete static identity again after the gate. It does not fetch remotes,
assert ancestry review, inspect images, use a GPU, launch an agent, or manufacture
live/showcase qualifications; those fields remain absent and therefore blocked.

Intake provenance follows Magpie's run mode. An explicit Docker image is
inspected when available; a Docker config that delegates image selection to
Magpie remains `partial` with `runtime_image_selection`, while local and Ray
configs remain `partial` until their runtime or worker identities are observed.
Those modes do not require a fictitious Docker image at intake. Missing runtime,
source-lock, model-revision, and loaded-byte proof remains explicit and prevents
formal source delivery, but does not incorrectly reject a valid raw Magpie
config before baseline execution.
`RunProvenance` schema version 2 records the resolved `run_mode` as a typed field;
deployment registries consume it together with the active source component so
local, Ray, and Docker authorities cannot be confused.
These generic provenance adapters are lower-level future inventory; Plan V2's
`apex optimize e2e` rejects Local and Ray before reaching them.

`FormalResultsRootValidator` protects evaluator-owned output placement. It
canonicalizes a caller-selected absolute root without creating it, rejects every
existing symlink component, and rejects either direction of overlap with the
Apex checkout, task workspace, verified dependency roots, and exact E2E source
roots supplied by the composition root. A new live campaign can additionally
require that the selected root not already exist. In-tree ignored directories
are not an exception: they are suitable only for explicitly non-authoritative
preflight material.

## Entrypoints and tests

The fresh-checkout launcher is `scripts/bootstrap_dependencies.py`. It prepares
Apex itself in the selected venv, then executes this module's CLI. Detailed user
instructions are in `docs/dependencies.md`.

Focused CPU tests:

```bash
pytest -q tests/test_bootstrap_dependencies.py
pytest -q tests/unit/runtime/test_magpie_corpus.py
pytest -q tests/unit/runtime/test_magpie_compatibility.py
pytest -q tests/unit/runtime/test_evaluator_lock.py
pytest -q tests/unit/runtime/test_lm_eval_runtime.py
pytest -q tests/unit/runtime/test_source_locks.py
pytest -q tests/unit/runtime/test_release_candidate.py
pytest -q tests/unit/runtime/test_formal_results.py
pytest -q tests/unit/runtime/test_hsa_inventory.py \
  tests/unit/runtime/test_gpu_ownership.py tests/unit/runtime/test_gpu_lease.py
```

The tests perform no remote access and include a real temporary-venv
install/repeat/verify cycle.

## Purpose

Runtime resolves pinned dependencies, repositories, provenance, receipts, and
run-scoped cooperative GPU leases needed by higher-level adapters.

## Public API

The lazy `apex.runtime` facade exports bootstrap, dependency, repository,
provenance, receipt, GPU-lease, and read-only qualification-authority contracts
without eager executable imports. The facade also exports the evaluator-policy
lock and GPU lifecycle/bracket types used by formal measurement. Production
composition installs the shared RL backend verifier only for Codex, Claude, and
Cursor on gfx950.

## Invariants

Dependency commits, source trees, and lock digests are exact; repository resolution is confined,
imports perform no I/O, and GPU leases are scoped to visible devices and run IDs.
Qualification manifests are locators, never claims: every accepted backend gate
is recomputed from canonical CAS receipts and a replayed evaluator reward.

## Dependencies

Runtime depends downward on core, ports, storage receipts, and supervised
execution. It never imports benchmark, optimization, CLI, delivery, or RL
policy; the composition root injects the higher-level episode verifier.

## Failure semantics

Missing/mismatched checkouts, evaluator task/dataset lock drift, Python
incompatibility, invalid or missing raw qualification receipts, dirty source
where cleanliness is required, lease conflict, or expired/drifted measurement
authority fails before workload use.

## Tests

Bootstrap and runtime unit tests use temporary repositories/environments and
exercise install, repeat, source materialization, tree/dirty-state rejection,
verification, evaluator lock/task drift, provenance, lease contention/lifecycle,
and read-only CAS tamper rejection.

## Provenance

Receipts record lock/schema hash, Python identity, canonical roots, exact commits and trees,
container identity, evaluator task/dataset/runtime identity, source trees, device
lease ownership and measurement brackets, and the installed qualification
verifier identity digest.
