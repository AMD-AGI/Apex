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
| `BootstrapError` | `repositories.py` | Deterministic bootstrap failure boundary |
| `canonical_repository`, `inspect_repository` | `repositories.py` | Repository identity helpers |
| `probe_errors`, `version_matches` | `dependencies.py` | Package version/import-root verification helpers |
| dependency CLI parser/composition | `dependency_cli.py` | Executable setup/verify surface kept out of the lock domain module |
| `DependencyReceipt`, `verify_runtime_dependencies` | `receipt.py` | One verified identity consumed by all runtime adapters |
| `LmEvalRuntimeLock`, `load_lm_eval_runtime_lock` | `lm_eval_lock.py` | Strict source, wheel, base-image, ABI, installed-tree, and evaluator-identity lock |
| `LmEvalRuntimePreparer` | `lm_eval_prepare.py` | Networkless exact wheel build/install/smoke producer and atomic CAS publisher |
| `LmEvalRuntimeReceipt`, `verify_lm_eval_runtime` | `lm_eval_runtime.py` | Independent byte/mode/tree verification for the immutable evaluator CAS |
| `SourceLockSet`, `SourceLockSpec`, `load_source_lock` | `source_locks.py` | Strict formal E2E source lock and checked-in content digest |
| `SourceLockManager`, `SourceLockReceipt` | `source_locks.py` | Managed exact-checkout materialization, read-only verification, and receipts |
| `RunProvenance`, `ProvenanceResolver`, `RepositoryLock` | `provenance.py` | Best-effort image observation and strict clean-source locks |
| `GpuLeaseManager`, `LocalGpuLeaseManager`, `GpuLeaseReceipt` | `gpu.py` | Run-scoped physical-GPU lock bound to ownership evidence |
| `GpuDeviceIdentity`, `GpuSelectorRequest`, `RsmiDeviceIdentity` | `gpu_topology.py` | Ordered HSA selector composition and HSA/KFD/DRM/RSMI identity join |
| Internal bounded RSMI adapter | `gpu_rsmi.py` | Fixed-signature ctypes calls for monitor identities and KFD process maps |
| `HsaInventoryEvidence`, `CleanHsaInventoryProvider` | `hsa_inventory.py` | Hash-bound, unfiltered HSA agent enumeration in a fresh helper process |
| `GpuOwnershipReceipt`, `RocmSmiGpuOwnershipInspector` | `gpu_ownership.py` | Race-checked physical identity and RSMI PID-to-GPU preflight |

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

The V2 lease receipt deliberately separates `execution_scope` from
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

The product never terminates a foreign process. `gpu_foreign_owner` includes the
typed ownership receipt so an authorized operator can resolve the exact PID and
start-time tuple outside Apex and retry. A successful lease records the second
post-lock ownership observation in the canonical run artifacts.

`PythonEnvironment` removes `PYTHONPATH` and disables user-site packages for
package probes, preventing an old editable Magpie checkout from silently winning
import resolution. TraceLens uses a locked Git commit plus a base-version prefix
because its upstream editable version includes build date and commit metadata.
InferenceX is a repository-only dependency: its exact clean Git identity is
verified and receipted, but the bootstrapper does not invent a Python package
installation for it. Resolved benchmark views carry that exact root so Magpie's
moving-branch auto-clone path is never used.

## Entrypoints and tests

The fresh-checkout launcher is `scripts/bootstrap_dependencies.py`. It prepares
Apex itself in the selected venv, then executes this module's CLI. Detailed user
instructions are in `docs/dependencies.md`.

Focused CPU tests:

```bash
pytest -q tests/test_bootstrap_dependencies.py
pytest -q tests/unit/runtime/test_lm_eval_runtime.py
pytest -q tests/unit/runtime/test_source_locks.py
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
provenance, receipt, and GPU-lease contracts without eager executable imports.

## Invariants

Dependency commits, source trees, and lock digests are exact; repository resolution is confined,
imports perform no I/O, and GPU leases are scoped to visible devices and run IDs.

## Dependencies

Runtime depends on core and supervised execution only. It never imports benchmark,
optimization, CLI, delivery, or storage policy.

## Failure semantics

Missing/mismatched checkouts, Python incompatibility, invalid receipts, dirty
source where cleanliness is required, or lease conflict fails before workload use.

## Tests

Bootstrap and runtime unit tests use temporary repositories/environments and
exercise install, repeat, source materialization, tree/dirty-state rejection,
verification, provenance, and lease contention.

## Provenance

Receipts record lock/schema hash, Python identity, canonical roots, exact commits and trees,
container identity, source trees, and device lease ownership.
