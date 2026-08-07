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
| `GpuLeaseManager`, `LocalGpuLeaseManager`, `GpuLeaseReceipt` | `gpu.py` | Run-scoped cooperative cross-process GPU ownership |

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
