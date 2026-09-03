# Pinned runtime dependencies and E2E source locks

Apex uses reviewed Magpie, TraceLens, and InferenceX checkouts. Magpie owns
benchmarking/grading, TraceLens owns trace analysis, and InferenceX supplies the
serving benchmark harness. Formal E2E delivery additionally uses reviewed vLLM and
AITER source trees. None may resolve from a moving `main` branch.

## One-command setup

From the Apex repository root:

```bash
python3 scripts/bootstrap_dependencies.py install
source .venv/bin/activate
apex dependencies prepare-runtime --json
```

The install command creates or reuses `.venv`, installs the two locked Python
projects, materializes the repository-only InferenceX pin, and publishes exact
vLLM/AITER source trees under `~/.cache/apex/source-locks`. It
then verifies all of the following:

- Git origin, exact 40-character commit, and clean worktree;
- installed Python distribution version for Magpie and TraceLens;
- the filesystem root from which `Magpie` and `TraceLens` are actually imported.
- the exact clean InferenceX root supplied to every resolved benchmark view.
- vLLM/AITER origin, exact commit, exact `HEAD^{tree}`, and clean worktree.

Running the same command again is safe. If the environment already imports the
locked checkout, installation is skipped and the action is reported as
`already-installed`.

`prepare-runtime` is the one-time serving-quality setup. It downloads only
artifacts named in `scripts/lm_eval_runtime.lock.json`, verifies each byte, builds
lm-eval plus source-only dependency wheels in the exact pinned vLLM image, and
installs the target with `--no-index --no-deps --no-compile --target`. Every
build/install/smoke container has `--network=none`. The smoke imports lm-eval
from the isolated target, checks all reused parent-image distribution versions,
constructs the OpenAI-compatible model adapter, and discovers InferenceX's
locked `gsm8k` task.

The result is published at
`.cache/apex-runtime/lm-eval/sha256/<runtime_sha256>` with 0444 files and 0555
directories. Its root contains only `site-packages/` and
`lm_eval_runtime_manifest.json`. Symlinks, hardlinks, special files, writable
bits, unexpected files, wheel/base-package overlap, and digest drift all fail
closed. No GPU is used to prepare or verify this runtime.

Current pins live in
[`scripts/dependencies.lock.json`](../scripts/dependencies.lock.json). The lock
also records the receipt schema, repository URL, package version policy, Python
distribution, import root, extras, and runtime root environment variable.
Formal source pins live separately in
[`scripts/e2e_source_locks.json`](../scripts/e2e_source_locks.json). Keeping this
lock separate prevents source-only delivery inputs from masquerading as installed
Python dependencies. Its receipt binds the lock-file SHA-256 and each selected
root, expected repository, observed origin, commit, tree, cleanliness, and
resolution method.

## Source resolution

For each dependency, the bootstrapper uses this precedence:

1. `--magpie-root`, `--tracelens-root`, or `--inferencex-root`;
2. `MAGPIE_ROOT`, `TRACELENS_REPO_PATH`, or `MAGPIE_INFERENCEX_PATH`;
3. an exact sibling checkout under `--sibling-root` (default: Apex's parent);
4. an exact managed checkout under `--checkout-root`;
5. a newly created managed checkout at the locked commit.

An explicit path must match the lock exactly; it never silently falls back. A
sibling checkout is never reset or switched. If the sibling has moved ahead but
still contains the locked commit, the bootstrapper clones that commit locally
into `.cache/apex-dependencies`, leaving the sibling untouched. Otherwise it
uses the locked repository URL unless `--offline` was requested.

The default layout therefore works directly with:

```text
/home/viouyang/
├── Apex/
├── Magpie/
└── TraceLens/
```

Case matters: the supported checkout is `TraceLens`, not `Tracelens` or
`tracelens`.

### Formal E2E source resolution

vLLM and AITER use a stricter managed-source flow. Apex first accepts an exact
`--vllm-source-root` / `--aiter-source-root` (or
`APEX_VLLM_SOURCE_ROOT` / `APEX_AITER_SOURCE_ROOT`). Otherwise it verifies the
managed path under `--source-lock-root`, whose default is:

```text
~/.cache/apex/source-locks/
├── vllm-v0.19.1/
└── aiter-v0.1.10.post2/
```

If a managed checkout is missing, a same-origin `vllm/` or `aiter/` sibling may
supply the exact commit object. Apex clones it with `--no-hardlinks` into a temporary
directory, detaches at the pin, verifies origin/commit/tree/cleanliness, and
atomically renames it into the managed location. The sibling is only read: it is
never reset, cleaned, checked out, fetched, or otherwise modified. This works
offline even when the sibling branch has advanced, provided the pinned commit
remains reachable. Without a suitable local object, online setup clones the
reviewed URL.

## Verification and machine-readable receipts

Verify without installing:

```bash
python3 scripts/bootstrap_dependencies.py verify
```

Emit a stable JSON receipt for logs or a runtime preflight:

```bash
python3 scripts/bootstrap_dependencies.py verify --json
```

Require and fully rehash the evaluator runtime as a separate preflight:

```bash
apex dependencies verify-runtime --json
```

`verify` reports the runtime when the default CAS entry exists;
`verify-runtime` treats its absence as an error. An operator may select another
absolute CAS root with `--lm-eval-runtime` or `APEX_LM_EVAL_RUNTIME`, but the
same lock identity and every file byte/mode are still recomputed.

Each dependency entry contains the lock digest, selected root, resolution
method, exact commit, version rule, observed installed version, import file, and
the environment variable used by runtime adapters. A successful receipt has
schema `apex.dependencies.receipt/v1` and `status: verified`.

The receipt also contains `magpie_corpus` and `magpie_compatibility`. The first
binds the exact benchmark subtree/path/hash inventory; the second joins every
config to its phase-view and orthogonal capability result under
`e2e_throughput_qos_v1`. Regenerate the latter with
`python scripts/build_magpie_compatibility_ledger.py` after a reviewed Magpie pin
change. A missing row, config hash drift, or `capability_upgrade_required` blocks
dependency verification. Workflow and formal-delivery qualification remain
separate live receipts and are not inferred by this CPU ledger.

For release qualification, dependency verification is one input to
`apex.release-candidate-receipt/v2`. Build the deterministic snapshot with
`scripts/build_release_candidate_receipt.py`; it also binds the clean Apex tree,
fresh reviewed Apex/Magpie remote tips, full CPU/static gate, installed CLI,
immutable images, live qualifications, and all four canonical showcase
verifications. The command never fetches, launches an agent/GPU, builds an image,
or upgrades a pending claim. Without separately supplied typed evidence it emits
a truthful `blocked` receipt. `--require-ready` fails unless current source and
lock bytes reconstruct a blocker-free receipt.

The receipt separates `baseline_status` from final release `status`.
`--require-baseline` checks whether a matching experimental run may be admitted
as release qualification evidence; `--require-ready` additionally checks the live
qualifications and showcase outputs. Optimization itself records dependency and
execution identities without consuming this release receipt.

The same JSON contains `e2e_source_locks`, with nested receipt schema
`apex.e2e-source-locks.receipt/v1`. Production `apex optimize e2e` consumes these
verified roots from `DependencyReceipt`; it no longer reconstructs unverified cache
paths. `verify` and `verify-runtime` never materialize a missing formal source and
fail closed. `install` materializes it, and `prepare-runtime` also materializes it
so preparation is safe to repeat after a cache cleanup.

Preview dependency source selection without cloning or installing:

```bash
python3 scripts/bootstrap_dependencies.py install --dry-run --json
```

On a fresh checkout the small launcher may still create `--venv` and install
Apex itself there so it can execute the authoritative `apex.runtime` resolver.
It does not modify either dependency during a dry run.

## Offline setup

When sibling checkouts and Python requirements are already present:

```bash
python3 scripts/bootstrap_dependencies.py install --offline
```

Offline mode forbids remote clones and passes `--no-index --no-deps
--no-build-isolation` to pip. It can clone the locked commit from a local sibling
even when that sibling's current branch has advanced. It fails clearly if the
commit or an already-installed Python requirement is unavailable; it never
substitutes a different revision.

After the locked download cache is populated, a clean runtime can also be
reproduced without network access:

```bash
apex dependencies prepare-runtime --offline --json
```

The reviewed installed-tree digest is
`23dc17079da4619a4cb37100f66f015dd9dd818df46e9f0ea16b541deaf27f60`
and its identity-bound runtime digest is
`ca744a9e0ab994eba275a0fc0b01b762247f76f9cd0129b31b5dc2969b23732e`.
These replace an unreproducible intermediate pair discovered during producer
validation; source, wheel, base-image, Python ABI, and InferenceX pins were not
changed. Two builds from empty runtime directories produced byte-identical
manifests accepted independently by Apex and Magpie.

## Useful overrides

```bash
# Use a dedicated environment.
python3 scripts/bootstrap_dependencies.py install --venv /path/to/venv

# Use explicitly selected source trees. All must match the lock.
python3 scripts/bootstrap_dependencies.py install \
  --magpie-root /path/to/Magpie \
  --tracelens-root /path/to/TraceLens \
  --inferencex-root /path/to/InferenceX

# Keep managed checkouts somewhere else.
python3 scripts/bootstrap_dependencies.py install \
  --checkout-root /path/to/apex-dependencies

# Keep formal E2E sources somewhere else.
python3 scripts/bootstrap_dependencies.py install \
  --source-lock-root /path/to/apex-source-locks

# Select already-materialized exact sources explicitly.
python3 scripts/bootstrap_dependencies.py install \
  --vllm-source-root /path/to/vllm \
  --aiter-source-root /path/to/aiter
```

The script never accepts a dirty or wrong-revision source. If an explicit or
managed checkout fails validation, either restore it outside Apex or select a
different clean checkout; the bootstrapper deliberately does not run `reset`,
`clean`, or branch-changing commands in an existing repository.

## Tests

The bootstrap contract is CPU-only and does not access the network:

```bash
pytest -q tests/test_bootstrap_dependencies.py
```

The tests cover lock validation, HTTPS/SSH origin equivalence, sibling priority,
offline cloning without sibling mutation, commit-tree mismatch rejection before
publication, dirty-source rejection, explicit-path fail-fast behavior, split-brain
import detection, TraceLens dynamic version matching, offline pip flags, and
repeat-install idempotence.
