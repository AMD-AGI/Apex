# Delivery

## Purpose

`apex.delivery` turns independently measured source changes into machine-readable
results and content-digested, unapplied artifacts. Delivery never edits the
caller's checkout. A site-packages overlay, agent claim, mutable image tag, or
config-only change is evidence for neither a standalone kernel reward nor a
verified E2E source delivery.

The package owns two distinct outputs. A standalone `KernelBundle` is the small
source-only patch contract consumed by AgentKernelArena or another external
evaluator. An `E2EPatchBundle` captures exact multi-repository source locks,
patches, a trusted fixed-argv build recipe, immutable image identities, runtime
engagement evidence, and a clean replay. Both bundles are new directories whose
declared bytes determine their digest; neither is automatically applied.

## Public API

Only names exported from `apex.delivery.__all__` are supported:

- `TaskResult` and `write_task_result` provide the atomic machine result for a
  standalone run, including safety and robust-measurement fields.
- `build_kernel_bundle`, `load_and_verify_kernel_bundle`, and
  `compute_bundle_digest` own the generic source-only kernel bundle;
  `apply_verified_kernel_bundle` is the explicit exact-clean-baseline mutation.
- `capture_repository_patch`, `CleanPatchMaterializer`, and the E2E bundle
  functions capture and verify exact Git source changes.
- `E2EBundleVerifierRouter` selects a reviewed profile only from the exact frozen
  recipe digest. `E2EBundleVerifier`, `SupervisedRecipeBuildBackend`, and their
  build, engagement, and replay ports then perform independent source-delivery
  verification.
- Receipt and lock dataclasses are immutable evidence values. Callers should not
  import private module helpers or infer a verdict from presentation text.
- `capture_portable_bundle` first runs the kind-specific official loader, stores
  every verified bundle file in CAS, and emits exact `winner_bundle` and
  `bundle_verification` receipts. `verify_portable_bundle` reconstructs those
  bytes and reruns the same loader offline.
- `kernel_reproduction_declaration` and `e2e_reproduction_declaration` project
  only bundle-bound source, recipe, image, config, authority, and fixed-argv
  identities. The two task kinds deliberately have different replay gates.

For standalone measurement, the evaluator writes a fresh regular JSON file with
schema `apex.kernel-measurement/v1`, policy
`kernel_invocation_nearest_rank_v1`, sample unit `kernel_invocation`, quantile
method `nearest_rank_v1`, a timer/resolution/method hash, `inner_repeats=1`, an
explicit warmup count and seeded, health-bracketed paired ABBA blocks. Blocks
carry raw positive finite samples and explicit invalid-sample counts; optional
workload counts remain case-level. Agent-reported summaries are not part of this
API.

## Invariants

Standalone grading requires at least 300 valid invocation samples for each
implementation in every case. Apex computes a true median p50 (including the
average of the two middle values for an even sample count) and a versioned
nearest-rank p99 with rank `ceil(0.99 * N)`. Thus 300 samples provide three tail
observations. Missing p99, fewer than 300 valid samples, a non-finite/non-positive
sample, or a policy/unit mismatch yields no reward; evidence is never substituted
with p50, `1.0x`, or `0.0x`.

For each case:

```text
S50     = Tref,p50 / Topt,p50
S99     = Tref,p99 / Topt,p99
Srobust = min(S50, S99)

Reward = 20 * Icompile
       + Icorrect * (100 + 200 * clip(Srobust - 1, -0.25, 1.00))
```

`Icorrect` represents the complete correctness, integrity, anti-tampering, and
applicable safety gate, not an agent assertion. A valid multi-case grade also
records `worst_case_srobust` and the declared equal-case or workload-weighted
aggregation. The serialized `TaskResult` binds the raw report SHA-256 and
`kernel_robust_v1`; status `valid` requires complete finite grade fields, and
`Srobust` must equal `min(S50, S99)`.

The reward formula above is unchanged. Internal promotion additionally requires
strict `Srobust > 1.05`, a seeded paired/block-bootstrap confidence lower bound
above `1.0`, population CV at or below the frozen limit (default `0.10`), and
`worst_case_srobust >= 1.0`. A valid non-promoted grade retains its point reward
and typed gate evidence. `TaskResult` also carries the run ID, baseline lock,
internal verdict event reference, verification-summary receipts, event-journal
head, artifact-store receipt set, and structured terminal error. None of these
lineage fields modifies or applies the standalone bundle.

The evidence event names are intentionally non-interchangeable:

```text
performance_command_result  normal-runtime command completed; no grade implied
measurement_result          evaluator parsed raw samples and recomputed a grade
reward_committed             evaluator committed an eligible scalar/vector reward
```

Only the evaluator owns `measurement_result` and `reward_committed`. A successful
performance command cannot create either event by itself. Missing, invalid, or
insufficient p99 may produce `measurement_result`, but never `reward_committed`.

The standalone bundle remains immutable and unapplied (`applied=false`); an
external evaluator must verify and apply it in its own isolated workspace. An
E2E result reaches `source_rebuild_verified` only when all of the following hold:

1. every changed repository starts at its declared clean commit/tree and blobs;
2. ordered patches apply, reverse to the byte-exact base, and reapply;
3. a controller-trusted argv recipe builds declared artifacts and an SBOM from
   an immutable parent image;
4. a fresh runtime proves it loaded those exact artifact bytes or build IDs;
5. measurement and replay configs differ only in the immutable derived image;
6. a second fresh environment rebuilds and passes quality, latency, and
   objective-improvement gates with normal, non-instrumented measurement.

Runtime-overlay evidence may truthfully retain
`validation_level=runtime_overlay_verified`, but cannot become a formally
verified source-rebuild success. Advisory safety gaps remain visible as
`safety_certified=false`.

A successful clean-replay receipt enumerates each raw benchmark and quality file
with role, run identity, measurement/quality identity, digest, size, and media
type. Verification rejects path escapes, symlinks, missing files, and byte drift.
Optimization copies verified E2E files into the main CAS before publishing its
unique task-terminal reward. Standalone delivery similarly carries the terminal
result/vector/source/policy identities produced by evaluation; selected/no-op
grades bind raw invocation evidence while trusted gate failures bind their exact
command receipts. Delivery cannot replace raw proof with a summary, aggregate
attempt rewards, or infer a terminal scalar itself.

Delivery carries either the exact receipt supplied by an independent trusted
safety evaluator or the explicit uncertified no-tool state. It cannot infer tool
coverage, promote not-applicable/inconclusive to clean, or manufacture a safety
receipt from correctness output. Instrumented or simulated artifacts are
`timing_eligible=false` and can become neither delivery runtime bytes nor scoring
measurements. See the [primary safety contract](../evaluation/safety/README.md).

A role label is not bundle evidence. Showcase qualification requires one event
with the portable bundle declaration, its delivery-owned verification receipt,
and a contiguous binding for every declared file. Export verifies this contract
against the source CAS; offline verification reconstructs the exported tree and
reruns `load_and_verify_kernel_bundle` or `load_and_verify_e2e_bundle`. Missing
or invalid portable evidence cannot be published.

## Dependencies

Delivery depends inward on `core`, intake contracts, the shell-free execution
supervisor, and runtime Git URL normalization. It may use PyYAML to compare
benchmark documents. It does not depend on orchestration, agents, Magpie or
TraceLens internals, AKA task/scoring code, or a mutable user workspace.

Build, runtime engagement, and clean replay are injected ports. The composition
root owns the trusted repository and recipe registries: bundle content may not
authorize its own clone URL or command. Commands are argv arrays with
`shell=False` semantics; there is no arbitrary shell-string fallback.

The trusted recipe registry also supplies each component's language,
engagement kind, and build-ID requirement. `SourceComponentCapability` is
compared with every frozen source lock; a bundle cannot lower this policy by
declaring weaker loaded-byte or build-ID requirements. Built and loaded artifact
sets must match exactly when engagement is verified.

Concrete workload profiles are composed lazily after static bundle verification.
An unrelated recipe cannot silently use the reviewed Qwen vertical-slice
backends; an unknown recipe fails with `e2e_verifier_profile_unavailable`, and
two profiles may not claim the same recipe digest.

## Failure semantics

Static schema, path, file-set, digest, or tree tampering raises an integrity or
contract error before build or GPU work. Bundles reject symlinks, Gitlinks,
submodules, binary source patches, undeclared files, path traversal, `.git`
paths, unsupported modes, untrusted repository URLs, and untrusted recipes.

For standalone results, `insufficient_samples`, `invalid`, `unsupported`, and
`error` are explicit measurement statuses with `reward=null`; they do not emit a
bundle as a locally proven optimization. `not_configured` carries no report,
policy, grade, or reward evidence. A `valid` result without an exact report
digest, the canonical policy, all robust metrics, or internally coherent
`Srobust` is rejected during `TaskResult` construction.

Once independent E2E verification starts, evidence failures produce a structured
`verification_failed` result and no final bundle. Unknown source locks map to
`provenance_unresolved`. Known locks with failed patch application, build,
engagement, or replay map to `verification_failed`. Retry uses a new results
path, so partial evidence cannot masquerade as resumed success.

The production CLI reaches this use case through the sole bootstrap composition
root. `apex bundle verify --bundle /abs/bundle --results /abs/new-results`
requires an unused absolute evidence directory. It exits successfully only when
the verifier's typed result is `succeeded`; a structurally valid but terminally
unverified E2E bundle is an input to verification, not a successful verdict.

## Tests

Run the deterministic delivery suite and architecture documentation gate with:

```bash
pytest -q -p no:cacheprovider --import-mode=importlib \
  tests/unit/delivery tests/architecture/test_source_architecture.py
```

Tests cover `TaskResult` measurement serialization and coherence; atomic writes;
single- and multi-repository apply/reverse; add/delete/rename/mode metadata;
dirty or wrong bases; path, symlink, submodule, manifest, recipe, and config
attacks; image/SBOM and loaded-byte mismatch; clean replay; and terminal status
policy. Temporary Git repositories and fake ports keep this suite independent of
Docker and GPUs.

Fresh replay evidence binds a distinct clean-source materialization digest, the
primary runtime identity, and a unique runtime identity for every A/B/B/A
observation. Every raw replay measurement also binds its protected execution
attestation artifact; caller-provided `fresh=true` flags are not proof.

`apex.optimization.e2e.SourceRebuildFinalDelivery` owns the authoring-side
composition: cumulative clean source materialization, canonical patch capture,
primary source-build evidence, candidate bundle assembly, and invocation of this
package's independent verifier. Production still must inject reviewed
`SourceBuildBackend`, `EngagementBackend`, and `CleanReplayBackend`
implementations, plus a primary source-build/benchmark attestor, for each exact
immutable image and vLLM/AITER repository set. There is deliberately no guessed
recipe or fallback that mutates installed files.

## Provenance

This package is an Apex-native implementation of the delivery and robust-grade
contracts in `tmp/plan.md`, including its source-rebuild Definition of Done. It
does not copy Hyperloom, GEAK, Magpie, TraceLens, or AgentKernelArena source. The
standalone `KernelBundle` shape remains deliberately compatible with the Apex
launcher contract in AKA, while AKA retains sole authority for its own score.
