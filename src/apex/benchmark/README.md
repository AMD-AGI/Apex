# Benchmark

This module is Apex's only E2E benchmark execution boundary. It delegates
execution to the exact Magpie checkout recorded by `DependencyReceipt`; it does
not import an in-tree fallback, manipulate kernels, or implement another
profiler.

## Phase views

`build_config_views` writes four immutable artifacts:

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
is never installed into or inferred from the workload image. For serving
frameworks, measurement and replay freeze `RUN_EVAL=true` and an explicit
`MAGPIE_EVAL_TASKS` value (default `gsm8k`). The diagnostic view keeps that task
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
from Magpie's `quality_gate`. An immutable evaluator-only policy is activated
when the exact reviewed Qwen config hash is selected. That policy keeps the
server `MAX_MODEL_LEN=2248`, while binding lm-eval `max_length=2248`,
`max_gen_tokens=480`, and `exact_match,strict-match`. Its digest is part of
workload and accuracy semantics; the original benchmark YAML is copied
byte-for-byte. Apex independently rehashes the results and raw sample receipts
and requires Magpie's outcome and sample-set digests to match.

Required measurement/replay quality evidence that is missing,
ambiguous, skipped, or empty makes the run fail closed even if Magpie reports
process success. The report must also agree with the requested evidence lane:
measurement/replay require `run_kind=measurement`, `reward_eligible=true`, and
profiling off; diagnostics require the inverse and accept only Magpie's explicit,
artifact-free `lm_eval_runtime.status=not_requested` receipt. Candidate-vs-baseline
regression policy belongs to the E2E optimization module, not this adapter.

Docker evidence uses `magpie.serving-runtime-receipt/v2`. The receipt keeps the
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

## Purpose

The package is Apex's Magpie boundary: it freezes phase-specific workload views
and normalizes evaluator-owned performance and quality evidence.

## Public API

Use `build_config_views`, `validate_phase_set_contract`,
`validate_resolved_view`, `MagpieBenchmarkAdapter`, and the immutable result
types exported by `apex.benchmark`.

## Invariants

Measurement views disable profiling and require quality; serving diagnostic
views cannot supply rewards or quality claims; replay changes only the allowed
image locator; derived runtime images are diagnostic-only and evidence-bound;
host environment state cannot silently change the imported evaluator.

## Dependencies

Benchmark code depends downward on core contracts, benchmark ports, supervised
execution, and pinned runtime receipts; it never imports optimization policy.

## Failure semantics

Malformed YAML, dependency drift, phase leakage, workspace escape, ambiguous
quality files, or missing required quality evidence fail closed with reason codes.

## Tests

Run `pytest -q -p no:cacheprovider tests/unit/benchmark`; fixtures are CPU-only
and use temporary immutable views and reports.

## Provenance

Resolved views record original/config semantic hashes plus exact Magpie,
TraceLens, Python, and dependency-lock identities.

Formal reports also fail closed unless Apex can independently re-read the
workspace `model_revision_receipt.json` and
`inferencex_runtime_receipt.json`. The latter must bind the configured clean
InferenceX checkout, exact commit and tree, unchanged empty Git status, and the
run-scoped private-index materialization method; a copied or dirty source tree
cannot become reward evidence.
The Magpie `lm_eval_runtime_receipt` report evidence is independently rehashed
against the snapshotted runtime manifest and execution receipt, then compared
with Apex's verified runtime identity. Both artifacts are retained in the
normalized result and canonical E2E run record; missing, writable-mounted, or
tampered evaluator evidence cannot become reward truth.
