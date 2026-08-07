# Contributing to Apex

Thank you for improving Apex. Changes should strengthen the evidence-driven kernel
optimizer and RL environment while preserving its module boundaries and fail-closed
evaluation model.

## Development setup

```bash
git switch -c your-branch
./setup.sh
source .venv/bin/activate
apex dependencies verify --json
```

For CPU-only work that does not exercise Magpie or TraceLens, install with
`.venv/bin/pip install -e '.[dev]'`. Do not substitute unpinned external checkouts in
test evidence.

## Before opening a change

- Keep behavior in the module that owns the policy; `apex.bootstrap` is the sole
  concrete composition root.
- Keep source files at most 600 lines and functions at most 80 lines.
- Use immutable typed contracts at boundaries and supervised fixed-argv processes.
- Add stable reason codes for expected failures. Missing evidence must fail closed.
- Update the owning module's README when its public API, artifacts, invariants,
  dependencies, failure semantics, or provenance changes.
- Preserve exact upstream attribution and generated-card manifests.
- Do not add compatibility readers or writers for removed formats.

Run the hermetic CPU suite:

```bash
pytest -q -p no:cacheprovider --import-mode=importlib \
  tests/unit tests/contract tests/integration tests/architecture \
  tests/test_bootstrap_dependencies.py
python -m compileall -q src/apex main.py scripts
```

Focused test commands are listed in each package README. A pull request should state
which focused and full gates ran and include the exact failure if a required external
campaign could not run.

## Evidence-sensitive changes

Changes to evaluation, orchestration, safety, benchmark adapters, provenance, or
delivery need negative tests in addition to the happy path. At minimum, cover missing
proof, malformed evidence, stale state or anchors, timeout/failure, and tampering or
path-escape behavior appropriate to the component.

Kernel reward changes must retain raw-sample replay tests. E2E acceptance changes must
test throughput, accuracy, TTFT p99, TPOT p99, normal-versus-diagnostic measurement,
and KEEP/REVERT behavior. Delivery changes must prove bundle digest validation and
that the caller checkout is not modified.

## GPU and live verification

Real GPU and model runs are separate from the CPU gate. Record:

- exact Apex, Magpie, TraceLens, and workload source commits;
- immutable baseline and derived image identities;
- GPU architecture, ROCm/runtime versions, and visible device lease;
- fixed benchmark config and agent backend/model/budget;
- raw measurement and safety receipts;
- final bundle digest and validation level.

Do not report an instrumented trace as a normal benchmark, an overlay as a source
rebuild, or an agent-claimed speedup as an evaluator result. Do not commit model data,
credentials, private traces, or run directories.

## Pull requests and issues

Keep pull requests focused and explain motivation, contract impact, compatibility
impact (normally none), test evidence, and provenance changes. Report security issues
privately using [SECURITY.md](SECURITY.md). Public bug reports should include a
minimal reproduction, exact commits/configuration, expected and observed behavior,
and redacted logs.

Contributions are licensed under the repository [MIT License](LICENSE).
