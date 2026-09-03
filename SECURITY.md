# Security policy

## Report a vulnerability

Do not open a public issue. Report privately through one of these channels:

- [GitHub Private Vulnerability Reporting](https://github.com/AMD-AGI/Apex/security/advisories/new)
- [AMD Product Security](https://www.amd.com/en/resources/product-security.html)

Include the affected revision, impact, reproduction steps, and any relevant redacted
receipts. Do not include API keys, model credentials, private workload data, or an
unnecessary weaponized GPU kernel.

## Scope

This policy covers the Apex Python package, CLI and bootstrap scripts, static
knowledge-card build, agent execution adapters, supervised command execution,
benchmark/trace adapters, safety policy, event and artifact storage, RL export, and
kernel/E2E bundle delivery.

High-value properties include:

- agent code and advisory knowledge cannot redefine trusted commands or evaluator
  policy;
- subprocesses are fixed argv with bounded time/output and confined workspaces;
- editable paths, measurement destinations, artifacts, and bundles resist path
  escape, symlink/hardlink substitution, and time-of-check/time-of-use changes;
- event history and content-addressed artifacts detect mutation or causal forks;
- raw timing, correctness, safety, and reward evidence cannot be forged from agent
  text or command stdout;
- E2E patches cannot alter protected benchmark semantics or claim stronger
  provenance/validation than the receipts prove;
- secrets and private/held-out episodes do not enter exported training data;
- dependency bootstrap uses exact reviewed commits and detects split import roots.

Please also report credential leakage, sandbox/container escape, unsafe source-build
recipes, malicious dependency/knowledge content, unauthorized host mutation, bundle
verification bypass, or denial-of-service against shared GPU runners.

Issues in Codex, Claude, Cursor, Docker, ROCm, Magpie, TraceLens, vLLM, SGLang,
PyTorch, Triton, or another upstream project should be reported to that vendor as
well. AMD product issues unrelated to Apex belong at the AMD Product Security link
above.
