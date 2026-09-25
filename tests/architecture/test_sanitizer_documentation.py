"""Keep sanitizer claims aligned with the documentation-only product boundary."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ROOT / "src" / "apex" / "evaluation" / "safety" / "README.md"
RELATED = {
    ROOT / "README.md": "sanitizer_runtime=not_implemented",
    ROOT / "src" / "apex" / "evaluation" / "README.md": (
        "Safety is neither correctness nor a reward bonus"
    ),
    ROOT / "src" / "apex" / "ports" / "README.md": (
        "`SafetyVerificationPort` is the external-receipt validation boundary"
    ),
    ROOT / "src" / "apex" / "optimization" / "kernel" / "README.md": (
        "optional external safety-receipt validation"
    ),
    ROOT / "src" / "apex" / "optimization" / "e2e" / "README.md": (
        "The safety step does not launch a sanitizer"
    ),
    ROOT / "src" / "apex" / "orchestration" / "README.md": (
        "never schedules a sanitizer process"
    ),
    ROOT / "src" / "apex" / "delivery" / "README.md": (
        "It cannot infer tool"
    ),
    ROOT / "src" / "apex" / "reporting" / "README.md": (
        "keeps capability, execution, finding, policy satisfaction"
    ),
    ROOT / "src" / "apex" / "rl" / "README.md": (
        "never trained as clean or safety-positive evidence"
    ),
    ROOT / "src" / "apex" / "cli" / "README.md": (
        "Apex provides no sanitizer command"
    ),
}


def test_primary_safety_readme_is_the_complete_fact_source() -> None:
    content = PRIMARY.read_text(encoding="utf-8")
    required = (
        "## Purpose",
        "## Public API",
        "## Invariants",
        "## Dependencies",
        "## Failure semantics",
        "## Tests",
        "## Provenance",
        "sanitizer_runtime=not_implemented",
        "VerificationPolicy.no_tools()",
        "Four-dimensional capability",
        "Execution and finding are orthogonal",
        "Positive control is not candidate attestation",
        "Instrumentation and dispatch engagement",
        "JIT, AOT, and replay-capsule boundary",
        "Evaluator phase isolation",
        "Plan fingerprint and resume",
        "Policy truth table",
        "Performance isolation",
        "Coverage and held-out limitations",
        "External receipt minimum fields",
        "Conceptual support matrix",
        "Typed examples",
    )
    assert all(item in content for item in required)


def test_primary_readme_names_every_public_safety_export() -> None:
    module = ROOT / "src" / "apex" / "evaluation" / "safety" / "__init__.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    exports = ast.literal_eval(assignment.value)
    content = PRIMARY.read_text(encoding="utf-8")
    assert all(f"`{name}`" in content for name in exports)


def test_related_readmes_state_the_uncertified_external_receipt_boundary() -> None:
    for path, expected in RELATED.items():
        content = path.read_text(encoding="utf-8")
        assert expected in content, path.relative_to(ROOT)


def test_safety_docs_do_not_cite_research_change_snapshots() -> None:
    markers = (
        "AgentKernelArena PR",
        "research snapshot",
        "commit snapshot",
        "移植自",
        "github.com/AMD-AGI/AgentKernelArena/pull/",
    )
    for path in (PRIMARY, *RELATED):
        content = path.read_text(encoding="utf-8")
        assert not any(marker in content for marker in markers), path.relative_to(ROOT)


def test_docs_do_not_claim_a_built_in_sanitizer_product() -> None:
    banned = (
        "sanitizer execution and policy",
        "kernel.sanitize is supported",
        "qualified sanitizer architecture",
        "not_applicable means clean",
    )
    for path in (PRIMARY, *RELATED):
        content = path.read_text(encoding="utf-8")
        assert not any(claim in content for claim in banned), path.relative_to(ROOT)
