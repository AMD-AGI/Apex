from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from apex.core import sha256_bytes
from apex.knowledge import KnowledgeCard, PinnedSourceManifest, SourceEstatePin


def make_card(
    *,
    claim: str,
    kind: str = "fact",
    status: str = "imported_unverified",
    gpu_arch: tuple[str, ...] = ("gfx950",),
    operator: tuple[str, ...] = ("rms_norm",),
    language: tuple[str, ...] = ("triton",),
    path: str | None = None,
    evidence: tuple[str, ...] = (),
) -> KnowledgeCard:
    source_path = path or f"perf_knowledge/{claim.replace(' ', '_')}.md"
    return KnowledgeCard.from_mapping(
        {
            "kind": kind,
            "status": status,
            "scope": {
                "operator": list(operator),
                "gpu_arch": list(gpu_arch),
                "dtype": ["fp16"],
                "regime": ["decode"],
                "language": list(language),
                "framework": ["vllm"],
                "versions": {"rocm": "7.2"},
            },
            "claim": claim,
            "apply": f"Advisory application for {claim}",
            "verify": "Measure with the protected harness.",
            "caution": "Do not select a winner without measurement.",
            "source": {
                "repository": "https://example.invalid/geak",
                "git_sha": "1" * 40,
                "path": source_path,
                "license": "Apache-2.0",
                "content_sha256": sha256_bytes(claim.encode()),
                "transform_version": "test_v1",
            },
            "evidence_receipts": list(evidence),
            "executable": False,
            "supersedes": [],
            "contradicts": [],
        }
    )


@pytest.fixture
def card_factory():
    return make_card


@pytest.fixture
def pinned_geak_fixture(tmp_path: Path) -> tuple[Path, PinnedSourceManifest, Path]:
    root = tmp_path / "geak"
    root.mkdir()
    _run(root, "init", "-q")
    _run(root, "config", "user.email", "test@example.invalid")
    _run(root, "config", "user.name", "Apex Test")
    license_content = b"Apache License 2.0 fixture\n"
    (root / "LICENSE.md").write_bytes(license_content)
    files = {
        "perf_knowledge/operators/rms_norm/overview.md": b"# RMS norm\nUse a fused pass.\n",
        "perf_knowledge/tools/check.py": b"raise SystemExit('must not execute')\n",
        "perf_knowledge/expert_skills/unsafe/skill.md": b"# Unsafe skill\nRun arbitrary code.\n",
        "kernel_workflow/knowledge/pitfalls.md": b"# Pitfalls\nAvoid hidden copies.\n",
        "e2e_workflow/knowledge/learned/result.md": b"# Prior result\nRetest this result.\n",
        "e2e_workflow/knowledge/templates/sitecustomize.py": b"raise RuntimeError('never run')\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    _run(root, "add", ".")
    _run(root, "commit", "-q", "-m", "fixture")
    sha = _run(root, "rev-parse", "HEAD").strip()
    estates = (
        _estate("perf", "perf_knowledge", files),
        _estate("kernel", "kernel_workflow/knowledge", files),
        _estate("e2e", "e2e_workflow/knowledge", files),
    )
    pin = PinnedSourceManifest(
        repository="https://example.invalid/geak",
        git_sha=sha,
        license="Apache-2.0",
        license_path="LICENSE.md",
        license_sha256=sha256_bytes(license_content),
        transform_version="fixture_transform_v1",
        estates=estates,
    )
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin.to_dict()), encoding="utf-8")
    return root, pin, pin_path


def _estate(estate_id: str, prefix: str, files: dict[str, bytes]) -> SourceEstatePin:
    selected = [content for path, content in files.items() if path.startswith(f"{prefix}/")]
    return SourceEstatePin(estate_id, prefix, len(selected), sum(map(len, selected)))


def _run(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout
