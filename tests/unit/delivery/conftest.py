from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest
import yaml

from apex.delivery import (
    BuildRecipeLock,
    BuildStep,
    BundleProvenanceLock,
    DerivedImageIdentity,
    E2EPatchBundle,
    PrimaryVerificationEvidence,
    build_e2e_patch_bundle,
    capture_repository_patch,
    source_stack_digest,
    verify_replay_config_invariants,
)
from apex.core import canonical_json_bytes, sha256_file


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repository(tmp_path: Path, name: str) -> tuple[Path, Path]:
    base = tmp_path / f"{name}-base"
    base.mkdir()
    git(base, "init", "-q")
    git(base, "config", "user.name", "Apex Test")
    git(base, "config", "user.email", "apex@example.com")
    git(base, "remote", "add", "origin", f"https://example.com/{name}.git")
    (base / "src").mkdir()
    (base / "src" / "modify.py").write_text("VALUE = 1\n", encoding="utf-8")
    (base / "src" / "delete.py").write_text("DELETE = True\n", encoding="utf-8")
    (base / "src" / "rename.py").write_text("RENAMED = False\n", encoding="utf-8")
    (base / "src" / "mode.py").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    git(base, "add", ".")
    git(base, "commit", "-q", "-m", "baseline")

    candidate = tmp_path / f"{name}-candidate"
    subprocess.run(("git", "clone", "-q", "--no-hardlinks", str(base), str(candidate)), check=True)
    git(candidate, "remote", "set-url", "origin", f"https://example.com/{name}.git")
    (candidate / "src" / "modify.py").write_text("VALUE = 2\n", encoding="utf-8")
    (candidate / "src" / "delete.py").unlink()
    git(candidate, "mv", "src/rename.py", "src/renamed.py")
    (candidate / "src" / "new.py").write_text("NEW = True\n", encoding="utf-8")
    os.chmod(candidate / "src" / "mode.py", 0o755)
    return base, candidate


def configs(tmp_path: Path, image_locator: str) -> dict[str, Path]:
    root = tmp_path / "configs"
    root.mkdir()
    original = {"benchmark": {"framework": "vllm", "model": "Qwen/test", "docker_image": "base:v1", "envs": {"RUN_EVAL": "true", "MAGPIE_EVAL_TASKS": "gsm8k"}, "server_args": {"tp": 1}}}
    measurement = {
        "benchmark": {
            **original["benchmark"],
            "profiler": {"torch_profiler": {"enabled": False}},
            "gap_analysis": {"enabled": False},
        },
        "apex": {"benchmark_view": {"kind": "measurement"}},
    }
    diagnostic = {
        "benchmark": {
            **original["benchmark"],
            "profiler": {"torch_profiler": {"enabled": True}},
            "gap_analysis": {"enabled": True},
        },
        "apex": {"benchmark_view": {"kind": "diagnostic"}},
    }
    replay = yaml.safe_load(yaml.safe_dump(measurement))
    replay["benchmark"]["docker_image"] = image_locator
    replay["apex"]["benchmark_view"]["kind"] = "replay"
    documents = {
        "benchmark_original": original,
        "benchmark_measurement": measurement,
        "benchmark_diagnostic": diagnostic,
        "benchmark_replay": replay,
    }
    paths: dict[str, Path] = {}
    for role, document in documents.items():
        path = root / f"{role}.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        paths[role] = path
    return paths


@dataclass
class BundleFixture:
    bundle: E2EPatchBundle
    recipe: BuildRecipeLock
    image: DerivedImageIdentity
    bases: dict[str, Path]


@pytest.fixture
def make_e2e_bundle(tmp_path: Path) -> Callable[..., BundleFixture]:
    def make(
        *,
        count: int = 1,
        overlay: bool = False,
        engagement_kind: str = "python_import",
        build_id_required: bool = False,
    ) -> BundleFixture:
        derived_digest = "sha256:" + "5" * 64
        derived_locator = "apex-derived@" + derived_digest
        recipe = BuildRecipeLock(
            "vllm-python-source-v1",
            "sha256:" + "1" * 64,
            derived_locator,
            tuple(
                BuildStep(
                    ("python3", "build.py"),
                    f"repo{index}",
                    timeout_seconds=60,
                )
                for index in range(count)
            ),
        )
        captures = []
        bases: dict[str, Path] = {}
        for index in range(count):
            name = f"repo{index}"
            base, candidate = repository(tmp_path, name)
            bases[name] = base
            dependencies = (f"repo{index - 1}",) if index else ()
            captures.append(
                capture_repository_patch(
                    repository_id=name,
                    base_root=base,
                    candidate_root=candidate,
                    patch_path=f"patches/{index:03d}-{name}.patch",
                    order=index,
                    dependencies=dependencies,
                    editable_allowlist=("src/",),
                    build_recipe_sha256=recipe.computed_sha256,
                    accepted_candidate_id=f"candidate-{index}",
                    anchor_generation=index,
                    license_id="Apache-2.0",
                    runtime_component=name,
                    engagement_kind=engagement_kind,
                    build_id_required=build_id_required,
                )
            )
        stack = source_stack_digest(tuple(item.lock for item in captures))
        receipt_root = tmp_path / "primary-receipts"
        receipt_root.mkdir()
        primary_receipts: dict[str, Path] = {}
        for role, value in (
            ("primary_build_receipt", {"kind": "source_build", "passed": True}),
            ("primary_engagement_receipt", {"kind": "engagement", "passed": True}),
            ("primary_benchmark_receipt", {"kind": "benchmark", "passed": True}),
        ):
            path = receipt_root / f"{role}.json"
            path.write_bytes(canonical_json_bytes(value) + b"\n")
            primary_receipts[role] = path
        primary = PrimaryVerificationEvidence(
            environment_id="primary-environment",
            runtime_identity_sha256="d" * 64,
            source_stack_sha256=stack,
            build_receipt_sha256=sha256_file(primary_receipts["primary_build_receipt"]),
            engagement_receipt_sha256=sha256_file(primary_receipts["primary_engagement_receipt"]),
            benchmark_receipt_sha256=sha256_file(primary_receipts["primary_benchmark_receipt"]),
            safety_source_sha256=None,
            performance_source_sha256=stack,
            deployed_source_sha256=stack,
            engagement_verified=True,
            normal_runtime_measurement=True,
            accuracy_passed=True,
            latency_gates_passed=True,
            objective_improved=True,
            overlay_verified=overlay,
            overlay_source_sha256=stack if overlay else None,
            overlay_rebuild_parity_passed=True if overlay else None,
            safety_certified=False,
        )
        sbom = tmp_path / "derived-image.sbom.json"
        sbom.write_bytes(canonical_json_bytes({"spdxVersion": "SPDX-2.3", "packages": []}) + b"\n")
        image = DerivedImageIdentity(
            derived_locator,
            recipe.parent_image_digest,
            derived_digest,
            sha256_file(sbom),
            ("build-id-1",),
        )
        config_paths = configs(tmp_path, image.locator)
        _, _, semantics_sha = verify_replay_config_invariants(
            config_paths["benchmark_measurement"],
            config_paths["benchmark_replay"],
            expected_image_locator=image.locator,
        )
        provenance = BundleProvenanceLock(
            primary_run_id="primary-run",
            framework="vllm",
            model_id="Qwen/test",
            model_revision="a" * 40,
            gpu_arch="gfx950",
            baseline_image_digest=recipe.parent_image_digest,
            original_config_sha256=sha256_file(config_paths["benchmark_original"]),
            workload_semantics_sha256=semantics_sha,
            accuracy_policy_sha256="7" * 64,
            performance_policy_sha256="8" * 64,
            safety_policy_sha256=None,
            agent_backend="codex",
            agent_model="gpt-5",
        )
        bundle = build_e2e_patch_bundle(
            bundle_id="e2e-test-bundle",
            bundle_dir=tmp_path / "candidate-bundle",
            repositories=captures,
            recipe=recipe,
            derived_image=image,
            provenance=provenance,
            configs=config_paths,
            primary_evidence=primary,
            primary_receipts=primary_receipts,
            image_sbom=sbom,
        )
        return BundleFixture(bundle, recipe, image, bases)

    return make
