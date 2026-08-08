from __future__ import annotations

import copy
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import yaml

from apex.core import (
    AgentBackendName,
    TaskStatus,
    ValidationLevel,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from apex.delivery import (
    BuildRecipeLock,
    BuildStep,
    BuildStepReceipt,
    BuiltArtifact,
    CleanReplayReceipt,
    DerivedImageIdentity,
    E2EBundleVerifier,
    LoadedArtifact,
    LoadedByteEngagementReceipt,
    SourceBuildReceipt,
    load_and_verify_e2e_bundle,
)
from apex.evaluation import E2EMeasurement
from apex.optimization.e2e.candidate import E2ECandidate, FrozenCandidateSource
from apex.optimization.e2e.kernel_lane import KernelOpportunity
from apex.optimization.e2e.services import (
    AcceptedCandidate,
    CandidateDeployment,
    DeploymentConfigDigests,
    FinalDeliveryRequest,
    MicroQualification,
    SafetyQualification,
)
from apex.optimization.e2e.source_delivery import (
    FormalDeliveryBinding,
    SourceRebuildFinalDelivery,
)
from apex.optimization.e2e.source_delivery_models import (
    FormalRepositoryProfile,
    FormalSourceDeliveryProfile,
    PrimarySourceBuildOutput,
)
from apex.ports import AgentResult
from apex.runtime import ContainerIdentity, RepositoryLock, RunProvenance


PARENT = "sha256:" + "a" * 64
DERIVED = "sha256:" + "b" * 64


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "vllm-source"
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "apex@example.invalid")
    _git(root, "config", "user.name", "Apex Test")
    for name in ("op_a.py", "op_b.py"):
        path = root / "vllm" / "kernels" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"VALUE = '{name}:base'\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")
    _git(root, "remote", "add", "origin", str(root))
    return root, _git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", "HEAD^{tree}")


def _source_digest(root: Path, relative: str) -> str:
    path = root / relative
    return sha256_json(
        {
            "schema_version": 1,
            "files": [
                {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "mode": path.stat().st_mode & 0o777,
                }
            ],
        }
    )


def _configs(tmp_path: Path, image: str) -> tuple[dict[str, Path], str]:
    benchmark = {
        "framework": "vllm",
        "model": "Qwen/test",
        "docker_image": image,
        "envs": {"RUN_EVAL": "true", "MAGPIE_EVAL_TASKS": "gsm8k"},
        "profiler": {"torch_profiler": {"enabled": False}},
        "gap_analysis": {"enabled": False},
    }
    projected = copy.deepcopy(benchmark)
    for key in ("docker_image", "profiler", "gap_analysis"):
        projected.pop(key)
    semantics = sha256_json(projected)
    paths: dict[str, Path] = {}
    for role in ("original", "measurement", "diagnostic", "replay"):
        document = {
            "benchmark": copy.deepcopy(benchmark),
            "apex": {
                "benchmark_view": {
                    "kind": role,
                    "workload_semantics_sha256": semantics,
                }
            },
        }
        path = (tmp_path / "configs" / f"{role}.yaml").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        paths[role] = path
    return paths, semantics


def _candidate(
    tmp_path: Path,
    root: Path,
    relative: str,
    value: str,
    index: int,
    configs: dict[str, Path],
    semantics: str,
) -> AcceptedCandidate:
    workspace = tmp_path / f"candidate-{index}"
    subprocess.run(("git", "clone", "-q", str(root), str(workspace)), check=True)
    path = workspace / relative
    path.write_text(f"VALUE = '{value}'\n", encoding="utf-8")
    baseline_digest = _source_digest(root, relative)
    candidate_digest = _source_digest(workspace, relative)
    content = path.read_bytes()
    frozen = FrozenCandidateSource(
        relative,
        sha256_bytes(content),
        path.stat().st_mode & 0o777,
        content,
    )
    candidate = E2ECandidate(
        f"attempt-{index}",
        f"candidate-{index}",
        True,
        "candidate_frozen",
        workspace,
        (relative,),
        (relative,),
        baseline_digest,
        candidate_digest,
        AgentResult(
            AgentBackendName.CODEX,
            "gpt-test",
            0,
            False,
            (),
            "",
            "",
            1.0,
        ),
        (frozen,),
    )
    path.chmod(0o444)
    opportunity = KernelOpportunity(
        f"kernel-{index}",
        f"evidence-{index}",
        f"runtime-{index}",
        "operator",
        "decode",
        index,
        "triton",
        "vllm",
        ("[1,128]",),
        ("fp16",),
        "eager",
        "exact",
        5.0,
        2.0,
        root / relative,
        root,
        root / relative,
        "pytest -q",
        "eligible",
        "eligible",
    )
    micro = MicroQualification(
        candidate_id=candidate.candidate_id or "",
        grade=None,
        evidence={},
        qualification_mode="e2e_quality_deferred",
        deferred_candidate_valid=True,
    )
    safety = SafetyQualification(
        candidate.candidate_id or "",
        True,
        True,
        False,
        False,
        (),
        {"policy_fingerprint": "6" * 64},
    )
    config_sha256 = DeploymentConfigDigests.capture(
        configs["measurement"], configs["diagnostic"], configs["replay"]
    )
    deployment = CandidateDeployment(
        candidate.candidate_id or "",
        True,
        "runtime_overlay_loaded_bytes_verified",
        configs["measurement"],
        configs["diagnostic"],
        configs["replay"],
        semantics,
        candidate_digest,
        "sha256:" + "f" * 64,
        ValidationLevel.RUNTIME_OVERLAY_VERIFIED,
        True,
        {
            "formal_source_rebuild": False,
            "derived_image": {"image_id": "sha256:" + "f" * 64},
            "config_sha256": config_sha256.to_dict(),
        },
        config_sha256=config_sha256,
    )
    return AcceptedCandidate(
        candidate,
        opportunity,
        micro,
        safety,
        deployment,
        _measurement(101.0 + index, semantics, f"candidate-{index}"),
        "d" * 64,
    )


def _measurement(throughput: float, protocol: str, receipt: str) -> E2EMeasurement:
    return E2EMeasurement(throughput, 10.0, 2.0, 1.0, 32, protocol, receipt, receipt)


class PrimaryBuilder:
    def __init__(self, *, missing_safety: bool = False, parity: bool = True) -> None:
        self.missing_safety = missing_safety
        self.parity = parity
        self.calls = 0
        self.observed: dict[str, str] = {}
        self.observed_modes: dict[str, int] = {}

    def build_and_validate(self, request):
        self.calls += 1
        for root in request.repository_roots.values():
            for path in sorted(root.glob("vllm/kernels/*.py")):
                self.observed[path.name] = path.read_text(encoding="utf-8")
                self.observed_modes[path.name] = path.stat().st_mode & 0o777
        output = request.artifact_root
        output.mkdir(parents=True)
        sbom = output / "sbom.json"
        sbom.write_text('{"spdxVersion":"SPDX-2.3"}\n', encoding="utf-8")
        image = DerivedImageIdentity(
            f"apex/formal@{DERIVED}", PARENT, DERIVED, sha256_file(sbom)
        )
        configs = {}
        for role, source in (
            ("measurement", request.benchmark_measurement),
            ("diagnostic", request.benchmark_diagnostic),
            ("replay", request.benchmark_replay),
        ):
            document = yaml.safe_load(source.read_text(encoding="utf-8"))
            document["benchmark"]["docker_image"] = image.locator
            path = (output / f"{role}.yaml").resolve()
            path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            configs[role] = path
        receipts = {}
        roles = [
            "primary_build_receipt",
            "primary_engagement_receipt",
            "primary_benchmark_receipt",
            "primary_safety_receipt",
        ]
        if self.missing_safety:
            roles.pop()
        for role in roles:
            path = (output / f"{role}.json").resolve()
            path.write_text(
                json.dumps(
                    {"role": role, "source_stack_sha256": request.source_stack_sha256},
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            receipts[role] = path
        return PrimarySourceBuildOutput(
            "primary-clean-environment",
            request.source_stack_sha256,
            image,
            sbom.resolve(),
            configs["measurement"],
            configs["diagnostic"],
            configs["replay"],
            receipts,
            True,
            True,
            True,
            True,
            True,
            self.parity,
            False,
        )


class MutatingPrimaryBuilder(PrimaryBuilder):
    def build_and_validate(self, request):
        output = super().build_and_validate(request)
        path = request.repository_roots["vllm"] / "vllm/kernels/op_a.py"
        path.write_text("VALUE = 'mutated-during-build'\n", encoding="utf-8")
        return output


class CrashingPrimaryBuilder:
    def build_and_validate(self, _request):
        raise RuntimeError("attestor unavailable")


class FakeBuild:
    def build(self, request):
        artifacts = tuple(
            BuiltArtifact(
                item.repository_id,
                f"/opt/{item.repository_id}/kernel.py",
                str(index + 1) * 64,
                None,
                request.source_stack_sha256,
            )
            for index, item in enumerate(request.repository_receipts)
        )
        steps = tuple(
            BuildStepReceipt(
                index,
                step.repository_id,
                step.cwd,
                sha256_json(list(step.argv)),
                0,
                False,
                "e" * 64,
                "f" * 64,
            )
            for index, step in enumerate(request.recipe.steps)
        )
        return SourceBuildReceipt(
            request.bundle_digest,
            request.recipe.computed_sha256,
            request.expected_image.parent_digest,
            request.expected_image.parent_digest,
            request.expected_image.image_digest,
            request.expected_image.image_digest,
            request.expected_image.sbom_sha256,
            request.expected_image.sbom_sha256,
            request.source_stack_sha256,
            True,
            artifacts,
            steps,
        )


class FakeEngagement:
    def verify_loaded_bytes(self, request):
        return LoadedByteEngagementReceipt(
            request.bundle_digest,
            request.expected_image.image_digest,
            request.source_stack_sha256,
            True,
            tuple(
                LoadedArtifact(
                    item.component,
                    item.runtime_path,
                    item.sha256,
                    item.sha256,
                    item.build_id,
                    item.build_id,
                    "python_import",
                    f"import:{item.component}",
                    True,
                )
                for item in request.build_receipt.artifacts
            ),
        )


class FakeReplay:
    def __init__(self, *, objective_improved: bool = True) -> None:
        self.objective_improved = objective_improved

    def replay(self, request):
        return CleanReplayReceipt(
            request.bundle_digest,
            request.primary_environment_id,
            "independent-clean-environment",
            request.expected_image.image_digest,
            request.config_receipt.replay_config_sha256,
            "c" * 64,
            request.source_stack_sha256,
            True,
            True,
            True,
            True,
            True,
            True,
            self.objective_improved,
        )


def _fixture(tmp_path: Path, *, replay_gain: bool = True):
    root, commit, tree = _repository(tmp_path)
    configs, semantics = _configs(tmp_path, "sha256:" + "9" * 64)
    accepted = (
        _candidate(
            tmp_path,
            root,
            "vllm/kernels/op_a.py",
            "optimized-a",
            1,
            configs,
            semantics,
        ),
        _candidate(
            tmp_path,
            root,
            "vllm/kernels/op_b.py",
            "optimized-b",
            2,
            configs,
            semantics,
        ),
    )
    lock = RepositoryLock("vllm", str(root), str(root), commit, tree, True)
    provenance = RunProvenance(
        1,
        str(configs["original"]),
        sha256_file(configs["original"]),
        "vllm",
        "Qwen/test",
        "1" * 40,
        "gfx950",
        ContainerIdentity(PARENT, PARENT, (), ()),
        ("vllm",),
        (lock,),
        "partial",
        ("runtime_loaded_bytes",),
    )
    recipe = BuildRecipeLock(
        "vllm-python-source-v1",
        PARENT,
        "apex/formal",
        (BuildStep(("python3", "build.py"), "vllm"),),
    )
    repository = FormalRepositoryProfile(
        "vllm", "vllm", str(root), ("vllm/kernels/",)
    )
    profile = FormalSourceDeliveryProfile("vllm-source", (repository,), recipe)
    verifier = E2EBundleVerifier(
        trusted_recipes={recipe.computed_sha256: recipe},
        trusted_source_urls={"vllm": str(root)},
        build_backend=FakeBuild(),
        engagement_backend=FakeEngagement(),
        replay_backend=FakeReplay(objective_improved=replay_gain),
    )
    builder = PrimaryBuilder()
    binding = FormalDeliveryBinding(
        profile, builder, verifier, {"vllm": root}
    )
    request = FinalDeliveryRequest(
        "e2e-test",
        accepted,
        provenance,
        configs["original"],
        configs["measurement"],
        configs["diagnostic"],
        configs["replay"],
        _measurement(100.0, semantics, "baseline"),
        _measurement(103.0, semantics, "final"),
        (tmp_path / "results" / "formal-delivery").resolve(),
        "codex",
        "gpt-test",
        "4" * 64,
        "5" * 64,
        "6" * 64,
    )
    return root, profile, binding, builder, request


def test_formal_delivery_accumulates_source_and_requires_second_clean_replay(
    tmp_path: Path,
) -> None:
    root, _, binding, builder, request = _fixture(tmp_path)

    result = SourceRebuildFinalDelivery((binding,)).finalize(request)

    assert result.verified is True
    assert result.status is TaskStatus.SUCCEEDED
    assert result.validation_level is ValidationLevel.SOURCE_REBUILD_VERIFIED
    assert result.clean_replay_verified is True
    assert result.bundle_path and result.bundle_digest
    bundle = load_and_verify_e2e_bundle(Path(result.bundle_path))
    assert bundle.verified is True
    assert {item.new_path for item in bundle.repositories[0].changes} == {
        "vllm/kernels/op_a.py",
        "vllm/kernels/op_b.py",
    }
    assert "optimized-a" in builder.observed["op_a.py"]
    assert "optimized-b" in builder.observed["op_b.py"]
    assert builder.observed_modes["op_a.py"] == (
        (root / "vllm/kernels/op_a.py").stat().st_mode & 0o777
    )
    assert ":base" in (root / "vllm/kernels/op_a.py").read_text(encoding="utf-8")
    assert (
        request.artifact_root
        / "independent-verification/worktrees/vllm/vllm/kernels/op_a.py"
    ).is_file()


def test_formal_delivery_uses_frozen_bytes_after_workspace_mutation(
    tmp_path: Path,
) -> None:
    _, _, binding, builder, request = _fixture(tmp_path)
    candidate = request.accepted[0].candidate
    mutable_path = candidate.workspace / candidate.editable_files[0]
    mutable_path.chmod(0o644)
    mutable_path.write_text("VALUE = 'late-workspace-mutation'\n", encoding="utf-8")

    result = SourceRebuildFinalDelivery((binding,)).finalize(request)

    assert result.verified is True
    assert "optimized-a" in builder.observed["op_a.py"]
    assert "late-workspace-mutation" not in builder.observed["op_a.py"]


def test_formal_delivery_never_follows_agent_workspace_symlink(
    tmp_path: Path,
) -> None:
    _, _, binding, builder, request = _fixture(tmp_path)
    candidate = request.accepted[0].candidate
    workspace = candidate.workspace
    hidden_workspace = tmp_path / "discarded-agent-workspace"
    workspace.rename(hidden_workspace)
    secret_root = tmp_path / "outside-agent-workspace"
    secret_path = secret_root / candidate.editable_files[0]
    secret_path.parent.mkdir(parents=True)
    secret_path.write_text("VALUE = 'must-not-leak'\n", encoding="utf-8")
    workspace.symlink_to(secret_root, target_is_directory=True)

    result = SourceRebuildFinalDelivery((binding,)).finalize(request)

    assert result.verified is True
    assert "optimized-a" in builder.observed["op_a.py"]
    assert "must-not-leak" not in builder.observed["op_a.py"]
    assert not any(
        b"must-not-leak" in path.read_bytes()
        for path in request.artifact_root.rglob("*")
        if path.is_file()
    )
    assert workspace.is_symlink()
    shutil.rmtree(hidden_workspace)


def test_formal_delivery_recomputes_digest_from_frozen_content(
    tmp_path: Path,
) -> None:
    _, _, binding, builder, request = _fixture(tmp_path)
    accepted = request.accepted[0]
    candidate = accepted.candidate
    frozen = candidate.frozen_sources[0]
    tampered = replace(frozen, content=b"VALUE = 'tampered-snapshot'\n")
    candidate = replace(candidate, frozen_sources=(tampered,))
    accepted = replace(accepted, candidate=candidate)
    request = replace(request, accepted=(accepted, *request.accepted[1:]))

    result = SourceRebuildFinalDelivery((binding,)).finalize(request)

    assert result.status is TaskStatus.VERIFICATION_FAILED
    assert result.reason_code == "candidate_source_capture_drift"
    assert builder.calls == 0
    assert not request.artifact_root.exists()


def test_missing_actual_agent_model_is_provenance_unresolved_before_build(
    tmp_path: Path,
) -> None:
    _, _, binding, builder, request = _fixture(tmp_path)
    request = replace(request, agent_model=None)

    result = SourceRebuildFinalDelivery((binding,)).finalize(request)

    assert result.status is TaskStatus.PROVENANCE_UNRESOLVED
    assert result.reason_code == "source_provenance_unresolved"
    assert builder.calls == 0
    assert not request.artifact_root.exists()


def test_missing_fixed_recipe_binding_fails_closed_without_artifacts(tmp_path: Path) -> None:
    _, profile, binding, builder, request = _fixture(tmp_path)
    wrong_recipe = replace(profile.recipe, parent_image_digest="sha256:" + "8" * 64)
    wrong_profile = replace(profile, recipe=wrong_recipe)
    wrong_binding = replace(binding, profile=wrong_profile)

    result = SourceRebuildFinalDelivery((wrong_binding,)).finalize(request)

    assert result.status is TaskStatus.VERIFICATION_FAILED
    assert result.reason_code == "untrusted_build_recipe"
    assert result.validation_level is ValidationLevel.RUNTIME_OVERLAY_VERIFIED
    assert builder.calls == 0
    assert not request.artifact_root.exists()


def test_primary_evidence_must_include_safety_and_overlay_rebuild_parity(
    tmp_path: Path,
) -> None:
    _, profile, binding, _, request = _fixture(tmp_path)
    builder = PrimaryBuilder(missing_safety=True)
    adapter = SourceRebuildFinalDelivery((replace(binding, primary_builder=builder),))

    result = adapter.finalize(request)

    assert result.status is TaskStatus.VERIFICATION_FAILED
    assert result.reason_code == "primary_verification_failed"
    assert result.bundle_path is None
    assert builder.calls == 1

    other = tmp_path / "other"
    _, _, binding2, _, request2 = _fixture(other)
    parity_builder = PrimaryBuilder(parity=False)
    result2 = SourceRebuildFinalDelivery(
        (replace(binding2, primary_builder=parity_builder),)
    ).finalize(request2)
    assert result2.reason_code == "primary_verification_failed"


def test_failed_independent_replay_never_claims_formal_success(tmp_path: Path) -> None:
    _, _, binding, _, request = _fixture(tmp_path, replay_gain=False)

    result = SourceRebuildFinalDelivery((binding,)).finalize(request)

    assert result.verified is False
    assert result.status is TaskStatus.VERIFICATION_FAILED
    assert result.reason_code == "second_clean_replay_failed"
    assert result.validation_level is ValidationLevel.RUNTIME_OVERLAY_VERIFIED
    assert result.bundle_path is None
    assert result.evidence["verification"]["replay_receipt"]["objective_improved"] is False


def test_primary_builder_cannot_mutate_frozen_source_inputs(tmp_path: Path) -> None:
    root, _, binding, _, request = _fixture(tmp_path)
    builder = MutatingPrimaryBuilder()

    result = SourceRebuildFinalDelivery(
        (replace(binding, primary_builder=builder),)
    ).finalize(request)

    assert result.status is TaskStatus.VERIFICATION_FAILED
    assert result.reason_code == "primary_build_input_mutation"
    assert result.bundle_path is None
    assert ":base" in (root / "vllm/kernels/op_a.py").read_text(encoding="utf-8")


def test_missing_or_crashed_attestor_is_structured_verification_failure(
    tmp_path: Path,
) -> None:
    _, _, binding, _, request = _fixture(tmp_path)

    result = SourceRebuildFinalDelivery(
        (replace(binding, primary_builder=CrashingPrimaryBuilder()),)
    ).finalize(request)

    assert result.status is TaskStatus.VERIFICATION_FAILED
    assert result.reason_code == "source_delivery_backend_error"
    assert result.bundle_path is None
    assert result.evidence["details"]["error_type"] == "RuntimeError"


def test_config_only_final_output_is_rejected(tmp_path: Path) -> None:
    _, _, binding, builder, request = _fixture(tmp_path)
    request = replace(request, accepted=())

    result = SourceRebuildFinalDelivery((binding,)).finalize(request)

    assert result.status is TaskStatus.VERIFICATION_FAILED
    assert result.reason_code == "config_only_candidate"
    assert builder.calls == 0


def test_safety_evidence_must_match_the_configured_policy(tmp_path: Path) -> None:
    _, _, binding, builder, request = _fixture(tmp_path)
    first = request.accepted[0]
    mismatched = replace(
        first,
        safety=replace(
            first.safety, evidence={"policy_fingerprint": "7" * 64}
        ),
    )
    request = replace(request, accepted=(mismatched, *request.accepted[1:]))

    result = SourceRebuildFinalDelivery((binding,)).finalize(request)

    assert result.status is TaskStatus.VERIFICATION_FAILED
    assert result.reason_code == "safety_policy_mismatch"
    assert builder.calls == 0
