from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import apex.delivery.git_patch as git_patch
from apex.delivery.e2e_models import SourceComponentCapability

from apex.core import (
    IntegrityError,
    TaskStatus,
    ValidationLevel,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from apex.delivery import (
    BuiltArtifact,
    BuildStepReceipt,
    CleanReplayReceipt,
    E2EBundleVerifier,
    LoadedArtifact,
    LoadedByteEngagementReceipt,
    ReplayArtifactReceipt,
    CleanPatchMaterializer,
    SourceBuildReceipt,
    SourceBuildRequest,
    SupervisedRecipeBuildBackend,
    capture_portable_bundle,
    e2e_reproduction_declaration,
    compute_e2e_bundle_digest,
    delivery_terminal_policy,
    load_and_verify_e2e_bundle,
    verify_portable_bundle,
)
from apex.execution import ProcessResult
from apex.evaluation import (
    E2EAcceptancePolicy,
    E2EObservation,
    E2EPairedMeasurement,
    E2EPairedWindow,
    evaluate_paired_current_anchor,
)
from apex.storage import ArtifactStore


def _paired_replay(keep: bool) -> tuple[dict, dict]:
    policy = E2EAcceptancePolicy()
    windows = []
    for window in range(3):
        values = []
        for position, candidate in enumerate((False, True, True, False)):
            receipt = f"{window}-{position}"
            values.append(
                E2EObservation(
                    101.0 if candidate and keep else 100.0,
                    10.0,
                    1.0,
                    1.0,
                    10,
                    "a" * 64,
                    f"quality-{receipt}",
                    f"measurement-{receipt}",
                )
            )
        windows.append(E2EPairedWindow(f"window-{window}", *values))
    measurement = E2EPairedMeasurement(tuple(windows), policy.digest, 3)
    verdict = evaluate_paired_current_anchor(measurement, policy)
    return measurement.to_dict(), verdict.to_dict()


class FakeBuild:
    def __init__(self, *, old_image: bool = False, wrong_source: bool = False) -> None:
        self.old_image = old_image
        self.wrong_source = wrong_source

    def build(self, request):
        source = "9" * 64 if self.wrong_source else request.source_stack_sha256
        image = "sha256:" + "9" * 64 if self.old_image else request.expected_image.image_digest
        artifacts = tuple(
            BuiltArtifact(
                item.repository_id,
                f"/opt/{item.repository_id}/kernel.py",
                str(index + 1) * 64,
                None,
                source,
            )
            for index, item in enumerate(request.repository_receipts)
        )
        return SourceBuildReceipt(
            bundle_digest=request.bundle_digest,
            recipe_sha256=request.recipe.computed_sha256,
            expected_parent_digest=request.expected_image.parent_digest,
            observed_parent_digest=request.expected_image.parent_digest,
            expected_image_digest=request.expected_image.image_digest,
            observed_image_digest=image,
            expected_sbom_sha256=request.expected_image.sbom_sha256,
            observed_sbom_sha256=request.expected_image.sbom_sha256,
            source_stack_sha256=request.source_stack_sha256,
            clean_worktrees=True,
            artifacts=artifacts,
            step_receipts=tuple(
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
            ),
        )


class FakeEngagement:
    def __init__(
        self,
        *,
        old_bytes: bool = False,
        wrong_image: bool = False,
        engagement_kind: str = "python_import",
    ) -> None:
        self.old_bytes = old_bytes
        self.wrong_image = wrong_image
        self.engagement_kind = engagement_kind

    def verify_loaded_bytes(self, request):
        artifacts = tuple(
            LoadedArtifact(
                item.component,
                item.runtime_path,
                item.sha256,
                "b" * 64 if self.old_bytes else item.sha256,
                item.build_id,
                item.build_id,
                self.engagement_kind,
                f"import:{item.component}",
                True,
            )
            for item in request.build_receipt.artifacts
        )
        return LoadedByteEngagementReceipt(
            request.bundle_digest,
            "sha256:" + "8" * 64 if self.wrong_image else request.expected_image.image_digest,
            request.source_stack_sha256,
            True,
            artifacts,
        )


class FakeReplay:
    def __init__(
        self,
        *,
        same_environment: bool = False,
        objective_improved: bool = True,
        quality_passed: bool = True,
        reused_runtime: bool = False,
        wrong_source_materialization: bool = False,
        omit_execution_attestation: bool = False,
    ) -> None:
        self.same_environment = same_environment
        self.objective_improved = objective_improved
        self.quality_passed = quality_passed
        self.reused_runtime = reused_runtime
        self.wrong_source_materialization = wrong_source_materialization
        self.omit_execution_attestation = omit_execution_attestation

    def replay(self, request):
        environment = request.primary_environment_id if self.same_environment else "independent-replay-environment"
        measurement, verdict = _paired_replay(self.objective_improved)
        artifacts = _raw_replay_artifacts(request, measurement)
        if self.omit_execution_attestation:
            artifacts = tuple(
                item for item in artifacts if item.role != "execution_attestation"
            )
        runtime_ids = tuple(
            sha256_json({"runtime": 0 if self.reused_runtime else index})
            for index in range(len(measurement["raw_measurement_receipts"]))
        )
        return CleanReplayReceipt(
            bundle_digest=request.bundle_digest,
            primary_environment_id=request.primary_environment_id,
            replay_environment_id=environment,
            image_digest=request.expected_image.image_digest,
            replay_config_sha256=request.config_receipt.replay_config_sha256,
            benchmark_receipt_sha256="c" * 64,
            source_stack_sha256=request.source_stack_sha256,
            source_materialization_sha256=sha256_json(
                [item.to_dict() for item in request.repository_receipts]
            )
            if not self.wrong_source_materialization
            else "0" * 64,
            primary_runtime_identity_sha256="d" * 64,
            replay_runtime_identity_sha256s=runtime_ids,
            normal_runtime_measurement=True,
            quality_passed=self.quality_passed,
            accuracy_passed=True,
            latency_gates_passed=True,
            objective_improved=self.objective_improved,
            paired_measurement=measurement,
            paired_verdict=verdict,
            raw_artifacts=artifacts,
        )


def _raw_replay_artifacts(request, measurement: dict) -> tuple[ReplayArtifactReceipt, ...]:
    root = request.output_dir
    assert root is not None
    root.mkdir(parents=True, exist_ok=True)
    values = []
    observations = [
        item
        for window in measurement["windows"]
        for item in window["observations"]
    ]
    for index, observation in enumerate(observations):
        for role in (
            "benchmark_report",
            "execution_attestation",
            "quality_result",
        ):
            path = root / f"{index}-{role}.json"
            path.write_text(json.dumps({"index": index, "role": role}), encoding="utf-8")
            values.append(
                ReplayArtifactReceipt(
                    role,
                    f"run-{index}",
                    observation["measurement_receipt"],
                    observation["quality_receipt"],
                    path.relative_to(root).as_posix(),
                    sha256_file(path),
                    path.stat().st_size,
                    "application/json",
                )
            )
    return tuple(values)


class RecordingSupervisor:
    def __init__(self) -> None:
        self.calls = []

    def run(self, argv, *, cwd, environment, timeout_seconds, stdin_text=None):
        self.calls.append((tuple(argv), cwd, dict(environment), timeout_seconds))
        return ProcessResult(tuple(argv), 0, False, "", "", False, False, 0.01)


def test_patch_git_child_revokes_agent_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-secret")
    monkeypatch.setenv("CURSOR_API_KEY", "cursor-test-secret")
    monkeypatch.setenv("HF_TOKEN", "hf-test-secret")
    supervisor = RecordingSupervisor()

    git_patch._Git(supervisor).run(tmp_path, "status")

    environment = supervisor.calls[0][2]
    assert all(
        key not in environment
        for key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "CURSOR_API_KEY",
            "HF_TOKEN",
        )
    )
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"


class FakeAttestor:
    def attest(self, request):
        return FakeBuild().build(request)


def verifier(fixture, *, build=None, engagement=None, replay=None):
    return E2EBundleVerifier(
        trusted_recipes={fixture.recipe.computed_sha256: fixture.recipe},
        trusted_source_urls={item.repository_id: item.url for item in fixture.bundle.repositories},
        trusted_recipe_capabilities={
            fixture.recipe.computed_sha256: tuple(
                item.component_capability for item in fixture.bundle.repositories
            )
        },
        build_backend=build or FakeBuild(),
        engagement_backend=engagement or FakeEngagement(),
        replay_backend=replay or FakeReplay(),
    )


def test_full_second_clean_replay_finalizes_verified_bundle(make_e2e_bundle, tmp_path: Path) -> None:
    fixture = make_e2e_bundle(count=2)
    outcome = verifier(fixture).verify(
        bundle_dir=fixture.bundle.path,
        results_dir=tmp_path / "verify",
        expected_digest=fixture.bundle.digest,
        source_overrides=fixture.bases,
    )

    assert outcome.result.verified
    assert outcome.result.status is TaskStatus.SUCCEEDED
    assert outcome.result.validation_level is ValidationLevel.SOURCE_REBUILD_VERIFIED
    assert outcome.verified_bundle is not None
    assert outcome.verified_bundle.verified
    assert outcome.verified_bundle.digest == fixture.bundle.digest
    assert load_and_verify_e2e_bundle(outcome.verified_bundle.path).verified
    serialized = json.loads(outcome.result_path.read_text())
    assert serialized["build_receipt"]["clean_worktrees"] is True
    assert serialized["engagement_receipt"]["artifacts"][0]["verified"] is True
    assert serialized["replay_receipt"]["fresh_source_materialization"] is True

    artifacts = ArtifactStore(tmp_path / "portable-cas")
    portable = capture_portable_bundle(
        artifacts,
        outcome.verified_bundle.path,
        bundle_kind="e2e",
        expected_digest=outcome.verified_bundle.digest,
    )
    replayed = verify_portable_bundle(
        artifacts, portable.evidence_receipt, portable.verification_receipt
    )
    assert replayed.bundle_kind == "e2e"
    assert replayed.bundle_digest == outcome.verified_bundle.digest
    assert replayed.file_count > 10
    reproduction = e2e_reproduction_declaration(outcome.verified_bundle, portable)
    assert reproduction["task_kind"] == "e2e_kernel_only"
    assert reproduction["parent_image_digest"] == fixture.recipe.parent_image_digest
    assert reproduction["derived_image_digest"] == fixture.image.image_digest
    assert {item["name"] for item in reproduction["commands"]} >= {
        "verify_bundle",
        "build_image",
        "clean_replay",
    }


def test_composed_default_sources_are_cloned_into_fresh_verifier_worktrees(
    make_e2e_bundle, tmp_path: Path
) -> None:
    fixture = make_e2e_bundle()
    verifier_instance = E2EBundleVerifier(
        trusted_recipes={fixture.recipe.computed_sha256: fixture.recipe},
        trusted_source_urls={
            item.repository_id: item.url for item in fixture.bundle.repositories
        },
        trusted_recipe_capabilities={
            fixture.recipe.computed_sha256: tuple(
                item.component_capability for item in fixture.bundle.repositories
            )
        },
        build_backend=FakeBuild(),
        engagement_backend=FakeEngagement(),
        replay_backend=FakeReplay(),
        default_source_overrides=fixture.bases,
    )

    outcome = verifier_instance.verify(
        bundle_dir=fixture.bundle.path,
        results_dir=tmp_path / "verify-default-sources",
    )

    assert outcome.result.verified
    worktree = tmp_path / "verify-default-sources" / "worktrees" / "repo0"
    assert worktree.is_dir()
    assert worktree.resolve() != fixture.bases["repo0"].resolve()


def test_fixed_recipe_executor_uses_argv_supervisor_without_shell(
    monkeypatch: pytest.MonkeyPatch, make_e2e_bundle, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-secret")
    monkeypatch.setenv("CURSOR_API_KEY", "cursor-test-secret")
    monkeypatch.setenv("HF_TOKEN", "hf-test-secret")
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/apex-test-docker.sock")
    monkeypatch.setenv("ROCM_PATH", "/opt/rocm-test")
    monkeypatch.setenv("PYTHONPATH", "/tmp/untrusted-python")
    fixture = make_e2e_bundle()
    roots, receipts = CleanPatchMaterializer().materialize(
        bundle_root=fixture.bundle.path,
        locks=fixture.bundle.repositories,
        destination=tmp_path / "build-roots",
        source_overrides=fixture.bases,
    )
    request = SourceBuildRequest(
        fixture.bundle.digest,
        fixture.bundle.primary_evidence.source_stack_sha256,
        fixture.recipe,
        fixture.image,
        roots,
        receipts,
    )
    supervisor = RecordingSupervisor()
    receipt = SupervisedRecipeBuildBackend(FakeAttestor(), supervisor).build(request)

    assert receipt.verified
    assert [call[0] for call in supervisor.calls] == [("python3", "build.py")]
    assert supervisor.calls[0][1] == roots["repo0"]
    environment = supervisor.calls[0][2]
    assert environment["DOCKER_HOST"] == "unix:///tmp/apex-test-docker.sock"
    assert environment["ROCM_PATH"] == "/opt/rocm-test"
    assert all(
        key not in environment
        for key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "CURSOR_API_KEY",
            "HF_TOKEN",
            "PYTHONPATH",
        )
    )
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"


@pytest.mark.parametrize(
    ("build", "engagement", "replay", "reason"),
    [
        (FakeBuild(old_image=True), None, None, "source_build_receipt_mismatch"),
        (FakeBuild(wrong_source=True), None, None, "source_build_receipt_mismatch"),
        (None, FakeEngagement(old_bytes=True), None, "loaded_byte_engagement_failed"),
        (None, FakeEngagement(wrong_image=True), None, "loaded_byte_engagement_failed"),
        (None, FakeEngagement(engagement_kind="process_map"), None, "loaded_byte_engagement_failed"),
        (None, None, FakeReplay(same_environment=True), "second_clean_replay_failed"),
        (None, None, FakeReplay(reused_runtime=True), "second_clean_replay_failed"),
        (
            None,
            None,
            FakeReplay(wrong_source_materialization=True),
            "second_clean_replay_failed",
        ),
        (
            None,
            None,
            FakeReplay(omit_execution_attestation=True),
            "invalid_replay_receipt",
        ),
        (None, None, FakeReplay(objective_improved=False), "second_clean_replay_failed"),
        (None, None, FakeReplay(quality_passed=False), "second_clean_replay_failed"),
    ],
)
def test_any_required_receipt_failure_is_verification_failed(
    make_e2e_bundle,
    tmp_path: Path,
    build,
    engagement,
    replay,
    reason: str,
) -> None:
    fixture = make_e2e_bundle(overlay=True)
    outcome = verifier(fixture, build=build, engagement=engagement, replay=replay).verify(
        bundle_dir=fixture.bundle.path,
        results_dir=tmp_path / "failed-verify",
        source_overrides=fixture.bases,
    )

    assert not outcome.result.verified
    assert outcome.result.status is TaskStatus.VERIFICATION_FAILED
    assert outcome.result.validation_level is ValidationLevel.RUNTIME_OVERLAY_VERIFIED
    assert outcome.result.reason_code == reason
    assert outcome.verified_bundle is None
    assert json.loads(outcome.result_path.read_text())["verified"] is False


def test_build_id_capability_rejects_digest_only_engagement(
    make_e2e_bundle, tmp_path: Path
) -> None:
    fixture = make_e2e_bundle(
        engagement_kind="linker_build_id", build_id_required=True
    )
    outcome = verifier(
        fixture,
        engagement=FakeEngagement(engagement_kind="linker_build_id"),
    ).verify(
        bundle_dir=fixture.bundle.path,
        results_dir=tmp_path / "missing-build-id",
        source_overrides=fixture.bases,
    )

    assert outcome.result.reason_code == "loaded_byte_engagement_failed"
    assert outcome.verified_bundle is None


def test_wrong_exact_source_override_fails_before_build(make_e2e_bundle, tmp_path: Path) -> None:
    fixture = make_e2e_bundle()
    base = fixture.bases["repo0"]
    (base / "later.txt").write_text("later\n")
    subprocess.run(("git", "-C", str(base), "add", "later.txt"), check=True)
    subprocess.run(("git", "-C", str(base), "commit", "-q", "-m", "later"), check=True)

    outcome = verifier(fixture).verify(
        bundle_dir=fixture.bundle.path,
        results_dir=tmp_path / "wrong-base",
        source_overrides=fixture.bases,
    )
    assert outcome.result.status is TaskStatus.VERIFICATION_FAILED
    assert outcome.result.reason_code == "repository_commit_mismatch"
    assert outcome.result.build_receipt is None


def test_untrusted_recipe_returns_structured_failure_before_build(make_e2e_bundle, tmp_path: Path) -> None:
    fixture = make_e2e_bundle(overlay=True)
    verifier_instance = E2EBundleVerifier(
        trusted_recipes={},
        trusted_source_urls={item.repository_id: item.url for item in fixture.bundle.repositories},
        trusted_recipe_capabilities={},
        build_backend=FakeBuild(),
        engagement_backend=FakeEngagement(),
        replay_backend=FakeReplay(),
    )
    outcome = verifier_instance.verify(
        bundle_dir=fixture.bundle.path,
        results_dir=tmp_path / "untrusted",
        source_overrides=fixture.bases,
    )
    assert outcome.result.status is TaskStatus.VERIFICATION_FAILED
    assert outcome.result.validation_level is ValidationLevel.RUNTIME_OVERLAY_VERIFIED
    assert outcome.result.reason_code == "untrusted_build_recipe"
    assert not (tmp_path / "untrusted" / "worktrees").exists()


def test_trusted_recipe_rejects_an_unregistered_repository_set(
    make_e2e_bundle, tmp_path: Path
) -> None:
    fixture = make_e2e_bundle(count=2)
    verifier_instance = E2EBundleVerifier(
        trusted_recipes={fixture.recipe.computed_sha256: fixture.recipe},
        trusted_source_urls={
            item.repository_id: item.url for item in fixture.bundle.repositories
        },
        trusted_recipe_capabilities={
            fixture.recipe.computed_sha256: (
                fixture.bundle.repositories[0].component_capability,
                SourceComponentCapability(
                    "repo1", "wrong-runtime", "python_import"
                ),
            )
        },
        build_backend=FakeBuild(),
        engagement_backend=FakeEngagement(),
        replay_backend=FakeReplay(),
    )

    outcome = verifier_instance.verify(
        bundle_dir=fixture.bundle.path,
        results_dir=tmp_path / "wrong-repository-set",
        source_overrides=fixture.bases,
    )

    assert outcome.result.status is TaskStatus.VERIFICATION_FAILED
    assert outcome.result.reason_code == "untrusted_build_recipe"
    assert not (tmp_path / "wrong-repository-set" / "worktrees").exists()


def test_untrusted_source_url_is_rejected_before_clone(make_e2e_bundle, tmp_path: Path) -> None:
    fixture = make_e2e_bundle()
    verifier_instance = E2EBundleVerifier(
        trusted_recipes={fixture.recipe.computed_sha256: fixture.recipe},
        trusted_source_urls={"repo0": "https://example.com/different.git"},
        trusted_recipe_capabilities={
            fixture.recipe.computed_sha256: tuple(
                item.component_capability for item in fixture.bundle.repositories
            )
        },
        build_backend=FakeBuild(),
        engagement_backend=FakeEngagement(),
        replay_backend=FakeReplay(),
    )
    outcome = verifier_instance.verify(
        bundle_dir=fixture.bundle.path,
        results_dir=tmp_path / "untrusted-source",
        source_overrides=fixture.bases,
    )
    assert outcome.result.status is TaskStatus.VERIFICATION_FAILED
    assert outcome.result.reason_code == "untrusted_source_repository"
    assert not (tmp_path / "untrusted-source" / "worktrees").exists()


def test_terminal_policy_never_conflates_overlay_and_source_verification() -> None:
    unresolved = delivery_terminal_policy(
        source_locks_resolved=False,
        overlay_verified=True,
        repositories_verified=False,
        build_verified=False,
        engagement_verified=False,
        config_verified=False,
        clean_replay_verified=False,
    )
    assert unresolved.status is TaskStatus.PROVENANCE_UNRESOLVED
    assert unresolved.validation_level is ValidationLevel.RUNTIME_OVERLAY_VERIFIED
    assert not unresolved.verified

    failed = delivery_terminal_policy(
        source_locks_resolved=True,
        overlay_verified=False,
        repositories_verified=True,
        build_verified=True,
        engagement_verified=True,
        config_verified=True,
        clean_replay_verified=False,
    )
    assert failed.status is TaskStatus.VERIFICATION_FAILED
    assert failed.validation_level is ValidationLevel.NONE

    passed = delivery_terminal_policy(
        source_locks_resolved=True,
        overlay_verified=False,
        repositories_verified=True,
        build_verified=True,
        engagement_verified=True,
        config_verified=True,
        clean_replay_verified=True,
    )
    assert passed.status is TaskStatus.SUCCEEDED
    assert passed.validation_level is ValidationLevel.SOURCE_REBUILD_VERIFIED
    assert passed.verified


def test_tampered_final_receipt_fails_even_when_attacker_rewrites_file_hash(
    make_e2e_bundle, tmp_path: Path
) -> None:
    fixture = make_e2e_bundle()
    outcome = verifier(fixture).verify(
        bundle_dir=fixture.bundle.path,
        results_dir=tmp_path / "verify",
        source_overrides=fixture.bases,
    )
    assert outcome.verified_bundle is not None
    root = outcome.verified_bundle.path
    manifest_path = root / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    receipt_path = root / manifest["verification_receipt"]["path"]
    receipt = json.loads(receipt_path.read_text())
    receipt["replay_receipt"]["objective_improved"] = False
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    new_hash = sha256_file(receipt_path)
    manifest["verification_receipt"]["sha256"] = new_hash
    for entry in manifest["files"]:
        if entry["role"] == "second_clean_replay_receipt":
            entry["sha256"] = new_hash
    assert compute_e2e_bundle_digest(manifest, root) == fixture.bundle.digest
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(IntegrityError) as failure:
        load_and_verify_e2e_bundle(root)
    assert failure.value.reason_code == "invalid_clean_replay_receipt"
