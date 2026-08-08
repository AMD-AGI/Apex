from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

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
    CleanPatchMaterializer,
    SourceBuildReceipt,
    SourceBuildRequest,
    SupervisedRecipeBuildBackend,
    compute_e2e_bundle_digest,
    delivery_terminal_policy,
    load_and_verify_e2e_bundle,
)
from apex.execution import ProcessResult


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
    def __init__(self, *, old_bytes: bool = False, wrong_image: bool = False) -> None:
        self.old_bytes = old_bytes
        self.wrong_image = wrong_image

    def verify_loaded_bytes(self, request):
        artifacts = tuple(
            LoadedArtifact(
                item.component,
                item.runtime_path,
                item.sha256,
                "b" * 64 if self.old_bytes else item.sha256,
                item.build_id,
                item.build_id,
                "python_import",
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
    ) -> None:
        self.same_environment = same_environment
        self.objective_improved = objective_improved
        self.quality_passed = quality_passed

    def replay(self, request):
        environment = request.primary_environment_id if self.same_environment else "independent-replay-environment"
        return CleanReplayReceipt(
            request.bundle_digest,
            request.primary_environment_id,
            environment,
            request.expected_image.image_digest,
            request.config_receipt.replay_config_sha256,
            "c" * 64,
            request.source_stack_sha256,
            True,
            True,
            True,
            self.quality_passed,
            True,
            True,
            self.objective_improved,
        )


class RecordingSupervisor:
    def __init__(self) -> None:
        self.calls = []

    def run(self, argv, *, cwd, environment, timeout_seconds, stdin_text=None):
        self.calls.append((tuple(argv), cwd, dict(environment), timeout_seconds))
        return ProcessResult(tuple(argv), 0, False, "", "", False, False, 0.01)


class FakeAttestor:
    def attest(self, request):
        return FakeBuild().build(request)


def verifier(fixture, *, build=None, engagement=None, replay=None):
    return E2EBundleVerifier(
        trusted_recipes={fixture.recipe.computed_sha256: fixture.recipe},
        trusted_source_urls={item.repository_id: item.url for item in fixture.bundle.repositories},
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


def test_composed_default_sources_are_cloned_into_fresh_verifier_worktrees(
    make_e2e_bundle, tmp_path: Path
) -> None:
    fixture = make_e2e_bundle()
    verifier_instance = E2EBundleVerifier(
        trusted_recipes={fixture.recipe.computed_sha256: fixture.recipe},
        trusted_source_urls={
            item.repository_id: item.url for item in fixture.bundle.repositories
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
    make_e2e_bundle, tmp_path: Path
) -> None:
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


@pytest.mark.parametrize(
    ("build", "engagement", "replay", "reason"),
    [
        (FakeBuild(old_image=True), None, None, "source_build_receipt_mismatch"),
        (FakeBuild(wrong_source=True), None, None, "source_build_receipt_mismatch"),
        (None, FakeEngagement(old_bytes=True), None, "loaded_byte_engagement_failed"),
        (None, FakeEngagement(wrong_image=True), None, "loaded_byte_engagement_failed"),
        (None, None, FakeReplay(same_environment=True), "second_clean_replay_failed"),
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
        build_backend=FakeBuild(),
        engagement_backend=FakeEngagement(),
        replay_backend=FakeReplay(),
        trusted_recipe_repositories={
            fixture.recipe.computed_sha256: frozenset({"repo0"})
        },
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
