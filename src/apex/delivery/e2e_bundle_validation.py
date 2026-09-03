"""Independent validation of serialized second-clean-replay evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from apex.core import IntegrityError, sha256_json

from .e2e_bundle_common import verify_replay_config_invariants
from .e2e_models import BuildRecipeLock, DerivedImageIdentity, SourceRepositoryLock
from .e2e_receipts import PrimaryVerificationEvidence


def validate_final_verification_mapping(
    value: Mapping[str, Any],
    digest: str,
    locks: Sequence[SourceRepositoryLock],
    recipe: BuildRecipeLock,
    image: DerivedImageIdentity,
    primary: PrimaryVerificationEvidence,
    configs: Mapping[str, Path],
) -> None:
    """Re-evaluate all serialized terminal gates instead of trusting booleans."""

    _validate_terminal_claim(value, digest)
    _validate_repository_receipts(value, locks)
    build, engagement, config, replay = _required_receipts(value)
    measurement_sha, replay_sha, semantics_sha = verify_replay_config_invariants(
        configs["benchmark_measurement"],
        configs["benchmark_replay"],
        expected_image_locator=image.locator,
    )
    _validate_receipt_lineage(
        build=build,
        engagement=engagement,
        config=config,
        replay=replay,
        digest=digest,
        recipe=recipe,
        image=image,
        primary=primary,
        measurement_sha=measurement_sha,
        replay_sha=replay_sha,
        semantics_sha=semantics_sha,
        source_materialization_sha=sha256_json(value["repository_receipts"]),
    )
    _validate_build_steps(build, recipe)
    _validate_loaded_bytes(build, engagement, locks, primary)


def _validate_terminal_claim(value: Mapping[str, Any], digest: str) -> None:
    expected = {
        "schema_version": 1,
        "bundle_digest": digest,
        "verified": True,
        "status": "succeeded",
        "validation_level": "source_rebuild_verified",
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise IntegrityError(
            "Final bundle verdict is not source-rebuild verified",
            "invalid_bundle_verdict",
        )


def _validate_repository_receipts(
    value: Mapping[str, Any],
    locks: Sequence[SourceRepositoryLock],
) -> None:
    receipts = value.get("repository_receipts")
    ids = [item.get("repository_id") for item in receipts if isinstance(item, Mapping)] if isinstance(receipts, list) else []
    if not isinstance(receipts, list) or ids != [item.repository_id for item in locks]:
        raise IntegrityError(
            "Repository verification receipts are incomplete",
            "invalid_clean_replay_receipt",
        )
    required_true = (
        "before_blobs_verified",
        "after_blobs_verified",
        "apply_check_passed",
        "reverse_check_passed",
        "reverse_restored_clean_base",
        "reapplied_for_build",
    )
    for serialized, lock in zip(receipts, locks, strict=True):
        identity_ok = isinstance(serialized, Mapping) and all(
            (
                serialized.get("verified") is True,
                serialized.get("base_commit") == lock.base_commit,
                serialized.get("base_tree") == lock.base_tree,
                serialized.get("patch_sha256") == lock.patch_sha256,
                serialized.get("patched_tree") == lock.patched_tree,
            )
        )
        if not identity_ok or any(serialized.get(field) is not True for field in required_true):
            raise IntegrityError(
                "Repository apply/reverse receipt failed",
                "invalid_clean_replay_receipt",
            )


def _required_receipts(value: Mapping[str, Any]):
    receipts = tuple(
        value.get(key)
        for key in (
            "build_receipt",
            "engagement_receipt",
            "config_receipt",
            "replay_receipt",
        )
    )
    if not all(isinstance(item, Mapping) and item.get("verified") is True for item in receipts):
        raise IntegrityError(
            "A required clean replay receipt failed",
            "invalid_clean_replay_receipt",
        )
    return receipts


def _validate_receipt_lineage(
    *,
    build,
    engagement,
    config,
    replay,
    digest,
    recipe,
    image,
    primary,
    measurement_sha,
    replay_sha,
    semantics_sha,
    source_materialization_sha,
) -> None:
    build_ok = all(
        (
            build.get("bundle_digest") == digest,
            build.get("recipe_sha256") == recipe.computed_sha256,
            build.get("expected_parent_digest") == image.parent_digest,
            build.get("observed_parent_digest") == image.parent_digest,
            build.get("expected_image_digest") == image.image_digest,
            build.get("observed_image_digest") == image.image_digest,
            build.get("expected_sbom_sha256") == image.sbom_sha256,
            build.get("observed_sbom_sha256") == image.sbom_sha256,
            build.get("source_stack_sha256") == primary.source_stack_sha256,
            build.get("clean_worktrees") is True,
            build.get("steps_succeeded") is True,
        )
    )
    engagement_ok = all(
        (
            engagement.get("bundle_digest") == digest,
            engagement.get("image_digest") == image.image_digest,
            engagement.get("source_stack_sha256") == primary.source_stack_sha256,
            engagement.get("runtime_started_from_image") is True,
        )
    )
    config_ok = all(
        (
            config.get("measurement_config_sha256") == measurement_sha,
            config.get("replay_config_sha256") == replay_sha,
            config.get("workload_semantics_sha256") == semantics_sha,
            config.get("replay_image_locator") == image.locator,
            config.get("unchanged_except_image_locator") is True,
        )
    )
    if not all(
        (
            build_ok,
            engagement_ok,
            config_ok,
            _replay_lineage_ok(
                replay,
                digest,
                image,
                primary,
                replay_sha,
                source_materialization_sha,
            ),
        )
    ):
        raise IntegrityError(
            "Serialized clean replay evidence is inconsistent",
            "invalid_clean_replay_receipt",
        )


def _replay_lineage_ok(
    replay, digest, image, primary, replay_sha, source_materialization_sha
) -> bool:
    required_true = (
        "fresh_source_materialization",
        "fresh_runtime",
        "normal_runtime_measurement",
        "quality_passed",
        "accuracy_passed",
        "latency_gates_passed",
        "objective_improved",
    )
    return all(
        (
            replay.get("bundle_digest") == digest,
            replay.get("primary_environment_id") == primary.environment_id,
            replay.get("replay_environment_id") != primary.environment_id,
            replay.get("image_digest") == image.image_digest,
            replay.get("replay_config_sha256") == replay_sha,
            replay.get("source_stack_sha256") == primary.source_stack_sha256,
            replay.get("source_materialization_sha256")
            == source_materialization_sha,
            replay.get("primary_runtime_identity_sha256")
            == primary.runtime_identity_sha256,
            _fresh_runtime_identities(replay),
            _valid_replay_artifact_bindings(replay),
            all(replay.get(field) is True for field in required_true),
        )
    )


def _fresh_runtime_identities(replay: Mapping[str, Any]) -> bool:
    primary = replay.get("primary_runtime_identity_sha256")
    identities = replay.get("replay_runtime_identity_sha256s")
    raw = replay.get("paired_measurement", {}).get("raw_measurement_receipts")
    return bool(
        isinstance(primary, str)
        and isinstance(identities, list)
        and isinstance(raw, list)
        and len(identities) == len(raw)
        and len(set(identities)) == len(identities)
        and primary not in identities
        and replay.get("fresh_runtime") is True
    )


def _valid_replay_artifact_bindings(replay: Mapping[str, Any]) -> bool:
    measurement = replay.get("paired_measurement")
    artifacts = replay.get("raw_artifacts")
    if not isinstance(measurement, Mapping) or not isinstance(artifacts, list):
        return False
    raw = measurement.get("raw_measurement_receipts")
    if not isinstance(raw, list):
        return False
    reports = [
        item.get("measurement_receipt")
        for item in artifacts
        if isinstance(item, Mapping) and item.get("role") == "benchmark_report"
    ]
    attestations = [
        item.get("measurement_receipt")
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("role") == "execution_attestation"
    ]
    keys = [
        (item.get("role"), item.get("relative_path"), item.get("measurement_receipt"))
        for item in artifacts
        if isinstance(item, Mapping)
    ]
    return bool(
        len(keys) == len(artifacts)
        and len(set(keys)) == len(keys)
        and sorted(reports) == sorted(raw)
        and sorted(attestations) == sorted(raw)
    )


def _validate_build_steps(build, recipe: BuildRecipeLock) -> None:
    receipts = build.get("step_receipts")
    if not isinstance(receipts, list) or len(receipts) != len(recipe.steps):
        raise IntegrityError(
            "Build step receipts are incomplete",
            "invalid_clean_replay_receipt",
        )
    for index, (serialized, step) in enumerate(zip(receipts, recipe.steps, strict=True)):
        digests_valid = isinstance(serialized, Mapping) and all(
            isinstance(serialized.get(field), str)
            and len(str(serialized.get(field))) == 64
            for field in ("stdout_sha256", "stderr_sha256")
        )
        fixed_fields = isinstance(serialized, Mapping) and all(
            (
                serialized.get("index") == index,
                serialized.get("repository_id") == step.repository_id,
                serialized.get("cwd") == step.cwd,
                serialized.get("argv_sha256") == sha256_json(list(step.argv)),
                serialized.get("exit_code") == 0,
                serialized.get("timed_out") is False,
                serialized.get("verified") is True,
            )
        )
        if not digests_valid or not fixed_fields:
            raise IntegrityError(
                "Build step receipt differs from fixed recipe",
                "invalid_clean_replay_receipt",
            )


def _validate_loaded_bytes(build, engagement, locks, primary) -> None:
    artifacts = build.get("artifacts")
    loaded = engagement.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or not isinstance(loaded, list)
        or not artifacts
        or len(artifacts) != len(loaded)
    ):
        raise IntegrityError(
            "Loaded-byte evidence is incomplete",
            "loaded_byte_engagement_failed",
        )
    components = {item.get("component") for item in artifacts if isinstance(item, Mapping)}
    if not {item.runtime_component for item in locks}.issubset(components):
        raise IntegrityError(
            "Changed repository lacks a deployed artifact",
            "loaded_byte_engagement_failed",
        )
    built_keys = {
        (item.get("component"), item.get("runtime_path"), item.get("sha256"), item.get("build_id"))
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("source_stack_sha256") == primary.source_stack_sha256
    }
    loaded_keys = _verified_loaded_keys(loaded)
    if built_keys != loaded_keys or len(built_keys) != len(artifacts):
        raise IntegrityError(
            "Runtime loaded old or unexpected bytes",
            "loaded_byte_engagement_failed",
        )
    policies = {
        item.runtime_component: (item.engagement_kind, item.build_id_required)
        for item in locks
    }
    for item in loaded:
        policy = policies.get(item.get("component")) if isinstance(item, Mapping) else None
        if (
            policy is None
            or item.get("engagement_kind") != policy[0]
            or policy[1]
            and (
                not item.get("expected_build_id")
                or not item.get("observed_build_id")
            )
        ):
            raise IntegrityError(
                "Loaded artifact violates its component engagement capability",
                "loaded_byte_engagement_failed",
            )


def _verified_loaded_keys(loaded) -> set[tuple[Any, ...]]:
    return {
        (
            item.get("component"),
            item.get("runtime_path"),
            item.get("expected_sha256"),
            item.get("expected_build_id"),
        )
        for item in loaded
        if isinstance(item, Mapping)
        and item.get("expected_sha256") == item.get("observed_sha256")
        and item.get("expected_build_id") == item.get("observed_build_id")
        and item.get("actually_loaded") is True
        and item.get("engagement_kind")
        in {"python_import", "process_map", "linker_build_id"}
        and bool(item.get("runtime_identity"))
        and item.get("verified") is True
    }


__all__ = ["validate_final_verification_mapping"]
