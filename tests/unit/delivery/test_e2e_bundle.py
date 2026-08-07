from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_bytes, sha256_file
from apex.delivery import (
    BuildRecipeLock,
    BuildStep,
    CleanPatchMaterializer,
    DerivedImageIdentity,
    SourceFileChange,
    PrimaryVerificationEvidence,
    compute_e2e_bundle_digest,
    detect_bundle_kind,
    load_and_verify_e2e_bundle,
    validate_lock_order,
)


def _manifest(root: Path) -> dict:
    return json.loads((root / "bundle.json").read_text())


def _resign(root: Path, manifest: dict) -> None:
    for entry in manifest["files"]:
        entry["sha256"] = sha256_file(root / entry["path"])
    manifest["bundle_digest"] = compute_e2e_bundle_digest(manifest, root)
    (root / "bundle.json").write_bytes(canonical_json_bytes(manifest) + b"\n")


def test_capture_records_add_delete_rename_modify_and_mode(make_e2e_bundle) -> None:
    fixture = make_e2e_bundle()
    lock = fixture.bundle.repositories[0]
    kinds = {item.kind for item in lock.changes}

    assert kinds == {"added", "deleted", "renamed", "modified"}
    assert any(item.old_mode == "100644" and item.new_mode == "100755" for item in lock.changes)
    assert all(item.before_blob for item in lock.changes if item.kind != "added")
    assert all(item.after_blob for item in lock.changes if item.kind != "deleted")
    assert fixture.bundle.verified is False
    assert detect_bundle_kind(fixture.bundle.path) == "e2e"


def test_multi_repository_clean_apply_reverse_and_reapply(make_e2e_bundle, tmp_path: Path) -> None:
    fixture = make_e2e_bundle(count=2)
    roots, receipts = CleanPatchMaterializer().materialize(
        bundle_root=fixture.bundle.path,
        locks=fixture.bundle.repositories,
        destination=tmp_path / "fresh-worktrees",
        source_overrides=fixture.bases,
    )

    assert [item.repository_id for item in receipts] == ["repo0", "repo1"]
    assert all(item.verified for item in receipts)
    for root in roots.values():
        assert (root / "src" / "modify.py").read_text() == "VALUE = 2\n"
        assert not (root / "src" / "delete.py").exists()
        assert (root / "src" / "renamed.py").is_file()
        assert (root / "src" / "new.py").is_file()
        assert (root / "src" / "mode.py").stat().st_mode & 0o111


def test_wrong_or_dirty_base_and_wrong_blob_fail_closed(make_e2e_bundle, tmp_path: Path) -> None:
    fixture = make_e2e_bundle()
    (fixture.bases["repo0"] / "dirty.txt").write_text("dirty\n")
    with pytest.raises(IntegrityError) as dirty:
        CleanPatchMaterializer().materialize(
            bundle_root=fixture.bundle.path,
            locks=fixture.bundle.repositories,
            destination=tmp_path / "dirty-replay",
            source_overrides=fixture.bases,
        )
    assert dirty.value.reason_code == "dirty_source_base"
    (fixture.bases["repo0"] / "dirty.txt").unlink()

    lock = fixture.bundle.repositories[0]
    first = lock.changes[0]
    field = "before_blob" if first.before_blob else "after_blob"
    broken_change = replace(first, **{field: "0" * 40})
    broken_lock = replace(lock, changes=(broken_change, *lock.changes[1:]))
    with pytest.raises(IntegrityError) as blob:
        CleanPatchMaterializer().materialize(
            bundle_root=fixture.bundle.path,
            locks=(broken_lock,),
            destination=tmp_path / "wrong-blob",
            source_overrides=fixture.bases,
        )
    assert blob.value.reason_code == "source_blob_mismatch"


def test_patch_cannot_smuggle_an_undeclared_changed_file(make_e2e_bundle, tmp_path: Path) -> None:
    fixture = make_e2e_bundle()
    lock = fixture.bundle.repositories[0]
    original = fixture.bundle.path / lock.patch_path
    injected = original.read_bytes() + (
        b"diff --git a/src/extra.py b/src/extra.py\n"
        b"new file mode 100644\n"
        b"index 0000000..f515fc4\n"
        b"--- /dev/null\n"
        b"+++ b/src/extra.py\n"
        b"@@ -0,0 +1 @@\n"
        b"+EXTRA = True\n"
    )
    original.write_bytes(injected)
    altered_lock = replace(lock, patch_sha256=sha256_bytes(injected))

    with pytest.raises(IntegrityError) as failure:
        CleanPatchMaterializer().materialize(
            bundle_root=fixture.bundle.path,
            locks=(altered_lock,),
            destination=tmp_path / "smuggled-file",
            source_overrides=fixture.bases,
        )
    assert failure.value.reason_code == "source_change_set_mismatch"


def test_patch_file_manifest_and_verdict_tampering_are_detected(make_e2e_bundle) -> None:
    fixture = make_e2e_bundle()
    root = fixture.bundle.path
    patch = root / fixture.bundle.repositories[0].patch_path
    patch.write_bytes(patch.read_bytes() + b"tamper\n")
    with pytest.raises(IntegrityError) as changed:
        load_and_verify_e2e_bundle(root)
    assert changed.value.reason_code == "bundle_file_digest_mismatch"


def test_wrong_expected_canonical_digest_is_rejected(make_e2e_bundle) -> None:
    fixture = make_e2e_bundle()
    with pytest.raises(IntegrityError) as failure:
        load_and_verify_e2e_bundle(fixture.bundle.path, expected_digest="0" * 64)
    assert failure.value.reason_code == "bundle_digest_mismatch"


def test_patch_tamper_still_fails_when_manifest_hash_and_bundle_digest_are_rewritten(
    make_e2e_bundle,
) -> None:
    fixture = make_e2e_bundle()
    root = fixture.bundle.path
    patch = root / fixture.bundle.repositories[0].patch_path
    patch.write_bytes(patch.read_bytes() + b"tamper\n")
    _resign(root, _manifest(root))

    with pytest.raises(IntegrityError) as changed:
        load_and_verify_e2e_bundle(root)
    assert changed.value.reason_code == "bundle_patch_digest_mismatch"


def test_path_traversal_and_undeclared_files_are_detected(make_e2e_bundle) -> None:
    fixture = make_e2e_bundle()
    manifest = _manifest(fixture.bundle.path)
    manifest["files"][0]["path"] = "../sources.lock.json"
    (fixture.bundle.path / "bundle.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    with pytest.raises(IntegrityError) as traversal:
        load_and_verify_e2e_bundle(fixture.bundle.path)
    assert traversal.value.reason_code == "unsafe_bundle_path"


def test_undeclared_file_and_symlink_are_rejected(make_e2e_bundle) -> None:
    fixture = make_e2e_bundle()
    extra = fixture.bundle.path / "unexpected.txt"
    extra.write_text("unexpected\n")
    with pytest.raises(IntegrityError) as undeclared:
        load_and_verify_e2e_bundle(fixture.bundle.path)
    assert undeclared.value.reason_code == "bundle_file_set_mismatch"
    extra.unlink()

    link = fixture.bundle.path / "linked"
    link.symlink_to(fixture.bundle.path / "sources.lock.json")
    with pytest.raises(IntegrityError) as symlink:
        load_and_verify_e2e_bundle(fixture.bundle.path)
    assert symlink.value.reason_code == "bundle_symlink"


def test_replay_workload_quality_tamper_detected_even_if_hashes_are_rewritten(make_e2e_bundle) -> None:
    fixture = make_e2e_bundle()
    root = fixture.bundle.path
    replay = fixture.bundle.config_paths["benchmark_replay"]
    document = yaml.safe_load(replay.read_text())
    document["benchmark"]["server_args"]["tp"] = 8
    replay.write_text(yaml.safe_dump(document), encoding="utf-8")
    manifest = _manifest(root)
    _resign(root, manifest)

    with pytest.raises(IntegrityError) as failure:
        load_and_verify_e2e_bundle(root)
    assert failure.value.reason_code == "replay_config_tampered"


def test_tampering_both_measurement_and_replay_is_caught_by_provenance_lock(
    make_e2e_bundle,
) -> None:
    fixture = make_e2e_bundle()
    root = fixture.bundle.path
    for role in ("benchmark_measurement", "benchmark_replay"):
        path = fixture.bundle.config_paths[role]
        document = yaml.safe_load(path.read_text())
        document["benchmark"]["server_args"]["tp"] = 8
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
    _resign(root, _manifest(root))

    with pytest.raises(IntegrityError) as failure:
        load_and_verify_e2e_bundle(root)
    assert failure.value.reason_code == "benchmark_provenance_mismatch"


def test_recipe_drift_and_parent_drift_are_semantically_rejected(make_e2e_bundle) -> None:
    fixture = make_e2e_bundle()
    root = fixture.bundle.path
    recipe_path = root / "build" / "recipe.lock.json"
    recipe = json.loads(recipe_path.read_text())
    recipe["output_image_locator"] = "attacker@sha256:" + "9" * 64
    recipe_path.write_bytes(canonical_json_bytes(recipe) + b"\n")
    _resign(root, _manifest(root))

    with pytest.raises(IntegrityError) as failure:
        load_and_verify_e2e_bundle(root)
    assert failure.value.reason_code == "build_recipe_drift"


def test_primary_receipt_tamper_cannot_be_hidden_by_rewriting_manifest_hashes(
    make_e2e_bundle,
) -> None:
    fixture = make_e2e_bundle()
    root = fixture.bundle.path
    receipt = fixture.bundle.primary_receipt_paths["primary_engagement_receipt"]
    receipt.write_text('{"kind":"engagement","passed":false}\n', encoding="utf-8")
    _resign(root, _manifest(root))

    with pytest.raises(IntegrityError) as failure:
        load_and_verify_e2e_bundle(root)
    assert failure.value.reason_code == "primary_receipt_mismatch"


def test_sbom_tamper_cannot_be_hidden_by_rewriting_manifest_hashes(make_e2e_bundle) -> None:
    fixture = make_e2e_bundle()
    root = fixture.bundle.path
    fixture.bundle.sbom_path.write_text('{"packages":["unexpected"]}\n', encoding="utf-8")
    _resign(root, _manifest(root))

    with pytest.raises(IntegrityError) as failure:
        load_and_verify_e2e_bundle(root)
    assert failure.value.reason_code == "image_sbom_mismatch"


def test_config_only_submodule_symlink_and_shell_recipe_are_rejected() -> None:
    with pytest.raises(ContractError) as config_only:
        validate_lock_order(())
    assert config_only.value.reason_code == "config_only_candidate"

    with pytest.raises(IntegrityError) as submodule:
        SourceFileChange(
            "modified",
            "submodule",
            "submodule",
            "1" * 40,
            "2" * 40,
            "3" * 64,
            "4" * 64,
            "160000",
            "160000",
        )
    assert submodule.value.reason_code == "submodule_boundary"

    with pytest.raises(IntegrityError) as symlink:
        SourceFileChange(
            "modified",
            "link",
            "link",
            "1" * 40,
            "2" * 40,
            "3" * 64,
            "4" * 64,
            "120000",
            "120000",
        )
    assert symlink.value.reason_code == "unsupported_source_mode"

    with pytest.raises(ContractError) as shell:
        BuildRecipeLock(
            "bad-recipe",
            "sha256:" + "1" * 64,
            "derived:test",
            (BuildStep(("bash", "-c", "make && install"), "repo"),),
        )
    assert shell.value.reason_code == "shell_build_step_forbidden"

    with pytest.raises(ContractError) as mutable_image:
        DerivedImageIdentity(
            "derived:latest",
            "sha256:" + "1" * 64,
            "sha256:" + "2" * 64,
            "3" * 64,
        )
    assert mutable_image.value.reason_code == "invalid_image_identity"


def test_primary_safety_and_overlay_lineage_mismatch_are_rejected() -> None:
    common = {
        "environment_id": "primary",
        "source_stack_sha256": "1" * 64,
        "build_receipt_sha256": "2" * 64,
        "engagement_receipt_sha256": "3" * 64,
        "benchmark_receipt_sha256": "4" * 64,
        "performance_source_sha256": "1" * 64,
        "deployed_source_sha256": "1" * 64,
        "engagement_verified": True,
        "normal_runtime_measurement": True,
        "accuracy_passed": True,
        "latency_gates_passed": True,
        "objective_improved": True,
    }
    with pytest.raises(ContractError) as safety:
        PrimaryVerificationEvidence(
            **common,
            safety_source_sha256="9" * 64,
            safety_certified=True,
        )
    assert safety.value.reason_code == "candidate_lineage_mismatch"

    with pytest.raises(ContractError) as overlay:
        PrimaryVerificationEvidence(
            **common,
            safety_source_sha256=None,
            overlay_verified=True,
            overlay_source_sha256="1" * 64,
            overlay_rebuild_parity_passed=False,
        )
    assert overlay.value.reason_code == "overlay_rebuild_mismatch"


def test_verified_flag_without_second_receipt_is_rejected(make_e2e_bundle) -> None:
    fixture = make_e2e_bundle()
    manifest = _manifest(fixture.bundle.path)
    manifest["verified"] = True
    (fixture.bundle.path / "bundle.json").write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(IntegrityError) as failure:
        load_and_verify_e2e_bundle(fixture.bundle.path)
    assert failure.value.reason_code == "missing_clean_replay_receipt"
