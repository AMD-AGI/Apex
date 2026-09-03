"""Read-only formal qualification artifact resolution contracts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from apex.core import ContractError, canonical_json_bytes, sha256_file, sha256_json
from apex.runtime import (
    EvaluatorQualificationArtifactAuthority,
    FormalResultsRootValidator,
    QualificationArtifactSet,
    build_qualification_evidence,
)
from apex.runtime.qualification_artifacts import INDEX_NAME, INDEX_SCHEMA
from apex.storage import ArtifactStore


_TREE = "b" * 40
_VERIFIER = "c" * 64
_OTHER = "d" * 64


class _AblationVerifier:
    qualification_id = "knowledge-ablation"
    verifier_identity_sha256 = _VERIFIER

    def recompute(self, artifacts: QualificationArtifactSet):
        raw = artifacts.manifest
        assert set(raw) == {"schema", "report_sha256", "episode_manifest_sha256"}
        assert raw["schema"] == "fixture.knowledge-ablation-raw/v1"
        subject = artifacts.manifest_receipt.digest
        return build_qualification_evidence(
            qualification_id=self.qualification_id,
            apex_tree=_TREE,
            subject_sha256=subject,
            status="qualified",
            coverage_count=6,
            formal_delivery_count=0,
            details={
                "schema": "apex.knowledge-ablation-qualification/v1",
                "qualification_manifest_sha256": subject,
                "arms": [
                    "disabled", "static_cards", "static_cards_plus_experience",
                ],
                "task_kinds": ["e2e_kernel_only", "single_kernel"],
                "matched_dimensions": [
                    "backend_model", "budget", "cohort", "gpu_identity",
                    "measurement_policy", "seed",
                ],
                "report_sha256": raw["report_sha256"],
                "episode_manifest_sha256": raw["episode_manifest_sha256"],
                "measured_outcomes_only": True,
                "evaluator_owned_experience_updates": True,
            },
        )


def _root(tmp_path: Path, manifest: dict | None = None) -> tuple[Path, object | None]:
    source = tmp_path / "Apex"
    source.mkdir()
    root = tmp_path / "formal-results"
    root.mkdir()
    receipt = None
    entries = []
    if manifest is not None:
        receipt = ArtifactStore(root / "artifacts").put_bytes(
            canonical_json_bytes(manifest), media_type="application/json"
        )
        entries.append({
            "qualification_id": "knowledge-ablation",
            "manifest_receipt": receipt.to_dict(),
        })
    payload = {"schema": INDEX_SCHEMA, "apex_tree": _TREE, "entries": entries}
    (root / INDEX_NAME).write_bytes(canonical_json_bytes({
        **payload, "manifest_sha256": sha256_json(payload),
    }))
    return root, receipt


def _authority(
    tmp_path: Path,
    manifest: dict | None = None,
    *,
    verifiers=(),
):
    root, receipt = _root(tmp_path, manifest)
    policy = FormalResultsRootValidator((tmp_path / "Apex",))
    return EvaluatorQualificationArtifactAuthority(
        artifact_root=root,
        results_policy=policy,
        verifiers=verifiers,
    ), receipt


def test_self_digested_json_in_cas_is_not_qualification_authority(
    tmp_path: Path,
) -> None:
    fake_claim = {
        "schema": "apex.release-qualification/v2",
        "qualification_id": "knowledge-ablation",
        "claim": "qualified",
    }
    authority, _ = _authority(tmp_path, fake_claim)
    collection = authority.collect()
    ablation = next(
        item for item in collection.entries
        if item.qualification_id == "knowledge-ablation"
    )

    assert ablation.status == "unavailable"
    assert ablation.reason_code == "qualification_artifacts_unavailable"
    assert ablation.evidence is None


def test_trusted_kind_verifier_recomputes_and_binds_exact_evidence(
    tmp_path: Path,
) -> None:
    authority, receipt = _authority(
        tmp_path,
        {
            "schema": "fixture.knowledge-ablation-raw/v1",
            "report_sha256": _OTHER,
            "episode_manifest_sha256": "e" * 64,
        },
        verifiers=(_AblationVerifier(),),
    )
    collection = authority.collect()
    ablation = next(
        item for item in collection.entries
        if item.qualification_id == "knowledge-ablation"
    )

    assert ablation.status == "verified"
    assert ablation.evidence is not None
    result = authority.verify(ablation.evidence)
    assert result.evidence_receipt_sha256 == ablation.evidence["receipt_sha256"]
    assert result.artifact_manifest_sha256 == receipt.digest
    assert result.verifier_identity_sha256 == _VERIFIER


def test_recomputed_evidence_mismatch_is_typed_invalid(tmp_path: Path) -> None:
    authority, _ = _authority(
        tmp_path,
        {
            "schema": "fixture.knowledge-ablation-raw/v1",
            "report_sha256": _OTHER,
            "episode_manifest_sha256": "e" * 64,
        },
        verifiers=(_AblationVerifier(),),
    )
    evidence = next(
        item.evidence for item in authority.collect().entries if item.evidence
    )
    forged = build_qualification_evidence(
        qualification_id="knowledge-ablation",
        apex_tree=_TREE,
        subject_sha256=evidence["subject_sha256"],
        status="qualified",
        coverage_count=7,
        formal_delivery_count=0,
        details=evidence["details"],
    )

    with pytest.raises(ContractError) as caught:
        authority.verify(forged.to_dict())

    assert caught.value.reason_code == "qualification_artifacts_invalid"


def test_cas_symlink_is_invalid_even_when_target_bytes_match(tmp_path: Path) -> None:
    root, receipt = _root(tmp_path, None)
    content = canonical_json_bytes({"schema": "fixture.raw/v1"})
    digest = hashlib.sha256(content).hexdigest()
    external = tmp_path / "external.json"
    external.write_bytes(content)
    cas_path = root / "artifacts" / "sha256" / digest[:2] / digest
    cas_path.parent.mkdir(parents=True)
    cas_path.symlink_to(external)
    entry_receipt = {
        "digest": digest,
        "size": len(content),
        "media_type": "application/json",
        "relative_path": f"sha256/{digest[:2]}/{digest}",
    }
    payload = {
        "schema": INDEX_SCHEMA,
        "apex_tree": _TREE,
        "entries": [{
            "qualification_id": "knowledge-ablation",
            "manifest_receipt": entry_receipt,
        }],
    }
    (root / INDEX_NAME).write_bytes(canonical_json_bytes({
        **payload, "manifest_sha256": sha256_json(payload),
    }))
    authority = EvaluatorQualificationArtifactAuthority(
        artifact_root=root,
        results_policy=FormalResultsRootValidator((tmp_path / "Apex",)),
    )

    ablation = next(
        item for item in authority.collect().entries
        if item.qualification_id == "knowledge-ablation"
    )
    assert receipt is None
    assert ablation.status == "invalid"
    assert ablation.reason_code == "qualification_artifacts_invalid"


def test_collection_does_not_modify_formal_result_tree(tmp_path: Path) -> None:
    authority, _ = _authority(
        tmp_path,
        {
            "schema": "fixture.knowledge-ablation-raw/v1",
            "report_sha256": _OTHER,
            "episode_manifest_sha256": "e" * 64,
        },
        verifiers=(_AblationVerifier(),),
    )
    before = _tree_identity(authority.root)

    authority.collect()

    assert _tree_identity(authority.root) == before


def test_missing_index_is_reported_without_manufacturing_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "Apex"
    source.mkdir()
    root = tmp_path / "formal-results"
    root.mkdir()
    authority = EvaluatorQualificationArtifactAuthority(
        artifact_root=root,
        results_policy=FormalResultsRootValidator((source,)),
    )

    collection = authority.collect()

    assert collection.artifact_index_sha256 is None
    assert {item.status for item in collection.entries} == {"unavailable"}
    assert all(item.evidence is None for item in collection.entries)


def test_artifact_root_must_exist_and_stay_outside_source(tmp_path: Path) -> None:
    source = tmp_path / "Apex"
    source.mkdir()
    policy = FormalResultsRootValidator((source,))

    with pytest.raises(ContractError) as missing:
        EvaluatorQualificationArtifactAuthority(
            artifact_root=tmp_path / "missing", results_policy=policy
        )
    with pytest.raises(ContractError) as overlap:
        EvaluatorQualificationArtifactAuthority(
            artifact_root=source, results_policy=policy
        )

    assert missing.value.reason_code == "qualification_artifacts_unavailable"
    assert overlap.value.reason_code == "formal_results_overlap"


def _tree_identity(root: Path) -> tuple[tuple[object, ...], ...]:
    entries = []
    for path in sorted((root, *root.rglob("*")), key=str):
        info = os.lstat(path)
        entries.append((
            str(path.relative_to(root)), info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns,
            sha256_file(path) if path.is_file() and not path.is_symlink() else None,
        ))
    return tuple(entries)
