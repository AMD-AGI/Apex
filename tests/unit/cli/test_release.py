from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from apex.cli import app
from apex.core import ContractError, sha256_json
from apex.runtime import ReleaseEvidence, build_qualification_evidence


ROOT = Path(__file__).resolve().parents[3]


def test_release_check_reports_distinct_baseline_and_final_gates(capsys) -> None:
    status = app.main([
        "release", "check", "--apex-root", str(ROOT), "--json",
    ])

    assert status == 0
    value = json.loads(capsys.readouterr().out)
    assert value["schema"] == "apex.release-candidate-receipt/v2"
    assert (value["baseline_status"], value["status"]) == ("blocked", "blocked")
    assert value["static"]["magpie"]["config_compatible_count"] == 27
    assert ("apex_source_dirty" in value["baseline_blockers"]) is (
        not value["static"]["apex_checkout"]["clean"]
    )
    assert "showcase_missing:e2e-qwen3-next-80b-fp8" in value["blockers"]


def test_release_check_verifies_existing_receipt(tmp_path: Path, capsys) -> None:
    assert app.main([
        "release", "check", "--apex-root", str(ROOT), "--json",
    ]) == 0
    value = json.loads(capsys.readouterr().out)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(value), encoding="utf-8")

    status = app.main([
        "release", "check", "--apex-root", str(ROOT),
        "--receipt", str(receipt), "--json",
    ])

    assert status == 0
    assert json.loads(capsys.readouterr().out) == value


def test_release_require_baseline_fails_before_live_work(capsys) -> None:
    status = app.main([
        "release", "check", "--apex-root", str(ROOT), "--require-baseline",
    ])

    assert status == 2
    error = json.loads(capsys.readouterr().err)
    assert error["reason_code"] == "campaign_baseline_blocked"


def test_release_evidence_symlink_is_rejected(tmp_path: Path, capsys) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(evidence)

    status = app.main([
        "release", "check", "--apex-root", str(ROOT), "--evidence", str(link),
    ])

    assert status == 2
    error = json.loads(capsys.readouterr().err)
    assert error["reason_code"] == "unsafe_release_evidence"


def test_release_collect_local_creates_one_typed_document(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(
        "apex.cli.release.collect_local_release_evidence",
        lambda root: ReleaseEvidence(),
    )
    output = tmp_path / "local-evidence.json"

    status = app.main([
        "release", "collect-local", "--apex-root", str(ROOT),
        "--output", str(output),
    ])

    assert status == 0
    assert json.loads(output.read_bytes()) == ReleaseEvidence().to_dict()
    assert str(output) in capsys.readouterr().out
    assert app.main([
        "release", "collect-local", "--apex-root", str(ROOT),
        "--output", str(output),
    ]) == 2
    assert json.loads(capsys.readouterr().err)["reason_code"] == "release_output_exists"


def test_release_collect_showcase_binds_clean_tree_and_offline_receipt(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    tree = "b" * 40
    receipt = {
        "schema": "apex.showcase-verification/v2",
        "showcase_id": "kernel-triton-paged-attention-2d",
        "status": "published",
        "file_count": 12,
        "checksums_sha256": "a" * 64,
        "event_count": 20,
        "artifact_count": 8,
        "reward_replayed": True,
        "bundle_verified": True,
        "reproduction_verified": True,
        "episode_sha256": "a" * 64,
        "artifact_manifest_sha256": "a" * 64,
        "reward_sha256": "a" * 64,
        "result_sha256": "a" * 64,
        "reproduction_sha256": "a" * 64,
    }
    receipt["verification_receipt_sha256"] = sha256_json(receipt)
    monkeypatch.setattr(
        "apex.cli.release.WorkspaceGitIdentityResolver",
        lambda: SimpleNamespace(inspect=lambda root: SimpleNamespace(
            resolved=True,
            root=str(root),
            tree=tree,
            dirty_paths=(),
        )),
    )
    monkeypatch.setattr(
        "apex.cli.release.verify_showcase",
        lambda path: SimpleNamespace(to_receipt=lambda: receipt),
    )
    output = tmp_path / "showcase-release.json"

    status = app.main([
        "release", "collect-showcase", "--apex-root", str(ROOT),
        "--path", str(tmp_path / "showcase"), "--output", str(output),
    ])

    assert status == 0
    value = json.loads(output.read_bytes())
    assert value["schema"] == "apex.release-showcase-verification/v2"
    assert value["apex_tree"] == tree
    assert value["verification_receipt_sha256"] == receipt[
        "verification_receipt_sha256"
    ]
    assert str(output) in capsys.readouterr().out

    base = tmp_path / "base.json"
    base.write_text(json.dumps(ReleaseEvidence().to_dict()), encoding="utf-8")
    joined = tmp_path / "joined.json"
    assert app.main([
        "release", "join-evidence", "--base", str(base),
        "--showcase", str(output), "--output", str(joined),
    ]) == 0
    joined_value = json.loads(joined.read_bytes())
    assert [item["showcase_id"] for item in joined_value["showcases"]] == [
        "kernel-triton-paged-attention-2d"
    ]
    assert str(joined) in capsys.readouterr().out


def test_release_collect_showcase_rejects_dirty_source(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.setattr(
        "apex.cli.release.WorkspaceGitIdentityResolver",
        lambda: SimpleNamespace(inspect=lambda root: SimpleNamespace(
            resolved=True,
            root=str(root),
            tree="b" * 40,
            dirty_paths=(" M src/apex/runtime/release.py",),
        )),
    )

    status = app.main([
        "release", "collect-showcase", "--apex-root", str(ROOT),
        "--path", str(tmp_path / "showcase"),
        "--output", str(tmp_path / "evidence.json"),
    ])

    assert status == 2
    assert json.loads(capsys.readouterr().err)["reason_code"] == (
        "release_showcase_source_invalid"
    )


def test_release_join_requires_fragment(
    tmp_path: Path, capsys
) -> None:
    base = tmp_path / "base.json"
    base.write_text(json.dumps(ReleaseEvidence().to_dict()), encoding="utf-8")
    output = tmp_path / "joined.json"

    assert app.main([
        "release", "join-evidence", "--base", str(base),
        "--output", str(output),
    ]) == 2
    assert json.loads(capsys.readouterr().err)["reason_code"] == (
        "release_evidence_fragment_required"
    )


def test_release_check_uses_explicit_artifact_root_and_stays_blocked(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    qualification = build_qualification_evidence(
        qualification_id="knowledge-ablation",
        apex_tree="b" * 40,
        subject_sha256="a" * 64,
        status="qualified",
        coverage_count=6,
        formal_delivery_count=0,
        details={
            "schema": "apex.knowledge-ablation-qualification/v1",
            "qualification_manifest_sha256": "a" * 64,
            "arms": ["disabled", "static_cards", "static_cards_plus_experience"],
            "task_kinds": ["e2e_kernel_only", "single_kernel"],
            "matched_dimensions": [
                "backend_model", "budget", "cohort", "gpu_identity",
                "measurement_policy", "seed",
            ],
            "report_sha256": "a" * 64,
            "episode_manifest_sha256": "a" * 64,
            "measured_outcomes_only": True,
            "evaluator_owned_experience_updates": True,
        },
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(ReleaseEvidence(
        qualifications=(qualification,),
    ).to_dict()), encoding="utf-8")
    artifact_root = tmp_path / "formal-results"
    artifact_root.mkdir()
    observed = []

    class UnavailableAuthority:
        def verify(self, value):
            raise ContractError(
                "No verifier", "qualification_artifacts_unavailable"
            )

    def build(**kwargs):
        observed.append(kwargs)
        return UnavailableAuthority()

    monkeypatch.setattr(
        "apex.cli.release.build_qualification_artifact_authority", build
    )

    assert app.main([
        "release", "check", "--apex-root", str(ROOT),
        "--evidence", str(evidence),
        "--qualification-artifact-root", str(artifact_root), "--json",
    ]) == 0

    value = json.loads(capsys.readouterr().out)
    assert observed == [{"apex_root": ROOT, "artifact_root": artifact_root}]
    assert "qualification_authority_missing:knowledge-ablation" in value["blockers"]
    assert value["qualification_authorities"] == []


def test_release_collect_qualifications_writes_path_free_report(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    artifact_root = tmp_path / "formal-results"
    artifact_root.mkdir()
    output = tmp_path / "qualification-resolution.json"
    document = {
        "schema": "apex.formal-qualification-artifact-resolution/v1",
        "artifact_index_sha256": None,
        "entries": [],
        "collection_sha256": "a" * 64,
    }
    collection = SimpleNamespace(
        entries=(),
        to_dict=lambda: document,
    )
    monkeypatch.setattr(
        "apex.cli.release.build_qualification_artifact_authority",
        lambda **kwargs: SimpleNamespace(collect=lambda: collection),
    )

    assert app.main([
        "release", "collect-qualifications", "--apex-root", str(ROOT),
        "--artifact-root", str(artifact_root), "--output", str(output),
    ]) == 0

    assert json.loads(output.read_bytes()) == document
    assert f"qualification_artifacts={output} verified=0" in capsys.readouterr().out
