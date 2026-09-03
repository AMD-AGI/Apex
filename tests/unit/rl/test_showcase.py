from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from apex.core import (
    ContractError,
    IntegrityError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
)
from apex.reporting import ShowcaseExporter, verify_showcase
from apex.rl import EpisodeArtifact, EpisodeGraphMaterializer
from apex.runtime import build_showcase_evidence

from .conftest import append_event, artifact_binding


def _graph(canonical_run):
    return EpisodeGraphMaterializer(
        canonical_run["journal"], canonical_run["artifacts"]
    ).materialize(canonical_run["run_id"])


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_pending_showcase_export_is_byte_stable_and_verifiable(
    canonical_run, tmp_path: Path
) -> None:
    graph = _graph(canonical_run)
    exporter = ShowcaseExporter(canonical_run["artifacts"])

    first = exporter.export(
        graph, tmp_path / "first", showcase_id="kernel-pending-example"
    )
    second = exporter.export(
        graph, tmp_path / "second", showcase_id="kernel-pending-example"
    )

    assert first.status == "pending"
    assert "terminal_reward_missing" in first.blockers
    assert "winner_bundle_missing" in first.blockers
    assert _tree_bytes(first.output_dir) == _tree_bytes(second.output_dir)
    verified = verify_showcase(first.output_dir)
    assert verified.showcase_id == "kernel-pending-example"
    assert verified.status == "pending"
    assert verified.checksums_sha256 == first.checksums_sha256
    assert verified.event_count == graph.high_water_mark
    assert verified.artifact_count > 0
    assert verified.reward_replayed is True
    assert verified.reproduction_verified is True
    assert verified.verification_receipt_sha256 == sha256_json(
        verified.receipt_payload()
    )
    release_evidence = build_showcase_evidence(
        apex_tree="a" * 40,
        verifier_receipt=verified.to_receipt(),
    )
    assert release_evidence.episode_sha256 == verified.episode_sha256
    assert release_evidence.verification_receipt_sha256 == (
        verified.verification_receipt_sha256
    )
    assert (first.output_dir / "trajectory" / "episode.json").is_file()
    assert (first.output_dir / "trajectory" / "artifact_manifest.json").is_file()
    required = {
        "README.md",
        "template/raw_config_snapshot.json",
        "winner/winner.diff",
        "receipts/dependencies.json",
        "receipts/sources.json",
        "receipts/images.json",
        "receipts/gpu.json",
    }
    assert required <= set(_tree_bytes(first.output_dir))
    assert "config_snapshot_missing" in first.blockers
    assert "gpu_receipts_missing" not in first.blockers
    assert b"Status: `pending`" in (first.output_dir / "README.md").read_bytes()


def test_showcase_required_projection_tamper_fails_after_checksum_rewrite(
    canonical_run, tmp_path: Path
) -> None:
    root = tmp_path / "showcase"
    ShowcaseExporter(canonical_run["artifacts"]).export(
        _graph(canonical_run), root, showcase_id="inventory-tamper"
    )
    projection = root / "receipts" / "gpu.json"
    document = json.loads(projection.read_bytes())
    document["status"] = "missing"
    projection.write_bytes(canonical_json_bytes(document))
    checksums_path = root / "checksums.json"
    checksums = json.loads(checksums_path.read_bytes())
    content = projection.read_bytes()
    checksums["files"]["receipts/gpu.json"] = {
        "sha256": sha256_bytes(content),
        "size": len(content),
    }
    checksums_path.write_bytes(canonical_json_bytes(checksums))

    with pytest.raises(IntegrityError) as error:
        verify_showcase(root)

    assert error.value.reason_code == "showcase_trajectory_mismatch"


def test_repeat_export_to_same_clean_tree_is_identical(
    canonical_run, tmp_path: Path
) -> None:
    graph = _graph(canonical_run)
    exporter = ShowcaseExporter(canonical_run["artifacts"])
    root = tmp_path / "showcase"

    exporter.export(graph, root, showcase_id="repeatable-showcase")
    before = _tree_bytes(root)
    exporter.export(graph, root, showcase_id="repeatable-showcase")

    assert _tree_bytes(root) == before


def test_showcase_tamper_and_extra_file_fail_verification(
    canonical_run, tmp_path: Path
) -> None:
    root = tmp_path / "showcase"
    ShowcaseExporter(canonical_run["artifacts"]).export(
        _graph(canonical_run), root, showcase_id="tamper-showcase"
    )
    reward = root / "reward.json"
    reward.write_bytes(reward.read_bytes() + b"\n")

    with pytest.raises(IntegrityError) as changed:
        verify_showcase(root)
    assert changed.value.reason_code == "showcase_checksum_mismatch"

    reward.write_bytes(reward.read_bytes().rstrip() + b"\n")
    (root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(IntegrityError) as extra:
        verify_showcase(root)
    assert extra.value.reason_code == "showcase_file_inventory_mismatch"


def test_showcase_extra_file_fails_even_when_added_to_checksums(
    canonical_run, tmp_path: Path
) -> None:
    root = tmp_path / "showcase"
    ShowcaseExporter(canonical_run["artifacts"]).export(
        _graph(canonical_run), root, showcase_id="strict-inventory"
    )
    extra = root / "untrusted.txt"
    extra.write_text("untrusted", encoding="utf-8")
    checksums_path = root / "checksums.json"
    checksums = json.loads(checksums_path.read_bytes())
    content = extra.read_bytes()
    checksums["files"]["untrusted.txt"] = {
        "sha256": sha256_bytes(content),
        "size": len(content),
    }
    checksums_path.write_bytes(canonical_json_bytes(checksums))

    with pytest.raises(IntegrityError) as error:
        verify_showcase(root)

    assert error.value.reason_code == "showcase_file_inventory_mismatch"


def test_showcase_cannot_promote_pending_status_by_rehashing_files(
    canonical_run, tmp_path: Path
) -> None:
    root = tmp_path / "showcase"
    ShowcaseExporter(canonical_run["artifacts"]).export(
        _graph(canonical_run), root, showcase_id="status-forgery"
    )
    changed: list[tuple[str, Path]] = []
    for relative in ("showcase.json", "result.json"):
        path = root / relative
        document = json.loads(path.read_bytes())
        if relative == "showcase.json":
            document["status"] = "published"
        else:
            document["showcase_status"] = "published"
        document["qualification_blockers"] = []
        path.write_bytes(canonical_json_bytes(document))
        changed.append((relative, path))
    checksums_path = root / "checksums.json"
    checksums = json.loads(checksums_path.read_bytes())
    for relative, path in changed:
        content = path.read_bytes()
        checksums["files"][relative] = {
            "sha256": sha256_bytes(content),
            "size": len(content),
        }
    checksums_path.write_bytes(canonical_json_bytes(checksums))

    with pytest.raises(IntegrityError) as raised:
        verify_showcase(root)
    assert raised.value.reason_code == "showcase_trajectory_mismatch"


def test_showcase_rejects_secret_in_materialized_text_artifact(
    canonical_run, tmp_path: Path
) -> None:
    secret = canonical_run["artifacts"].put_bytes(
        b"authorization=Bearer abcdefghijklmnopqrstuvwxyz",
        media_type="text/plain",
    )
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "tool_result",
        {
            "attempt_id": "attempt-1",
            "artifacts": [artifact_binding("tool_result", secret)],
        },
        "showcase-secret-artifact",
    )

    with pytest.raises(ContractError) as raised:
        ShowcaseExporter(canonical_run["artifacts"]).export(
            _graph(canonical_run), tmp_path / "secret", showcase_id="secret-showcase"
        )

    assert raised.value.reason_code == "showcase_secret_detected"
    assert not (tmp_path / "secret").exists()


def test_showcase_rejects_structured_credential_field(
    canonical_run, tmp_path: Path
) -> None:
    secret = canonical_run["artifacts"].put_bytes(
        b'{"password":"not-a-real-password"}',
        media_type="application/json",
    )
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "tool_result",
        {
            "attempt_id": "attempt-1",
            "artifacts": [artifact_binding("benchmark_config", secret)],
        },
        "showcase-structured-secret",
    )

    with pytest.raises(ContractError) as error:
        ShowcaseExporter(canonical_run["artifacts"]).export(
            _graph(canonical_run), tmp_path / "secret", showcase_id="secret-config"
        )

    assert error.value.reason_code == "showcase_secret_detected"
    assert not (tmp_path / "secret").exists()


def test_showcase_redacts_host_path_artifact_but_preserves_original_identity(
    canonical_run, tmp_path: Path
) -> None:
    private = canonical_run["artifacts"].put_bytes(
        b'{"workspace":"/home/alice/private/kernel.py"}',
        media_type="application/json",
    )
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "tool_result",
        {
            "attempt_id": "attempt-1",
            "artifacts": [artifact_binding("tool_result", private)],
        },
        "showcase-private-path-artifact",
    )

    result = ShowcaseExporter(canonical_run["artifacts"]).export(
        _graph(canonical_run), tmp_path / "redacted", showcase_id="redacted-showcase"
    )
    manifest = json.loads(
        (result.output_dir / "trajectory" / "artifact_manifest.json").read_bytes()
    )
    entry = next(item for item in manifest["artifacts"] if item["digest"] == private.digest)
    exported = result.output_dir / entry["portable_path"]

    assert entry["redaction_policy_id"] == "host_path_redaction_v1"
    assert entry["export_sha256"] != private.digest
    assert b"/home/alice" not in exported.read_bytes()
    assert b"[REDACTED_PATH]" in exported.read_bytes()
    assert verify_showcase(result.output_dir).reward_replayed is True


def test_showcase_rejects_private_or_heldout_child(canonical_run, tmp_path: Path) -> None:
    graph = _graph(canonical_run)
    private_child = replace(graph.children[0], visibility="private")
    private_graph = replace(graph, children=(private_child, *graph.children[1:]))

    with pytest.raises(ContractError) as raised:
        ShowcaseExporter(canonical_run["artifacts"]).export(
            private_graph, tmp_path / "private", showcase_id="private-showcase"
        )

    assert raised.value.reason_code == "showcase_private_evidence"


def test_fabricated_parent_reward_cannot_be_marked_published(
    canonical_run, tmp_path: Path
) -> None:
    graph = _graph(canonical_run)
    source = canonical_run["packet_receipt"]
    parent = replace(
        graph.parent,
        task_reward=140.0,
        reward_vector={"srobust": 1.1},
        reward_policy_id="kernel_robust_v1",
        reward_policy_digest="1" * 64,
        reward_source_receipt=source.digest,
        raw_measurement_receipts=(source.digest,),
        trainability="complete",
    )
    qualified = replace(graph, parent=parent)

    with pytest.raises(IntegrityError) as raised:
        ShowcaseExporter(canonical_run["artifacts"]).export(
            qualified, tmp_path / "published", showcase_id="published-showcase"
        )

    assert raised.value.reason_code == "showcase_reward_replay_mismatch"


def test_fabricated_bundle_roles_are_not_qualification_evidence(
    canonical_run, tmp_path: Path
) -> None:
    graph = _graph(canonical_run)
    bundle = canonical_run["artifacts"].put_bytes(
        b'{"schema":"bundle"}', media_type="application/json"
    )
    verification = canonical_run["artifacts"].put_bytes(
        b'{"verified":true}', media_type="application/json"
    )
    event = graph.children[0].events[0]
    changed_event = replace(
        event,
        artifacts=(
            *event.artifacts,
            EpisodeArtifact("winner_bundle", bundle, event.event_id),
            EpisodeArtifact("bundle_verification", verification, event.event_id),
        ),
    )
    child = replace(
        graph.children[0], events=(changed_event, *graph.children[0].events[1:])
    )

    with pytest.raises(IntegrityError) as raised:
        ShowcaseExporter(canonical_run["artifacts"]).export(
            replace(graph, children=(child, *graph.children[1:])),
            tmp_path / "forged-bundle",
            showcase_id="forged-bundle-showcase",
        )

    assert raised.value.reason_code == "invalid_portable_bundle"
