from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from apex.core import ContractError, IntegrityError
from apex.rl import DatasetExportConfig, DatasetExporter, EpisodeGraphMaterializer

from .conftest import append_event, artifact_binding


def _graph(canonical_run):
    return EpisodeGraphMaterializer(
        canonical_run["journal"], canonical_run["artifacts"]
    ).materialize(canonical_run["run_id"])


def test_exports_deterministic_json_jsonl_and_real_sft(canonical_run, tmp_path: Path):
    graph = _graph(canonical_run)
    exporter = DatasetExporter(canonical_run["artifacts"])
    first = exporter.export(
        graph,
        tmp_path / "one",
        config=DatasetExportConfig(split="train"),
    )
    second = exporter.export(
        graph,
        tmp_path / "two",
        config=DatasetExportConfig(split="train"),
    )

    assert first.record_count == 2
    assert first.sft_count == 1
    assert first.dataset_sha256 == second.dataset_sha256
    assert first.manifest_sha256 == second.manifest_sha256
    assert (tmp_path / "one" / "dataset.jsonl").read_bytes() == (
        tmp_path / "two" / "dataset.jsonl"
    ).read_bytes()
    document = json.loads((tmp_path / "one" / "dataset.json").read_text())
    assert document["parent_episode"]["workload_id"] == "workload-1"
    assert json.loads((tmp_path / "one" / "parent_episode.json").read_text())[
        "episode_id"
    ] == document["parent_episode"]["episode_id"]
    assert len(document["records"]) == 2
    success = next(item for item in document["records"] if item["attempt_id"] == "attempt-1")
    assert success["observation"]["context_packet_id"] == canonical_run[
        "packet"
    ].context_packet_id
    assert success["reward"]["scalar"] == 140.0
    assert success["tools"]
    sft = json.loads((tmp_path / "one" / "sft.jsonl").read_text().strip())
    assert "def optimized" in sft["response"]
    assert "placeholder" not in sft["response"].lower()
    validation = json.loads(
        (tmp_path / "one" / "validation_report.json").read_text()
    )
    assert validation["quality_gates"]["schema_validation_pct"] == 100
    assert validation["quality_gates"]["stdout_transition_recovery_count"] == 0


def test_empty_filter_fails_closed(canonical_run, tmp_path: Path):
    with pytest.raises(ContractError) as error:
        DatasetExporter(canonical_run["artifacts"]).export(
            _graph(canonical_run),
            tmp_path / "empty",
            config=DatasetExportConfig(split="heldout", on_incomplete="skip"),
        )
    assert error.value.reason_code == "empty_dataset_export"


def test_incomplete_episode_fails_or_skips_with_reason(canonical_run, tmp_path: Path):
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "candidate_frozen",
        {"attempt_id": "attempt-incomplete", "split": "train"},
        "incomplete-candidate",
    )
    graph = _graph(canonical_run)
    exporter = DatasetExporter(canonical_run["artifacts"])
    with pytest.raises(ContractError) as error:
        exporter.export(graph, tmp_path / "strict", config=DatasetExportConfig(split="train"))
    assert error.value.reason_code == "episode_export_incomplete"
    result = exporter.export(
        graph,
        tmp_path / "skip",
        config=DatasetExportConfig(split="train", on_incomplete="skip"),
    )
    assert result.record_count == 2
    assert result.skipped[0]["attempt_id"] == "attempt-incomplete"
    assert "context_packet_missing" in result.skipped[0]["reason"]


def test_secret_in_artifact_fails_export(canonical_run, tmp_path: Path):
    secret = canonical_run["artifacts"].put_bytes(
        b"Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
        media_type="text/plain",
    )
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "agent_message",
        {
            "attempt_id": "attempt-1",
            "artifacts": [artifact_binding("agent_message", secret)],
        },
        "attempt-1-secret",
    )
    with pytest.raises(IntegrityError) as error:
        DatasetExporter(canonical_run["artifacts"]).export(
            _graph(canonical_run), tmp_path / "secret"
        )
    assert error.value.reason_code == "dataset_secret_detected"


def test_kernel_reward_is_replayed_not_trusted(canonical_run, tmp_path: Path):
    graph = _graph(canonical_run)
    child = replace(
        graph.children[0],
        scalar_reward=999.0,
        reward_vector={
            "compile": True,
            "correctness": True,
            "integrity": True,
            "anti_tampering": True,
            "safety": {"finding": False},
            "kernel_srobust": 1.1,
            "kernel_robust_reward": 999.0,
        },
    )
    graph = replace(graph, children=(child, *graph.children[1:]))
    with pytest.raises(IntegrityError) as error:
        DatasetExporter(canonical_run["artifacts"]).export(
            graph, tmp_path / "bad-reward"
        )
    assert error.value.reason_code == "reward_replay_mismatch"
