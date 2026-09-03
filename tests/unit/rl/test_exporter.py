from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from apex.core import ContractError, IntegrityError
from apex.intake import RegressionGates
from apex.ports import BenchmarkPass
from apex.rl import DatasetExportConfig, DatasetExporter, EpisodeGraphMaterializer

from .conftest import append_event, artifact_binding
from .measured_evidence import build_measured_run


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


def test_exports_measured_experience_and_knowledge_association(
    canonical_run, tmp_path: Path
) -> None:
    evidence = canonical_run["artifacts"].put_bytes(
        b'{"verdict":"keep"}', media_type="application/json"
    )
    common = {
        "attempt_id": "attempt-1",
        "candidate_id": "candidate-1",
        "split": "train",
        "visibility": "public",
    }
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "experience.measured",
        {
            **common,
            "evidence_class": "measured",
            "dry_run": False,
            "identity": {
                "task_id": "task-1",
                "operator": "attention",
                "gpu_arch": "gfx950",
                "framework": "vllm",
                "versions": {},
                "shape_hash": "1" * 64,
                "source_hash": "2" * 64,
                "harness_hash": "3" * 64,
                "policy_hash": "4" * 64,
            },
            "outcome": "success",
            "strategy_fingerprint": evidence.digest,
            "mechanism": "Reduce repeated loads.",
            "micro_verdict": "passed",
            "e2e_verdict": "keep",
            "evidence_receipts": [evidence.digest],
            "failure_reason": None,
            "retry_condition": None,
            "artifacts": [artifact_binding("experience_evidence", evidence)],
        },
        "attempt-1-experience",
    )
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "knowledge_outcome_linked",
        {
            **common,
            "read_id": "read-attempt-1",
            "card_id": "card-1",
            "outcome": "inconclusive",
            "evidence_receipt": evidence.digest,
            "evidence_class": "derived",
            "artifacts": [artifact_binding("knowledge_outcome_evidence", evidence)],
        },
        "attempt-1-knowledge-outcome",
    )

    DatasetExporter(canonical_run["artifacts"]).export(
        _graph(canonical_run),
        tmp_path / "learning",
        config=DatasetExportConfig(split="train"),
    )
    document = json.loads((tmp_path / "learning/dataset.json").read_text())
    record = next(
        item for item in document["records"] if item["attempt_id"] == "attempt-1"
    )

    assert any(
        event["event_type"] == "experience.measured" for event in record["outcomes"]
    )
    link = next(
        event
        for event in record["observations"]
        if event["event_type"] == "knowledge_outcome_linked"
    )
    assert link["payload"]["outcome"] == "inconclusive"


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
        {
            "attempt_id": "attempt-incomplete",
            "split": "train",
            "visibility": "public",
        },
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


@pytest.mark.parametrize("visibility", ("private", "heldout_private"))
def test_public_export_rejects_private_evidence_for_every_split(
    canonical_run, tmp_path: Path, visibility: str
) -> None:
    graph = _graph(canonical_run)
    private = replace(graph.children[0], visibility=visibility, split="validation")

    with pytest.raises(ContractError) as error:
        DatasetExporter(canonical_run["artifacts"]).export(
            replace(graph, children=(private, *graph.children[1:])),
            tmp_path / visibility,
            config=DatasetExportConfig(split="train", on_incomplete="skip"),
        )

    assert error.value.reason_code == "private_dataset_evidence"
    assert not (tmp_path / visibility).exists()


def test_export_manifest_self_describes_policy_and_host_path_redaction(
    canonical_run, tmp_path: Path
) -> None:
    private_path = canonical_run["artifacts"].put_bytes(
        b'{"lease_path":"/tmp/apex/lease.lock"}',
        media_type="application/json",
    )
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "tool_result",
        {
            "attempt_id": "attempt-1",
            "artifacts": [artifact_binding("tool_result", private_path)],
        },
        "dataset-private-host-path",
    )
    output = tmp_path / "redacted"

    DatasetExporter(canonical_run["artifacts"]).export(_graph(canonical_run), output)

    manifest = json.loads((output / "export_manifest.json").read_bytes())
    dataset = (output / "dataset.json").read_text(encoding="utf-8")
    assert manifest["schema_version"] == 2
    assert manifest["visibility_policy"]["policy_id"] == (
        "public_episode_only_fail_closed_v1"
    )
    assert manifest["redaction_policy"]["policy_id"] == (
        "host_absolute_path_redaction_v1"
    )
    assert manifest["license_policy"]["summary"]
    assert manifest["retention_policy"]["summary"]
    assert manifest["summary"]["visibility_counts"] == {"public": 2}
    assert manifest["summary"]["redacted_artifact_count"] == 1
    assert "/tmp/apex" not in dataset
    assert "[REDACTED_PATH]" in dataset


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


def test_exports_source_free_e2e_reject_without_sft(
    e2e_no_source_run, tmp_path: Path
):
    graph = EpisodeGraphMaterializer(
        e2e_no_source_run["journal"], e2e_no_source_run["artifacts"]
    ).materialize(e2e_no_source_run["run_id"])

    result = DatasetExporter(e2e_no_source_run["artifacts"]).export(
        graph,
        tmp_path / "e2e-no-source",
        config=DatasetExportConfig(policy_id="e2e_throughput_qos_v1"),
    )

    assert result.record_count == 1
    assert result.sft_count == 0
    record = json.loads((tmp_path / "e2e-no-source" / "dataset.jsonl").read_text())
    assert record["candidate_id"] is None
    assert record["reward"]["scalar"] == 0.0
    assert record["artifacts_by_role"]["candidate_manifest"]
    assert "candidate_source" not in record["artifacts_by_role"]
    assert (tmp_path / "e2e-no-source" / "sft.jsonl").read_bytes() == b""


def test_e2e_reward_is_replayed_not_trusted(e2e_no_source_run, tmp_path: Path):
    graph = EpisodeGraphMaterializer(
        e2e_no_source_run["journal"], e2e_no_source_run["artifacts"]
    ).materialize(e2e_no_source_run["run_id"])
    child = replace(graph.children[0], scalar_reward=999.0)
    graph = replace(graph, children=(child,))

    with pytest.raises(IntegrityError) as error:
        DatasetExporter(e2e_no_source_run["artifacts"]).export(
            graph, tmp_path / "bad-e2e-reward"
        )
    assert error.value.reason_code == "reward_replay_mismatch"


def test_measured_e2e_export_replays_raw_cas_evidence(tmp_path: Path):
    run = build_measured_run(tmp_path / "run")
    graph = EpisodeGraphMaterializer(run["journal"], run["artifacts"]).materialize(
        run["run_id"]
    )

    child = graph.children[0]
    assert child.opportunity_id == "opportunity-1"
    assert child.trainability == "complete"

    result = DatasetExporter(run["artifacts"]).export(
        graph,
        tmp_path / "export",
        config=DatasetExportConfig(split="train", include_sft=False),
    )

    assert result.record_count == 1


def test_measured_e2e_export_replays_multi_window_point_and_confidence(
    tmp_path: Path,
) -> None:
    run = build_measured_run(
        tmp_path / "run",
        candidate_throughputs=(103.0, 100.6),
    )
    pair = run["pair"]

    assert pair["schema"] == "apex.e2e-paired-promotion/v1"
    assert pair["window_order"] == ["anchor", "candidate", "candidate", "anchor"]
    assert len(pair["observations"]) == 12
    assert len(pair["measurement"]["windows"]) == 3
    assert pair["verdict"]["keep"] is True
    assert pair["verdict"]["confidence"]["passed"] is True

    graph = EpisodeGraphMaterializer(run["journal"], run["artifacts"]).materialize(
        run["run_id"]
    )
    result = DatasetExporter(run["artifacts"]).export(
        graph,
        tmp_path / "export",
        config=DatasetExportConfig(include_sft=False),
    )

    assert result.record_count == 1


def test_measured_e2e_export_uses_non_default_frozen_gates(tmp_path: Path):
    strict = RegressionGates(ttft_p99_regression_pct=3.0)
    run = build_measured_run(
        tmp_path / "run",
        candidate_ttft_p99_ms=1.04,
        acceptance_gates=strict,
    )
    graph = EpisodeGraphMaterializer(run["journal"], run["artifacts"]).materialize(
        run["run_id"]
    )

    assert graph.children[0].verdict == "revert"
    result = DatasetExporter(run["artifacts"]).export(
        graph,
        tmp_path / "export",
        config=DatasetExportConfig(include_sft=False),
    )

    assert result.record_count == 1


def test_measured_e2e_export_rejects_decision_from_different_gates(tmp_path: Path):
    run = build_measured_run(
        tmp_path / "run",
        candidate_ttft_p99_ms=1.04,
        acceptance_gates=RegressionGates(ttft_p99_regression_pct=3.0),
        decision_gates=RegressionGates(),
    )
    graph = EpisodeGraphMaterializer(run["journal"], run["artifacts"]).materialize(
        run["run_id"]
    )

    with pytest.raises(IntegrityError) as error:
        DatasetExporter(run["artifacts"]).export(graph, tmp_path / "export")

    assert error.value.reason_code == "e2e_measurement_evidence_mismatch"


@pytest.mark.parametrize(
    "overrides",
    (
        {"runtime_image": "sha256:" + "d" * 64},
        {"candidate_config_matches_delivery": False},
        {"tamper": "decision_pair_receipt"},
        {"candidate_lane": BenchmarkPass.DIAGNOSTIC},
        {"decision_candidate_throughput": 102.0},
        {"raw_candidate_accuracy": 0.79},
        {"report_candidate_throughput": 102.0},
        {"objective_hash_matches_request": False},
        {"official_private_fields": True},
        {"attestation_run_id_drift": True},
    ),
)
def test_measured_e2e_export_rejects_unbound_or_fabricated_evidence(
    tmp_path: Path,
    overrides: dict[str, object],
):
    run = build_measured_run(tmp_path / "run", **overrides)
    graph = EpisodeGraphMaterializer(run["journal"], run["artifacts"]).materialize(
        run["run_id"]
    )

    with pytest.raises(IntegrityError):
        DatasetExporter(run["artifacts"]).export(graph, tmp_path / "export")


@pytest.mark.parametrize(
    "tamper",
    (
        "pair_schema",
        "pair_extra_field",
        "selection_policy",
        "selected_comparison",
        "observation",
        "action_id",
        "pair_binding",
        "aggregate_extra_role",
        "leg_order",
        "pair_before_final_leg",
        "missing_leg",
        "duplicate_leg",
        "duplicate_pair",
        "gpu_scope",
        "gpu_inventory",
        "missing_measurement_bracket",
        "reward_pair_missing",
        "legacy_benchmark_receipt",
    ),
)
def test_measured_e2e_export_rejects_malformed_paired_promotion(
    tmp_path: Path,
    tamper: str,
) -> None:
    run = build_measured_run(tmp_path / "run", tamper=tamper)
    graph = EpisodeGraphMaterializer(run["journal"], run["artifacts"]).materialize(
        run["run_id"]
    )

    with pytest.raises(IntegrityError) as error:
        DatasetExporter(run["artifacts"]).export(graph, tmp_path / "export")

    assert error.value.reason_code == "e2e_measurement_evidence_mismatch"


def test_measured_e2e_opportunity_mismatch_fails_materialization(tmp_path: Path):
    run = build_measured_run(
        tmp_path / "run",
        reward_opportunity_id="opportunity-other",
    )

    with pytest.raises(IntegrityError) as error:
        EpisodeGraphMaterializer(run["journal"], run["artifacts"]).materialize(
            run["run_id"]
        )

    assert error.value.reason_code == "opportunity_id_mismatch"


def test_legacy_e2e_raw_measurement_role_is_not_trainable(tmp_path: Path):
    run = build_measured_run(tmp_path / "run", add_legacy_raw_role=True)
    graph = EpisodeGraphMaterializer(run["journal"], run["artifacts"]).materialize(
        run["run_id"]
    )

    child = graph.children[0]
    assert child.trainability == "truncated"
    assert "legacy_raw_measurement_role" in child.validation_reasons


def test_e2e_reward_vector_artifact_must_equal_reward_event(
    e2e_no_source_run, tmp_path: Path
):
    graph = EpisodeGraphMaterializer(
        e2e_no_source_run["journal"], e2e_no_source_run["artifacts"]
    ).materialize(e2e_no_source_run["run_id"])
    child = graph.children[0]
    assert child.reward_vector is not None
    vector = dict(child.reward_vector)
    metrics = dict(vector["metrics"])
    metrics["accuracy_regression_pct"] = 1.0
    vector["metrics"] = metrics
    graph = replace(graph, children=(replace(child, reward_vector=vector),))

    with pytest.raises(IntegrityError) as error:
        DatasetExporter(e2e_no_source_run["artifacts"]).export(
            graph, tmp_path / "bad-e2e-grade"
        )

    assert error.value.reason_code == "reward_replay_mismatch"
