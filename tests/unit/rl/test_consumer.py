from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_bytes
from apex.rl import (
    DatasetExportConfig,
    DatasetExporter,
    EpisodeGraphMaterializer,
    ReferenceDatasetLoader,
)

from .measured_evidence import build_measured_run


def _export(canonical_run, output: Path) -> Path:
    graph = EpisodeGraphMaterializer(
        canonical_run["journal"], canonical_run["artifacts"]
    ).materialize(canonical_run["run_id"])
    DatasetExporter(canonical_run["artifacts"]).export(
        graph,
        output,
        config=DatasetExportConfig(split="train"),
    )
    return output


def test_reference_loader_builds_three_trainer_neutral_views(
    canonical_run, tmp_path: Path
) -> None:
    dataset = ReferenceDatasetLoader().load(_export(canonical_run, tmp_path / "export"))

    terminal = dataset.terminal_episode()
    assert terminal["task_kind"] == "single_kernel"
    assert terminal["scalar_reward"] is None
    assert terminal["trainability"] == "unscored"

    transitions = dataset.attempt_transitions()
    assert [item["attempt_id"] for item in transitions] == ["attempt-1", "attempt-2"]
    assert transitions[0]["attempt_reward"] == 140.0
    assert transitions[0]["advantage_mode"] == "episode_mean"
    assert transitions[0]["advantage"] == 0.0
    assert transitions[0]["performance_trainable"] is True
    assert transitions[1]["attempt_reward"] is None
    assert transitions[1]["advantage"] is None
    assert transitions[1]["performance_trainable"] is False

    supervision = dataset.supervision_view()
    assert len(supervision) == 1
    assert supervision[0]["attempt_id"] == "attempt-1"
    assert [
        event["payload"]["call_id"] for event in supervision[0]["tool_targets"]
    ] == ["tool-1", "tool-1"]
    assert supervision[0]["decision_targets"][0]["payload"]["verdict"] == "keep"


def test_reference_advantage_can_use_explicit_zero_baseline(
    canonical_run, tmp_path: Path
) -> None:
    dataset = ReferenceDatasetLoader().load(_export(canonical_run, tmp_path / "export"))

    transitions = dataset.attempt_transitions(advantage_mode="zero")

    assert transitions[0]["advantage_baseline"] == 0.0
    assert transitions[0]["advantage"] == 140.0
    with pytest.raises(ContractError) as error:
        dataset.attempt_transitions(advantage_mode="gae")
    assert error.value.reason_code == "invalid_advantage_mode"


def test_reference_loader_preserves_unscored_e2e_terminal_state(tmp_path: Path) -> None:
    run = build_measured_run(tmp_path / "run")
    graph = EpisodeGraphMaterializer(run["journal"], run["artifacts"]).materialize(
        run["run_id"]
    )
    output = tmp_path / "export"
    DatasetExporter(run["artifacts"]).export(
        graph,
        output,
        config=DatasetExportConfig(include_sft=False),
    )

    terminal = ReferenceDatasetLoader().load(output).terminal_episode()

    assert terminal["task_kind"] == "e2e_kernel_only"
    assert terminal["trainability"] == "unscored"
    assert terminal["scalar_reward"] is None
    assert terminal["reward_policy_id"] is None
    assert terminal["raw_measurement_receipts"] == []


def test_reference_loader_rejects_digest_tamper(canonical_run, tmp_path: Path) -> None:
    output = _export(canonical_run, tmp_path / "export")
    with (output / "dataset.jsonl").open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(IntegrityError) as error:
        ReferenceDatasetLoader().load(output)

    assert error.value.reason_code == "rl_export_tampered"


def test_reference_loader_rejects_cross_projection_tamper_even_with_new_hash(
    canonical_run, tmp_path: Path
) -> None:
    output = _export(canonical_run, tmp_path / "export")
    dataset_path = output / "dataset.json"
    document = json.loads(dataset_path.read_bytes())
    document["parent_episode"]["terminal_status"] = "tampered"
    dataset_path.write_bytes(canonical_json_bytes(document))
    manifest_path = output / "export_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"]["dataset.json"] = sha256_bytes(dataset_path.read_bytes())
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(IntegrityError) as error:
        ReferenceDatasetLoader().load(output)

    assert error.value.reason_code == "rl_export_tampered"


def test_reference_loader_rejects_extra_file(canonical_run, tmp_path: Path) -> None:
    output = _export(canonical_run, tmp_path / "export")
    (output / "notes.txt").write_text("untracked")

    with pytest.raises(IntegrityError) as error:
        ReferenceDatasetLoader().load(output)

    assert error.value.reason_code == "rl_export_tampered"
