from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex.cli.app import main
from apex.cli.projections import resolve_run_source
from apex.context import (
    AnchorView,
    ContextBudget,
    ContextContract,
    ContextPacket,
    Hypothesis,
    TargetEvidence,
    freeze_metrics,
)
from apex.core import ContractError
from apex.orchestration import RunController, RunPhase
from apex.storage import ArtifactReceipt, ArtifactStore, EventJournal, SnapshotStore


def _binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


def _packet(run_id: str) -> ContextPacket:
    return ContextPacket(
        run_id=run_id,
        workload_id="task-cli",
        phase="executing",
        cycle=0,
        state_generation=1,
        role_kind="kernel_optimizer",
        role_objective="Improve the frozen kernel.",
        primary_metric="kernel_srobust",
        hard_constraints=("correctness must pass",),
        target=TargetEvidence("kernel-cli", "kernel.py:kernel", "shape-1", ("a" * 64,)),
        hypothesis=Hypothesis("hypothesis-cli", "reduce traffic", "no change"),
        current_anchor=AnchorView("anchor-cli", 0, freeze_metrics({"latency": 1.0})),
        attempts=(),
        dead_ends=(),
        knowledge_cards=(),
        knowledge_selection_receipt=None,
        knowledge_unavailable_reason="disabled_for_test",
        budget=ContextBudget(1024, 512, 2, 60, 30),
        contract=ContextContract(
            ("edit", "test"),
            ("kernel.py",),
            "evaluator decides",
            "stop after one attempt",
        ),
        artifact_refs=(),
    )


@pytest.fixture
def canonical_run(tmp_path: Path) -> tuple[Path, str]:
    run_id = "run-cli-1"
    root = tmp_path / run_id
    journal = EventJournal(root / "events" / "run.db")
    artifacts = ArtifactStore(root / "artifacts")
    controller = RunController.create(
        run_id,
        journal,
        SnapshotStore(root / "state.snapshot.json"),
        initial_anchor_id="anchor-cli",
    )
    packet = _packet(run_id)
    packet_receipt = artifacts.put_bytes(packet.canonical_bytes, media_type="application/json")
    candidate = artifacts.put_bytes(b"def kernel(x):\n    return x\n", media_type="text/x-python")
    common = {
        "attempt_id": "attempt-cli",
        "candidate_id": "candidate-cli",
        "task_id": "task-cli",
        "kernel_id": "kernel-cli",
        "state_generation": 1,
        "anchor_generation": 0,
        "split": "train",
        "visibility": "public",
    }
    controller.record_domain_event(
        "context_packet_created",
        {
            **common,
            "context_packet_id": packet.context_packet_id,
            "artifacts": [_binding("context_packet", packet_receipt)],
        },
        idempotency_key="context",
    )
    controller.record_domain_event(
        "candidate_frozen",
        {**common, "artifacts": [_binding("candidate", candidate)]},
        idempotency_key="candidate",
    )
    controller.record_domain_event(
        "decision",
        {**common, "verdict": "keep"},
        idempotency_key="decision",
    )
    controller.finish(RunPhase.SUCCEEDED, reason="test_complete")
    return root, run_id


def test_report_command_rebuilds_only_disposable_views(
    canonical_run: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, run_id = canonical_run
    output = tmp_path / "report"
    before = len(EventJournal(root / "events" / "run.db").iter_events(run_id))
    evidence_paths = (
        root / "events" / "run.db",
        root / "state.snapshot.json",
        *tuple(path for path in (root / "artifacts").rglob("*") if path.is_file()),
    )
    evidence_mtimes = {path: path.stat().st_mtime_ns for path in evidence_paths}
    assert main(
        ["report", "--run-root", str(root), "--output", str(output), "--json"]
    ) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["status"] == "reported"
    assert envelope["run_id"] == run_id
    assert (output / "report.json").is_file()
    assert (output / "replication_guide.md").is_file()
    assert len(EventJournal(root / "events" / "run.db").iter_events(run_id)) == before
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["workload_state_hash"] is not None
    assert {path: path.stat().st_mtime_ns for path in evidence_paths} == evidence_mtimes


def test_export_rl_command_uses_real_candidate_and_context(
    canonical_run: tuple[Path, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, run_id = canonical_run
    output = tmp_path / "dataset"
    assert main(
        [
            "export-rl",
            "--run-root",
            str(root),
            "--output",
            str(output),
            "--split",
            "train",
            "--json",
        ]
    ) == 0
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["run_id"] == run_id
    assert envelope["record_count"] == 1
    assert envelope["sft_count"] == 1
    sft = json.loads((output / "sft.jsonl").read_text(encoding="utf-8"))
    assert "def kernel" in sft["response"]


def test_projection_source_never_creates_missing_evidence(tmp_path: Path) -> None:
    root = tmp_path / "run-missing"
    root.mkdir()
    with pytest.raises(ContractError) as error:
        resolve_run_source(root, run_id="run-missing")
    assert error.value.reason_code == "projection_journal_missing"
    assert list(root.iterdir()) == []


def test_projection_output_cannot_overlap_canonical_evidence(
    canonical_run: tuple[Path, str], capsys: pytest.CaptureFixture[str]
) -> None:
    root, _ = canonical_run
    assert main(
        [
            "report",
            "--run-root",
            str(root),
            "--output",
            str(root / "artifacts" / "views"),
        ]
    ) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["reason_code"] == "projection_output_overlaps_evidence"


def test_explicit_run_id_conflict_fails_closed(
    canonical_run: tuple[Path, str], tmp_path: Path
) -> None:
    root, run_id = canonical_run
    (root / "result.json").write_text(
        json.dumps({"run_id": run_id}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ContractError) as error:
        resolve_run_source(root, run_id="run-other")
    assert error.value.reason_code == "projection_run_id_conflict"
