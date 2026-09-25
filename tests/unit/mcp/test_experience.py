from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import ContractError, sha256_json
from apex.mcp import (
    CapabilityRegistry,
    CapabilityScope,
    ExperienceRetrieveHandler,
    experience_retrieve_descriptor,
)
from apex.ports import CapabilityAuthority, CapabilityRequest
from apex.storage import EventJournal


def _identity(source: str = "source") -> dict[str, object]:
    return {
        "task_id": "rms-norm",
        "operator": "rms_norm",
        "gpu_arch": "gfx950",
        "framework": "vllm",
        "versions": {"rocm": "7.2"},
        "shape_hash": sha256_json("shape"),
        "source_hash": sha256_json(source),
        "harness_hash": sha256_json("harness"),
        "policy_hash": sha256_json("policy"),
    }


def _experience_payload() -> dict[str, object]:
    return {
        "evidence_class": "measured",
        "dry_run": False,
        "candidate_id": "candidate-1",
        "identity": _identity(),
        "outcome": "no_gain",
        "strategy_fingerprint": sha256_json("strategy"),
        "mechanism": "Increase vector width.",
        "micro_verdict": "correct_no_gain",
        "e2e_verdict": None,
        "evidence_receipts": [sha256_json("receipt")],
        "failure_reason": "No measured improvement.",
        "retry_condition": "Retry after the source or shape changes.",
    }


def _registry(workspace: Path, results: Path) -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(
        experience_retrieve_descriptor(),
        ExperienceRetrieveHandler(CapabilityScope(workspace, results)),
    )
    return registry


def test_experience_retrieval_reads_verified_journal_with_exact_identity(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    results = tmp_path / "results"
    journal_path = results / "run-1" / "events" / "run.db"
    workspace.mkdir()
    journal = EventJournal(journal_path)
    journal.append(
        run_id="run-1",
        event_type="experience.measured",
        payload=_experience_payload(),
        idempotency_key="experience-1",
    )
    before = (journal_path.stat().st_mtime_ns, journal_path.stat().st_size)
    registry = _registry(workspace, results)
    authority = frozenset({CapabilityAuthority.WORKSPACE_USER})

    result = registry.invoke(
        CapabilityRequest(
            "experience.retrieve",
            {"run_path": "run-1", "run_id": "run-1", "identity": _identity()},
            authority,
        )
    )
    mismatch = registry.invoke(
        CapabilityRequest(
            "experience.retrieve",
            {
                "run_path": "run-1",
                "run_id": "run-1",
                "identity": _identity("different"),
            },
            authority,
        )
    )

    assert result.content["record_count"] == 1
    assert result.content["records"][0]["candidate_id"] == "candidate-1"
    assert result.content["event_journal"] == "run-1/events/run.db"
    assert result.content["evidence_only"] is True
    assert mismatch.content["record_count"] == 0
    assert (journal_path.stat().st_mtime_ns, journal_path.stat().st_size) == before


def test_experience_retrieval_rejects_scope_escape_and_missing_authority(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    results = tmp_path / "results"
    workspace.mkdir()
    results.mkdir()
    registry = _registry(workspace, results)
    arguments = {"run_path": "../escape", "run_id": "run-1", "identity": _identity()}

    with pytest.raises(ContractError) as authority:
        registry.invoke(CapabilityRequest("experience.retrieve", arguments))
    assert authority.value.reason_code == "capability_authority_missing"

    with pytest.raises(ContractError) as path:
        registry.invoke(
            CapabilityRequest(
                "experience.retrieve",
                arguments,
                frozenset({CapabilityAuthority.WORKSPACE_USER}),
            )
        )
    assert path.value.reason_code == "unsafe_capability_path"
