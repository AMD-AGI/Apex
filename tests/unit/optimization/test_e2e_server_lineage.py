from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import ContractError, IntegrityError, canonical_json_bytes, sha256_json
from apex.benchmark.inferencex_runtime import InferenceXRuntimeEvidence
from apex.benchmark.lm_eval_runtime import LmEvalRuntimeEvidence
from apex.benchmark.local_runtime import LocalRuntimeEvidence
from apex.benchmark.model_revision import ModelRevisionEvidence
from apex.benchmark.quality import QualityEvidence
from apex.benchmark.results import (
    LatencyDistribution,
    LatencyMetrics,
    NormalizedBenchmarkResult,
    ThroughputMetrics,
)
from apex.benchmark.serving_runtime import ServingRuntimeEvidence
from apex.ports import BenchmarkPass
from apex.optimization.e2e.benchmark_artifacts import BenchmarkEvidenceReceipts
from apex.optimization.e2e.server_lineage import (
    LINEAGE_SCHEMA,
    LocalServerLineageEvidence,
    capture_local_server_lineage,
    replay_local_server_lineage,
    require_resumable_server_lineage,
)
from apex.storage import ArtifactStore, EventJournal


RUN_ID = "e2e-lineage"
ANCHOR = "anchor-original"
DIGEST = "a" * 64


def test_reuse_and_cleanup_share_server_generation_not_client_config(
    tmp_path: Path,
) -> None:
    journal, store, head = _run(tmp_path)
    reuse, attestation = _evidence(store, mode="reuse", sequence=1)
    head = _measurement(journal, head, reuse, attestation)
    cleanup, cleanup_attestation = _evidence(
        store,
        mode="cleanup",
        sequence=2,
        config_sha256="b" * 64,
    )
    _measurement(journal, head, cleanup, cleanup_attestation)

    projection = replay_local_server_lineage(
        journal.iter_events(RUN_ID, verify=True), store
    )

    assert projection.observation_count == 2
    assert projection.active is None
    assert projection.retired_generations == (
        reuse.document["server_generation_sha256"],
    )
    assert reuse.document["client_config_sha256"] != cleanup.document[
        "client_config_sha256"
    ]
    assert reuse.document["server_source_generation_sha256"] == cleanup.document[
        "server_source_generation_sha256"
    ]
    assert cleanup.document["cleanup_succeeded"] is True


def test_capture_consumes_typed_local_v2_receipt_and_gpu_lease(
    tmp_path: Path,
) -> None:
    journal, store, _ = _run(tmp_path)
    runtime = _runtime("reuse")
    attestation = store.put_bytes(
        canonical_json_bytes(
            {"runtime": {"serving_runtime_receipt": runtime}}
        ),
        media_type="application/json",
    )
    config = store.put_bytes(b"benchmark: {}\n", media_type="application/yaml")
    placeholder = store.put_bytes(b"{}", media_type="application/json")
    evidence = BenchmarkEvidenceReceipts(
        placeholder,
        placeholder,
        config,
        (
            {
                "role": "benchmark_execution_attestation",
                "receipt": attestation.to_dict(),
            },
        ),
        (placeholder, config, attestation),
    )
    captured = capture_local_server_lineage(
        store=store,
        events=journal.iter_events(RUN_ID),
        result=_result(runtime),
        evidence=evidence,
        run_id=RUN_ID,
        action_id="reuse-1",
        owner_kind="anchor",
        owner_id=ANCHOR,
        anchor_id=ANCHOR,
        anchor_generation=0,
    )

    assert captured is not None
    assert captured.document["gpu_lease_digest"] == "5" * 64
    assert captured.document["server_source_generation_sha256"] == "1" * 64
    assert captured.document["server_generation_sha256"] == runtime[
        "lifecycle_receipt"
    ]["server_generation_sha256"]


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("server_source_generation_sha256", "active generation"),
        ("server_identity_sha256", "exact active generation"),
        ("gpu_lease_digest", "exact active generation"),
    ],
)
def test_reuse_or_cleanup_rejects_server_generation_drift(
    tmp_path: Path, field: str, reason: str
) -> None:
    journal, store, head = _run(tmp_path)
    reuse, attestation = _evidence(store, mode="reuse", sequence=1)
    head = _measurement(journal, head, reuse, attestation)
    mode = "reuse" if field == "server_source_generation_sha256" else "cleanup"
    next_value, next_attestation = _evidence(store, mode=mode, sequence=2)
    changed = dict(next_value.document)
    changed[field] = "f" * 64
    changed_receipt = store.put_bytes(
        canonical_json_bytes(changed), media_type="application/json"
    )
    changed_evidence = LocalServerLineageEvidence(changed, changed_receipt)
    _measurement(journal, head, changed_evidence, next_attestation)

    with pytest.raises(IntegrityError, match=reason):
        replay_local_server_lineage(journal.iter_events(RUN_ID), store)


def test_duplicate_lineage_sequence_and_retired_generation_are_rejected(
    tmp_path: Path,
) -> None:
    journal, store, head = _run(tmp_path)
    reuse, attestation = _evidence(store, mode="reuse", sequence=1)
    head = _measurement(journal, head, reuse, attestation)
    duplicate, second_attestation = _evidence(store, mode="reuse", sequence=2)
    changed = dict(duplicate.document)
    changed["lineage_sequence"] = 1
    receipt = store.put_bytes(
        canonical_json_bytes(changed), media_type="application/json"
    )
    _measurement(
        journal,
        head,
        LocalServerLineageEvidence(changed, receipt),
        second_attestation,
    )

    with pytest.raises(IntegrityError, match="duplicated or out of order"):
        replay_local_server_lineage(journal.iter_events(RUN_ID), store)


def test_resume_rejects_open_server_from_previous_gpu_lease(tmp_path: Path) -> None:
    journal, store, head = _run(tmp_path)
    reuse, attestation = _evidence(store, mode="reuse", sequence=1)
    _measurement(journal, head, reuse, attestation)

    with pytest.raises(ContractError, match="interrupted GPU lease"):
        require_resumable_server_lineage(
            journal.iter_events(RUN_ID), store, "f" * 64
        )


def test_replay_rejects_missing_lineage_for_normalized_local_reuse(
    tmp_path: Path,
) -> None:
    journal, store, head = _run(tmp_path)
    normalized = store.put_bytes(
        canonical_json_bytes(
            {
                "local_runtime": {
                    "required": True,
                    "passed": True,
                    "lifecycle": "reuse",
                }
            }
        ),
        media_type="application/json",
    )
    journal.append(
        run_id=RUN_ID,
        event_type="measurement_result",
        payload={
            "action_id": "reuse-missing",
            "artifacts": [
                {
                    "role": "normalized_benchmark",
                    "receipt": normalized.to_dict(),
                }
            ],
        },
        idempotency_key="measurement.reuse-missing",
        parent_event_id=head,
    )

    with pytest.raises(IntegrityError, match="missing server lineage"):
        replay_local_server_lineage(journal.iter_events(RUN_ID), store)


def test_failed_unverified_local_action_does_not_establish_a_generation(
    tmp_path: Path,
) -> None:
    journal, store, head = _run(tmp_path)
    normalized = store.put_bytes(
        canonical_json_bytes(
            {
                "local_runtime": {
                    "required": True,
                    "passed": False,
                    "lifecycle": "reuse",
                }
            }
        ),
        media_type="application/json",
    )
    journal.append(
        run_id=RUN_ID,
        event_type="measurement_result",
        payload={
            "action_id": "reuse-failed",
            "artifacts": [
                {
                    "role": "normalized_benchmark",
                    "receipt": normalized.to_dict(),
                }
            ],
        },
        idempotency_key="measurement.reuse-failed",
        parent_event_id=head,
    )

    projection = replay_local_server_lineage(journal.iter_events(RUN_ID), store)
    assert projection.observation_count == 0
    assert projection.active is None


def _run(tmp_path: Path) -> tuple[EventJournal, ArtifactStore, str]:
    journal = EventJournal(tmp_path / "events.db")
    store = ArtifactStore(tmp_path / "artifacts")
    started = journal.append(
        run_id=RUN_ID,
        event_type="run.started",
        payload={"initial_anchor_id": ANCHOR},
        idempotency_key="run.started",
    )
    return journal, store, started.event_id


def _evidence(
    store: ArtifactStore,
    *,
    mode: str,
    sequence: int,
    config_sha256: str = DIGEST,
) -> tuple[LocalServerLineageEvidence, object]:
    runtime = _runtime(mode)
    attestation = store.put_bytes(
        canonical_json_bytes(
            {"runtime": {"serving_runtime_receipt": runtime}}
        ),
        media_type="application/json",
    )
    identity = sha256_json(runtime["lifecycle_receipt"]["server_state"])
    source = runtime["lifecycle_receipt"]["server_source_generation_sha256"]
    generation = runtime["lifecycle_receipt"]["server_generation_sha256"]
    value = {
        "schema": LINEAGE_SCHEMA,
        "run_id": RUN_ID,
        "action_id": f"{mode}-{sequence}",
        "lineage_sequence": sequence,
        "lifecycle": mode,
        "framework": "vllm",
        "model": "org/model",
        "owner": {
            "kind": "anchor",
            "id": ANCHOR,
            "anchor_id": ANCHOR,
            "anchor_generation": 0,
        },
        "client_config_sha256": config_sha256,
        "server_source_generation_sha256": source,
        "server_generation_sha256": generation,
        "server_identity_sha256": identity,
        "dependency_receipt_sha256": "4" * 64,
        "gpu_lease_digest": "5" * 64,
        "execution_attestation_sha256": attestation.digest,
        "local_runtime_receipt_sha256": sha256_json(runtime),
        "reward_eligible": mode == "reuse",
        "cleanup_verified": mode == "cleanup",
        "cleanup_succeeded": mode == "cleanup",
    }
    receipt = store.put_bytes(
        canonical_json_bytes(value), media_type="application/json"
    )
    return LocalServerLineageEvidence(value, receipt), attestation


def _runtime(mode: str) -> dict[str, object]:
    source = "1" * 64
    process = {"pid": 101, "start_time_ticks": 202}
    server = sha256_json(
        {
            "server_source_generation_sha256": source,
            "server_process": process,
            "compatibility_sha256": "3" * 64,
            "port": 8888,
        }
    )
    return {
        "schema": "apex.magpie-local-runtime-observation/v2",
        "execution_mode": "local",
        "lifecycle": mode,
        "input_config_sha256": DIGEST,
        "dependency_receipt_sha256": "4" * 64,
        "gpu_lease_digest": "5" * 64,
        "inferencex_source": {
            "root": "/source", "commit": "6" * 40, "tree": "7" * 40,
        },
        "benchmark_process": {},
        "runtime_processes": [],
        "lifecycle_receipt": {
            "mode": mode,
            "port": 8888,
            "observed_listener_pids": [101],
            "server_state": {
                "process": process,
                "listener_pids": [101],
                "compatibility_sha256": "3" * 64,
            },
            "quiescence_receipt": (
                {"verified": True} if mode == "cleanup" else None
            ),
            "server_source_generation_sha256": source,
            "server_generation_sha256": server,
        },
        "process_succeeded": True,
        "verified": True,
        "errors": [],
    }


def _result(runtime: dict[str, object]) -> NormalizedBenchmarkResult:
    distribution = LatencyDistribution(1.0, 1.0, 1.0, 0.0)
    lifecycle = runtime["lifecycle_receipt"]
    return NormalizedBenchmarkResult(
        1,
        RUN_ID,
        BenchmarkPass.MEASUREMENT,
        True,
        "vllm",
        "org/model",
        Path("/tmp/workspace"),
        None,
        ThroughputMetrics(1.0, 1.0, 1.0, 1, 1.0),
        LatencyMetrics(distribution, distribution, distribution, distribution),
        QualityEvidence(False, "none", True, (), ()),
        False,
        "measurement",
        True,
        ModelRevisionEvidence(False, True, None, None, None),
        InferenceXRuntimeEvidence(False, True, None, None, None, None, None),
        (),
        (),
        command_exit_code=0,
        timed_out=False,
        lm_eval_runtime=LmEvalRuntimeEvidence(
            False, True, None, None, None, None, None, None
        ),
        serving_runtime=ServingRuntimeEvidence(
            False, True, None, None, None, None, None, None, None
        ),
        local_runtime=LocalRuntimeEvidence(
            True,
            True,
            "reuse",
            Path("/source"),
            "6" * 40,
            "7" * 40,
            101,
            1,
            "5" * 64,
            str(lifecycle["server_source_generation_sha256"]),
            str(lifecycle["server_generation_sha256"]),
            None,
            None,
        ),
    )


def _measurement(
    journal: EventJournal,
    parent: str,
    lineage: LocalServerLineageEvidence,
    attestation: object,
) -> str:
    value = lineage.document
    event = journal.append(
        run_id=RUN_ID,
        event_type="measurement_result",
        payload={
            "action_id": value["action_id"],
            "config_sha256": value["client_config_sha256"],
            "reward_eligible": value["reward_eligible"],
            "server_lineage": lineage.reference,
            "artifacts": [
                lineage.binding,
                {
                    "role": "benchmark_execution_attestation",
                    "receipt": attestation.to_dict(),
                },
            ],
        },
        idempotency_key=f"measurement.{value['action_id']}",
        parent_event_id=parent,
    )
    return event.event_id
