from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from apex.benchmark import (
    InferenceXRuntimeEvidence,
    LatencyDistribution,
    LatencyMetrics,
    ModelRevisionEvidence,
    NormalizedBenchmarkResult,
    QualityEvidence,
    QualityMetric,
    ServingRuntimeEvidence,
    ThroughputMetrics,
)
from apex.core import ValidationLevel, canonical_json_bytes, sha256_file, sha256_json
from apex.evaluation import (
    E2EAcceptancePolicy,
    E2ERewardPolicy,
    e2e_comparison_selection_policy,
    evaluate_current_anchor,
    grade_e2e_outcome,
    select_conservative_e2e_verdict,
)
from apex.intake import RegressionGates
from apex.optimization.e2e.benchmark_artifacts import persist_benchmark_evidence
from apex.optimization.e2e.benchmarking import measurement_from_result
from apex.optimization.e2e.services import CandidateDeployment, DeploymentConfigDigests
from apex.ports import BenchmarkPass
from apex.runtime import (
    GpuDeviceIdentity,
    GpuLeaseReceipt,
    GpuOwnershipReceipt,
    GpuSelectorRequest,
    HsaGpuIdentity,
    HsaInventoryEvidence,
    RsmiDeviceIdentity,
)
from apex.storage import ArtifactStore, EventJournal

from .conftest import (
    append_event,
    append_event_transaction,
    artifact_binding,
    make_packet,
)


_BASE_IMAGE = "sha256:" + "b" * 64
_CANDIDATE_IMAGE = "sha256:" + "c" * 64
_PROTOCOL = "a" * 64


def build_measured_run(
    root: Path,
    *,
    runtime_image: str = _CANDIDATE_IMAGE,
    candidate_config_matches_delivery: bool = True,
    candidate_lane: BenchmarkPass = BenchmarkPass.MEASUREMENT,
    decision_candidate_throughput: float | None = None,
    raw_candidate_accuracy: float = 0.81,
    report_candidate_throughput: float | None = None,
    candidate_ttft_p99_ms: float = 1.0,
    candidate_throughputs: tuple[float, float] = (101.0, 101.0),
    acceptance_gates: RegressionGates = RegressionGates(),
    decision_gates: RegressionGates | None = None,
    objective_hash_matches_request: bool = True,
    reward_opportunity_id: str = "opportunity-1",
    add_legacy_raw_role: bool = False,
    tamper: str | None = None,
) -> dict[str, object]:
    run_id = "run-e2e-measured"
    attempt_id = "attempt-measured"
    candidate_id = "candidate-measured"
    journal = EventJournal(root / "events" / "run.db")
    artifacts = ArtifactStore(root / "artifacts")
    baseline = _benchmark(
        root,
        artifacts,
        "baseline",
        100.0,
        0.8,
        _BASE_IMAGE,
    )
    legs = (
        _benchmark(root, artifacts, "ab-anchor", 100.0, 0.8, _BASE_IMAGE),
        _benchmark(
            root,
            artifacts,
            "ab-candidate",
            candidate_throughputs[0],
            0.81,
            runtime_image,
            lane=candidate_lane,
            raw_accuracy=raw_candidate_accuracy,
            report_throughput=report_candidate_throughput,
            ttft_p99_ms=candidate_ttft_p99_ms,
        ),
        _benchmark(
            root,
            artifacts,
            "ba-candidate",
            candidate_throughputs[1],
            0.81,
            runtime_image,
            ttft_p99_ms=candidate_ttft_p99_ms,
        ),
        _benchmark(root, artifacts, "ba-anchor", 100.0, 0.8, _BASE_IMAGE),
    )
    packet = make_packet(run_id)
    packet_receipt = artifacts.put_bytes(packet.canonical_bytes, media_type="application/json")
    source = artifacts.put_bytes(b"def kernel(x): return x\n", media_type="text/x-python")
    manifest = artifacts.put_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "attempt_id": attempt_id,
                "candidate_id": candidate_id,
                "succeeded": True,
                "reason_code": "candidate_frozen",
                "source_receipts": [source.to_dict()],
            }
        ),
        media_type="application/json",
    )
    micro = artifacts.put_bytes(b'{"qualified":true}', media_type="application/json")
    safety = artifacts.put_bytes(b'{"qualified":true}', media_type="application/json")
    delivery, delivery_bindings = _delivery(
        root,
        artifacts,
        candidate_id,
        candidate_config=(
            legs[1]["config_path"]
            if candidate_config_matches_delivery
            else baseline["config_path"]
        ),
    )
    pair, pair_receipt, lease, comparisons, selected = _matched_pair(
        artifacts,
        run_id=run_id,
        attempt_id=attempt_id,
        candidate_id=candidate_id,
        legs=legs,
        acceptance_gates=decision_gates or acceptance_gates,
        decision_candidate_throughput=decision_candidate_throughput,
        tamper=tamper,
    )
    verdict = comparisons[selected]
    decision_name = "keep" if verdict.keep else "revert"
    grade = grade_e2e_outcome(
        verdict=decision_name,
        reason_code=verdict.reason_code,
        candidate_present=True,
        measurement_verdict=verdict,
    )
    decision_pair_receipt = (
        artifacts.put_bytes(b"{}", media_type="application/json")
        if tamper == "decision_pair_receipt"
        else pair_receipt
    )
    decision_doc = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "opportunity_id": "opportunity-1",
        "candidate_manifest_receipt": manifest.digest,
        "verdict": decision_name,
        "reason": verdict.reason_code,
        "measurement_verdict": verdict.to_dict(),
        "micro_receipt": micro.digest,
        "safety_receipt": safety.digest,
        "delivery_receipt": delivery.digest,
        "promotion_pair_receipt": decision_pair_receipt.digest,
    }
    if tamper == "legacy_benchmark_receipt":
        decision_doc["benchmark_receipt"] = legs[1]["evidence"].normalized.digest
    decision = artifacts.put_bytes(
        canonical_json_bytes(decision_doc), media_type="application/json"
    )
    grade_receipt = artifacts.put_bytes(
        canonical_json_bytes(grade.to_dict()), media_type="application/json"
    )
    policy = artifacts.put_bytes(
        canonical_json_bytes(E2ERewardPolicy().to_dict()),
        media_type="application/json",
    )
    goal = {
        "primary": "throughput",
        "direction": "maximize",
        "gates": asdict(acceptance_gates),
    }
    run_request = artifacts.put_bytes(
        canonical_json_bytes(
            {
                "schema": "apex.e2e-run-request/v1",
                "run_id": run_id,
                "spec": {"goal": goal},
            }
        ),
        media_type="application/json",
    )
    _append_history(
        journal,
        run_id,
        attempt_id,
        candidate_id,
        packet,
        packet_receipt,
        run_request,
        goal,
        objective_hash_matches_request,
        source,
        manifest,
        micro,
        safety,
        delivery,
        delivery_bindings,
        baseline,
        legs,
        pair,
        pair_receipt,
        lease,
        tamper,
    )
    _append_outcome(
        journal,
        run_id,
        attempt_id,
        candidate_id,
        decision_name,
        verdict.reason_code,
        reward_opportunity_id,
        decision,
        grade,
        grade_receipt,
        policy,
        manifest,
        source,
        micro,
        safety,
        delivery,
        pair_receipt,
        tamper,
        add_legacy_raw_role,
    )
    append_event(
        journal,
        run_id,
        "run_finished",
        {"workload_id": "workload-1", "status": "succeeded"},
        "run-finished",
    )
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "journal": journal,
        "artifacts": artifacts,
        "decision": decision,
        "pair": pair,
        "pair_receipt": pair_receipt,
        "legs": legs,
        "selected_comparison": selected,
    }


def _matched_pair(
    artifacts: ArtifactStore,
    *,
    run_id: str,
    attempt_id: str,
    candidate_id: str,
    legs: Sequence[dict[str, Any]],
    acceptance_gates: RegressionGates,
    decision_candidate_throughput: float | None,
    tamper: str | None,
):
    measurements = tuple(_measurement(item) for item in legs)
    decision_candidates = (measurements[1], measurements[2])
    if decision_candidate_throughput is not None:
        decision_candidates = tuple(
            measurement_from_result(
                _result_for_decision(legs[position]["result"], decision_candidate_throughput),
                _PROTOCOL,
                quality_receipt=legs[position]["evidence"].quality.digest,
                measurement_receipt=legs[position]["evidence"].normalized.digest,
            )
            for position in (1, 2)
        )
    acceptance = E2EAcceptancePolicy(acceptance_gates)
    comparisons = (
        evaluate_current_anchor(measurements[0], decision_candidates[0], acceptance),
        evaluate_current_anchor(measurements[3], decision_candidates[1], acceptance),
    )
    selected = select_conservative_e2e_verdict(comparisons)
    if tamper == "selected_comparison":
        selected = 1 - selected
    lease_document = _gpu_lease_document(run_id)
    if tamper == "gpu_inventory":
        lease_document["ownership"]["device_inventory"][0]["unique_id"] = (
            "GPU-ffffffffffffffff"
        )
    lease = artifacts.put_bytes(
        canonical_json_bytes(lease_document), media_type="application/json"
    )
    window_id = "window-attempt-measured-10"
    observations = [
        _promotion_observation(position, side, window_id, legs[position], measurements[position])
        for position, side in enumerate(("anchor", "candidate", "candidate", "anchor"))
    ]
    if tamper == "observation":
        observations[1] = dict(observations[1])
        observations[1]["measurement"] = dict(observations[1]["measurement"])
        observations[1]["measurement"]["throughput"] += 1.0
    if tamper == "action_id":
        observations[2] = dict(observations[2])
        observations[2]["action_id"] += "-tampered"
    pair = {
        "schema": (
            "apex.e2e-matched-promotion/v1"
            if tamper == "pair_schema"
            else "apex.e2e-matched-promotion/v2"
        ),
        "pair_id": "pair-attempt-measured-20",
        "window_id": window_id,
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "opportunity_id": "opportunity-1",
        "anchor_id": "anchor-0",
        "anchor_generation": 0,
        "gpu_lease_digest": lease.digest,
        "gpu_device_scope": (
            "amd-gpu-set=GPU-ffffffffffffffff"
            if tamper == "gpu_scope"
            else lease_document["execution_scope"]
        ),
        "order": ["anchor", "candidate", "candidate", "anchor"],
        "anchor_config_sha256": legs[0]["evidence"].config.digest,
        "candidate_config_sha256": legs[1]["evidence"].config.digest,
        "anchor_image": _bundle_image(legs[0]),
        "candidate_image": _bundle_image(legs[1]),
        "observations": observations,
        "comparisons": [item.to_dict() for item in comparisons],
        "selection_policy": (
            {**e2e_comparison_selection_policy(), "policy_id": "tampered"}
            if tamper == "selection_policy"
            else e2e_comparison_selection_policy()
        ),
        "selected_comparison": selected,
        "verdict": comparisons[selected].to_dict(),
    }
    if tamper == "pair_extra_field":
        pair["unexpected"] = True
    receipt = artifacts.put_bytes(
        canonical_json_bytes(pair), media_type="application/json"
    )
    return pair, receipt, lease, comparisons, selected


def _promotion_observation(position, side, window_id, bundle, measurement):
    evidence = bundle["evidence"]
    return {
        "position": position,
        "side": side,
        "action_id": _promotion_action(window_id, position),
        "measurement": measurement.to_dict(),
        "normalized_receipt": evidence.normalized.digest,
        "quality_receipt": evidence.quality.digest,
        "config_receipt": evidence.config.digest,
        **_bundle_image(bundle),
    }


def _bundle_image(bundle):
    runtime = bundle["result"].serving_runtime
    return {
        "requested_image": runtime.requested_image,
        "resolved_image_id": runtime.resolved_image_id,
    }


def _promotion_action(window_id: str, position: int) -> str:
    slots = ("ab-anchor", "ab-candidate", "ba-candidate", "ba-anchor")
    return f"promotion-attempt-measured-{window_id}-{slots[position]}"


def _gpu_lease_document(run_id: str) -> dict[str, Any]:
    unique_id = "GPU-0123456789abcdef"
    hsa_device = HsaGpuIdentity(0, 2, 2, 1, 0, unique_id)
    hsa = HsaInventoryEvidence(
        1,
        "clean_unfiltered_hsa_gpu_inventory_v1",
        "/opt/apex/hsa-helper.py",
        "1" * 64,
        "/opt/rocm/lib/libhsa-runtime64.so",
        "2" * 64,
        (hsa_device,),
    )
    monitor = RsmiDeviceIdentity(0, 2, 1, unique_id, 128)
    device = GpuDeviceIdentity(0, 2, 0, unique_id, "/dev/dri/renderD128")
    selector = GpuSelectorRequest(requested=(unique_id,))
    ownership = GpuOwnershipReceipt(
        2,
        "clean_hsa_kfd_rsmi_process_gpu_map_v2",
        selector,
        1,
        "/opt/rocm/lib/librocm_smi64.so",
        "3" * 64,
        "/sys/class/kfd/kfd/topology/nodes",
        hsa,
        (monitor,),
        (device,),
        (device,),
        (),
        (),
    )
    lease = GpuLeaseReceipt(
        2,
        run_id,
        ownership.execution_scope,
        ownership.physical_scope,
        1234,
        1.0,
        "/tmp/apex-gpu-leases/GPU-0123456789abcdef.lock",
        ownership,
    )
    return lease.to_dict()


def _benchmark(
    root: Path,
    artifacts: ArtifactStore,
    label: str,
    throughput: float,
    accuracy: float,
    image_id: str,
    *,
    lane: BenchmarkPass = BenchmarkPass.MEASUREMENT,
    raw_accuracy: float | None = None,
    report_throughput: float | None = None,
    ttft_p99_ms: float = 1.0,
) -> dict[str, object]:
    workspace = root / label
    workspace.mkdir(parents=True)
    config = workspace / "benchmark.yaml"
    config.write_text(f"benchmark:\n  serving:\n    image: {image_id}\n", encoding="utf-8")
    result_path = workspace / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "results": {
                    "gsm8k": {
                        "exact_match,strict-match": (
                            accuracy if raw_accuracy is None else raw_accuracy
                        )
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    sample = workspace / "samples_gsm8k.jsonl"
    sample.write_text('{"doc_id":0,"target":"42"}\n', encoding="utf-8")
    report = workspace / "benchmark_report.json"
    report.write_text(
        json.dumps(
            _report_document(
                workspace,
                throughput if report_throughput is None else report_throughput,
                image_id,
                sha256_file(config),
                lane,
                ttft_p99_ms,
            )
        ),
        encoding="utf-8",
    )
    metric = QualityMetric("gsm8k", "exact_match,strict-match", accuracy, True)
    ttft = LatencyDistribution(1.0, 1.0, ttft_p99_ms, 0.0)
    tpot = LatencyDistribution(1.0, 1.0, 1.0, 0.0)
    result = NormalizedBenchmarkResult(
        1,
        f"{label}-run",
        lane,
        True,
        "vllm",
        "Qwen/example",
        workspace,
        report,
        ThroughputMetrics(1.0, throughput, throughput, 100, 1.0),
        LatencyMetrics(ttft, tpot, tpot, tpot),
        QualityEvidence(
            True,
            "lm_eval",
            True,
            (metric,),
            (result_path,),
            None,
            (metric,),
            (result_path, sample),
            "d" * 64,
            "e" * 64,
        ),
        lane is BenchmarkPass.DIAGNOSTIC,
        lane.value,
        lane is BenchmarkPass.MEASUREMENT,
        ModelRevisionEvidence(False, True, None, None, None),
        InferenceXRuntimeEvidence(False, True, None, None, None, None, None),
        (report, result_path, sample),
        (),
        0,
        False,
        serving_runtime=ServingRuntimeEvidence(
            True,
            True,
            sha256_file(config),
            image_id,
            image_id,
            f"magpie-benchmark-{label}",
            "f" * 64,
            True,
            None,
            image_id,
            image_id,
            _direct_image_derivation(image_id),
        ),
    )
    evidence = persist_benchmark_evidence(artifacts, result, config)
    return {
        "result": result,
        "evidence": evidence,
        "config_path": config,
    }


def _report_document(
    workspace: Path,
    throughput: float,
    image_id: str,
    config_digest: str,
    lane: BenchmarkPass,
    ttft_p99_ms: float,
) -> dict[str, object]:
    ttft = {"mean_ms": 1.0, "median_ms": 1.0, "p99_ms": ttft_p99_ms}
    tpot = {"mean_ms": 1.0, "median_ms": 1.0, "p99_ms": 1.0}
    return {
        "success": True,
        "errors": [],
        "framework": "vllm",
        "model": "Qwen/example",
        "workspace_dir": str(workspace),
        "profiling_enabled": lane is BenchmarkPass.DIAGNOSTIC,
        "run_kind": lane.value,
        "reward_eligible": lane is BenchmarkPass.MEASUREMENT,
        "throughput": {
            "request_throughput": 1.0,
            "output_throughput": throughput,
            "total_token_throughput": throughput,
            "completed_requests": 100,
            "duration_seconds": 1.0,
        },
        "latency": {"ttft": ttft, "tpot": tpot},
        "serving_runtime_receipt": {
            "schema": "magpie.serving-runtime-receipt/v2",
            "execution_mode": "docker",
            "input_config_sha256": config_digest,
            "input_image": image_id,
            "input_image_id": image_id,
            "requested_image": image_id,
            "resolved_image_id": image_id,
            "image_derivation": _direct_image_derivation(image_id),
            "container_name": f"magpie-benchmark-{workspace.name}",
            "docker_argv_sha256": "f" * 64,
            "process_succeeded": True,
            "verified": True,
            "errors": [],
        },
    }


def _direct_image_derivation(image_id: str) -> dict[str, object]:
    return {
        "kind": "direct",
        "framework": "vllm",
        "runtime_schema": None,
        "base_image": image_id,
        "base_image_id": image_id,
        "base_image_locator": image_id,
        "derived_image": image_id,
        "derived_image_id": image_id,
        "tracelens_source_commit": None,
        "tracelens_source_tree": None,
        "patch_version": None,
        "patch_path": None,
        "patch_sha256": None,
        "dependency_wheel_manifest_sha256": None,
        "validator": "docker-image-id",
        "verified": True,
    }


def _delivery(
    root: Path,
    artifacts: ArtifactStore,
    candidate_id: str,
    *,
    candidate_config: Path,
):
    diagnostic = root / "delivery-diagnostic.yaml"
    replay = root / "delivery-replay.yaml"
    diagnostic.write_text(candidate_config.read_text(encoding="utf-8") + "# diagnostic\n")
    replay.write_text(candidate_config.read_text(encoding="utf-8") + "# replay\n")
    digests = DeploymentConfigDigests.capture(candidate_config, diagnostic, replay)
    deployment = CandidateDeployment(
        candidate_id,
        True,
        "deployed",
        candidate_config,
        diagnostic,
        replay,
        _PROTOCOL,
        "1" * 64,
        _CANDIDATE_IMAGE,
        ValidationLevel.SOURCE_REBUILD_VERIFIED,
        True,
        {
            "derived_image": {"image_id": _CANDIDATE_IMAGE},
            "config_sha256": digests.to_dict(),
        },
        False,
        digests,
    )
    receipt = artifacts.put_bytes(
        canonical_json_bytes(deployment.to_dict()), media_type="application/json"
    )
    bindings = tuple(
        artifact_binding(role, artifacts.put_file(path, media_type="application/yaml"))
        for role, path in (
            ("delivery_measurement_config", candidate_config),
            ("delivery_diagnostic_config", diagnostic),
            ("delivery_replay_config", replay),
        )
    )
    return receipt, bindings


def _measurement(bundle: dict[str, object]):
    evidence = bundle["evidence"]
    return measurement_from_result(
        bundle["result"],
        _PROTOCOL,
        quality_receipt=evidence.quality.digest,
        measurement_receipt=evidence.normalized.digest,
    )


def _result_for_decision(result: NormalizedBenchmarkResult, throughput: float):
    return NormalizedBenchmarkResult(
        result.schema_version,
        result.run_id,
        BenchmarkPass.MEASUREMENT,
        True,
        result.framework,
        result.model,
        result.workspace_path,
        result.report_path,
        ThroughputMetrics(1.0, throughput, throughput, 100, 1.0),
        result.latency,
        result.quality,
        False,
        "measurement",
        True,
        result.model_revision,
        result.inferencex_runtime,
        result.artifacts,
        (),
        0,
        False,
        serving_runtime=result.serving_runtime,
    )


def _append_history(
    journal,
    run_id,
    attempt_id,
    candidate_id,
    packet,
    packet_receipt,
    run_request,
    goal,
    objective_hash_matches_request,
    source,
    manifest,
    micro,
    safety,
    delivery,
    delivery_bindings,
    baseline,
    legs,
    pair,
    pair_receipt,
    lease,
    tamper,
) -> None:
    append_event(journal, run_id, "run_started", {"workload_id": "workload-1"}, "run-started")
    append_event(
        journal,
        run_id,
        "e2e.initialized",
        {
            "workload_id": "workload-1",
            "measurement_protocol_hash": _PROTOCOL,
            "objective_policy_hash": (
                sha256_json(goal)
                if objective_hash_matches_request
                else "9" * 64
            ),
        },
        "e2e-initialized",
    )
    append_event(
        journal,
        run_id,
        "dependency_verified",
        {
            "kind": "resolved_e2e_run_request",
            "artifacts": [artifact_binding("run_request", run_request)],
        },
        "run-request",
    )
    append_event(
        journal,
        run_id,
        "measurement_result",
        _benchmark_payload(baseline, {}),
        "baseline-measurement",
    )
    common = {
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "opportunity_id": "opportunity-1",
        "split": "train",
        "visibility": "public",
    }
    append_event(
        journal,
        run_id,
        "context_packet_created",
        {
            **common,
            "context_packet_id": packet.context_packet_id,
            "artifacts": [artifact_binding("context_packet", packet_receipt)],
        },
        "attempt-context",
    )
    append_event(
        journal,
        run_id,
        "candidate_frozen",
        {
            **common,
            "artifacts": [
                artifact_binding("candidate_manifest", manifest),
                artifact_binding("candidate_source", source),
            ],
        },
        "attempt-candidate",
    )
    append_event(
        journal,
        run_id,
        "tool_result",
        {**common, "artifacts": [artifact_binding("micro_qualification", micro)]},
        "attempt-micro",
    )
    append_event(
        journal,
        run_id,
        "safety_result",
        {**common, "artifacts": [artifact_binding("safety_qualification", safety)]},
        "attempt-safety",
    )
    append_event(
        journal,
        run_id,
        "delivery_result",
        {
            **common,
            "deployed": True,
            "engagement_verified": True,
            "infrastructure_failure": False,
            "config_sha256": _delivery_digests(delivery_bindings),
            "artifacts": [artifact_binding("primary_delivery", delivery), *delivery_bindings],
        },
        "attempt-delivery",
    )
    order = (0, 2, 1, 3) if tamper == "leg_order" else (0, 1, 2, 3)
    if tamper == "pair_before_final_leg":
        for position in order[:3]:
            _append_leg(journal, run_id, common, legs, pair, position)
        _append_pair(journal, run_id, common, legs, pair, pair_receipt, lease, tamper)
        _append_leg(journal, run_id, common, legs, pair, order[3])
        return
    for position in order:
        if tamper == "missing_leg" and position == 2:
            continue
        _append_leg(journal, run_id, common, legs, pair, position)
    _append_pair(journal, run_id, common, legs, pair, pair_receipt, lease, tamper)
    if tamper == "duplicate_pair":
        _append_pair(
            journal,
            run_id,
            common,
            legs,
            pair,
            pair_receipt,
            lease,
            tamper,
            event_key="attempt-promotion-pair-duplicate",
        )
    if tamper == "duplicate_leg":
        _append_leg(
            journal,
            run_id,
            common,
            legs,
            pair,
            1,
            event_key="attempt-measurement-1-duplicate",
        )


def _append_leg(journal, run_id, common, legs, pair, position, *, event_key=None):
    payload = _benchmark_payload(
        legs[position],
        {
            **common,
            "action_id": pair["observations"][position]["action_id"],
            "anchor_generation": 0,
        },
    )
    append_event(
        journal,
        run_id,
        "measurement_result",
        payload,
        event_key or f"attempt-measurement-{position}",
    )


def _append_pair(
    journal,
    run_id,
    common,
    legs,
    pair,
    pair_receipt,
    lease,
    tamper,
    *,
    event_key="attempt-promotion-pair",
):
    artifacts = [
        artifact_binding("matched_promotion_pair", pair_receipt),
        artifact_binding("promotion_gpu_lease", lease),
    ]
    if tamper == "aggregate_extra_role":
        artifacts.append(artifact_binding("unexpected_pair_artifact", lease))
    sides = ("anchor", "candidate", "candidate", "anchor")
    for position, side in enumerate(sides):
        evidence = legs[position]["evidence"]
        normalized = (
            legs[0]["evidence"].normalized
            if tamper == "pair_binding" and position == 1
            else evidence.normalized
        )
        prefix = f"promotion_{position}_{side}"
        artifacts.extend(
            (
                artifact_binding(f"{prefix}_normalized", normalized),
                artifact_binding(f"{prefix}_quality", evidence.quality),
                artifact_binding(f"{prefix}_config", evidence.config),
            )
        )
    append_event(
        journal,
        run_id,
        "measurement_result",
        {
            **common,
            "anchor_id": pair["anchor_id"],
            "anchor_generation": pair["anchor_generation"],
            "measurement_kind": "matched_promotion_ab_ba",
            "pair_id": pair["pair_id"],
            "window_id": pair["window_id"],
            "gpu_lease_digest": pair["gpu_lease_digest"],
            "order": pair["order"],
            "verdict": pair["verdict"],
            "artifacts": artifacts,
        },
        event_key,
    )


def _delivery_digests(bindings):
    by_role = {item["role"]: item["receipt"]["digest"] for item in bindings}
    return {
        "measurement": by_role["delivery_measurement_config"],
        "diagnostic": by_role["delivery_diagnostic_config"],
        "replay": by_role["delivery_replay_config"],
    }


def _benchmark_payload(bundle, lineage):
    result = bundle["result"]
    evidence = bundle["evidence"]
    return {
        **lineage,
        "pass_type": result.pass_type.value,
        "succeeded": result.succeeded,
        "metrics": {
            key: value for key, value in result.metric_mapping().items() if value is not None
        },
        "evidence_class": "diagnostic" if result.profiling_enabled else "measured",
        "run_kind": result.run_kind,
        "reward_eligible": result.reward_eligible,
        "config_sha256": evidence.config.digest,
        "normalized_benchmark_receipt": evidence.normalized.digest,
        "quality_receipt": evidence.quality.digest,
        "artifacts": [dict(item) for item in evidence.bindings],
    }


def _append_outcome(
    journal,
    run_id,
    attempt_id,
    candidate_id,
    verdict,
    reason,
    reward_opportunity_id,
    decision,
    grade,
    grade_receipt,
    policy,
    manifest,
    source,
    micro,
    safety,
    delivery,
    pair_receipt,
    tamper,
    add_legacy_raw_role,
) -> None:
    common = {
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "split": "train",
        "visibility": "public",
    }
    reward_artifacts = [
        artifact_binding("decision_evidence", decision),
        artifact_binding("e2e_grade", grade_receipt),
        artifact_binding("reward_policy", policy),
        artifact_binding("candidate_manifest", manifest),
        artifact_binding("candidate_source", source),
        artifact_binding("micro_qualification", micro),
        artifact_binding("safety_qualification", safety),
        artifact_binding("primary_delivery", delivery),
    ]
    if tamper != "reward_pair_missing":
        reward_artifacts.append(artifact_binding("matched_promotion_pair", pair_receipt))
    if add_legacy_raw_role:
        reward_artifacts.append(artifact_binding("raw_measurement", pair_receipt))
    append_event_transaction(
        journal,
        run_id,
        (
            (
                "e2e.candidate_decided",
                {
                    **common,
                    "opportunity_id": "opportunity-1",
                    "receipt": decision.digest,
                    "verdict": verdict,
                    "reason": reason,
                    "artifacts": [artifact_binding("decision_evidence", decision)],
                },
                "attempt-decision",
            ),
            (
                "reward_committed",
                {
                    **common,
                    "opportunity_id": reward_opportunity_id,
                    "verdict": verdict,
                    "reason_code": reason,
                    "policy_id": grade.policy_id,
                    "policy_digest": grade.policy_digest,
                    "scalar_reward": grade.scalar_reward,
                    "reward_vector": grade.to_dict(),
                    "evidence_class": "derived",
                    "artifacts": reward_artifacts,
                },
                "attempt-reward",
            ),
        ),
    )


__all__ = ["build_measured_run"]
