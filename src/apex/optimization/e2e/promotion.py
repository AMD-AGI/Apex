"""Matched current-anchor/candidate promotion windows for E2E search."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from apex.benchmark import NormalizedBenchmarkResult
from apex.core import ContractError, IntegrityError, sha256_file
from apex.evaluation import (
    E2EAcceptancePolicy,
    E2EMeasurement,
    E2EVerdict,
    evaluate_current_anchor,
)
from apex.runtime import GpuLeaseReceipt
from apex.storage import ArtifactReceipt
from apex.ports import BenchmarkPass

from .benchmark_artifacts import BenchmarkEvidenceReceipts
from .benchmarking import E2EBenchmarkSession, measurement_from_result
from .run_record import E2ERunRecord
from .services import CandidateDeployment


_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_ORDER = ("anchor", "candidate", "candidate", "anchor")
_SLOTS = ("ab-anchor", "ab-candidate", "ba-candidate", "ba-anchor")


@dataclass(frozen=True, slots=True)
class PromotionObservation:
    """One immutable benchmark observation inside a matched window."""

    position: int
    side: str
    action_id: str
    measurement: E2EMeasurement
    normalized: ArtifactReceipt
    quality: ArtifactReceipt
    config: ArtifactReceipt
    requested_image: str | None
    resolved_image_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "side": self.side,
            "action_id": self.action_id,
            "measurement": self.measurement.to_dict(),
            "normalized_receipt": self.normalized.digest,
            "quality_receipt": self.quality.digest,
            "config_receipt": self.config.digest,
            "requested_image": self.requested_image,
            "resolved_image_id": self.resolved_image_id,
        }


@dataclass(frozen=True, slots=True)
class MatchedPromotion:
    """Evaluator-owned AB/BA comparison and its canonical CAS identity."""

    pair_id: str
    window_id: str
    attempt_id: str
    candidate_id: str
    opportunity_id: str
    anchor_id: str
    anchor_generation: int
    gpu_lease_digest: str
    gpu_device_scope: str
    anchor_config_sha256: str
    candidate_config_sha256: str
    anchor_image: Mapping[str, str | None]
    candidate_image: Mapping[str, str | None]
    observations: tuple[PromotionObservation, ...]
    comparisons: tuple[E2EVerdict, E2EVerdict]
    selected_comparison: int
    verdict: E2EVerdict
    receipt: ArtifactReceipt

    @property
    def primary_measurement(self) -> E2EMeasurement:
        return self.observations[1 if self.selected_comparison == 0 else 2].measurement

    def document(self) -> dict[str, Any]:
        return {
            "schema": "apex.e2e-matched-promotion/v1",
            "pair_id": self.pair_id,
            "window_id": self.window_id,
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "opportunity_id": self.opportunity_id,
            "anchor_id": self.anchor_id,
            "anchor_generation": self.anchor_generation,
            "gpu_lease_digest": self.gpu_lease_digest,
            "gpu_device_scope": self.gpu_device_scope,
            "order": list(_ORDER),
            "anchor_config_sha256": self.anchor_config_sha256,
            "candidate_config_sha256": self.candidate_config_sha256,
            "anchor_image": dict(self.anchor_image),
            "candidate_image": dict(self.candidate_image),
            "observations": [item.to_dict() for item in self.observations],
            "comparisons": [item.to_dict() for item in self.comparisons],
            "selected_comparison": self.selected_comparison,
            "verdict": self.verdict.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PromotionRunResult:
    promotion: MatchedPromotion | None
    evidence_receipt: ArtifactReceipt
    reason_code: str


@dataclass(frozen=True, slots=True)
class _WindowRequest:
    attempt_id: str
    candidate_id: str
    opportunity_id: str
    anchor_config: Path
    anchor_config_sha256: str
    anchor_image_id: str | None
    deployment: CandidateDeployment
    anchor_id: str
    anchor_generation: int


class MatchedPromotionRunner:
    """Run one counterbalanced matched window under the caller-held GPU lease."""

    def __init__(
        self,
        *,
        session: E2EBenchmarkSession,
        record: E2ERunRecord,
        gpu_lease: GpuLeaseReceipt,
        policy: E2EAcceptancePolicy,
    ) -> None:
        self.session = session
        self.record = record
        self.gpu_lease = gpu_lease
        self.policy = policy

    def run(
        self,
        *,
        attempt_id: str,
        candidate_id: str,
        opportunity_id: str,
        anchor_config: Path,
        anchor_image_id: str | None,
        deployment: CandidateDeployment,
    ) -> PromotionRunResult:
        state = self.record.controller.state
        request = _WindowRequest(
            attempt_id,
            candidate_id,
            opportunity_id,
            anchor_config,
            sha256_file(anchor_config),
            anchor_image_id,
            deployment,
            state.anchor_id,
            state.anchor_generation,
        )
        sequence = state.sequence
        window_id = f"window-{attempt_id}-{sequence}"
        observations: list[PromotionObservation] = []
        for position, side in enumerate(_ORDER):
            observed = self._observe(request, window_id, position, side)
            if isinstance(observed, PromotionRunResult):
                return observed
            observations.append(observed)
        promotion = self._complete(request, window_id, tuple(observations))
        return PromotionRunResult(promotion, promotion.receipt, promotion.verdict.reason_code)

    def _observe(
        self,
        request: _WindowRequest,
        window_id: str,
        position: int,
        side: str,
    ) -> PromotionObservation | PromotionRunResult:
        action_id = f"promotion-{request.attempt_id}-{window_id}-{_SLOTS[position]}"
        config = (
            request.anchor_config
            if side == "anchor"
            else request.deployment.measurement_config
        )
        result, evidence = self.session.action(
            action_id,
            config,
            BenchmarkPass.MEASUREMENT,
            attempt_id=request.attempt_id,
            candidate_id=request.candidate_id,
            opportunity_id=request.opportunity_id,
        )
        expected_image = (
            request.anchor_image_id
            if side == "anchor"
            else request.deployment.deployed_image_id
        )
        expected_config = (
            request.anchor_config_sha256
            if side == "anchor"
            else (
                request.deployment.config_sha256.measurement
                if request.deployment.config_sha256 is not None
                else ""
            )
        )
        runtime = _runtime_identity(
            result, evidence, config, expected_config, expected_image, side
        )
        if not result.succeeded:
            return PromotionRunResult(
                None, evidence.normalized, "candidate_e2e_measurement_failed"
            )
        try:
            measurement = measurement_from_result(
                result,
                self.session.protocol_hash,
                quality_receipt=evidence.quality.digest,
                measurement_receipt=evidence.normalized.digest,
            )
        except ContractError:
            return PromotionRunResult(
                None, evidence.normalized, "candidate_e2e_measurement_failed"
            )
        return PromotionObservation(
            position,
            side,
            action_id,
            measurement,
            evidence.normalized,
            evidence.quality,
            evidence.config,
            runtime[0],
            runtime[1],
        )

    def _complete(
        self,
        request: _WindowRequest,
        window_id: str,
        observations: tuple[PromotionObservation, ...],
    ) -> MatchedPromotion:
        _validate_observation_set(observations)
        comparisons = (
            evaluate_current_anchor(
                observations[0].measurement, observations[1].measurement, self.policy
            ),
            evaluate_current_anchor(
                observations[3].measurement, observations[2].measurement, self.policy
            ),
        )
        selected = _selected_comparison(comparisons)
        anchor_image = _side_image(observations, "anchor")
        candidate_image = _side_image(observations, "candidate")
        pair_id = f"pair-{request.attempt_id}-{self.record.controller.state.sequence}"
        placeholder = ArtifactReceipt("", 0, "application/json", "")
        value = MatchedPromotion(
            pair_id,
            window_id,
            request.attempt_id,
            request.candidate_id,
            request.opportunity_id,
            request.anchor_id,
            request.anchor_generation,
            self.gpu_lease.digest,
            self.gpu_lease.execution_scope,
            observations[0].config.digest,
            observations[1].config.digest,
            anchor_image,
            candidate_image,
            observations,
            comparisons,
            selected,
            comparisons[selected],
            placeholder,
        )
        receipt = self.record.put_json(value.document())
        promotion = _replace_receipt(value, receipt)
        _record_pair(self.record, promotion, self.gpu_lease)
        return promotion


def _runtime_identity(
    result: NormalizedBenchmarkResult,
    evidence: BenchmarkEvidenceReceipts,
    config: Path,
    expected_config: str,
    expected_image: str | None,
    side: str,
) -> tuple[str | None, str | None]:
    runtime = result.serving_runtime
    config_digest = sha256_file(config)
    if evidence.config.digest != config_digest or config_digest != expected_config:
        reason = (
            "candidate_runtime_config_mismatch"
            if side == "candidate"
            else "promotion_config_mismatch"
        )
        raise IntegrityError("Matched config bytes drifted", reason)
    if runtime.required:
        invalid = (
            runtime.input_config_sha256 != config_digest
            or not isinstance(runtime.requested_image, str)
            or not runtime.requested_image
            or not isinstance(runtime.resolved_image_id, str)
            or not _IMAGE_ID.fullmatch(runtime.resolved_image_id)
            or (result.succeeded and (not runtime.passed or runtime.process_succeeded is not True))
            or (
                expected_image is not None
                and (
                    runtime.requested_image != expected_image
                    or runtime.resolved_image_id != expected_image
                )
            )
        )
        if invalid:
            reason = (
                "candidate_runtime_image_mismatch"
                if side == "candidate"
                else "promotion_runtime_mismatch"
            )
            raise IntegrityError(
                "Matched benchmark runtime identity differs", reason
            )
    elif expected_image is not None:
        raise IntegrityError(
            "Expected anchor image lacks runtime proof", "promotion_runtime_mismatch"
        )
    return runtime.requested_image, runtime.resolved_image_id


def _validate_observation_set(
    observations: tuple[PromotionObservation, ...],
) -> None:
    if len(observations) != 4 or tuple(item.side for item in observations) != _ORDER:
        raise IntegrityError("Matched promotion order differs", "promotion_order_mismatch")
    if tuple(item.position for item in observations) != tuple(range(4)):
        raise IntegrityError("Matched promotion positions differ", "promotion_order_mismatch")
    for side in ("anchor", "candidate"):
        values = tuple(item for item in observations if item.side == side)
        if len({item.config.digest for item in values}) != 1:
            raise IntegrityError("Matched configs differ within window", "promotion_config_mismatch")
        if len({(item.requested_image, item.resolved_image_id) for item in values}) != 1:
            raise IntegrityError("Matched images differ within window", "promotion_image_mismatch")


def _selected_comparison(comparisons: tuple[E2EVerdict, E2EVerdict]) -> int:
    failures = tuple(index for index, value in enumerate(comparisons) if not value.keep)
    if failures:
        return failures[0]
    return min(range(2), key=lambda index: comparisons[index].throughput_gain_pct)


def _side_image(
    observations: tuple[PromotionObservation, ...], side: str
) -> dict[str, str | None]:
    value = next(item for item in observations if item.side == side)
    return {
        "requested_image": value.requested_image,
        "resolved_image_id": value.resolved_image_id,
    }


def _replace_receipt(
    value: MatchedPromotion, receipt: ArtifactReceipt
) -> MatchedPromotion:
    return replace(value, receipt=receipt)


def _record_pair(
    record: E2ERunRecord,
    promotion: MatchedPromotion,
    gpu_lease: GpuLeaseReceipt,
) -> None:
    lease = record.put_json(gpu_lease.to_dict())
    artifacts = [
        _binding("matched_promotion_pair", promotion.receipt),
        _binding("promotion_gpu_lease", lease),
    ]
    for item in promotion.observations:
        prefix = f"promotion_{item.position}_{item.side}"
        artifacts.extend(
            (
                _binding(f"{prefix}_normalized", item.normalized),
                _binding(f"{prefix}_quality", item.quality),
                _binding(f"{prefix}_config", item.config),
            )
        )
    record.controller.record_domain_event(
        "measurement_result",
        {
            "attempt_id": promotion.attempt_id,
            "candidate_id": promotion.candidate_id,
            "opportunity_id": promotion.opportunity_id,
            "anchor_id": promotion.anchor_id,
            "anchor_generation": promotion.anchor_generation,
            "measurement_kind": "matched_promotion_ab_ba",
            "pair_id": promotion.pair_id,
            "window_id": promotion.window_id,
            "gpu_lease_digest": promotion.gpu_lease_digest,
            "order": list(_ORDER),
            "verdict": promotion.verdict.to_dict(),
            "artifacts": artifacts,
        },
        idempotency_key=f"attempt.{promotion.attempt_id}.promotion_pair",
    )


def _binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


__all__ = [
    "MatchedPromotion",
    "MatchedPromotionRunner",
    "PromotionObservation",
    "PromotionRunResult",
]
