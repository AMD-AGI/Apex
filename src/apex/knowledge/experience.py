"""Read-only measured experience projected deterministically from events."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Protocol

from apex.core import ContractError, sha256_json, validate_identifier


_DIGEST = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


class EventLike(Protocol):
    """Structural event boundary used to avoid a second experience writer."""

    sequence: int
    event_id: str
    event_type: str
    payload: Mapping[str, Any]


class ExperienceOutcome(str, Enum):
    """Mutually exclusive attempt outcomes retained by the projection."""

    SUCCESS = "success"
    FAILURE = "failure"
    NO_GAIN = "no_gain"
    REGRESSION = "regression"
    INFRA_ERROR = "infra_error"


class KnowledgeOutcome(str, Enum):
    """Measured relation between a knowledge read and its later outcome."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class ExperienceIdentity:
    """Compatibility identity required before an attempt may be reused."""

    task_id: str
    operator: str
    gpu_arch: str
    framework: str
    versions: tuple[tuple[str, str], ...]
    shape_hash: str
    source_hash: str
    harness_hash: str
    policy_hash: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperienceIdentity":
        versions = value.get("versions", {})
        if not isinstance(versions, Mapping):
            raise ContractError(
                "Experience versions must be a mapping", "invalid_experience_identity"
            )
        try:
            identity = cls(
                task_id=str(value["task_id"]),
                operator=str(value["operator"]).lower(),
                gpu_arch=str(value["gpu_arch"]).lower(),
                framework=str(value["framework"]).lower(),
                versions=tuple(sorted((str(key), str(item)) for key, item in versions.items())),
                shape_hash=_digest(value["shape_hash"]),
                source_hash=_digest(value["source_hash"]),
                harness_hash=_digest(value["harness_hash"]),
                policy_hash=_digest(value["policy_hash"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(
                "Malformed experience identity", "invalid_experience_identity"
            ) from error
        validate_identifier(identity.task_id, field_name="experience.task_id")
        if not all((identity.operator, identity.gpu_arch, identity.framework)):
            raise ContractError("Incomplete experience identity", "invalid_experience_identity")
        return identity

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "operator": self.operator,
            "gpu_arch": self.gpu_arch,
            "framework": self.framework,
            "versions": dict(self.versions),
            "shape_hash": self.shape_hash,
            "source_hash": self.source_hash,
            "harness_hash": self.harness_hash,
            "policy_hash": self.policy_hash,
        }


@dataclass(frozen=True, slots=True)
class ExperienceRecord:
    """A measured positive or negative attempt linked only to receipts."""

    event_sequence: int
    event_id: str
    candidate_id: str
    identity: ExperienceIdentity
    outcome: ExperienceOutcome
    strategy_fingerprint: str
    mechanism: str
    micro_verdict: str
    e2e_verdict: str | None
    evidence_receipts: tuple[str, ...]
    failure_reason: str | None = None
    retry_condition: str | None = None

    @classmethod
    def from_event(cls, event: EventLike) -> "ExperienceRecord":
        value = event.payload
        try:
            record = cls(
                event_sequence=int(event.sequence),
                event_id=str(event.event_id),
                candidate_id=str(value["candidate_id"]),
                identity=ExperienceIdentity.from_mapping(_mapping(value["identity"])),
                outcome=ExperienceOutcome(str(value["outcome"])),
                strategy_fingerprint=_digest(value["strategy_fingerprint"]),
                mechanism=_bounded(value["mechanism"]),
                micro_verdict=str(value["micro_verdict"]),
                e2e_verdict=_optional_bounded(value.get("e2e_verdict")),
                evidence_receipts=tuple(_digest(item) for item in value["evidence_receipts"]),
                failure_reason=_optional_bounded(value.get("failure_reason")),
                retry_condition=_optional_bounded(value.get("retry_condition")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(
                "Malformed measured experience event", "invalid_experience_event"
            ) from error
        if record.event_sequence < 1 or not record.evidence_receipts:
            raise ContractError("Measured experience needs receipts", "missing_experience_evidence")
        validate_identifier(record.candidate_id, field_name="candidate_id")
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_sequence": self.event_sequence,
            "event_id": self.event_id,
            "candidate_id": self.candidate_id,
            "identity": self.identity.to_dict(),
            "outcome": self.outcome.value,
            "strategy_fingerprint": self.strategy_fingerprint,
            "mechanism": self.mechanism,
            "micro_verdict": self.micro_verdict,
            "e2e_verdict": self.e2e_verdict,
            "evidence_receipts": list(self.evidence_receipts),
            "failure_reason": self.failure_reason,
            "retry_condition": self.retry_condition,
        }


@dataclass(frozen=True, slots=True)
class KnowledgeOutcomeLink:
    """Append-only result attribution for one prior knowledge read."""

    event_sequence: int
    read_id: str
    card_id: str
    outcome: KnowledgeOutcome
    evidence_receipt: str

    @classmethod
    def from_event(cls, event: EventLike) -> "KnowledgeOutcomeLink":
        try:
            link = cls(
                event_sequence=int(event.sequence),
                read_id=str(event.payload["read_id"]),
                card_id=str(event.payload["card_id"]),
                outcome=KnowledgeOutcome(str(event.payload["outcome"])),
                evidence_receipt=_digest(event.payload["evidence_receipt"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ContractError(
                "Malformed knowledge outcome event", "invalid_knowledge_outcome"
            ) from error
        validate_identifier(link.read_id, field_name="knowledge.read_id")
        validate_identifier(link.card_id, field_name="knowledge.card_id")
        return link

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_sequence": self.event_sequence,
            "read_id": self.read_id,
            "card_id": self.card_id,
            "outcome": self.outcome.value,
            "evidence_receipt": self.evidence_receipt,
        }


@dataclass(frozen=True, slots=True)
class ExperienceView:
    """Disposable projection rebuilt from committed canonical events."""

    records: tuple[ExperienceRecord, ...]
    knowledge_outcomes: tuple[KnowledgeOutcomeLink, ...]

    @classmethod
    def from_events(cls, events: Iterable[EventLike]) -> "ExperienceView":
        records: list[ExperienceRecord] = []
        outcomes: list[KnowledgeOutcomeLink] = []
        for event in sorted(events, key=lambda item: (item.sequence, item.event_id)):
            if event.event_type == "experience.measured":
                if (
                    event.payload.get("evidence_class") != "measured"
                    or event.payload.get("dry_run")
                ):
                    continue
                records.append(ExperienceRecord.from_event(event))
            elif event.event_type == "knowledge_outcome_linked":
                outcomes.append(KnowledgeOutcomeLink.from_event(event))
        return cls(tuple(records), tuple(outcomes))

    def compatible(
        self, identity: ExperienceIdentity, *, limit: int = 8
    ) -> tuple[ExperienceRecord, ...]:
        if limit < 1:
            raise ContractError("Experience limit must be positive", "invalid_experience_limit")
        matches = [record for record in self.records if record.identity == identity]
        ordered = sorted(matches, key=lambda item: (-item.event_sequence, item.candidate_id))
        return tuple(ordered[:limit])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "records": [record.to_dict() for record in self.records],
            "knowledge_outcomes": [item.to_dict() for item in self.knowledge_outcomes],
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


def _digest(value: object) -> str:
    digest = str(value).removeprefix("sha256:")
    if not _DIGEST.fullmatch(digest):
        raise ValueError("invalid digest")
    return digest


def _bounded(value: object) -> str:
    text = str(value).strip()
    if not text or len(text) > 2_048:
        raise ValueError("invalid bounded text")
    return text


def _optional_bounded(value: object) -> str | None:
    return None if value is None else _bounded(value)


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping")
    return value


__all__ = [
    "ExperienceIdentity",
    "ExperienceOutcome",
    "ExperienceRecord",
    "ExperienceView",
    "KnowledgeOutcome",
    "KnowledgeOutcomeLink",
]
