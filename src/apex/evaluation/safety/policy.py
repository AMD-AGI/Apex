"""Trusted safety policy and its promotion truth table.

Policy is deliberately not part of ``TaskSafetyProfile``.  A task or agent may
describe a candidate boundary, but it cannot disable a required tool, qualify a
tool/runtime pair, or reinterpret incomplete evidence as clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from apex.core import ContractError, canonical_json_bytes, sha256_json

from .profile import CapabilityStatus


POLICY_SCHEMA_VERSION = "apex.safety-verification-policy/v1"


class SafetyRequirement(str, Enum):
    DISABLED = "disabled"
    ADVISORY = "advisory"
    REQUIRED = "required"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    tool: str
    requirement: SafetyRequirement
    qualified: bool = False
    qualification_digest: str | None = None

    def __post_init__(self) -> None:
        tool = self.tool.strip().lower().replace("-", "_")
        if not tool:
            raise ContractError("tool policy requires a tool", "invalid_safety_policy")
        object.__setattr__(self, "tool", tool)
        try:
            requirement = SafetyRequirement(str(self.requirement))
        except ValueError as exc:
            raise ContractError("invalid safety requirement", "invalid_safety_policy") from exc
        object.__setattr__(self, "requirement", requirement)
        if self.qualified and not _is_digest(self.qualification_digest):
            raise ContractError(
                "qualified tool policy requires an exact qualification digest",
                reason_code="invalid_safety_policy",
                details={"tool": tool},
            )
        if not self.qualified and self.qualification_digest is not None:
            raise ContractError(
                "unqualified tool policy cannot carry qualification evidence",
                reason_code="invalid_safety_policy",
                details={"tool": tool},
            )
        if requirement is SafetyRequirement.REQUIRED and not self.qualified:
            raise ContractError(
                "a tool/runtime pair must be qualified before it can be required",
                reason_code="unqualified_required_safety_tool",
                details={"tool": tool},
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "tool": self.tool,
            "requirement": self.requirement.value,
            "qualified": self.qualified,
            "qualification_digest": self.qualification_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ToolPolicy":
        return cls(
            tool=str(value.get("tool", "")),
            requirement=str(value.get("requirement", "")),  # type: ignore[arg-type]
            qualified=value.get("qualified") is True,
            qualification_digest=(
                str(value["qualification_digest"])
                if value.get("qualification_digest") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class VerificationPolicy:
    """Evaluator-owned per-tool requirements; there is no global required flag."""

    rules: tuple[ToolPolicy, ...]
    schema_version: str = POLICY_SCHEMA_VERSION

    @classmethod
    def no_tools(cls) -> "VerificationPolicy":
        """Return the honest default when no qualified safety backend is bound."""

        return cls(rules=())

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_SCHEMA_VERSION:
            raise ContractError("unsupported safety policy schema", "unsupported_safety_schema")
        object.__setattr__(self, "rules", tuple(self.rules))
        tools = tuple(rule.tool for rule in self.rules)
        if len(set(tools)) != len(tools) or tools != tuple(sorted(tools)):
            raise ContractError("safety policy rules must be unique and sorted", "invalid_safety_policy")

    @property
    def fingerprint(self) -> str:
        return sha256_json(self._body())

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def rule_for(self, tool: str) -> ToolPolicy:
        for rule in self.rules:
            if rule.tool == tool:
                return rule
        raise ContractError(
            f"tool {tool!r} is absent from the trusted safety policy",
            reason_code="untrusted_safety_tool",
            details={"tool": tool},
        )

    def _body(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "rules": [rule.to_dict() for rule in self.rules],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._body(), "policy_fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "VerificationPolicy":
        raw_rules = value.get("rules")
        if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, (str, bytes)):
            raise ContractError("safety policy rules must be a sequence", "invalid_safety_policy")
        rules: list[ToolPolicy] = []
        for raw in raw_rules:
            if not isinstance(raw, Mapping):
                raise ContractError("invalid safety policy rule", "invalid_safety_policy")
            rules.append(ToolPolicy.from_dict(raw))
        policy = cls(rules=tuple(rules), schema_version=str(value.get("schema_version", "")))
        claimed = value.get("policy_fingerprint")
        if claimed is not None and str(claimed) != policy.fingerprint:
            raise ContractError("safety policy fingerprint mismatch", "safety_policy_tampered")
        return policy


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    """Gate outcome, separate from correctness and kernel reward."""

    allowed_to_measure: bool
    promotion_eligible: bool
    safety_certified: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    @property
    def reject(self) -> bool:
        return not self.promotion_eligible

    @property
    def allow_ordinary_keep(self) -> bool:
        return self.promotion_eligible

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed_to_measure": self.allowed_to_measure,
            "promotion_eligible": self.promotion_eligible,
            "safety_certified": self.safety_certified,
            "reason_codes": list(self.reason_codes),
        }


def decide_safety(
    evaluations: Sequence[object],
    *,
    policy: VerificationPolicy,
    blocking_errors: Sequence[str] = (),
) -> SafetyDecision:
    """Apply the fail-closed truth table without changing performance reward.

    ``evaluations`` use structural attributes to keep this policy module free of
    a results-module import cycle.  They must expose ``tool``, ``capability``,
    ``execution``, and ``finding`` enum values.
    """

    reasons = list(dict.fromkeys(str(reason) for reason in blocking_errors if reason))
    if reasons:
        return SafetyDecision(False, False, False, tuple(reasons))

    enabled_rules = tuple(rule for rule in policy.rules if rule.requirement is not SafetyRequirement.DISABLED)
    by_tool = {str(getattr(evaluation, "tool")): evaluation for evaluation in evaluations}
    if set(by_tool) != {rule.tool for rule in enabled_rules}:
        return SafetyDecision(False, False, False, ("safety_result_set_mismatch",))

    from .results import ExecutionStatus, FindingStatus  # local import avoids a cycle

    if any(getattr(evaluation, "finding") is FindingStatus.FOUND for evaluation in evaluations):
        return SafetyDecision(False, False, False, ("confirmed_safety_finding",))

    applicable: list[tuple[object, ToolPolicy]] = []
    required_failures: list[str] = []
    advisory_uncertain = False
    for rule in enabled_rules:
        evaluation = by_tool[rule.tool]
        capability = getattr(evaluation, "capability")
        if capability is CapabilityStatus.NOT_APPLICABLE:
            # Not applicable is its own recorded state.  It neither passes nor
            # blocks the required truth table, and never counts as clean.
            continue
        applicable.append((evaluation, rule))
        clean = (
            capability is CapabilityStatus.READY
            and getattr(evaluation, "execution") is ExecutionStatus.COMPLETED
            and getattr(evaluation, "finding") is FindingStatus.CLEAN
        )
        if rule.requirement is SafetyRequirement.REQUIRED and not clean:
            required_failures.append(f"required_safety_incomplete:{rule.tool}")
        elif rule.requirement is SafetyRequirement.ADVISORY and not clean:
            advisory_uncertain = True

    if required_failures:
        return SafetyDecision(False, False, False, tuple(required_failures))

    certified = bool(applicable) and all(
        rule.qualified
        and getattr(evaluation, "capability") is CapabilityStatus.READY
        and getattr(evaluation, "execution") is ExecutionStatus.COMPLETED
        and getattr(evaluation, "finding") is FindingStatus.CLEAN
        for evaluation, rule in applicable
    )
    if certified:
        return SafetyDecision(True, True, True, ("qualified_safety_clean",))
    if advisory_uncertain:
        reasons.append("advisory_safety_uncertain")
    elif not applicable:
        reasons.append("no_applicable_safety_check")
    else:
        reasons.append("safety_clean_not_qualified")
    return SafetyDecision(True, True, False, tuple(dict.fromkeys(reasons)))


def _is_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


__all__ = [
    "POLICY_SCHEMA_VERSION",
    "SafetyDecision",
    "SafetyRequirement",
    "ToolPolicy",
    "VerificationPolicy",
    "decide_safety",
]
