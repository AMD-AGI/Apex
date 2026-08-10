"""Canonical serialization for recoverable kernel-opportunity plans."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from apex.core import IntegrityError

from .kernel_lane import (
    KernelOpportunity,
    KernelOpportunityPlan,
    KernelPlanningCoverage,
)


def plan_dict(plan: KernelOpportunityPlan) -> dict[str, Any]:
    opportunities = []
    for item in plan.opportunities:
        value = asdict(item)
        for name in ("source_path", "source_root", "test_file"):
            value[name] = str(value[name]) if value[name] else None
        opportunities.append(value)
    return {
        "schema": "apex.kernel-opportunity-plan/v1",
        "opportunities": opportunities,
        "measured_order": list(plan.measured_order),
        "recoverable_order": list(plan.recoverable_order),
        "coverage": plan.coverage.to_dict(),
        "correctness_oracle_policy_sha256": plan.correctness_oracle_policy_sha256,
    }


def plan_from_mapping(value: Mapping[str, Any]) -> KernelOpportunityPlan:
    if value.get("schema") != "apex.kernel-opportunity-plan/v1":
        raise IntegrityError("Opportunity plan schema is invalid", "invalid_diagnosis")
    raw_opportunities = value.get("opportunities")
    if not isinstance(raw_opportunities, list):
        raise IntegrityError("Opportunity plan entries are invalid", "invalid_diagnosis")
    opportunities = []
    for raw in raw_opportunities:
        item = dict(_mapping(raw, "opportunity"))
        for name in ("source_path", "source_root", "test_file"):
            item[name] = Path(item[name]) if item.get(name) else None
        opportunities.append(KernelOpportunity(**item))
    return KernelOpportunityPlan(
        opportunities=tuple(opportunities),
        measured_order=_text_tuple(value.get("measured_order"), "measured_order"),
        recoverable_order=_text_tuple(
            value.get("recoverable_order"), "recoverable_order"
        ),
        coverage=KernelPlanningCoverage.from_mapping(
            _mapping(value.get("coverage"), "planning coverage")
        ),
        correctness_oracle_policy_sha256=_optional_text(
            value.get("correctness_oracle_policy_sha256")
        ),
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IntegrityError(f"{label} is invalid", "invalid_diagnosis")
    return value


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise IntegrityError(f"{label} is invalid", "invalid_diagnosis")
    return tuple(value)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise IntegrityError("Optional plan digest is invalid", "invalid_diagnosis")
    return value


__all__ = ["plan_dict", "plan_from_mapping"]
