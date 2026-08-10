"""Small immutable result shared by formal evaluator phase services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from apex.storage import ArtifactReceipt


@dataclass(frozen=True, slots=True)
class FormalEvaluatorResult:
    receipt: Mapping[str, object]
    artifacts: tuple[ArtifactReceipt, ...] = ()
    reward_eligible: bool = False


__all__ = ["FormalEvaluatorResult"]
