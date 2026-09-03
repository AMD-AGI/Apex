"""Trusted, one-shot authority boundary for chat-started formal campaigns."""

from __future__ import annotations

from threading import Lock
from typing import Protocol

from apex.evaluation import EvaluationAuthorityReceipt, EvaluationContractDraft


class FormalEvaluationAuthorityProvider(Protocol):
    """Consume authority supplied by the trusted local composition boundary."""

    def consume(
        self, *, run_id: str, draft: EvaluationContractDraft
    ) -> EvaluationAuthorityReceipt | None: ...


class OneShotEvaluationAuthorityProvider:
    """Release one pre-issued receipt to one exact run and draft at most once."""

    def __init__(
        self,
        *,
        run_id: str,
        receipt: EvaluationAuthorityReceipt,
    ) -> None:
        self._run_id = run_id
        self._receipt = receipt
        self._consumed = False
        self._lock = Lock()

    def consume(
        self, *, run_id: str, draft: EvaluationContractDraft
    ) -> EvaluationAuthorityReceipt | None:
        if run_id != self._run_id or draft.digest != self._receipt.draft_digest:
            return None
        with self._lock:
            if self._consumed:
                return None
            self._consumed = True
            return self._receipt


__all__ = [
    "FormalEvaluationAuthorityProvider",
    "OneShotEvaluationAuthorityProvider",
]
