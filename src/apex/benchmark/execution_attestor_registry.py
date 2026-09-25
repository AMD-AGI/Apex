"""Route one Magpie execution to exactly one trusted observer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from apex.core import ContractError
from apex.ports import (
    MagpieAttestationRequest,
    MagpieExecutionAttestor,
    MagpieFormalMeasurementSupport,
    MagpieReportLocation,
)


@dataclass(frozen=True, slots=True)
class _RoutedSession:
    attestor: MagpieExecutionAttestor
    session: object


class MagpieExecutionAttestorRegistry:
    """Compose mode/lifecycle-specific observers without a fallback lane."""

    def __init__(self, attestors: tuple[MagpieExecutionAttestor, ...]) -> None:
        if not attestors:
            raise ValueError("At least one Magpie execution attestor is required")
        self._attestors = attestors

    @property
    def is_available(self) -> bool:
        return any(attestor.is_available for attestor in self._attestors)

    def supports(self, execution_mode: str, lifecycle: str) -> bool:
        return len(self._matches(execution_mode, lifecycle)) == 1

    def formal_measurement_support(
        self, execution_mode: str, lifecycle: str
    ) -> MagpieFormalMeasurementSupport:
        matches = self._matches(execution_mode, lifecycle)
        if len(matches) != 1:
            return MagpieFormalMeasurementSupport(
                False,
                "magpie_execution_attestor_unavailable",
                None,
                ("magpie_execution_attestor_unavailable",),
            )
        return matches[0].formal_measurement_support(execution_mode, lifecycle)

    def prepare(self, request: MagpieAttestationRequest) -> object:
        matches = self._matches(request.execution_mode, request.lifecycle)
        if not matches:
            raise ContractError(
                "No execution attestor supports this Magpie lane",
                "magpie_execution_attestor_unavailable",
            )
        if len(matches) != 1:
            raise ContractError(
                "Multiple execution attestors claim the same Magpie lane",
                "ambiguous_magpie_execution_attestor",
            )
        attestor = matches[0]
        return _RoutedSession(attestor, attestor.prepare(request))

    def complete(
        self,
        session: object,
        *,
        report_path: Path | None,
        command_exit_code: int | None,
        timed_out: bool,
    ) -> Path | None:
        if not isinstance(session, _RoutedSession):
            raise ContractError(
                "Magpie routed observer session is invalid",
                "invalid_magpie_execution_attestor_session",
            )
        return session.attestor.complete(
            session.session,
            report_path=report_path,
            command_exit_code=command_exit_code,
            timed_out=timed_out,
        )

    def launch_argv(self, session: object) -> tuple[str, ...]:
        if not isinstance(session, _RoutedSession):
            raise ContractError(
                "Magpie routed observer session is invalid",
                "invalid_magpie_execution_attestor_session",
            )
        return session.attestor.launch_argv(session.session)

    def abort(self, session: object, *, reason: str) -> None:
        if not isinstance(session, _RoutedSession):
            raise ContractError(
                "Magpie routed observer session is invalid",
                "invalid_magpie_execution_attestor_session",
            )
        session.attestor.abort(session.session, reason=reason)

    def locate_report(self, session: object) -> MagpieReportLocation:
        if not isinstance(session, _RoutedSession):
            raise ContractError(
                "Magpie routed observer session is invalid",
                "invalid_magpie_execution_attestor_session",
            )
        return session.attestor.locate_report(session.session)

    def _matches(
        self, execution_mode: str, lifecycle: str
    ) -> tuple[MagpieExecutionAttestor, ...]:
        matches = []
        for attestor in self._attestors:
            supports = getattr(attestor, "supports", None)
            if (
                attestor.is_available
                and callable(supports)
                and supports(execution_mode, lifecycle)
            ):
                matches.append(attestor)
        return tuple(matches)


__all__ = ["MagpieExecutionAttestorRegistry"]
