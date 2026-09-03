"""Bounded, read-only health checks for native coding backends."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from apex.core import AgentBackendName, sha256_file

from .environment import AGENT_CONFIG_ENVIRONMENT_KEYS, build_subprocess_environment
from .supervisor import ProcessResult, SubprocessSupervisor


_CLI_NAMES = {
    AgentBackendName.CODEX: "codex",
    AgentBackendName.CLAUDE: "claude",
    AgentBackendName.CURSOR: "cursor-agent",
}
_CREDENTIAL_KEYS = {
    AgentBackendName.CODEX: "OPENAI_API_KEY",
    AgentBackendName.CLAUDE: "ANTHROPIC_API_KEY",
    AgentBackendName.CURSOR: "CURSOR_API_KEY",
}
_AUTH_ARGUMENTS = {
    AgentBackendName.CODEX: ("login", "status"),
    AgentBackendName.CLAUDE: ("auth", "status", "--json"),
    AgentBackendName.CURSOR: ("status", "--format", "json"),
}


@dataclass(frozen=True, slots=True)
class BackendFeature:
    """One native surface that the launcher can use without emulation."""

    available: bool
    unavailable_reason: str | None = None
    verification: str = "launcher_contract_only"

    def __post_init__(self) -> None:
        if self.available == bool(self.unavailable_reason):
            raise ValueError("backend feature availability is incoherent")

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "verification": self.verification,
        }


@dataclass(frozen=True, slots=True)
class BackendDoctorReport:
    """Credential-redacted identity, authentication, and feature receipt."""

    backend: AgentBackendName
    status: str
    cli_name: str
    installed: bool
    authenticated: bool | None
    version: str | None
    executable_path: str | None
    resolved_executable_path: str | None
    entrypoint_sha256: str | None
    credential_environment_key: str
    features: Mapping[str, BackendFeature]
    probes: Mapping[str, Mapping[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "apex.backend-doctor/v1",
            "backend": self.backend.value,
            "status": self.status,
            "cli_name": self.cli_name,
            "installed": self.installed,
            "authenticated": self.authenticated,
            "version": self.version,
            "executable_path": self.executable_path,
            "resolved_executable_path": self.resolved_executable_path,
            "entrypoint_sha256": self.entrypoint_sha256,
            "credential_environment_key": self.credential_environment_key,
            "credential_value_recorded": False,
            "features": {
                name: feature.to_dict()
                for name, feature in sorted(self.features.items())
            },
            "probes": {name: dict(value) for name, value in self.probes.items()},
        }


class NativeBackendDoctor:
    """Inspect exact local CLIs with fixed argv, time, and output bounds."""

    def __init__(
        self,
        *,
        supervisor: SubprocessSupervisor | None = None,
        executable_resolver: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self._supervisor = supervisor or SubprocessSupervisor(max_output_bytes=64 * 1024)
        self._resolve = executable_resolver

    def inspect(self, backend: AgentBackendName, *, workspace: Path) -> BackendDoctorReport:
        cli_name = _CLI_NAMES[backend]
        credential_key = _CREDENTIAL_KEYS[backend]
        executable = self._resolve(cli_name)
        if executable is None:
            return BackendDoctorReport(
                backend,
                "cli_missing",
                cli_name,
                False,
                None,
                None,
                None,
                None,
                None,
                credential_key,
                _features(backend, "cli_missing"),
                {},
            )
        discovered = Path(executable).absolute()
        try:
            resolved = discovered.resolve(strict=True)
            entrypoint_sha256 = sha256_file(resolved)
        except OSError:
            return _identity_failed_report(
                backend, cli_name, credential_key, discovered
            )
        environment = build_subprocess_environment(
            {},
            inherit=AGENT_CONFIG_ENVIRONMENT_KEYS,
            inherit_secrets=(credential_key,),
        )
        version_result = self._probe(
            (str(discovered), "--version"), workspace, environment
        )
        version = _bounded_version(version_result)
        probes = {"version": _probe_receipt(version_result)}
        if version is None:
            return BackendDoctorReport(
                backend,
                "version_probe_failed",
                cli_name,
                True,
                None,
                None,
                str(discovered),
                str(resolved),
                entrypoint_sha256,
                credential_key,
                _features(backend, "version_probe_failed"),
                probes,
            )
        auth_result = self._probe(
            (str(discovered), *_AUTH_ARGUMENTS[backend]), workspace, environment
        )
        probes["authentication"] = _probe_receipt(auth_result)
        authenticated = _authentication(backend, auth_result)
        status = (
            "ready"
            if authenticated is True
            else "authentication_required"
            if authenticated is False
            else "authentication_probe_failed"
        )
        return BackendDoctorReport(
            backend,
            status,
            cli_name,
            True,
            authenticated,
            version,
            str(discovered),
            str(resolved),
            entrypoint_sha256,
            credential_key,
            _features(backend, None if authenticated is True else status),
            probes,
        )

    def _probe(self, argv, workspace: Path, environment) -> ProcessResult:
        return self._supervisor.run(
            argv,
            cwd=workspace,
            environment=environment,
            timeout_seconds=15,
        )


def _features(
    backend: AgentBackendName, prerequisite_failure: str | None = None
) -> dict[str, BackendFeature]:
    available = (
        BackendFeature(True)
        if prerequisite_failure is None
        else BackendFeature(False, prerequisite_failure, "doctor_prerequisite")
    )
    features = {
        "interactive": available,
        "headless_text": available,
        "headless_jsonl": available,
        "native_resume": available,
        "native_approval_and_sandbox": available,
        "effort": available,
        "run_scoped_mcp": available,
        "run_scoped_skills": available,
    }
    if backend is AgentBackendName.CURSOR:
        features["effort"] = BackendFeature(False, "agent_effort_unsupported")
        features["run_scoped_mcp"] = BackendFeature(
            False, "per_invocation_mcp_configuration_unavailable"
        )
    return features


def _bounded_version(result: ProcessResult) -> str | None:
    output = result.stdout.strip() or result.stderr.strip()
    if not _probe_succeeded(result) or not output or len(output) > 512:
        return None
    return output


def _authentication(
    backend: AgentBackendName, result: ProcessResult
) -> bool | None:
    if result.timed_out or result.stdout_truncated or result.stderr_truncated:
        return None
    if not result.cleanup_succeeded or result.exit_code is None:
        return None
    output = (result.stdout.strip() or result.stderr.strip()).strip()
    if not output:
        return None
    if backend is AgentBackendName.CODEX:
        normalized = output.casefold()
        if any(value in normalized for value in ("not logged in", "login required")):
            return False
        if result.exit_code == 0 and any(
            value in normalized for value in ("logged in", "authenticated")
        ):
            return True
        return None
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        return None
    return _json_authentication(value, exit_code=result.exit_code)


def _json_authentication(value: object, *, exit_code: int) -> bool | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("authenticated", "loggedIn", "isAuthenticated", "logged_in"):
        observed = value.get(key)
        if isinstance(observed, bool):
            return observed if not observed or exit_code == 0 else None
    status = value.get("status")
    if isinstance(status, str):
        normalized = status.casefold().replace("-", "_").replace(" ", "_")
        if normalized in {"authenticated", "logged_in"} and exit_code == 0:
            return True
        if normalized in {"unauthenticated", "not_authenticated", "logged_out"}:
            return False
    return None


def _probe_succeeded(result: ProcessResult) -> bool:
    return bool(
        result.exit_code == 0
        and not result.timed_out
        and not result.stdout_truncated
        and not result.stderr_truncated
        and result.cleanup_succeeded
    )


def _probe_receipt(result: ProcessResult) -> dict[str, object]:
    return {
        "argv": list(result.argv[1:]),
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "output_truncated": result.stdout_truncated or result.stderr_truncated,
        "cleanup_succeeded": result.cleanup_succeeded,
        "output_recorded": False,
    }


def _identity_failed_report(
    backend: AgentBackendName,
    cli_name: str,
    credential_key: str,
    discovered: Path,
) -> BackendDoctorReport:
    return BackendDoctorReport(
        backend,
        "cli_identity_failed",
        cli_name,
        True,
        None,
        None,
        str(discovered),
        None,
        None,
        credential_key,
        _features(backend, "cli_identity_failed"),
        {},
    )


__all__ = ["BackendDoctorReport", "BackendFeature", "NativeBackendDoctor"]
