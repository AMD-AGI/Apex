from __future__ import annotations

from pathlib import Path

from apex.core import AgentBackendName
from apex.execution import NativeBackendDoctor, ProcessResult


class _Supervisor:
    def __init__(
        self,
        *,
        auth_exit_code: int = 0,
        auth_stdout: str = "Logged in using ChatGPT\n",
    ) -> None:
        self.auth_exit_code = auth_exit_code
        self.auth_stdout = auth_stdout
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv, **_values) -> ProcessResult:
        command = tuple(argv)
        self.calls.append(command)
        version = command[-1] == "--version"
        return ProcessResult(
            argv=command,
            exit_code=0 if version else self.auth_exit_code,
            timed_out=False,
            stdout="codex-cli 1.2.3\n" if version else self.auth_stdout,
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=0.01,
        )


def _executable(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    return path


def test_doctor_records_identity_and_redacts_probe_output(tmp_path: Path) -> None:
    executable = _executable(tmp_path, "codex")
    supervisor = _Supervisor()
    doctor = NativeBackendDoctor(
        supervisor=supervisor,
        executable_resolver=lambda _name: str(executable),
    )

    report = doctor.inspect(AgentBackendName.CODEX, workspace=tmp_path)
    document = report.to_dict()

    assert report.status == "ready"
    assert report.authenticated is True
    assert report.version == "codex-cli 1.2.3"
    assert len(report.entrypoint_sha256 or "") == 64
    assert document["credential_value_recorded"] is False
    assert document["probes"]["authentication"]["output_recorded"] is False
    assert document["probes"]["authentication"]["argv"] == ["login", "status"]
    assert document["features"]["run_scoped_mcp"]["available"] is True
    assert document["features"]["run_scoped_skills"]["available"] is True
    assert document["features"]["run_scoped_skills"]["verification"] == (
        "launcher_contract_only"
    )


def test_doctor_reports_authentication_and_cursor_feature_gaps(tmp_path: Path) -> None:
    executable = _executable(tmp_path, "cursor-agent")
    doctor = NativeBackendDoctor(
        supervisor=_Supervisor(
            auth_exit_code=1,
            auth_stdout='{"authenticated":false}\n',
        ),
        executable_resolver=lambda _name: str(executable),
    )

    report = doctor.inspect(AgentBackendName.CURSOR, workspace=tmp_path)

    assert report.status == "authentication_required"
    assert report.authenticated is False
    assert report.features["effort"].unavailable_reason == "agent_effort_unsupported"
    assert (
        report.features["run_scoped_mcp"].unavailable_reason
        == "per_invocation_mcp_configuration_unavailable"
    )
    assert report.features["run_scoped_skills"].available is False
    assert (
        report.features["run_scoped_skills"].unavailable_reason
        == "authentication_required"
    )
    assert report.probes["authentication"]["argv"] == [
        "status",
        "--format",
        "json",
    ]


def test_doctor_treats_missing_cli_as_an_inspectable_result(tmp_path: Path) -> None:
    doctor = NativeBackendDoctor(executable_resolver=lambda _name: None)

    report = doctor.inspect(AgentBackendName.CLAUDE, workspace=tmp_path)

    assert report.status == "cli_missing"
    assert report.installed is False
    assert report.authenticated is None
    assert report.probes == {}
    assert all(not feature.available for feature in report.features.values())
    assert report.features["interactive"].unavailable_reason == "cli_missing"
    assert report.features["interactive"].verification == "doctor_prerequisite"


def test_doctor_does_not_treat_unknown_auth_command_failure_as_logged_out(
    tmp_path: Path,
) -> None:
    executable = _executable(tmp_path, "cursor-agent")
    doctor = NativeBackendDoctor(
        supervisor=_Supervisor(
            auth_exit_code=1,
            auth_stdout="unknown command: status\n",
        ),
        executable_resolver=lambda _name: str(executable),
    )

    report = doctor.inspect(AgentBackendName.CURSOR, workspace=tmp_path)

    assert report.status == "authentication_probe_failed"
    assert report.authenticated is None
    assert report.features["interactive"].available is False
    assert (
        report.features["interactive"].unavailable_reason
        == "authentication_probe_failed"
    )


def test_doctor_requires_positive_json_auth_evidence_for_claude(
    tmp_path: Path,
) -> None:
    executable = _executable(tmp_path, "claude")
    unknown = NativeBackendDoctor(
        supervisor=_Supervisor(auth_stdout='{"account":"redacted"}\n'),
        executable_resolver=lambda _name: str(executable),
    ).inspect(AgentBackendName.CLAUDE, workspace=tmp_path)
    authenticated = NativeBackendDoctor(
        supervisor=_Supervisor(auth_stdout='{"loggedIn":true}\n'),
        executable_resolver=lambda _name: str(executable),
    ).inspect(AgentBackendName.CLAUDE, workspace=tmp_path)

    assert unknown.status == "authentication_probe_failed"
    assert unknown.authenticated is None
    assert authenticated.status == "ready"
    assert authenticated.authenticated is True
