"""Post-measurement Magpie diagnostics for standalone kernel attempts."""

from __future__ import annotations

from pathlib import Path

from apex.core import ApexError, IntegrityError, canonical_json_bytes
from apex.ports import (
    KernelDiagnosticCommand,
    KernelDiagnosticOutput,
    KernelDiagnosticRequest,
    KernelDiagnosticsPort,
)
from apex.storage import ArtifactReceipt

from .attempts import AttemptSession, PreparedCandidate
from .verification import candidate_source_digest
from .workspace import CandidateWorkspace, candidate_file_bytes


def run_kernel_diagnostics(
    adapter: KernelDiagnosticsPort | None,
    attempt: AttemptSession,
    prepared: PreparedCandidate,
) -> tuple[ArtifactReceipt, ...]:
    """Run compare on disposable projections; failures remain advisory."""

    if adapter is None:
        return ()
    _record_called(attempt, adapter.adapter_id)
    try:
        baseline, candidate = _projections(attempt, prepared)
        output = adapter.run(_request(attempt, baseline=baseline, candidate=candidate))
        return _record_output(attempt, output)
    except (ApexError, OSError) as error:
        reason_code = (
            error.reason_code
            if isinstance(error, ApexError)
            else "magpie_diagnostic_io_failed"
        )
        return (
            _record_failure(attempt, adapter.adapter_id, reason_code),
        )


def _projections(
    attempt: AttemptSession, prepared: PreparedCandidate
) -> tuple[Path, Path]:
    root = attempt.run.run_root / "diagnostics" / attempt.attempt_id
    baseline_workspace = CandidateWorkspace.create(
        attempt.run.resolved, destination=root / "baseline"
    )
    candidate_workspace = CandidateWorkspace.create(
        attempt.run.resolved, destination=root / "candidate"
    )
    files = candidate_file_bytes(
        attempt.candidate.root, attempt.run.resolved.task.editable_files
    )
    for relative, content in files.items():
        candidate_workspace.root.joinpath(*relative.split("/")).write_bytes(content)
    observed = candidate_source_digest(
        candidate_workspace.root, attempt.run.resolved.task.editable_files
    )
    if observed != prepared.normal_source_digest:
        raise IntegrityError(
            "Diagnostic projection differs from candidate",
            "magpie_candidate_projection_mismatch",
        )
    return baseline_workspace.root.resolve(), candidate_workspace.root.resolve()


def _request(
    attempt: AttemptSession, *, baseline: Path, candidate: Path
) -> KernelDiagnosticRequest:
    task = attempt.run.resolved.task

    def command(name: str) -> KernelDiagnosticCommand:
        item = task.commands[name]
        return KernelDiagnosticCommand(item.argv, item.cwd, item.env)

    timeout = max(item.timeout_seconds for item in task.commands.values())
    return KernelDiagnosticRequest(
        run_id=attempt.run.run_id,
        attempt_id=attempt.attempt_id,
        mode="compare",
        kernel_type=task.language,
        source_files=task.editable_files,
        candidate_root=candidate,
        baseline_root=baseline,
        output_root=(
            attempt.run.run_root
            / "diagnostic-results"
            / attempt.attempt_id
            / "magpie-compare"
        ).resolve(),
        compile=command("compile"),
        correctness=command("correctness"),
        performance=command("performance"),
        timeout_seconds=timeout,
    )


def _record_called(attempt: AttemptSession, adapter_id: str) -> None:
    record = attempt.run.record
    record.controller.record_domain_event(
        "tool_called",
        {
            **record.attempt_payload(attempt.attempt_id),
            "tool_name": "magpie.compare",
            "tool_call_id": f"magpie-compare-{attempt.attempt_id}",
            "adapter_id": adapter_id,
            "evidence_class": "diagnostic",
            "reward_eligible": False,
        },
        idempotency_key=f"attempt.{attempt.attempt_id}.magpie.compare.called",
    )


def _record_output(
    attempt: AttemptSession, output: KernelDiagnosticOutput
) -> tuple[ArtifactReceipt, ...]:
    record = attempt.run.record
    report = record.artifacts.put_file(output.report_path, media_type="application/json")
    config = record.artifacts.put_file(output.config_path, media_type="application/json")
    execution = record.artifacts.put_bytes(
        canonical_json_bytes(output.execution), media_type="application/json"
    )
    record.controller.record_domain_event(
        "tool_result",
        {
            **record.attempt_payload(attempt.attempt_id),
            "tool_name": f"magpie.{output.mode}",
            "tool_call_id": f"magpie-{output.mode}-{attempt.attempt_id}",
            "succeeded": True,
            "adapter_id": output.adapter_id,
            "evidence_class": "diagnostic",
            "reward_eligible": False,
            "artifacts": [
                _binding("magpie_kernel_report", report),
                _binding("magpie_kernel_config", config),
                _binding("magpie_kernel_execution", execution),
            ],
        },
        idempotency_key=f"attempt.{attempt.attempt_id}.magpie.{output.mode}.result",
    )
    return report, config, execution


def _record_failure(
    attempt: AttemptSession, adapter_id: str, reason_code: str
) -> ArtifactReceipt:
    record = attempt.run.record
    receipt = record.artifacts.put_bytes(
        canonical_json_bytes(
            {
                "schema": "apex.magpie-kernel-diagnostic-failure/v1",
                "adapter_id": adapter_id,
                "reason_code": reason_code,
                "reward_eligible": False,
            }
        ),
        media_type="application/json",
    )
    record.controller.record_domain_event(
        "tool_result",
        {
            **record.attempt_payload(attempt.attempt_id),
            "tool_name": "magpie.compare",
            "tool_call_id": f"magpie-compare-{attempt.attempt_id}",
            "succeeded": False,
            "adapter_id": adapter_id,
            "reason_code": reason_code,
            "evidence_class": "diagnostic",
            "reward_eligible": False,
            "artifacts": [_binding("magpie_kernel_failure", receipt)],
        },
        idempotency_key=f"attempt.{attempt.attempt_id}.magpie.compare.result",
    )
    return receipt


def _binding(role: str, receipt: ArtifactReceipt) -> dict[str, object]:
    return {"role": role, "receipt": receipt.to_dict()}


__all__ = ["run_kernel_diagnostics"]
