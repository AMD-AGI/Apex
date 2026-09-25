from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from apex.execution import MagpieKernelDiagnosticsAdapter
from apex.ports import KernelDiagnosticCommand, KernelDiagnosticRequest
from apex.runtime import DependencyReceipt


class _Containment:
    namespace_empty_verified = True

    def to_dict(self) -> dict[str, object]:
        return {"namespace_empty_verified": True}


class _Supervisor:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def run(self, argv, **kwargs):
        command = tuple(argv)
        self.calls.append((command, kwargs))
        output = Path(command[command.index("--output-dir") + 1])
        report_dir = output / "compare_fixture"
        report_dir.mkdir()
        (report_dir / "compare_report.json").write_text(
            json.dumps(
                {
                    "winner": 1,
                    "comparison_metrics": {"duration_ns_total": [100, 80]},
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            exit_code=0,
            timed_out=False,
            stdout="",
            stderr="",
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=1.25,
            process_containment=_Containment(),
            cleanup_succeeded=True,
        )


def _receipt(tmp_path: Path) -> DependencyReceipt:
    magpie = tmp_path / "Magpie"
    magpie.mkdir()
    return DependencyReceipt(
        schema="apex.dependency-receipt.v1",
        lock_sha256="a" * 64,
        python=Path(sys.executable),
        roots={"magpie": magpie},
        commits={"magpie": "b" * 40},
        raw={},
    )


def _request(tmp_path: Path) -> KernelDiagnosticRequest:
    baseline = (tmp_path / "baseline").resolve()
    candidate = (tmp_path / "candidate").resolve()
    baseline.mkdir()
    candidate.mkdir()
    for root in (baseline, candidate):
        (root / "kernel.py").write_text("def kernel(): pass\n", encoding="utf-8")
    command = KernelDiagnosticCommand((sys.executable, "runner.py"), ".", {})
    return KernelDiagnosticRequest(
        run_id="run-test",
        attempt_id="attempt-test",
        mode="compare",
        kernel_type="triton",
        source_files=("kernel.py",),
        candidate_root=candidate,
        baseline_root=baseline,
        output_root=(tmp_path / "private" / "magpie").resolve(),
        compile=command,
        correctness=command,
        performance=command,
        timeout_seconds=30,
    )


def test_compare_uses_pinned_python_and_keeps_result_advisory(tmp_path: Path) -> None:
    supervisor = _Supervisor()
    adapter = MagpieKernelDiagnosticsAdapter(
        lambda: _receipt(tmp_path),
        supervisor=supervisor,  # type: ignore[arg-type]
    )

    output = adapter.run(_request(tmp_path))

    argv, options = supervisor.calls[0]
    assert argv[:4] == (
        str(Path(sys.executable).resolve()),
        "-m",
        "Magpie",
        "compare",
    )
    assert argv[-2:] == ("--baseline", "0")
    assert options["require_pid_namespace"] is True
    config = json.loads(output.config_path.read_text(encoding="utf-8"))
    assert [item["id"] for item in config["kernels"]] == ["baseline", "candidate"]
    assert output.execution["magpie_commit"] == "b" * 40
    assert output.execution["evidence_class"] == "diagnostic"
    assert output.execution["reward_eligible"] is False


def test_analyze_emits_one_candidate_without_baseline_flag(tmp_path: Path) -> None:
    supervisor = _Supervisor()
    adapter = MagpieKernelDiagnosticsAdapter(
        lambda: _receipt(tmp_path),
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    request = _request(tmp_path)
    request = KernelDiagnosticRequest(
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        mode="analyze",
        kernel_type=request.kernel_type,
        source_files=request.source_files,
        candidate_root=request.candidate_root,
        baseline_root=None,
        output_root=(tmp_path / "private" / "analyze").resolve(),
        compile=request.compile,
        correctness=request.correctness,
        performance=request.performance,
        timeout_seconds=request.timeout_seconds,
    )

    # The fake models Magpie's mode-specific report filename.
    original_run = supervisor.run

    def run(argv, **kwargs):
        result = original_run(argv, **kwargs)
        output = Path(argv[argv.index("--output-dir") + 1])
        report = next(output.glob("**/compare_report.json"))
        report.rename(report.with_name("analyze_report.json"))
        return result

    supervisor.run = run  # type: ignore[method-assign]
    output = adapter.run(request)

    argv, _ = supervisor.calls[0]
    assert "--baseline" not in argv
    config = json.loads(output.config_path.read_text(encoding="utf-8"))
    assert [item["id"] for item in config["kernels"]] == ["candidate"]
