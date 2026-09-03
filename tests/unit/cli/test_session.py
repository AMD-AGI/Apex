from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from apex.cli import app
from apex.core import AgentBackendName, ApexError
from apex.ports import CodingSessionOutput, KernelEnhancement


class _Launcher:
    def __init__(self) -> None:
        self.request = None

    def launch(self, request) -> int:
        self.request = request
        return 23


def test_root_request_dispatches_native_session_without_formal_optimizer(
    tmp_path, monkeypatch
) -> None:
    launcher = _Launcher()
    calls = []

    def build(**values):
        calls.append(values)
        return SimpleNamespace(coding_session=launcher)

    monkeypatch.setattr(app, "build_application", build)

    status = app.main(
        [
            "Refactor parser.py",
            "--workspace",
            str(tmp_path),
            "--backend",
            "claude",
            "--print",
            "--plain",
            "--results",
            str(tmp_path / "capability-results"),
        ]
    )

    assert status == 23
    assert calls == [{"include_kernel": False, "include_coding_session": True}]
    assert launcher.request.backend is AgentBackendName.CLAUDE
    assert launcher.request.output is CodingSessionOutput.TEXT
    assert launcher.request.results_dir == (tmp_path / "capability-results").resolve()
    assert launcher.request.enhancement is KernelEnhancement.PLAIN
    assert launcher.request.prompt == "Refactor parser.py"


def test_chat_campaign_handoff_dispatches_without_natural_language_resolution(
    tmp_path, monkeypatch
) -> None:
    captured = []

    def handoff(args, build_application):
        captured.append((args, build_application))
        return 19

    monkeypatch.setattr(app, "run_kernel_campaign_handoff", handoff)
    status = app.main(
        [
            "optimize",
            "kernel",
            "--campaign",
            str(tmp_path / "results" / "campaigns" / "campaign-1"),
            "--workspace",
            str(tmp_path),
            "--results",
            str(tmp_path / "results"),
            "--evaluation-contract-draft-digest",
            "a" * 64,
        ]
    )

    assert status == 19
    assert len(captured) == 1
    assert captured[0][0].campaign.name == "campaign-1"
    assert not hasattr(captured[0][0], "release_candidate_receipt")


def test_interactive_unresolved_kernel_enters_native_discovery_only(
    tmp_path, monkeypatch
) -> None:
    launcher = _Launcher()
    calls = []

    def build(**values):
        calls.append(values)
        return SimpleNamespace(coding_session=launcher)

    monkeypatch.setattr(app, "build_application", build)
    results = tmp_path / "results"

    status = app.main(
        [
            "optimize",
            "kernel",
            "Optimize the fastest kernel",
            "--workspace",
            str(tmp_path),
            "--results",
            str(results),
            "--backend",
            "claude",
        ]
    )

    assert status == 23
    assert calls == [{"include_kernel": False, "include_coding_session": True}]
    assert launcher.request.workspace == tmp_path.resolve()
    assert launcher.request.results_dir == results.resolve()
    assert launcher.request.backend is AgentBackendName.CLAUDE
    assert launcher.request.output is CodingSessionOutput.INTERACTIVE
    assert launcher.request.enhancement is KernelEnhancement.KERNEL
    assert "Optimize the fastest kernel" in launcher.request.prompt
    assert "task_descriptor_missing" in launcher.request.prompt
    assert "without running or authorizing evaluation" in launcher.request.prompt
    assert not (results / "result.json").exists()


def test_interactive_ambiguous_target_uses_same_non_formal_discovery_boundary(
    tmp_path, monkeypatch
) -> None:
    launcher = _Launcher()
    calls = []

    def fail_resolution(*_values, **_options):
        raise ApexError(
            "More than one target matches",
            "ambiguous_kernel_target",
            {"task_ids": ["first", "second"]},
        )

    def build(**values):
        calls.append(values)
        return SimpleNamespace(coding_session=launcher)

    monkeypatch.setattr(app.NaturalLanguageTaskResolver, "resolve", fail_resolution)
    monkeypatch.setattr(app, "build_application", build)

    status = app.main(
        [
            "optimize",
            "kernel",
            "Optimize this kernel",
            "--workspace",
            str(tmp_path),
            "--results",
            str(tmp_path / "results"),
        ]
    )

    assert status == 23
    assert calls == [{"include_kernel": False, "include_coding_session": True}]
    assert launcher.request.enhancement is KernelEnhancement.KERNEL
    assert "ambiguous_kernel_target" in launcher.request.prompt


@pytest.mark.parametrize(
    "machine_arguments",
    (
        ("--non-interactive",),
        ("--json",),
        ("--dry-run",),
        ("--result-json", "machine-result.json"),
    ),
)
def test_unresolved_kernel_machine_modes_remain_typed_needs_input(
    tmp_path, monkeypatch, machine_arguments
) -> None:
    def unexpected_build(**_values):
        raise AssertionError("machine needs-input must not compose an agent or evaluator")

    monkeypatch.setattr(app, "build_application", unexpected_build)
    results = tmp_path / "results"
    arguments = [
        "optimize",
        "kernel",
        "Optimize this kernel",
        "--workspace",
        str(tmp_path),
        "--results",
        str(results),
        *machine_arguments,
    ]
    if machine_arguments[0] == "--result-json":
        arguments[-1] = str(tmp_path / arguments[-1])

    status = app.main(arguments)

    result_path = (
        tmp_path / "machine-result.json"
        if machine_arguments[0] == "--result-json"
        else results / "result.json"
    )
    document = json.loads(result_path.read_text(encoding="utf-8"))
    assert status == 2
    assert document["status"] == "needs_input"
    assert document["reason_code"] == "task_descriptor_missing"
    assert document["interaction_mode"] in {"non_interactive", "deferred"}


def test_doctor_dispatches_without_composing_an_optimizer(
    tmp_path, monkeypatch, capsys
) -> None:
    class Doctor:
        def inspect(self, backend, *, workspace):
            assert backend is AgentBackendName.CURSOR
            assert workspace == tmp_path.resolve()
            return SimpleNamespace(
                status="authentication_required",
                to_dict=lambda: {
                    "backend": "cursor",
                    "status": "authentication_required",
                    "authenticated": False,
                    "features": {
                        "run_scoped_mcp": {
                            "available": False,
                            "unavailable_reason": "bridge_unavailable",
                        }
                    },
                },
            )

    calls = []

    def build(**values):
        calls.append(values)
        return SimpleNamespace(backend_doctor=Doctor())

    monkeypatch.setattr(app, "build_application", build)

    status = app.main(
        ["doctor", "--backend", "cursor", "--workspace", str(tmp_path), "--json"]
    )

    assert status == 1
    assert calls == [{"include_kernel": False, "include_backend_doctor": True}]
    assert '"status": "authentication_required"' in capsys.readouterr().out


def test_gpu_doctor_emits_ownership_and_fails_closed_on_missing_health_evidence(
    tmp_path, monkeypatch, capsys
) -> None:
    selected = SimpleNamespace(unique_id="GPU-0000000000000001")
    ownership = SimpleNamespace(
        selected_devices=(selected,),
        foreign_owners=(),
        digest="a" * 64,
        to_dict=lambda: {"selected_devices": [{"unique_id": selected.unique_id}]},
    )
    receipt = SimpleNamespace(
        ownership=ownership,
        status="incomplete",
        formal_measurement_ready=False,
        rocm_health=None,
        digest="b" * 64,
        to_dict=lambda: {"ownership": ownership.to_dict()},
    )

    class Doctor:
        def inspect(self, selector, *, allowed_pids):
            assert selector == "amd-gpu-set=0"
            assert len(allowed_pids) == 1
            return receipt

    calls = []

    def build(**values):
        calls.append(values)
        return SimpleNamespace(gpu_doctor=Doctor())

    monkeypatch.setattr(app, "build_application", build)

    status = app.main(["doctor", "gpu", "--gpu-devices", "0", "--json"])

    output = capsys.readouterr().out
    assert status == 1
    assert calls == [{"include_kernel": False, "include_gpu_doctor": True}]
    assert '"ownership_status": "clean"' in output
    assert '"formal_measurement_ready": false' in output
    assert '"rocm_health"' in output


def test_capability_inventory_uses_caller_scope_without_creating_results(
    tmp_path, monkeypatch, capsys
) -> None:
    results = tmp_path / "selected-results"
    availability = SimpleNamespace(
        to_dict=lambda: {
            "capability_id": "campaign.status",
            "available": True,
            "unavailable_reason": None,
            "summary": "Inspect a campaign.",
        }
    )
    calls = []

    def build(**values):
        calls.append(values)
        return SimpleNamespace(
            capabilities=SimpleNamespace(inventory=lambda: (availability,))
        )

    monkeypatch.setattr(app, "build_application", build)

    status = app.main(
        [
            "capabilities",
            "--workspace",
            str(tmp_path),
            "--results",
            str(results),
            "--json",
        ]
    )

    assert status == 0
    assert calls == [
        {
            "include_kernel": False,
            "include_capabilities": True,
            "capability_workspace": tmp_path.resolve(),
            "capability_results": results.resolve(),
        }
    ]
    assert not results.exists()
    assert '"capability_id": "campaign.status"' in capsys.readouterr().out
