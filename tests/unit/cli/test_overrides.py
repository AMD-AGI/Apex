from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import json

import pytest

from apex.cli import app
from apex.core import (
    AgentBackendName,
    ApexError,
    TaskStatus,
    canonical_json_bytes,
    sha256_bytes,
)
from apex.intake import TaskSpec


ROOT = Path(__file__).resolve().parents[3]


def _task(tmp_path: Path) -> TaskSpec:
    return TaskSpec.from_mapping(
        {
            "task_id": "cli-budget",
            "workspace": str(tmp_path / "workspace"),
            "results_dir": str(tmp_path / "results"),
            "instructions": "Optimize kernel",
            "language": "triton",
            "editable_files": ["kernel.py"],
            "target_functions": ["kernel"],
            "commands": {
                phase: {"argv": ["true"]}
                for phase in ("compile", "correctness", "performance")
            },
        }
    )


def test_kernel_cli_budget_overrides_are_frozen_into_task(tmp_path: Path) -> None:
    args = app._parser().parse_args(
        [
            "optimize", "kernel", "Optimize kernel.py",
            "--workspace", str(tmp_path / "workspace"),
            "--results", str(tmp_path / "results"),
            "--max-iterations", "3", "--max-turns", "17",
            "--timeout-seconds", "901",
        ]
    )

    task = app._kernel_budget_overrides(_task(tmp_path), args)

    assert task.budget.max_iterations == 3
    assert task.budget.max_turns == 17
    assert task.budget.timeout_seconds == 901


def test_e2e_cli_builds_internal_spec_from_raw_magpie_config(tmp_path: Path) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text("benchmark: {framework: vllm}\n", encoding="utf-8")
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    args = app._parser().parse_args(
        [
            "optimize", "e2e", "--config", str(config),
            "--results", str(tmp_path / "results"),
            "--backend", "claude", "--model", "opus",
            "--effort", "high", "--max-iterations", "4",
            "--max-kernels", "6", "--max-turns", "19",
            "--timeout-seconds", "902", "--gpu-arch", "gfx942",
            "--gpu-devices", "2,3", "--hf-cache-path", str(cache),
            "--hf-offline",
        ]
    )

    updated = app._e2e_spec(args)

    assert updated.config_path == config.resolve()
    assert updated.results_dir == (tmp_path / "results").resolve()
    assert updated.agent_backend is AgentBackendName.CLAUDE
    assert updated.agent_model == "opus"
    assert updated.agent_effort == "high"
    assert updated.gpu_arch == "gfx942"
    assert updated.deployment_hints == {
        "gpu_devices": "2,3",
        "hf_cache_path": str(cache.resolve()),
        "hf_offline": True,
    }
    assert (updated.max_iterations, updated.max_kernels) == (4, 6)
    assert (updated.max_turns, updated.agent_timeout_seconds) == (19, 902)


def test_e2e_cli_has_no_apex_specific_spec_input(tmp_path: Path) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text("benchmark: {framework: vllm}\n", encoding="utf-8")
    parser = app._parser()

    args = parser.parse_args(
        [
            "optimize", "e2e", "--config", str(config),
            "--results", str(tmp_path / "results"),
        ]
    )

    assert not hasattr(args, "spec")
    assert args.agent_backend == "codex"


def test_e2e_live_spec_rejects_results_inside_apex_checkout(tmp_path: Path) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text("benchmark: {framework: vllm}\n", encoding="utf-8")
    args = app._parser().parse_args(
        [
            "optimize",
            "e2e",
            "--config",
            str(config),
            "--results",
            str(ROOT / "tmp" / "refactor" / "formal-e2e"),
        ]
    )

    with pytest.raises(ApexError) as caught:
        app._e2e_spec(args)

    assert caught.value.reason_code == "formal_results_overlap"


def test_e2e_dry_run_allows_non_authoritative_in_tree_preflight(
    tmp_path: Path,
) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text("benchmark: {framework: vllm}\n", encoding="utf-8")
    results = ROOT / "tmp" / "refactor" / "preflight"
    args = app._parser().parse_args(
        [
            "optimize",
            "e2e",
            "--config",
            str(config),
            "--results",
            str(results),
            "--dry-run",
        ]
    )

    assert app._e2e_spec(args).results_dir == results.resolve()


def test_e2e_dry_run_uses_preview_and_never_runs_campaign(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        "benchmark: {framework: vllm, model: example/model}\n",
        encoding="utf-8",
    )
    results = tmp_path / "results"
    calls: list[str] = []

    class Optimizer:
        def preview(self, spec):
            calls.append(f"preview:{spec.config_path.name}")
            return SimpleNamespace(
                to_dict=lambda: {
                    "schema": "apex.e2e-preflight/v1",
                    "status": "config_compatible",
                    "gpu_acquired": False,
                }
            )

        def run(self, _spec):
            raise AssertionError("dry-run must not start a campaign")

    monkeypatch.setattr(
        app,
        "build_application",
        lambda **kwargs: SimpleNamespace(e2e_optimizer=Optimizer()),
    )
    monkeypatch.setattr(
        app,
        "write_preflight_result",
        lambda preview, output: output / "preflight.json",
    )

    status = app.main(
        [
            "optimize", "e2e", "--config", str(config),
            "--results", str(results), "--dry-run",
        ]
    )

    assert status == 0
    output = json.loads(capsys.readouterr().out)
    assert output["gpu_acquired"] is False
    assert output["result_path"] == str(results / "preflight.json")
    assert calls == ["preview:benchmark.yaml"]


def test_e2e_live_run_requires_verified_campaign_baseline(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        "benchmark: {framework: vllm, model: example/model}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        app,
        "build_application",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must fail before composition")),
    )

    status = app.main([
        "optimize", "e2e", "--config", str(config),
        "--results", str(tmp_path / "results"),
    ])

    assert status == 2
    assert json.loads(capsys.readouterr().err)["reason_code"] == (
        "campaign_baseline_receipt_required"
    )


def test_e2e_live_run_carries_verified_baseline_into_spec(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        "benchmark: {framework: vllm, model: example/model}\n",
        encoding="utf-8",
    )
    payload = {
        "schema": "apex.release-candidate-receipt/v2",
        "baseline_status": "ready",
        "baseline_blockers": [],
    }
    payload["receipt_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    observed = []

    class Optimizer:
        def run(self, spec):
            observed.append(spec.campaign_baseline_receipt)
            return SimpleNamespace(
                status=TaskStatus.NO_GAIN,
                to_dict=lambda: {"status": "no_gain"},
            )

    monkeypatch.setattr(
        app,
        "require_campaign_baseline",
        lambda path: SimpleNamespace(to_dict=lambda: payload),
    )
    monkeypatch.setattr(
        app,
        "build_application",
        lambda **kwargs: SimpleNamespace(e2e_optimizer=Optimizer()),
    )

    status = app.main([
        "optimize", "e2e", "--config", str(config),
        "--results", str(tmp_path / "results"),
        "--release-candidate-receipt", str(tmp_path / "receipt.json"),
    ])

    assert status == 0
    assert observed == [payload]
    assert json.loads(capsys.readouterr().out)["status"] == "no_gain"


def test_resume_parser_requires_explicit_run_root() -> None:
    args = app._parser().parse_args(["run", "resume", "--run", "/tmp/apex-run"])

    assert args.command == "run"
    assert args.run_command == "resume"
    assert args.run == Path("/tmp/apex-run")


def test_resume_requires_current_original_campaign_baseline(capsys) -> None:
    status = app.main(["run", "resume", "--run", "/tmp/apex-run"])

    assert status == 2
    assert json.loads(capsys.readouterr().err)["reason_code"] == (
        "campaign_baseline_receipt_required"
    )


def test_pending_attributed_template_fails_before_execution(
    tmp_path: Path, capsys
) -> None:
    template = (
        ROOT
        / "examples"
        / "optimization_showcases"
        / "kernel_ck_moe_2stage"
    )

    exit_code = app.main(
        [
            "optimize",
            "kernel",
            "Optimize the declared kernel",
            "--template",
            str(template),
            "--results",
            str(tmp_path / "results"),
        ]
    )

    assert exit_code == 2
    assert '"reason_code": "template_not_materializable"' in capsys.readouterr().err


def test_template_cli_forbids_a_second_workspace(tmp_path: Path, capsys) -> None:
    exit_code = app.main(
        [
            "optimize",
            "kernel",
            "Optimize the declared kernel",
            "--template",
            str(tmp_path / "template"),
            "--workspace",
            str(tmp_path),
            "--results",
            str(tmp_path / "results"),
        ]
    )

    assert exit_code == 2
    assert '"reason_code": "kernel_template_input_invalid"' in capsys.readouterr().err
