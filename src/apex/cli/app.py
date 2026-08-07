"""Parse and dispatch Apex commands without owning domain state."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from apex.bootstrap import build_application
from apex.core import AgentBackendName, ApexError, TaskStatus
from apex.delivery import (
    apply_verified_kernel_bundle,
    detect_bundle_kind,
    load_and_verify_e2e_bundle,
    load_and_verify_kernel_bundle,
)
from apex.intake import (
    E2EOptimizeSpec,
    NaturalLanguageRequest,
    NaturalLanguageTaskResolver,
    TaskResolver,
    TaskSpec,
)
from apex.optimization.kernel import KernelOptimizeRequest
from apex.rl import DatasetExportConfig
from apex.runtime.dependencies import main as dependencies_main

from .projections import export_rl_dataset, rebuild_report


_NEEDS_INPUT_REASONS = {
    "task_descriptor_missing",
    "target_not_resolved",
    "ambiguous_kernel_target",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apex",
        description="Evidence-driven AMD GPU kernel optimization environment",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    optimize = commands.add_parser("optimize", help="Run an optimization use case")
    optimize_commands = optimize.add_subparsers(dest="optimize_command", required=True)
    kernel = optimize_commands.add_parser("kernel", help="Optimize one existing kernel")
    kernel.add_argument("request", nargs="?", help="Natural-language task (with discovery flags)")
    kernel.add_argument("--task-spec", type=Path, help="Caller-neutral TaskSpec JSON/YAML")
    kernel.add_argument("--workspace", type=Path, help="Workspace for a natural-language task")
    kernel.add_argument("--results", type=Path, help="Run output directory for a natural-language task")
    kernel.add_argument("--result-json", type=Path, help="Atomic machine result path")
    kernel.add_argument("--agent-backend", choices=[item.value for item in AgentBackendName])
    kernel.add_argument("--agent-model")
    kernel.add_argument("--agent-effort")
    kernel.add_argument("--max-iterations", type=int)
    kernel.add_argument("--max-turns", type=int)
    kernel.add_argument("--timeout-seconds", type=int)
    kernel.add_argument("--non-interactive", action="store_true")
    kernel.add_argument("--dry-run", action="store_true", help="Resolve and persist the task only")
    kernel.add_argument("--json", action="store_true", help="Emit a stable JSON envelope")

    e2e = optimize_commands.add_parser("e2e", help="Optimize kernels in one E2E workload")
    e2e.add_argument("--spec", type=Path, required=True)
    e2e.add_argument("--results", type=Path, help="Override spec.results_dir")
    e2e.add_argument("--agent-backend", choices=[item.value for item in AgentBackendName])
    e2e.add_argument("--agent-model")
    e2e.add_argument("--agent-effort")
    e2e.add_argument("--max-iterations", type=int)
    e2e.add_argument("--max-kernels", type=int)
    e2e.add_argument("--max-turns", type=int)
    e2e.add_argument("--timeout-seconds", type=int)

    bundle = commands.add_parser("bundle", help="Inspect or verify source bundles")
    bundle_commands = bundle.add_subparsers(dest="bundle_command", required=True)
    verify = bundle_commands.add_parser("verify", help="Verify a content-digested kernel bundle")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--digest")
    verify.add_argument("--json", action="store_true")
    apply_bundle = bundle_commands.add_parser(
        "apply", help="Explicitly apply a kernel bundle to an exact clean baseline"
    )
    apply_bundle.add_argument("--bundle", type=Path, required=True)
    apply_bundle.add_argument("--workspace", type=Path, required=True)
    apply_bundle.add_argument("--digest")
    apply_bundle.add_argument("--json", action="store_true")

    report = commands.add_parser("report", help="Rebuild reports from a canonical run")
    report.add_argument("--run-root", type=Path, required=True)
    report.add_argument("--run-id")
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--json", action="store_true")

    export = commands.add_parser("export-rl", help="Export RL data from a canonical run")
    export.add_argument("--run-root", type=Path, required=True)
    export.add_argument("--run-id")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--split", choices=("train", "validation", "heldout"))
    export.add_argument("--policy-id")
    export.add_argument("--on-incomplete", choices=("fail", "skip"), default="fail")
    export.add_argument("--no-sft", action="store_true")
    export.add_argument("--json", action="store_true")

    dependencies = commands.add_parser("dependencies", help="Install or verify pinned dependencies")
    dependencies.add_argument("dependency_args", nargs=argparse.REMAINDER)
    return parser


def _kernel(args: argparse.Namespace) -> int:
    if bool(args.task_spec) == bool(args.request):
        raise ApexError(
            "Provide exactly one of a natural-language request or --task-spec",
            "kernel_input_exactly_one",
        )
    if args.task_spec is not None:
        task = TaskSpec.from_file(args.task_spec.expanduser().resolve(strict=True))
    else:
        if args.workspace is None or args.results is None:
            raise ApexError(
                "Natural-language optimization requires --workspace and --results",
                "natural_language_paths_required",
            )
        intent = NaturalLanguageRequest(
            text=args.request,
            workspace=args.workspace.expanduser().resolve(strict=True),
            results_dir=args.results.expanduser().resolve(),
        )
        try:
            resolved = NaturalLanguageTaskResolver().resolve(
                intent,
                backend=AgentBackendName(
                    args.agent_backend or AgentBackendName.CODEX.value
                ),
            )
        except ApexError as error:
            if error.reason_code not in _NEEDS_INPUT_REASONS:
                raise
            result_path = _natural_language_result_path(args)
            return _write_needs_input(error, result_path, args=args)
        task = resolved.task
    task = _kernel_budget_overrides(task, args)
    result_path = (
        args.result_json.expanduser().resolve()
        if args.result_json is not None
        else task.results_dir / "result.json"
    )
    if args.dry_run:
        return _write_resolved_task(task, result_path)
    request = KernelOptimizeRequest(
        task=task,
        result_json=result_path,
        backend_override=AgentBackendName(args.agent_backend) if args.agent_backend else None,
        model_override=args.agent_model,
        effort_override=args.agent_effort,
    )
    result = build_application().kernel_optimizer.run(request)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return _status_exit_code(result.status)


def _write_resolved_task(task: TaskSpec, result_path: Path) -> int:
    resolved = TaskResolver().resolve(task)
    output = {
        "schema_version": 1,
        "status": "resolved",
        "resolution_hash": resolved.resolution_hash,
        "task": task.to_dict(),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_name(f".{result_path.name}.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _natural_language_result_path(args: argparse.Namespace) -> Path:
    if args.result_json is not None:
        return args.result_json.expanduser().resolve()
    assert args.results is not None
    return args.results.expanduser().resolve() / "result.json"


def _write_needs_input(
    error: ApexError,
    result_path: Path,
    *,
    args: argparse.Namespace,
) -> int:
    output = {
        "schema_version": 1,
        "status": TaskStatus.NEEDS_INPUT.value,
        "reason_code": error.reason_code,
        "message": error.message,
        "details": dict(error.details or {}),
        "next_action": (
            "Add or select one trusted task descriptor, then rerun with an "
            "explicit source path or target function."
        ),
        "interaction_mode": "non_interactive" if args.non_interactive else "deferred",
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_name(f".{result_path.name}.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    stream = sys.stdout if args.json else sys.stderr
    print(json.dumps(output, indent=2, sort_keys=True), file=stream)
    return _status_exit_code(TaskStatus.NEEDS_INPUT)


def _bundle_verify(args: argparse.Namespace) -> int:
    path = args.bundle.expanduser()
    kind = detect_bundle_kind(path)
    if kind == "kernel":
        bundle = load_and_verify_kernel_bundle(path, expected_digest=args.digest)
        result = {
            "schema_version": 1,
            "status": "verified",
            "bundle_kind": kind,
            "task_id": bundle.task_id,
            "bundle_path": str(bundle.path),
            "bundle_digest": bundle.digest,
            "changed_files": list(bundle.changed_files),
        }
    else:
        e2e_bundle = load_and_verify_e2e_bundle(path, expected_digest=args.digest)
        result = {
            "schema_version": 1,
            "status": "verified",
            "bundle_kind": kind,
            "bundle_id": e2e_bundle.bundle_id,
            "bundle_path": str(e2e_bundle.path),
            "bundle_digest": e2e_bundle.digest,
            "terminal_verified": e2e_bundle.verified,
            "repositories": [item.repository_id for item in e2e_bundle.repositories],
            "derived_image": e2e_bundle.derived_image.reference,
        }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"verified {result['bundle_digest']} ({kind} bundle)")
    return 0


def _bundle_apply(args: argparse.Namespace) -> int:
    if detect_bundle_kind(args.bundle.expanduser()) != "kernel":
        raise ApexError(
            "E2E bundles are applied only by the formal clean-replay verifier",
            "e2e_bundle_apply_unsupported",
        )
    receipt = apply_verified_kernel_bundle(
        args.bundle.expanduser(),
        args.workspace.expanduser().resolve(strict=True),
        expected_digest=args.digest,
    )
    result = {"schema_version": 1, "status": "applied", **receipt.to_dict()}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"applied {receipt.bundle_digest} to {receipt.workspace}")
    return 0


def _e2e(args: argparse.Namespace) -> int:
    spec = E2EOptimizeSpec.from_file(args.spec.expanduser().resolve(strict=True))
    spec = _e2e_overrides(spec, args)
    application = build_application(include_e2e=True)
    if application.e2e_optimizer is None:
        raise ApexError("E2E composition is unavailable", "e2e_not_composed")
    result = application.e2e_optimizer.run(spec)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return _status_exit_code(result.status)


def _kernel_budget_overrides(task: TaskSpec, args: argparse.Namespace) -> TaskSpec:
    values = {
        "max_iterations": args.max_iterations,
        "max_turns": args.max_turns,
        "timeout_seconds": args.timeout_seconds,
    }
    selected = {key: value for key, value in values.items() if value is not None}
    return replace(task, budget=replace(task.budget, **selected)) if selected else task


def _e2e_overrides(spec: E2EOptimizeSpec, args: argparse.Namespace) -> E2EOptimizeSpec:
    values = {
        "results_dir": args.results.expanduser().resolve() if args.results else None,
        "agent_backend": AgentBackendName(args.agent_backend) if args.agent_backend else None,
        "agent_model": args.agent_model,
        "agent_effort": args.agent_effort,
        "max_iterations": args.max_iterations,
        "max_kernels": args.max_kernels,
        "max_turns": args.max_turns,
        "agent_timeout_seconds": args.timeout_seconds,
    }
    selected = {key: value for key, value in values.items() if value is not None}
    return replace(spec, **selected) if selected else spec


def _report(args: argparse.Namespace) -> int:
    result = rebuild_report(
        args.run_root,
        args.output,
        run_id=args.run_id,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"reported {result['run_id']} to {args.output.expanduser().resolve()}")
    return 0


def _export_rl(args: argparse.Namespace) -> int:
    result = export_rl_dataset(
        args.run_root,
        args.output,
        run_id=args.run_id,
        config=DatasetExportConfig(
            split=args.split,
            policy_id=args.policy_id,
            on_incomplete=args.on_incomplete,
            include_sft=not args.no_sft,
        ),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"exported {result['record_count']} episodes "
            f"({result['sft_count']} SFT) to {result['output_dir']}"
        )
    return 0


def _status_exit_code(status: TaskStatus) -> int:
    if status in {TaskStatus.CANDIDATE_READY, TaskStatus.SUCCEEDED, TaskStatus.NO_GAIN}:
        return 0
    if status in {TaskStatus.NEEDS_INPUT, TaskStatus.INVALID_REQUEST, TaskStatus.UNSUPPORTED}:
        return 2
    if status in {TaskStatus.REJECTED, TaskStatus.NO_MEASUREMENT, TaskStatus.VERIFICATION_FAILED}:
        return 3
    if status is TaskStatus.TIMEOUT:
        return 124
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "dependencies":
            return dependencies_main(args.dependency_args)
        if args.command == "bundle" and args.bundle_command == "verify":
            return _bundle_verify(args)
        if args.command == "bundle" and args.bundle_command == "apply":
            return _bundle_apply(args)
        if args.command == "report":
            return _report(args)
        if args.command == "export-rl":
            return _export_rl(args)
        if args.command == "optimize" and args.optimize_command == "kernel":
            return _kernel(args)
        if args.command == "optimize" and args.optimize_command == "e2e":
            return _e2e(args)
        parser.error("unknown command")
    except ApexError as error:
        print(
            json.dumps(
                {"status": "error", "reason_code": error.reason_code, "message": error.message},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    return 2
