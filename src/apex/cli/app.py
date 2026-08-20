"""Parse and dispatch Apex commands without owning domain state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from apex.bootstrap import build_application
from apex.core import AgentBackendName, ApexError, TaskStatus
from apex.delivery import (
    apply_verified_kernel_bundle,
    detect_bundle_kind,
    load_and_verify_kernel_bundle,
)
from apex.evaluation import (
    EvaluationContractReceipt,
    reviewed_template_evaluation_authorizer,
    user_confirmed_evaluation_authorizer,
)
from apex.intake import (
    E2EOptimizeSpec,
    NaturalLanguageRequest,
    NaturalLanguageTaskResolver,
    TaskSpec,
    load_kernel_template,
)
from apex.optimization.kernel import KernelOptimizeRequest
from apex.optimization.e2e import write_preflight_result
from apex.runtime.dependencies import main as dependencies_main

from .formal import (
    formal_result_path as _formal_result_path,
    formal_results_root as _formal_results_root,
    kernel_budget_overrides as _kernel_budget_overrides,
    regular_e2e_config as _regular_e2e_config,
    status_exit_code as _status_exit_code,
)
from .kernel_handoff import run_kernel_campaign_handoff
from .projection_commands import (
    export_rl_command as _export_rl,
    report_command as _report,
)
from .release import add_release_commands, require_campaign_baseline, run_release_command
from .recovery import run_resume
from .showcase import add_showcase_commands, run_showcase_command
from .session import (
    FORMAL_COMMANDS as _FORMAL_COMMANDS,
    add_capability_commands as _add_capability_commands,
    kernel_discovery_is_interactive,
    launch_kernel_discovery,
    launch_session,
    serve_mcp,
    session_parser as _session_parser,
    show_backend_doctor,
    show_capabilities,
    show_gpu_doctor,
)
_NEEDS_INPUT_REASONS = {
    "task_descriptor_missing",
    "target_not_resolved",
    "ambiguous_kernel_target",
}
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apex", description="Evidence-driven AMD GPU kernel optimization environment",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    optimize = commands.add_parser("optimize", help="Run an optimization use case")
    optimize_commands = optimize.add_subparsers(dest="optimize_command", required=True)
    kernel = optimize_commands.add_parser("kernel", help="Optimize one existing kernel")
    kernel.add_argument("request", nargs="?", help="Natural-language task (with discovery flags)")
    kernel.add_argument("--task-spec", type=Path, help="Caller-neutral TaskSpec JSON/YAML")
    kernel.add_argument("--campaign", type=Path, help="Chat-created campaign draft")
    kernel.add_argument(
        "--template",
        type=Path,
        help="Reviewed attributed template directory (mutually exclusive with --task-spec)",
    )
    kernel.add_argument("--workspace", type=Path, help="Workspace for a natural-language task")
    kernel.add_argument("--results", type=Path, help="Run output directory for a natural-language task")
    kernel.add_argument("--result-json", type=Path, help="Atomic machine result path")
    kernel.add_argument(
        "--backend",
        dest="agent_backend",
        choices=[item.value for item in AgentBackendName],
    )
    kernel.add_argument("--model", dest="agent_model")
    kernel.add_argument("--effort", dest="agent_effort")
    kernel.add_argument("--max-iterations", type=int)
    kernel.add_argument("--max-turns", type=int)
    kernel.add_argument("--timeout-seconds", type=int)
    kernel.add_argument("--non-interactive", action="store_true")
    kernel.add_argument("--dry-run", action="store_true", help="Resolve and persist the task only")
    kernel.add_argument("--release-candidate-receipt", type=Path)
    kernel.add_argument(
        "--evaluation-contract-draft-digest",
        help="Explicitly confirm the exact draft digest emitted by --dry-run",
    )
    kernel.add_argument("--json", action="store_true", help="Emit a stable JSON envelope")
    _add_e2e_parser(optimize_commands)

    run = commands.add_parser("run", help="Recover or inspect a canonical run")
    run_commands = run.add_subparsers(dest="run_command", required=True)
    resume = run_commands.add_parser(
        "resume", help="Resume an interrupted E2E run from durable state"
    )
    resume.add_argument("--run", type=Path, required=True, help="Existing run root")
    resume.add_argument("--release-candidate-receipt", type=Path)
    bundle = commands.add_parser("bundle", help="Inspect or verify source bundles")
    bundle_commands = bundle.add_subparsers(dest="bundle_command", required=True)
    verify = bundle_commands.add_parser("verify", help="Independently verify a source bundle")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--results", type=Path, help="New absolute E2E evidence directory")
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
    _add_capability_commands(commands)
    add_release_commands(commands)
    add_showcase_commands(commands)
    return parser


def _add_e2e_parser(optimize_commands) -> None:
    e2e = optimize_commands.add_parser(
        "e2e", help="Optimize kernels in one E2E workload"
    )
    e2e.add_argument(
        "--config", type=Path, required=True, help="Raw Magpie benchmark YAML"
    )
    e2e.add_argument(
        "--results", type=Path, required=True, help="New run output directory"
    )
    e2e.add_argument(
        "--backend",
        dest="agent_backend",
        choices=[item.value for item in AgentBackendName],
        default=AgentBackendName.CODEX.value,
    )
    e2e.add_argument("--model", dest="agent_model")
    e2e.add_argument("--effort", dest="agent_effort")
    e2e.add_argument("--gpu-arch", default="gfx950")
    e2e.add_argument("--gpu-devices")
    e2e.add_argument("--hf-cache-path", type=Path)
    e2e.add_argument("--hf-offline", action="store_true")
    e2e.add_argument("--max-iterations", type=int)
    e2e.add_argument("--max-kernels", type=int)
    e2e.add_argument("--max-turns", type=int)
    e2e.add_argument("--timeout-seconds", type=int)
    e2e.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config and capability composition without acquiring a GPU",
    )
    e2e.add_argument("--release-candidate-receipt", type=Path)


def _session(args: argparse.Namespace) -> int:
    return launch_session(args, build_application)


def _capabilities(args: argparse.Namespace) -> int:
    return show_capabilities(args, build_application)


def _doctor(args: argparse.Namespace) -> int:
    if args.doctor_target == "gpu":
        return show_gpu_doctor(args, build_application)
    return show_backend_doctor(args, build_application)


def _mcp_server(args: argparse.Namespace) -> int:
    return serve_mcp(args, build_application)


def _kernel(args: argparse.Namespace) -> int:
    if args.campaign is not None:
        return run_kernel_campaign_handoff(args, build_application)
    if args.template is not None:
        return _kernel_template(args)
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
            if kernel_discovery_is_interactive(args):
                return launch_kernel_discovery(args, error, build_application)
            result_path = _natural_language_result_path(args)
            return _write_needs_input(error, result_path, args=args)
        task = resolved.task
    task = _kernel_budget_overrides(task, args)
    if not args.dry_run:
        _formal_results_root(task.results_dir, workspace=task.workspace)
    result_path = (
        _formal_result_path(args.result_json, task.results_dir)
        if args.result_json is not None
        else task.results_dir / "result.json"
    )
    authorizer = (
        user_confirmed_evaluation_authorizer(
            args.evaluation_contract_draft_digest
        )
        if args.evaluation_contract_draft_digest is not None
        else None
    )
    optimizer = build_application(
        kernel_evaluation_authorizer=authorizer
    ).kernel_optimizer
    if optimizer is None:
        raise ApexError("Kernel optimizer composition is unavailable", "kernel_not_composed")
    if args.dry_run:
        contract = optimizer.preview_evaluation_contract(task)
        return _write_resolved_task(task, result_path, contract)
    request = KernelOptimizeRequest(
        task=task,
        result_json=result_path,
        backend_override=AgentBackendName(args.agent_backend) if args.agent_backend else None,
        model_override=args.agent_model,
        effort_override=args.agent_effort,
        campaign_baseline=require_campaign_baseline(args.release_candidate_receipt),
    )
    result = optimizer.run(request)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return _status_exit_code(result.status)


def _kernel_template(args: argparse.Namespace) -> int:
    if (
        args.task_spec is not None
        or args.workspace is not None
        or args.evaluation_contract_draft_digest is not None
        or not args.request
    ):
        raise ApexError(
            "Template optimization requires one natural-language request and forbids --task-spec/--workspace",
            "kernel_template_input_invalid",
        )
    if args.results is None:
        raise ApexError(
            "Template optimization requires --results",
            "kernel_template_results_required",
        )
    template = load_kernel_template(args.template)
    template.require_materializable()
    if not args.dry_run:
        _formal_results_root(args.results, workspace=template.root)
    materializer = build_application(
        include_kernel=False,
        include_kernel_templates=True,
    ).kernel_template_materializer
    if materializer is None:
        raise ApexError(
            "Reviewed template materialization is not composed",
            "template_materializer_unavailable",
        )
    materialized = materializer.materialize(
        template,
        results_dir=args.results.expanduser().absolute(),
        instructions=args.request,
        backend=AgentBackendName(
            args.agent_backend or AgentBackendName.CODEX.value
        ),
        model=args.agent_model,
        effort=args.agent_effort,
    )
    task = _kernel_budget_overrides(materialized.task, args)
    result_path = (
        _formal_result_path(args.result_json, task.results_dir)
        if args.result_json is not None
        else task.results_dir / "result.json"
    )
    optimizer = build_application(
        kernel_evaluation_authorizer=reviewed_template_evaluation_authorizer(task)
    ).kernel_optimizer
    if optimizer is None:
        raise ApexError("Kernel optimizer composition is unavailable", "kernel_not_composed")
    if args.dry_run:
        return _write_resolved_task(
            task, result_path, optimizer.preview_evaluation_contract(task)
        )
    request = KernelOptimizeRequest(
        task=task,
        result_json=result_path,
        backend_override=(
            AgentBackendName(args.agent_backend) if args.agent_backend else None
        ),
        model_override=args.agent_model,
        effort_override=args.agent_effort,
        campaign_baseline=require_campaign_baseline(args.release_candidate_receipt),
    )
    result = optimizer.run(request)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return _status_exit_code(result.status)


def _write_resolved_task(
    task: TaskSpec,
    result_path: Path,
    contract: EvaluationContractReceipt,
) -> int:
    output = {
        "schema_version": 1,
        "status": "evaluation_contract_preview",
        "resolution_hash": contract.draft.resolution_hash,
        "evaluation_contract_draft_digest": contract.draft.digest,
        "evaluation_contract_receipt_digest": contract.digest,
        "evaluation_contract": contract.to_dict(),
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
        return _verify_kernel_bundle(path, args)
    return _verify_e2e_bundle(path, args)


def _verify_kernel_bundle(path: Path, args: argparse.Namespace) -> int:
    if args.results is not None:
        raise ApexError(
            "--results is used only for E2E bundle verification",
            "kernel_bundle_results_unsupported",
        )
    bundle = load_and_verify_kernel_bundle(path, expected_digest=args.digest)
    result = {
        "schema_version": 1,
        "status": "verified",
        "bundle_kind": "kernel",
        "task_id": bundle.task_id,
        "bundle_path": str(bundle.path),
        "bundle_digest": bundle.digest,
        "changed_files": list(bundle.changed_files),
    }
    _print_bundle_result(result, json_output=args.json)
    return 0


def _verify_e2e_bundle(path: Path, args: argparse.Namespace) -> int:
    results_dir = _e2e_verification_results(args.results)
    application = build_application(include_e2e_verifier=True)
    verifier = application.e2e_bundle_verifier
    if verifier is None:
        raise ApexError(
            "E2E bundle verification composition is unavailable",
            "e2e_bundle_verifier_not_composed",
        )
    outcome = verifier.verify(
        bundle_dir=path.resolve(strict=True),
        results_dir=results_dir,
        expected_digest=args.digest,
    )
    verified = outcome.verified_bundle
    result = {
        **outcome.result.to_dict(),
        "bundle_kind": "e2e",
        "input_bundle_path": str(path.resolve(strict=True)),
        "verification_result_path": str(outcome.result_path),
        "verified_bundle_path": str(verified.path) if verified else None,
        "verified_bundle_digest": verified.digest if verified else None,
    }
    _print_bundle_result(result, json_output=args.json)
    return _status_exit_code(outcome.result.status)


def _e2e_verification_results(value: Path | None) -> Path:
    if value is None:
        raise ApexError(
            "E2E bundle verification requires --results",
            "e2e_verification_results_required",
        )
    expanded = value.expanduser()
    if not expanded.is_absolute():
        raise ApexError(
            "E2E bundle verification --results must be absolute",
            "invalid_bundle_path",
        )
    return _formal_results_root(expanded)


def _print_bundle_result(result: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    status = "verified" if result.get("verified") is True else result["status"]
    reason = result.get("reason_code")
    detail = f"; {reason}" if reason and status != "verified" else ""
    print(f"{status} {result['bundle_digest']} ({result['bundle_kind']} bundle{detail})")


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
    spec = _e2e_spec(
        args, None if args.dry_run else require_campaign_baseline(args.release_candidate_receipt)
    )
    application = build_application(include_e2e=True)
    if application.e2e_optimizer is None:
        raise ApexError("E2E composition is unavailable", "e2e_not_composed")
    if args.dry_run:
        preview = application.e2e_optimizer.preview(spec)
        output = preview.to_dict()
        output["result_path"] = str(
            write_preflight_result(preview, spec.results_dir)
        )
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0
    result = application.e2e_optimizer.run(spec)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return _status_exit_code(result.status)


def _e2e_spec(args: argparse.Namespace, campaign_baseline=None) -> E2EOptimizeSpec:
    config = _regular_e2e_config(args.config)
    results = args.results.expanduser()
    if args.dry_run:
        if results.is_symlink():
            raise ApexError("E2E results cannot be a symlink", "unsafe_e2e_results")
        results = results.resolve()
    else:
        results = _formal_results_root(results)
    hints: dict[str, object] = {}
    if args.gpu_devices is not None:
        hints["gpu_devices"] = args.gpu_devices
    if args.hf_cache_path is not None:
        hints["hf_cache_path"] = str(
            args.hf_cache_path.expanduser().resolve(strict=True)
        )
    if args.hf_offline:
        hints["hf_offline"] = True
    values = {
        "schema_version": 1,
        "config_path": str(config),
        "results_dir": str(results),
        "agent_backend": args.agent_backend,
        "agent_model": args.agent_model,
        "agent_effort": args.agent_effort,
        "gpu_arch": args.gpu_arch,
        "deployment_hints": hints,
        "max_iterations": args.max_iterations,
        "max_kernels": args.max_kernels,
        "max_turns": args.max_turns,
        "agent_timeout_seconds": args.timeout_seconds,
        "campaign_baseline_receipt": campaign_baseline.to_dict() if campaign_baseline else None,
    }
    return E2EOptimizeSpec.from_mapping(
        {key: value for key, value in values.items() if value is not None}
    )


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    try:
        if not values or values[0] not in _FORMAL_COMMANDS:
            return _session(_session_parser().parse_args(values))
        parser = _parser()
        args = parser.parse_args(values)
        if args.command == "dependencies":
            return dependencies_main(args.dependency_args)
        if args.command == "capabilities":
            return _capabilities(args)
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "mcp-server":
            return _mcp_server(args)
        if args.command == "bundle" and args.bundle_command == "verify":
            return _bundle_verify(args)
        if args.command == "bundle" and args.bundle_command == "apply":
            return _bundle_apply(args)
        if args.command == "report":
            return _report(args)
        if args.command == "export-rl":
            return _export_rl(args)
        if args.command == "showcase":
            return run_showcase_command(args)
        if args.command == "release":
            return run_release_command(args)
        if args.command == "optimize" and args.optimize_command == "kernel":
            return _kernel(args)
        if args.command == "optimize" and args.optimize_command == "e2e":
            return _e2e(args)
        if args.command == "run" and args.run_command == "resume":
            return run_resume(args, build_application)
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
