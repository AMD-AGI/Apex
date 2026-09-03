"""Parser/dispatch helpers for native sessions and their MCP façade."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from apex.core import AgentBackendName, ApexError
from apex.execution import default_capability_results
from apex.mcp import KernelDraftSessionGrantAuthority
from apex.ports import CodingSessionOutput, CodingSessionRequest, KernelEnhancement


FORMAL_COMMANDS = (
    "bundle",
    "capabilities",
    "dependencies",
    "doctor",
    "export-rl",
    "mcp-server",
    "optimize",
    "report",
    "release",
    "run",
    "showcase",
)


def session_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apex",
        description=(
            "Start a native coding-agent session. Formal optimization remains "
            "available under 'apex optimize'."
        ),
    )
    parser.add_argument("request", nargs="?", help="Optional initial coding request")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--results",
        type=Path,
        help="Artifact root for Apex capabilities (default: a sibling of WORKSPACE)",
    )
    parser.add_argument(
        "--backend",
        choices=[item.value for item in AgentBackendName],
        default=AgentBackendName.CODEX.value,
    )
    parser.add_argument("--model")
    parser.add_argument("--effort")
    parser.add_argument("--print", action="store_true", dest="print_output")
    parser.add_argument("--json", action="store_true", help="Headless JSONL output")
    parser.add_argument("--resume", dest="resume_session")
    parser.add_argument("--continue", action="store_true", dest="resume_latest")
    enhancement = parser.add_mutually_exclusive_group()
    enhancement.add_argument("--plain", action="store_true")
    enhancement.add_argument("--kernel", action="store_true")
    parser.epilog = "Formal commands: " + ", ".join(FORMAL_COMMANDS)
    return parser


def add_capability_commands(commands) -> None:
    doctor = commands.add_parser(
        "doctor", help="Check a coding backend or inspect AMD GPU ownership"
    )
    doctor.add_argument(
        "doctor_target",
        nargs="?",
        choices=("backend", "gpu"),
        default="backend",
        help="Diagnostic target (default: backend)",
    )
    doctor.add_argument(
        "--backend",
        choices=[item.value for item in AgentBackendName],
        default=AgentBackendName.CODEX.value,
    )
    doctor.add_argument("--workspace", type=Path, default=Path.cwd())
    doctor.add_argument(
        "--gpu-devices",
        help="Requested HSA ordinals or GPU UUIDs for 'doctor gpu'",
    )
    doctor.add_argument("--json", action="store_true")
    dependencies = commands.add_parser(
        "dependencies", help="Install or verify pinned dependencies"
    )
    dependencies.add_argument("dependency_args", nargs=argparse.REMAINDER)
    capabilities = commands.add_parser(
        "capabilities", help="Inspect the canonical Apex capability inventory"
    )
    capabilities.add_argument("--workspace", type=Path, default=Path.cwd())
    capabilities.add_argument(
        "--results",
        type=Path,
        help=(
            "Artifact root used to resolve scoped capability availability "
            "(default: a sibling of WORKSPACE)"
        ),
    )
    capabilities.add_argument("--json", action="store_true")
    mcp_server = commands.add_parser("mcp-server", help=argparse.SUPPRESS)
    mcp_server.add_argument("--knowledge-catalog", type=Path)
    mcp_server.add_argument("--workspace", type=Path)
    mcp_server.add_argument("--results", type=Path)
    mcp_server.add_argument(
        "--session-kernel-draft-grants", action="store_true", help=argparse.SUPPRESS
    )


def launch_session(args: argparse.Namespace, build_application) -> int:
    output = (
        CodingSessionOutput.JSONL
        if args.json
        else CodingSessionOutput.TEXT
        if args.print_output
        else CodingSessionOutput.INTERACTIVE
    )
    enhancement = (
        KernelEnhancement.PLAIN
        if args.plain
        else KernelEnhancement.KERNEL
        if args.kernel
        else KernelEnhancement.AUTO
    )
    workspace = args.workspace.expanduser().resolve(strict=True)
    request = CodingSessionRequest(
        workspace=workspace,
        results_dir=(args.results.expanduser().resolve() if args.results else None),
        backend=AgentBackendName(args.backend),
        prompt=args.request,
        model=args.model,
        effort=args.effort,
        output=output,
        enhancement=enhancement,
        resume_session=args.resume_session,
        resume_latest=args.resume_latest,
    )
    application = build_application(
        include_kernel=False,
        include_coding_session=True,
    )
    if application.coding_session is None:
        raise ApexError("Coding session composition is unavailable", "session_not_composed")
    return application.coding_session.launch(request)


def kernel_discovery_is_interactive(args: argparse.Namespace) -> bool:
    """Return whether unresolved natural-language intake may open native UX."""

    return not any(
        (
            args.non_interactive,
            args.json,
            args.dry_run,
            args.result_json is not None,
        )
    )


def launch_kernel_discovery(
    args: argparse.Namespace,
    error: ApexError,
    build_application,
) -> int:
    """Open an explicitly non-formal kernel discovery session."""

    prompt = (
        "Resolve formal kernel intake interactively without running or authorizing "
        "evaluation. Identify the trusted task descriptor and one unambiguous kernel "
        f"target for this request: {args.request}\n"
        f"Intake stopped with: {error.reason_code}."
    )
    request = CodingSessionRequest(
        workspace=args.workspace.expanduser().resolve(strict=True),
        results_dir=args.results.expanduser().resolve(),
        backend=AgentBackendName(
            args.agent_backend or AgentBackendName.CODEX.value
        ),
        prompt=prompt,
        model=args.agent_model,
        effort=args.agent_effort,
        output=CodingSessionOutput.INTERACTIVE,
        enhancement=KernelEnhancement.KERNEL,
    )
    application = build_application(
        include_kernel=False,
        include_coding_session=True,
    )
    if application.coding_session is None:
        raise ApexError("Coding session composition is unavailable", "session_not_composed")
    return application.coding_session.launch(request)


def show_capabilities(args: argparse.Namespace, build_application) -> int:
    workspace = args.workspace.expanduser().resolve(strict=True)
    results = (
        args.results.expanduser().resolve()
        if args.results is not None
        else default_capability_results(workspace)
    )
    application = build_application(
        include_kernel=False,
        include_capabilities=True,
        capability_workspace=workspace,
        capability_results=results,
    )
    if application.capabilities is None:
        raise ApexError("Capability registry is unavailable", "capability_registry_unavailable")
    inventory = [item.to_dict() for item in application.capabilities.inventory()]
    output = {"schema": "apex.capability-inventory/v1", "capabilities": inventory}
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        for item in inventory:
            status = "available" if item["available"] else item["unavailable_reason"]
            print(f"{item['capability_id']}: {status} — {item['summary']}")
    return 0


def show_backend_doctor(args: argparse.Namespace, build_application) -> int:
    application = build_application(
        include_kernel=False,
        include_backend_doctor=True,
    )
    if application.backend_doctor is None:
        raise ApexError("Backend doctor is unavailable", "backend_doctor_unavailable")
    report = application.backend_doctor.inspect(
        AgentBackendName(args.backend),
        workspace=args.workspace.expanduser().resolve(strict=True),
    )
    output = report.to_dict()
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        authentication = (
            "authenticated"
            if output["authenticated"] is True
            else "authentication required"
            if output["authenticated"] is False
            else "authentication unknown"
        )
        print(f"{output['backend']}: {output['status']} ({authentication})")
        for name, feature in output["features"].items():
            status = (
                "available"
                if feature["available"]
                else feature["unavailable_reason"]
            )
            print(f"  {name}: {status}")
    return 0 if report.status == "ready" else 1


def show_gpu_doctor(args: argparse.Namespace, build_application) -> int:
    """Emit a read-only ownership receipt without acquiring a GPU lease."""

    from apex.runtime import resolve_gpu_device_scope

    application = build_application(
        include_kernel=False,
        include_gpu_doctor=True,
    )
    if application.gpu_doctor is None:
        raise ApexError("GPU doctor is unavailable", "gpu_doctor_unavailable")
    selector = resolve_gpu_device_scope(args.gpu_devices)
    receipt = application.gpu_doctor.inspect(selector, allowed_pids=(os.getpid(),))
    ownership = receipt.ownership
    ownership_status = "blocked" if ownership.foreign_owners else "clean"
    gaps = () if receipt.rocm_health is not None else ("rocm_health",)
    output = {
        "schema": "apex.gpu-doctor/v1",
        "status": receipt.status,
        "ownership_status": ownership_status,
        "formal_measurement_ready": receipt.formal_measurement_ready,
        "selector_scope": selector,
        "selected_physical_gpu_uuids": [
            item.unique_id for item in ownership.selected_devices
        ],
        "doctor_receipt_sha256": receipt.digest,
        "doctor_receipt": receipt.to_dict(),
        "evidence_gaps": list(gaps),
    }
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        devices = ", ".join(output["selected_physical_gpu_uuids"])
        print(f"GPU ownership: {ownership_status} ({devices})")
        readiness = "ready" if receipt.formal_measurement_ready else receipt.status
        print(f"Formal measurement readiness: {readiness}")
        for gap in gaps:
            print(f"  not captured: {gap}")
    return 0 if receipt.formal_measurement_ready else 1


def serve_mcp(args: argparse.Namespace, build_application) -> int:
    from apex.mcp import run_stdio_server

    catalog = (
        args.knowledge_catalog.expanduser().resolve(strict=True)
        if args.knowledge_catalog is not None
        else None
    )
    application = build_application(
        include_kernel=False,
        include_capabilities=True,
        knowledge_catalog=catalog,
        capability_workspace=(
            args.workspace.expanduser().resolve(strict=True)
            if args.workspace is not None
            else None
        ),
        capability_results=(
            args.results.expanduser().resolve()
            if args.results is not None
            else None
        ),
    )
    if application.capabilities is None:
        raise ApexError("Capability registry is unavailable", "capability_registry_unavailable")
    authority = (
        KernelDraftSessionGrantAuthority()
        if args.session_kernel_draft_grants
        else None
    )
    run_stdio_server(application.capabilities, grant_authority=authority)
    return 0


__all__ = [
    "FORMAL_COMMANDS",
    "add_capability_commands",
    "launch_session",
    "kernel_discovery_is_interactive",
    "launch_kernel_discovery",
    "serve_mcp",
    "session_parser",
    "show_capabilities",
    "show_backend_doctor",
    "show_gpu_doctor",
]
