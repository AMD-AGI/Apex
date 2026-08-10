"""Release baseline/readiness inspection and explicit local evidence collection."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from apex.bootstrap import build_qualification_artifact_authority
from apex.core import ApexError, canonical_json_bytes
from apex.reporting import verify_showcase
from apex.runtime import (
    ReleaseEvidence,
    QualificationEvidence,
    ShowcaseEvidence,
    WorkspaceGitIdentityResolver,
    build_showcase_evidence,
    collect_local_release_evidence,
    freeze_campaign_baseline,
    freeze_release_candidate,
    inspect_release_candidate,
    verify_release_candidate_receipt,
)
from apex.runtime.repositories import BootstrapError


def add_release_commands(commands) -> None:
    release = commands.add_parser(
        "release", help="Inspect or verify the campaign baseline and release gate"
    )
    release_commands = release.add_subparsers(dest="release_command", required=True)
    check = release_commands.add_parser(
        "check", help="Rebuild readiness from current source and typed evidence"
    )
    check.add_argument("--apex-root", type=Path, default=Path.cwd())
    inputs = check.add_mutually_exclusive_group()
    inputs.add_argument("--evidence", type=Path)
    inputs.add_argument("--receipt", type=Path)
    requirement = check.add_mutually_exclusive_group()
    requirement.add_argument("--require-baseline", action="store_true")
    requirement.add_argument("--require-ready", action="store_true")
    check.add_argument("--json", action="store_true")
    check.add_argument(
        "--qualification-artifact-root",
        type=Path,
        help=(
            "Existing external formal result root; only installed artifact "
            "verifiers can authorize qualification claims"
        ),
    )
    collect = release_commands.add_parser(
        "collect-local",
        help="Run the fixed dependency, CPU/static, and installed-CLI gates",
    )
    collect.add_argument("--apex-root", type=Path, default=Path.cwd())
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--json", action="store_true")
    qualifications = release_commands.add_parser(
        "collect-qualifications",
        help="Inspect evaluator-owned qualification manifests and CAS read-only",
    )
    qualifications.add_argument("--apex-root", type=Path, default=Path.cwd())
    qualifications.add_argument("--artifact-root", type=Path, required=True)
    qualifications.add_argument("--output", type=Path, required=True)
    qualifications.add_argument("--json", action="store_true")
    showcase = release_commands.add_parser(
        "collect-showcase",
        help="Verify one showcase tree and emit path-free release evidence",
    )
    showcase.add_argument("--apex-root", type=Path, default=Path.cwd())
    showcase.add_argument("--path", type=Path, required=True)
    showcase.add_argument("--output", type=Path, required=True)
    showcase.add_argument("--json", action="store_true")
    join = release_commands.add_parser(
        "join-evidence",
        help="Join typed claims; qualification authority is still required for ready",
    )
    join.add_argument("--base", type=Path, required=True)
    join.add_argument("--qualification", type=Path, action="append", default=[])
    join.add_argument("--showcase", type=Path, action="append", default=[])
    join.add_argument("--output", type=Path, required=True)
    join.add_argument("--json", action="store_true")


def run_release_command(args: argparse.Namespace) -> int:
    if args.release_command == "collect-local":
        return _collect_local(args)
    if args.release_command == "collect-showcase":
        return _collect_showcase(args)
    if args.release_command == "collect-qualifications":
        return _collect_qualifications(args)
    if args.release_command == "join-evidence":
        return _join_evidence(args)
    if args.release_command != "check":
        raise ApexError("Unknown release command", "release_command_unknown")
    root = _root(args.apex_root)
    authority = (
        build_qualification_artifact_authority(
            apex_root=root,
            artifact_root=args.qualification_artifact_root,
        )
        if args.qualification_artifact_root is not None
        else None
    )
    try:
        receipt = (
            verify_release_candidate_receipt(
                _mapping(args.receipt, "release receipt"),
                apex_root=root,
                qualification_authority=authority,
            )
            if args.receipt is not None
            else inspect_release_candidate(
                root,
                ReleaseEvidence.from_dict(_mapping(args.evidence, "release evidence"))
                if args.evidence is not None
                else ReleaseEvidence(),
                qualification_authority=authority,
            )
        )
        if args.require_ready:
            receipt = freeze_release_candidate(
                receipt.to_dict(),
                apex_root=root,
                qualification_authority=authority,
            )
        elif args.require_baseline:
            receipt = freeze_campaign_baseline(
                receipt.to_dict(),
                apex_root=root,
                qualification_authority=authority,
            )
    except BootstrapError as error:
        raise ApexError(str(error), "release_identity_invalid") from error
    _print(receipt.to_dict(), json_output=args.json)
    return 0


def _collect_local(args: argparse.Namespace) -> int:
    root = _root(args.apex_root)
    evidence = collect_local_release_evidence(root)
    output = _write_new(args.output, evidence.to_dict())
    if args.json:
        print(json.dumps(evidence.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"local_release_evidence={output}")
    return 0


def _collect_showcase(args: argparse.Namespace) -> int:
    root = _root(args.apex_root)
    identity = WorkspaceGitIdentityResolver().inspect(root)
    if (
        not identity.resolved
        or Path(identity.root) != root
        or identity.tree is None
        or identity.dirty_paths
    ):
        raise ApexError(
            "Showcase release evidence requires the exact clean Apex checkout",
            "release_showcase_source_invalid",
        )
    verification = verify_showcase(args.path)
    evidence = build_showcase_evidence(
        apex_tree=identity.tree,
        verifier_receipt=verification.to_receipt(),
    )
    output = _write_new(args.output, evidence.to_dict())
    if args.json:
        print(json.dumps(evidence.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"showcase_release_evidence={output}")
    return 0


def _collect_qualifications(args: argparse.Namespace) -> int:
    root = _root(args.apex_root)
    authority = build_qualification_artifact_authority(
        apex_root=root,
        artifact_root=args.artifact_root,
    )
    collection = authority.collect()
    output = _write_new(args.output, collection.to_dict())
    if args.json:
        print(json.dumps(collection.to_dict(), indent=2, sort_keys=True))
    else:
        verified = sum(item.status == "verified" for item in collection.entries)
        print(f"qualification_artifacts={output} verified={verified}")
    return 0


def _join_evidence(args: argparse.Namespace) -> int:
    """Join typed inputs without promoting self-asserted qualification claims."""
    if not args.qualification and not args.showcase:
        raise ApexError(
            "Evidence join requires at least one typed fragment",
            "release_evidence_fragment_required",
        )
    base = ReleaseEvidence.from_dict(_mapping(args.base, "base release evidence"))
    qualifications = tuple(
        QualificationEvidence.from_dict(_mapping(path, "qualification evidence"))
        for path in args.qualification
    )
    showcases = tuple(
        ShowcaseEvidence.from_dict(_mapping(path, "showcase evidence"))
        for path in args.showcase
    )
    joined = replace(
        base,
        qualifications=tuple(sorted(
            (*base.qualifications, *qualifications),
            key=lambda item: item.qualification_id,
        )),
        showcases=tuple(sorted(
            (*base.showcases, *showcases),
            key=lambda item: item.showcase_id,
        )),
    )
    output = _write_new(args.output, joined.to_dict())
    if args.json:
        print(json.dumps(joined.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"joined_release_evidence={output}")
    return 0


def _write_new(path: Path, value: Mapping[str, Any]) -> Path:
    selected = path.expanduser()
    if not selected.is_absolute() or selected.name in {"", ".", ".."}:
        raise ApexError(
            "Release evidence output must be an absolute file path",
            "invalid_release_output",
        )
    try:
        parent = selected.parent.resolve(strict=True)
    except OSError as error:
        raise ApexError(
            "Release evidence output parent does not exist",
            "release_output_parent_missing",
        ) from error
    if parent != selected.parent or not parent.is_dir():
        raise ApexError(
            "Release evidence output parent cannot traverse symlinks",
            "unsafe_release_output",
        )
    output = parent / selected.name
    try:
        with output.open("xb") as target:
            target.write(canonical_json_bytes(value) + b"\n")
    except FileExistsError as error:
        raise ApexError(
            "Release evidence output already exists", "release_output_exists"
        ) from error
    return output


def require_campaign_baseline(path: Path | None):
    """Rebuild one supplied receipt against the installed Apex source checkout."""

    if path is None:
        raise ApexError(
            "Live optimization requires --release-candidate-receipt",
            "campaign_baseline_receipt_required",
        )
    try:
        return freeze_campaign_baseline(
            _mapping(path, "release receipt"),
            apex_root=_source_root(),
        )
    except BootstrapError as error:
        raise ApexError(str(error), "release_identity_invalid") from error


def _root(value: Path) -> Path:
    selected = value.expanduser()
    if selected.is_symlink():
        raise ApexError("Apex root cannot be a symlink", "unsafe_release_root")
    try:
        root = selected.resolve(strict=True)
    except OSError as error:
        raise ApexError("Apex root does not exist", "release_root_missing") from error
    if not root.is_dir():
        raise ApexError("Apex root must be a directory", "invalid_release_root")
    return root


def _source_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _mapping(path: Path, field: str) -> Mapping[str, Any]:
    selected = path.expanduser()
    if selected.is_symlink():
        raise ApexError(f"{field} cannot be a symlink", "unsafe_release_evidence")
    try:
        resolved = selected.resolve(strict=True)
        if not resolved.is_file():
            raise OSError("not a file")
        value = json.loads(resolved.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ApexError(f"Invalid {field}: {error}", "invalid_release_evidence") from error
    if not isinstance(value, Mapping):
        raise ApexError(f"{field} root must be an object", "invalid_release_evidence")
    return value


def _print(value: Mapping[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    print(
        f"baseline={value['baseline_status']} release={value['status']} "
        f"receipt={value['receipt_sha256']}"
    )
    for field in ("baseline_blockers", "blockers"):
        items = value[field]
        if items:
            print(f"{field}=" + ",".join(str(item) for item in items))


__all__ = [
    "add_release_commands", "require_campaign_baseline", "run_release_command",
]
