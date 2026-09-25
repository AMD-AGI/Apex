#!/usr/bin/env python3
"""Join or verify a path-free release-candidate receipt without gate execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from apex.core import ApexError, canonical_json_bytes
from apex.runtime import (
    ReleaseEvidence,
    freeze_campaign_baseline,
    freeze_release_candidate,
    inspect_release_candidate,
    verify_release_candidate_receipt,
)
from apex.runtime.repositories import BootstrapError


ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Join/verify typed evidence into an Apex release-candidate receipt."
    )
    parser.add_argument("--apex-root", type=Path, default=ROOT)
    parser.add_argument(
        "--evidence",
        type=Path,
        help="JSON containing the apex.release evidence object; omissions stay blocked.",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="Verify an existing receipt instead of building one.",
    )
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--require-baseline", action="store_true")
    parser.add_argument("--output", type=Path, help="Create this new file; never overwrite.")
    return parser


def _mapping(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {field} JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} JSON root must be an object")
    return value


def _build(args: argparse.Namespace) -> Mapping[str, Any]:
    evidence = (
        ReleaseEvidence.from_dict(_mapping(args.evidence, "release evidence"))
        if args.evidence
        else ReleaseEvidence()
    )
    receipt = inspect_release_candidate(args.apex_root, evidence)
    if args.require_ready:
        receipt = freeze_release_candidate(receipt.to_dict(), apex_root=args.apex_root)
    elif args.require_baseline:
        receipt = freeze_campaign_baseline(receipt.to_dict(), apex_root=args.apex_root)
    return receipt.to_dict()


def _verify(args: argparse.Namespace) -> Mapping[str, Any]:
    receipt = verify_release_candidate_receipt(
        _mapping(args.verify, "release receipt"),
        apex_root=args.apex_root,
    )
    if args.require_ready:
        receipt = freeze_release_candidate(receipt.to_dict(), apex_root=args.apex_root)
    elif args.require_baseline:
        receipt = freeze_campaign_baseline(receipt.to_dict(), apex_root=args.apex_root)
    return receipt.to_dict()


def _emit(value: Mapping[str, Any], output: Path | None) -> None:
    content = canonical_json_bytes(value) + b"\n"
    if output is None:
        sys.stdout.buffer.write(content)
        return
    selected = output.expanduser()
    selected.parent.mkdir(parents=True, exist_ok=True)
    try:
        with selected.open("xb") as target:
            target.write(content)
    except FileExistsError as error:
        raise ValueError(f"output already exists: {selected}") from error


def main() -> int:
    args = _parser().parse_args()
    if args.evidence and args.verify:
        _parser().error("--evidence and --verify are mutually exclusive")
    if args.require_ready and args.require_baseline:
        _parser().error("--require-ready and --require-baseline are mutually exclusive")
    try:
        value = _verify(args) if args.verify else _build(args)
        _emit(value, args.output)
    except (ApexError, BootstrapError, OSError, ValueError) as error:
        print(f"release candidate error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
