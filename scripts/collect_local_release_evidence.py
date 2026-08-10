#!/usr/bin/env python3
"""Run the fixed local release gates and create one typed evidence document."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from apex.core import ApexError, canonical_json_bytes
from apex.runtime import collect_local_release_evidence
from apex.runtime.repositories import BootstrapError


ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect clean-tree dependency, complete CPU/static, and installed-CLI "
            "release evidence. This does not fetch remotes or run GPU/live gates."
        )
    )
    parser.add_argument("--apex-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _write_new(path: Path, content: bytes) -> None:
    selected = path.expanduser()
    if not selected.is_absolute():
        raise ValueError("--output must be absolute")
    parent = selected.parent.resolve(strict=True)
    if parent != selected.parent or not parent.is_dir():
        raise ValueError("--output parent cannot traverse symlinks")
    try:
        with selected.open("xb") as target:
            target.write(content)
    except FileExistsError as error:
        raise ValueError(f"output already exists: {selected}") from error


def main() -> int:
    args = _parser().parse_args()
    try:
        evidence = collect_local_release_evidence(args.apex_root)
        _write_new(args.output, canonical_json_bytes(evidence.to_dict()) + b"\n")
    except (ApexError, BootstrapError, OSError, ValueError) as error:
        print(f"local release evidence error: {error}", file=sys.stderr)
        return 2
    print(f"local_release_evidence={args.output.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
