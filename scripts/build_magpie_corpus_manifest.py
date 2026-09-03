#!/usr/bin/env python3
"""Build or check the frozen Magpie benchmark-corpus manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from apex.runtime.magpie_corpus import build_magpie_corpus_manifest
from apex.runtime.repositories import BootstrapError, canonical_repository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magpie-root", required=True, type=Path)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).resolve().parent / "dependencies.lock.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "magpie_corpus_manifest.json",
    )
    parser.add_argument("--check", action="store_true")
    return parser


def _locked_identity(path: Path) -> tuple[str, str]:
    try:
        raw = json.loads(path.read_bytes())
        magpie = raw["dependencies"]["magpie"]
        repository = magpie["repository"]
        commit = magpie["commit"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise BootstrapError(f"cannot read Magpie dependency lock: {error}") from error
    if not isinstance(repository, str) or not isinstance(commit, str):
        raise BootstrapError("Magpie dependency lock identity is invalid")
    return repository, commit


def _content(root: Path, lock: Path) -> bytes:
    repository, commit = _locked_identity(lock)
    manifest = build_magpie_corpus_manifest(root.expanduser().resolve())
    if (
        canonical_repository(manifest.repository) != canonical_repository(repository)
        or manifest.commit != commit
    ):
        raise BootstrapError("Magpie checkout differs from dependency lock")
    return json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode("utf-8") + b"\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        content = _content(args.magpie_root, args.lock.expanduser().resolve())
        output = args.output.expanduser().resolve()
        if args.check:
            if not output.is_file() or output.is_symlink() or output.read_bytes() != content:
                raise BootstrapError("Magpie corpus manifest is stale")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)
        return 0
    except BootstrapError as error:
        print(f"Magpie corpus manifest failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
