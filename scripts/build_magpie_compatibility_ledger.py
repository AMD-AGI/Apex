#!/usr/bin/env python3
"""Regenerate the config-compatibility ledger from exact pinned dependencies."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from apex.benchmark import MagpieConfigContract, MagpieMainConfigAdapter
from apex.runtime import (
    DependencyReceipt,
    MagpieCompatibilityEntry,
    build_magpie_compatibility_ledger,
    load_magpie_compatibility_ledger,
    load_magpie_corpus_manifest,
    verify_magpie_compatibility_ledger,
    verify_runtime_dependencies,
)
from apex.runtime.dependencies import load_lock
from apex.runtime.repositories import inspect_repository


def _arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--apex-root", type=Path, default=root)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "scripts" / "magpie_compatibility_ledger.json",
    )
    parser.add_argument(
        "--magpie-root",
        type=Path,
        help="Exact published Magpie checkout used while rotating the dependency pin",
    )
    return parser.parse_args(argv)


def _entry(path: str, resolved: MagpieConfigContract) -> MagpieCompatibilityEntry:
    plan = resolved.plan
    capability = resolved.capability_receipt
    identity = plan["identity"]
    source_runtime = plan["source_runtime"]
    framework = str(identity["framework"])
    run_mode = str(identity["run_mode"])
    image = source_runtime.get("requested_image")
    image_text = image.strip() if isinstance(image, str) and image.strip() else None
    return MagpieCompatibilityEntry(
        path=path,
        config_sha256=resolved.config_sha256,
        framework=framework,
        run_mode=run_mode,
        precision=str(identity["precision"]),
        lifecycle=str(plan["lifecycle"]),
        image_status=_image_status(run_mode, image_text),
        model_identity_sha256=str(identity["model_sha256"]),
        compatibility_status=str(capability["status"]),
    )


def _image_status(run_mode: str, image: str | None) -> str:
    if run_mode != "docker":
        return "not_applicable"
    if image is None:
        return "runtime_selection_required"
    if image.startswith("sha256:") or "@sha256:" in image:
        return "immutable"
    return "mutable_locator"


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"refusing output symlink: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _rotation_receipt(
    apex_root: Path, magpie_root: Path
) -> DependencyReceipt:
    lock = load_lock(apex_root / "scripts" / "dependencies.lock.json")
    dependency = next(item for item in lock.dependencies if item.key == "magpie")
    selected = magpie_root.expanduser().resolve(strict=True)
    state = inspect_repository(selected)
    if state.commit != dependency.commit or state.dirty_paths:
        raise RuntimeError("Magpie checkout differs from the exact clean dependency pin")
    return DependencyReceipt(
        schema=lock.receipt_schema,
        lock_sha256=lock.sha256,
        python=apex_root / ".venv" / "bin" / "python",
        roots={"magpie": selected},
        commits={"magpie": dependency.commit},
        raw={},
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    root = args.apex_root.expanduser().resolve(strict=True)
    receipt = (
        _rotation_receipt(root, args.magpie_root)
        if args.magpie_root is not None
        else verify_runtime_dependencies(apex_root=root)
    )
    lock = load_lock(root / "scripts" / "dependencies.lock.json")
    corpus = load_magpie_corpus_manifest(lock.magpie_corpus_manifest)
    resolver = MagpieMainConfigAdapter(receipt)
    entries: list[MagpieCompatibilityEntry] = []
    for item in corpus.files:
        resolved = resolver.resolve(receipt.root("magpie") / item.path)
        if resolved.config_sha256 != item.sha256:
            raise RuntimeError(f"Magpie resolver digest differs for {item.path}")
        entries.append(_entry(item.path, resolved))
    ledger = build_magpie_compatibility_ledger(
        magpie_commit=corpus.commit,
        benchmark_tree=corpus.benchmark_tree,
        corpus_manifest_sha256=corpus.manifest_sha256,
        entries=entries,
    )
    _write(args.output, ledger.to_dict())
    loaded = load_magpie_compatibility_ledger(args.output.expanduser().resolve())
    verify_magpie_compatibility_ledger(loaded, corpus)
    print(json.dumps(loaded.payload()["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
