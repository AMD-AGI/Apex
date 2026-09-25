#!/usr/bin/env python3
"""Run the production E2E CPU preview over the exact pinned Magpie corpus."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from apex.bootstrap import build_application
from apex.core import ApexError, canonical_json_bytes, sha256_json
from apex.intake import E2EOptimizeSpec
from apex.runtime import (
    load_magpie_corpus_manifest,
    verify_magpie_corpus_manifest,
    verify_runtime_dependencies,
)


SCHEMA = "apex.magpie-workflow-preflight/v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU-free production preview for every pinned Magpie config."
    )
    parser.add_argument("--apex-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _row(
    optimizer: object,
    path: Path,
    relative_path: str,
    staging: Path,
) -> dict[str, Any]:
    spec = E2EOptimizeSpec.from_mapping(
        {"config_path": str(path), "results_dir": str(staging / path.stem)}
    )
    try:
        preview = optimizer.preview(spec)
        document = preview.to_dict()
        return {
            "path": relative_path,
            "status": document["status"],
            "reason_code": None,
            "gpu_acquired": document["gpu_acquired"],
            "dimensions": document["dimensions"],
            "capabilities": document["capabilities"],
            "config": document["config"],
        }
    except ApexError as error:
        return {
            "path": relative_path,
            "status": "rejected",
            "reason_code": error.reason_code,
            "gpu_acquired": False,
            "dimensions": None,
            "capabilities": None,
            "config": None,
        }


def collect(apex_root: Path) -> dict[str, Any]:
    """Build one self-digested, explicitly non-authoritative preview manifest."""

    root = apex_root.expanduser().resolve(strict=True)
    receipt = verify_runtime_dependencies(apex_root=root)
    manifest = load_magpie_corpus_manifest(root / "scripts/magpie_corpus_manifest.json")
    verify_magpie_corpus_manifest(
        manifest,
        receipt.root("magpie"),
        repository=manifest.repository,
        commit=receipt.commits["magpie"],
    )
    application = build_application(include_e2e=True, include_kernel=False)
    if application.e2e_optimizer is None:
        raise RuntimeError("production E2E composition is unavailable")
    parent = root / "tmp" / "refactor" / "preflight"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".corpus-", dir=parent) as temporary:
        staging = Path(temporary)
        rows = tuple(
            _row(
                application.e2e_optimizer,
                receipt.root("magpie") / item.path,
                item.path,
                staging,
            )
            for item in manifest.files
        )
    payload = {
        "schema": SCHEMA,
        "non_authoritative": True,
        "qualification": "not_claimed",
        "magpie_commit": manifest.commit,
        "benchmark_tree": manifest.benchmark_tree,
        "corpus_manifest_sha256": manifest.manifest_sha256,
        "dependency_lock_sha256": receipt.lock_sha256,
        "rows": list(rows),
        "summary": {
            "config_count": len(rows),
            "config_compatible": sum(
                row["status"] == "config_compatible" for row in rows
            ),
            "rejected": sum(row["status"] == "rejected" for row in rows),
            "gpu_acquired": any(row["gpu_acquired"] for row in rows),
            **_capability_summary(rows),
        },
    }
    return {**payload, "manifest_sha256": sha256_json(payload)}


def _capability_summary(rows: tuple[dict[str, Any], ...]) -> dict[str, int]:
    capabilities = tuple(
        row["capabilities"] for row in rows if isinstance(row["capabilities"], dict)
    )
    source_statuses = tuple(
        item.get("source_optimization", {}).get("status") for item in capabilities
    )
    return {
        "benchmark_execution_available": sum(
            item.get("benchmark_execution", {}).get("available") is True
            for item in capabilities
        ),
        "benchmark_execution_unavailable": sum(
            item.get("benchmark_execution", {}).get("available") is False
            for item in capabilities
        ),
        "formal_measurement_available": sum(
            item.get("formal_measurement", {}).get("available") is True
            for item in capabilities
        ),
        "formal_measurement_unavailable": sum(
            item.get("formal_measurement", {}).get("available") is False
            for item in capabilities
        ),
        "source_evidence_pending": source_statuses.count("evidence_pending"),
        "source_capability_upgrade_required": source_statuses.count(
            "capability_upgrade_required"
        ),
    }


def _write_new(path: Path, value: dict[str, Any]) -> Path:
    selected = path.expanduser()
    if not selected.is_absolute() or selected.is_symlink() or selected.exists():
        raise ValueError("output must be one new absolute non-symlink path")
    selected.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".workflow.", dir=selected.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(canonical_json_bytes(value) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, selected)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return selected


def main() -> int:
    args = _parser().parse_args()
    try:
        value = collect(args.apex_root)
        output = _write_new(args.output, value)
    except (ApexError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "error", "message": str(error)}))
        return 2
    print(json.dumps({"status": "collected", "output": str(output), **value["summary"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
