#!/usr/bin/env python3
"""Import three attributed AgentKernelArena task-input snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "scripts" / "agent_kernel_templates.lock.json"
OUTPUT_ROOT = ROOT / "examples" / "optimization_showcases"
REGISTRY_PATH = ROOT / "src" / "apex" / "intake" / "data" / "kernel_template_registry.json"


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _git(source: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(source), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value}")
    return Path(*path.parts)


def _load_lock() -> Mapping[str, Any]:
    value = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if value.get("schema") != "apex.agent-kernel-template-import-lock/v1":
        raise ValueError("unsupported template import lock")
    return value


def _verify_checkout(source: Path, lock: Mapping[str, Any]) -> None:
    upstream = lock["upstream"]
    if source.is_symlink() or not source.resolve(strict=True).is_dir():
        raise ValueError("source checkout must be a non-symlink directory")
    observed = {
        "commit": _git(source, "rev-parse", "HEAD"),
        "tree": _git(source, "rev-parse", "HEAD^{tree}"),
        "remote": _git(source, "remote", "get-url", "origin"),
        "status": _git(source, "status", "--porcelain"),
    }
    if observed["commit"] != upstream["commit"] or observed["tree"] != upstream["tree"]:
        raise ValueError("AgentKernelArena checkout does not match the reviewed Git identity")
    if observed["remote"] not in upstream["allowed_source_remotes"]:
        raise ValueError("AgentKernelArena checkout has an unreviewed origin")
    if observed["status"]:
        raise ValueError("AgentKernelArena checkout must be clean")
    license_record = upstream["license"]
    _verify_file(source / _safe_relative(license_record["path"]), license_record)


def _verify_file(path: Path, record: Mapping[str, Any]) -> bytes:
    metadata = os.lstat(path)
    if path.is_symlink() or not path.is_file() or metadata.st_nlink != 1:
        raise ValueError(f"unsafe upstream file: {path}")
    content = path.read_bytes()
    if len(content) != int(record["size"]) or _sha256(content) != record["sha256"]:
        raise ValueError(f"upstream file does not match lock: {path}")
    return content


def _upstream_document(lock: Mapping[str, Any], template: Mapping[str, Any]) -> bytes:
    upstream = lock["upstream"]
    lines = [
        "# Upstream sources", "",
        f"- Repository: `{upstream['repository']}`",
        f"- Commit: `{upstream['commit']}`",
        f"- Repository tree: `{upstream['tree']}`",
        f"- Task tree: `{template['upstream_tree']}`",
        f"- Source path: `{template['upstream_path']}`",
        f"- Reviewed/imported: `{lock['reviewed_at_utc']}`", "",
        "Copied input files:", "",
    ]
    for item in template["files"]:
        lines.append(f"- `{item['path']}` — SHA-256 `{item['sha256']}` ({item['size']} bytes)")
    lines.extend([
        "", "The upstream task runners and Forge driver were deliberately not copied. They",
        "depend on AgentKernelArena workspace injection and are not Apex evaluator evidence.",
        "The mutable upstream image tag is provenance only and cannot launch a formal run.", "",
    ])
    return "\n".join(lines).encode()


def _notice_document(lock: Mapping[str, Any], template: Mapping[str, Any]) -> bytes:
    return (
        "# Third-party notices\n\n"
        "The files under `upstream/` are copied, unmodified task inputs from\n"
        f"AgentKernelArena commit `{lock['upstream']['commit']}`, task\n"
        f"`{template['upstream_path']}`. AgentKernelArena is licensed under Apache-2.0;\n"
        "the complete upstream license is included as `LICENSE.agent-kernel-arena`.\n\n"
        "Apex wrapper documentation and the generated template manifest are modifications\n"
        "created for attribution and fail-closed integration. No upstream runner, scorer,\n"
        "validator, optimized vLLM/AITER source, or AgentKernelArena result is included.\n"
    ).encode()


def _wrapper_readme(template: Mapping[str, Any]) -> bytes:
    blockers = "\n".join(f"- `{item}`" for item in template["blockers"])
    text = f"""# {template['showcase_id']}

Status: **pending**. This directory currently preserves reviewed, attributed task
inputs; it is not a published Apex winner and contains no reward evidence.

The original input bytes are under `template/upstream/`. `config.yaml` is a
provenance snapshot, not an Apex TaskSpec and not a trusted evaluator. Apex does
not import or execute AgentKernelArena runners, scorers, validators, or Forge code.

Formal materialization is blocked by:

{blockers}

Once those receipts and an Apex-owned evaluator are reviewed, this template is
intended to use the ordinary `apex optimize kernel ... --template ...` path. Until
then, the CLI fails before agent, container, GPU, or measurement execution.
"""
    return text.encode()


def _template_files(
    source: Path, lock: Mapping[str, Any], template: Mapping[str, Any]
) -> dict[str, bytes]:
    source_root = source / _safe_relative(template["upstream_path"])
    files: dict[str, bytes] = {}
    upstream_files: list[dict[str, object]] = []
    for record in template["files"]:
        content = _verify_file(source_root / _safe_relative(record["path"]), record)
        imported = f"template/upstream/{record['path']}"
        files[imported] = content
        upstream_files.append({
            "original_path": f"{template['upstream_path']}/{record['path']}",
            "imported_path": imported,
            "sha256": record["sha256"],
            "size": record["size"],
        })
    license_record = lock["upstream"]["license"]
    files["template/LICENSE.agent-kernel-arena"] = _verify_file(
        source / license_record["path"], license_record
    )
    files["template/UPSTREAM_SOURCES.md"] = _upstream_document(lock, template)
    files["template/THIRD_PARTY_NOTICES.md"] = _notice_document(lock, template)
    files["README.md"] = _wrapper_readme(template)
    manifest = _manifest(lock, template, upstream_files, files)
    files["template/template_manifest.json"] = _json_bytes(manifest)
    return files


def _manifest(
    lock: Mapping[str, Any],
    template: Mapping[str, Any],
    upstream_files: list[dict[str, object]],
    files: Mapping[str, bytes],
) -> dict[str, object]:
    snapshot_paths = sorted(path for path in files if path.startswith("template/"))
    unsigned: dict[str, object] = {
        "schema": "apex.kernel-template/v1",
        "template_id": template["template_id"],
        "showcase_id": template["showcase_id"],
        "status": "pending",
        "upstream": {
            "repository": lock["upstream"]["repository"],
            "commit": lock["upstream"]["commit"],
            "tree": template["upstream_tree"],
            "imported_at_utc": lock["reviewed_at_utc"],
            "source_path": template["upstream_path"],
            "files": upstream_files,
        },
        "runtime": {
            **template["runtime"],
            "immutable_locator": None,
            "image_id": None,
        },
        "source": {
            **template["source"],
            "baseline_tree_sha256": None,
        },
        "evaluator": None,
        "blockers": template["blockers"],
        "snapshot_files": [
            {"path": path, "sha256": _sha256(files[path]), "size": len(files[path])}
            for path in snapshot_paths
        ],
    }
    return {**unsigned, "manifest_sha256": _sha256(_canonical_json(unsigned))}


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _write_outputs(output: Path, generated: Mapping[str, bytes]) -> None:
    for relative, content in generated.items():
        destination = output / _safe_relative(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def _check_outputs(output: Path, generated: Mapping[str, bytes]) -> None:
    for relative, expected in generated.items():
        path = output / _safe_relative(relative)
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
            raise ValueError(f"generated template output differs: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    lock = _load_lock()
    source = args.source.expanduser().resolve(strict=True)
    _verify_checkout(source, lock)
    registry_entries: list[dict[str, object]] = []
    for template in lock["templates"]:
        observed_tree = _git(source, "rev-parse", f"HEAD:{template['upstream_path']}")
        if observed_tree != template["upstream_tree"]:
            raise ValueError(f"upstream task tree changed: {template['template_id']}")
        output = args.output_root / template["output_directory"]
        generated = _template_files(source, lock, template)
        _write_outputs(output, generated) if args.write else _check_outputs(output, generated)
        manifest = json.loads(generated["template/template_manifest.json"])
        registry_entries.append({
            "template_id": manifest["template_id"],
            "showcase_id": manifest["showcase_id"],
            "status": manifest["status"],
            "manifest_sha256": manifest["manifest_sha256"],
        })
    registry = _json_bytes({
        "schema": "apex.kernel-template-registry/v1",
        "entries": sorted(registry_entries, key=lambda item: str(item["template_id"])),
    })
    if args.write:
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY_PATH.write_bytes(registry)
    elif not REGISTRY_PATH.is_file() or REGISTRY_PATH.read_bytes() != registry:
        raise ValueError("generated kernel template registry differs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
