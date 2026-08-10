"""Checked-in and local-Git identity collection for a release candidate."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from apex.core import sha256_json

from .dependencies import DependencyLock, load_lock
from .evaluator_lock import load_evaluator_policy_lock
from .lm_eval_lock import LmEvalRuntimeLock, load_lm_eval_runtime_lock
from .magpie_compatibility import (
    load_magpie_compatibility_ledger,
    verify_magpie_compatibility_ledger,
)
from .magpie_corpus import MagpieCorpusManifest, load_magpie_corpus_manifest
from .repositories import BootstrapError, canonical_repository, inspect_repository
from .source_locks import SourceLockSet, load_source_lock


def collect_release_static_identity(
    root: Path,
    *,
    kernel_showcases: Sequence[str],
    required_showcases: Sequence[str],
    required_images: Sequence[str],
    required_qualifications: Sequence[str],
) -> dict[str, Any]:
    """Strictly load every checked-in identity and the current local Git state."""

    scripts = root / "scripts"
    dependency = load_lock(scripts / "dependencies.lock.json")
    corpus = load_magpie_corpus_manifest(dependency.magpie_corpus_manifest)
    ledger = load_magpie_compatibility_ledger(dependency.magpie_compatibility_ledger)
    verify_magpie_compatibility_ledger(ledger, corpus)
    sources = load_source_lock(scripts / "e2e_source_locks.json")
    lm_eval = load_lm_eval_runtime_lock(scripts / "lm_eval_runtime.lock.json")
    evaluator = load_evaluator_policy_lock(scripts / "evaluator_policy.lock.json")
    checkout = inspect_repository(root)
    _verify_magpie_dependency(dependency, corpus)
    templates = _template_identity(
        root, scripts / "agent_kernel_templates.lock.json", tuple(kernel_showcases)
    )
    return {
        "apex_checkout": {
            "repository": canonical_repository(checkout.remote),
            "commit": checkout.commit,
            "tree": checkout.tree,
            "clean": not checkout.dirty_paths,
            "dirty_path_count": len(checkout.dirty_paths),
        },
        "project": _project_identity(root),
        "local_cli": _installed_cli_identity(root),
        "locks": {
            "dependencies": dependency.sha256,
            "e2e_sources": sources.sha256,
            "lm_eval_runtime": lm_eval.sha256,
            "evaluator_policy": evaluator.lock_sha256,
            "agent_templates": templates["lock_sha256"],
        },
        "dependencies": _dependency_projection(dependency, corpus),
        "sources": _source_projection(sources),
        "lm_eval": _lm_eval_projection(lm_eval),
        "evaluator_policy": evaluator.to_dict(),
        "magpie": _magpie_projection(corpus, ledger.to_dict()),
        "templates": templates,
        "required": {
            "showcases": list(required_showcases),
            "images": list(required_images),
            "qualifications": list(required_qualifications),
        },
    }


def _project_identity(root: Path) -> dict[str, str]:
    path = root / "pyproject.toml"
    content = _read_regular(path, "Apex pyproject")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BootstrapError("Apex pyproject is not UTF-8") from error
    project = _toml_section(text, "project")
    scripts = _toml_section(text, "project.scripts")
    name = _toml_string(project, "name")
    version = _toml_string(project, "version")
    entrypoint = _toml_string(scripts, "apex")
    if entrypoint != "apex.cli:main":
        raise BootstrapError("Apex project entrypoint differs")
    return {
        "name": name,
        "version": version,
        "entrypoint": entrypoint,
        "pyproject_sha256": hashlib.sha256(content).hexdigest(),
        "import_file_sha256": hashlib.sha256(
            _read_regular(root / "src" / "apex" / "__init__.py", "Apex import file")
        ).hexdigest(),
    }


def _installed_cli_identity(root: Path) -> dict[str, str | None]:
    executable = root / ".venv" / "bin" / "apex"
    if not executable.exists():
        return {"status": "missing", "executable_sha256": None}
    try:
        content = _read_regular(executable, "installed Apex CLI")
    except BootstrapError:
        return {"status": "invalid", "executable_sha256": None}
    if not _valid_console_script(content, root / ".venv" / "bin" / "python"):
        return {"status": "invalid", "executable_sha256": None}
    return {
        "status": "observed",
        "executable_sha256": hashlib.sha256(content).hexdigest(),
    }


def _valid_console_script(content: bytes, python: Path) -> bool:
    first, separator, body = content.partition(b"\n")
    expected_body = (
        b"# -*- coding: utf-8 -*-\n"
        b"import re\n"
        b"import sys\n"
        b"from apex.cli import main\n"
        b"if __name__ == '__main__':\n"
        b"    sys.argv[0] = re.sub(r'(-script\\.pyw|\\.exe)?$', '', sys.argv[0])\n"
        b"    sys.exit(main())\n"
    )
    return separator == b"\n" and first == f"#!{python}".encode() and body == expected_body


def _toml_section(text: str, name: str) -> str:
    marker = f"[{name}]"
    start = text.find(marker)
    if start < 0 or (start and text[start - 1] not in "\r\n"):
        raise BootstrapError(f"Apex pyproject lacks [{name}]")
    body = text[start + len(marker):]
    next_section = re.search(r"(?m)^\[", body)
    return body[: next_section.start()] if next_section else body


def _toml_string(section: str, key: str) -> str:
    match = re.search(rf'(?m)^{re.escape(key)}\s*=\s*"([^"\r\n]+)"\s*$', section)
    if match is None:
        raise BootstrapError(f"Apex pyproject {key} is invalid")
    return match.group(1)


def _verify_magpie_dependency(
    lock: DependencyLock,
    corpus: MagpieCorpusManifest,
) -> None:
    magpie = next(item for item in lock.dependencies if item.key == "magpie")
    if (
        canonical_repository(magpie.repository) != canonical_repository(corpus.repository)
        or magpie.commit != corpus.commit
    ):
        raise BootstrapError("Magpie dependency differs from corpus manifest")


def _dependency_projection(
    lock: DependencyLock,
    corpus: MagpieCorpusManifest,
) -> list[dict[str, str]]:
    return [
        {
            "name": item.key,
            "repository": canonical_repository(item.repository),
            "commit": item.commit,
            **({"tree": corpus.repository_tree} if item.key == "magpie" else {}),
        }
        for item in sorted(lock.dependencies, key=lambda value: value.key)
    ]


def _source_projection(lock: SourceLockSet) -> list[dict[str, str]]:
    return [
        {
            "name": item.key,
            "repository": canonical_repository(item.repository),
            "commit": item.commit,
            "tree": item.tree,
        }
        for item in sorted(lock.sources, key=lambda value: value.key)
    ]


def _lm_eval_projection(lock: LmEvalRuntimeLock) -> dict[str, str]:
    return {
        "runtime_sha256": lock.runtime_sha256,
        "installed_tree_sha256": lock.installed_tree_sha256,
        "base_image_id": lock.identity["base_image_id"],
        "base_image_repo_digest": lock.identity["base_image_repo_digest"],
        "inferencex_commit": lock.identity["inferencex_commit"],
        "inferencex_tree": lock.identity["inferencex_tree"],
    }


def _magpie_projection(
    corpus: MagpieCorpusManifest,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    summary = ledger["summary"]
    return {
        "repository": canonical_repository(corpus.repository),
        "commit": corpus.commit,
        "repository_tree": corpus.repository_tree,
        "benchmark_tree": corpus.benchmark_tree,
        "corpus_manifest_sha256": corpus.manifest_sha256,
        "compatibility_ledger_sha256": ledger["ledger_sha256"],
        "config_count": len(corpus.files),
        "config_compatible_count": summary["config_compatible"],
        "compatibility_authority": "legacy_apex_projection_not_release_evidence",
        "apex_config_resolution_evidence_required": True,
        "configs": [item.to_dict() for item in corpus.files],
        "workflow_qualified_count": summary["workflow_qualified"],
        "formal_delivery_qualified_count": summary["formal_delivery_qualified"],
    }


def _template_identity(
    root: Path,
    lock_path: Path,
    expected_ids: tuple[str, ...],
) -> dict[str, Any]:
    payload = _read_regular(lock_path, "template import lock")
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as error:
        raise BootstrapError(f"invalid template import lock: {error}") from error
    if not isinstance(raw, Mapping) or raw.get("schema") != "apex.agent-kernel-template-import-lock/v1":
        raise BootstrapError("unsupported template import lock")
    upstream = raw.get("upstream")
    templates = raw.get("templates")
    if not isinstance(upstream, Mapping) or not isinstance(templates, list):
        raise BootstrapError("template import lock is incomplete")
    projected = tuple(sorted(
        (_verify_template(root, item, expected_ids) for item in templates),
        key=lambda item: item["template_id"],
    ))
    if tuple(item["template_id"] for item in projected) != expected_ids:
        raise BootstrapError("template import lock does not cover canonical showcases")
    return {
        "lock_sha256": hashlib.sha256(payload).hexdigest(),
        "upstream_commit": _git_object(upstream.get("commit"), "template upstream commit"),
        "upstream_tree": _git_object(upstream.get("tree"), "template upstream tree"),
        "snapshot_sha256": sha256_json(projected),
        "entries": list(projected),
    }


def _verify_template(
    root: Path,
    value: object,
    expected_ids: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BootstrapError("template import entry is invalid")
    template_id = str(value.get("template_id", ""))
    if template_id != value.get("showcase_id") or template_id not in expected_ids:
        raise BootstrapError("template/showcase identity differs")
    output = _safe_relative(value.get("output_directory"), "template output directory")
    records = value.get("files")
    blockers = value.get("blockers")
    if not isinstance(records, list) or not isinstance(blockers, list):
        raise BootstrapError("template files/blockers are invalid")
    files = [_verify_template_file(root, output, record) for record in records]
    manifest = root / "examples" / "optimization_showcases" / output / "template" / "template_manifest.json"
    _verify_template_manifest(manifest, template_id)
    return {
        "template_id": template_id,
        "upstream_tree": _git_object(value.get("upstream_tree"), "template task tree"),
        "source_snapshot_sha256": sha256_json(files),
        "blockers": sorted(str(item) for item in blockers),
    }


def _verify_template_file(root: Path, output: Path, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "size"}:
        raise BootstrapError("template file record differs")
    relative = _safe_relative(value["path"], "template file path")
    path = root / "examples" / "optimization_showcases" / output / "template" / "upstream" / relative
    content = _read_regular(path, "template source file")
    digest = str(value["sha256"])
    size = value["size"]
    if hashlib.sha256(content).hexdigest() != digest or len(content) != size:
        raise BootstrapError("template source file differs from import lock")
    return {"path": relative.as_posix(), "sha256": digest, "size": size}


def _verify_template_manifest(path: Path, template_id: str) -> None:
    try:
        value = json.loads(_read_regular(path, "template manifest"))
    except json.JSONDecodeError as error:
        raise BootstrapError(f"invalid template manifest: {error}") from error
    valid = (
        isinstance(value, Mapping)
        and value.get("schema") == "apex.kernel-template/v1"
        and value.get("template_id") == template_id
        and value.get("showcase_id") == template_id
        and value.get("status") == "pending"
    )
    if not valid:
        raise BootstrapError("template manifest identity differs")


def _safe_relative(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BootstrapError(f"{field} is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise BootstrapError(f"{field} is unsafe")
    return Path(*path.parts)


def _read_regular(path: Path, field: str) -> bytes:
    try:
        stat = os.lstat(path)
        if path.is_symlink() or not path.is_file() or stat.st_nlink != 1:
            raise BootstrapError(f"{field} is not a safe regular file")
        return path.read_bytes()
    except OSError as error:
        raise BootstrapError(f"cannot read {field}: {error}") from error


def _git_object(value: object, field: str) -> str:
    text = str(value)
    if len(text) != 40 or any(item not in "0123456789abcdef" for item in text):
        raise BootstrapError(f"{field} is invalid")
    return text


__all__ = ["collect_release_static_identity"]
