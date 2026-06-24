"""Docker-driven generation for fixed trace-kernel registries."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .discovery import discover_trace_kernel_entries
from .registry import (
    SGLANG_TRACE_IMAGE,
    TRACE_IMAGE_REGISTRY_PATHS,
    VLLM_TRACE_IMAGE,
    TraceKernelEntry,
    _load_raw_registry,
    _validate_raw_registry,
    supported_trace_images,
)


DEST_PACKAGE_ROOTS = {
    "aiter": Path("tools/rocm/aiter/aiter"),
    "vllm": Path("tools/rocm/vllm/vllm"),
    "sglang": Path("tools/rocm/sglang/python/sglang"),
}
REGISTRY_SORT_KEY = lambda e: (e["repo"], e["kernel_type"], e["kernel_file"], e["kernel_name"], e["id"])
FIXED_IMAGE_REPOS = {
    VLLM_TRACE_IMAGE: ("vllm", "aiter"),
    SGLANG_TRACE_IMAGE: ("sglang", "aiter"),
}
FIXED_SOURCE_PATHS = {
    (SGLANG_TRACE_IMAGE, "aiter"): "/sgl-workspace/aiter/aiter",
    (SGLANG_TRACE_IMAGE, "sglang"): "/sgl-workspace/sglang/python/sglang",
}
FIXED_GIT_ROOTS = {
    (SGLANG_TRACE_IMAGE, "aiter"): "/sgl-workspace/aiter",
    (SGLANG_TRACE_IMAGE, "sglang"): "/sgl-workspace/sglang",
}
PACKAGE_NAMES = {
    "aiter": ("amd-aiter", "amd_aiter", "aiter"),
    "vllm": ("vllm",),
    "sglang": ("sglang",),
}


class RegistryUpdateError(RuntimeError):
    """Raised when fixed registries cannot be generated reliably."""


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str


@dataclass(frozen=True)
class RegistryUpdateResult:
    output_paths: dict[str, Path]
    report_path: Path | None
    wrote_registry: bool
    repos_by_image: dict[str, list[str]]
    diffs: dict[str, dict[str, Any]]
    report: str


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 600,
) -> CommandResult:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        rendered = " ".join(cmd)
        details = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        raise RegistryUpdateError(f"Command failed: {rendered}\n{details}")
    return CommandResult(stdout=proc.stdout, stderr=proc.stderr)


def _docker_image_metadata(image: str) -> dict[str, str]:
    raw = _run(["docker", "image", "inspect", image], timeout=120).stdout
    data = json.loads(raw)
    if not data:
        raise RegistryUpdateError(f"Docker image not found: {image}")
    item = data[0]
    repo_digests = item.get("RepoDigests") or []
    return {
        "image": image,
        "image_id": str(item.get("Id") or ""),
        "image_created": str(item.get("Created") or ""),
        "repo_digest": str(repo_digests[0]) if repo_digests else "",
    }


def _docker_run_shell(image: str, script: str, *, timeout: int = 300) -> str:
    return _run(
        ["docker", "run", "--rm", "--entrypoint", "/bin/bash", image, "-lc", script],
        timeout=timeout,
    ).stdout.strip()


def _python_package_dir_from_image(image: str, module_name: str) -> str:
    code = (
        "import importlib.util\n"
        f"module_name = {module_name!r}\n"
        "spec = importlib.util.find_spec(module_name)\n"
        "locations = list(getattr(spec, 'submodule_search_locations', []) or []) if spec else []\n"
        "if not locations:\n"
        "    raise SystemExit(f'{module_name} package not found')\n"
        "print(locations[0])\n"
    )
    return _docker_run_shell(image, f"python3 - <<'PY'\n{code}PY", timeout=300).strip()


def _package_version_from_image(image: str, repo: str) -> str:
    names = PACKAGE_NAMES[repo]
    code = (
        "import importlib.metadata as m\n"
        f"names = {list(names)!r}\n"
        "for name in names:\n"
        "    try:\n"
        "        print(m.version(name))\n"
        "        raise SystemExit(0)\n"
        "    except Exception:\n"
        "        pass\n"
        "raise SystemExit(2)\n"
    )
    try:
        return _docker_run_shell(image, f"python3 - <<'PY'\n{code}PY", timeout=300).strip()
    except RegistryUpdateError:
        return ""


def _git_info_from_image(image: str, git_root: str) -> dict[str, str]:
    script = (
        f"git -C {git_root!r} rev-parse HEAD && "
        f"git -C {git_root!r} show -s --format=%cI HEAD"
    )
    try:
        lines = [
            line.strip()
            for line in _docker_run_shell(image, script, timeout=300).splitlines()
            if line.strip()
        ]
    except RegistryUpdateError:
        return {}
    if len(lines) < 2:
        return {}
    return {
        "commit": lines[0],
        "commit_date": lines[1],
        "commit_resolution": f"git:{git_root}",
    }


def _docker_cp_python_tree_from_image(image: str, container_path: str, destination: Path) -> None:
    """Copy regular Python files from an image source tree without following symlinks."""
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.mkdir(parents=True, exist_ok=True)
    script = (
        "set -euo pipefail\n"
        f"cd {container_path!r}\n"
        "find . -type f -name '*.py' -print0 | "
        "tar --null --files-from - -cf - | "
        "tar -C /apex_registry_dst -xf -\n"
    )
    _run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-v",
            f"{destination}:/apex_registry_dst",
            "--entrypoint",
            "/bin/bash",
            image,
            "-lc",
            script,
        ],
        timeout=900,
    )


def _source_path_for_repo(image: str, repo: str) -> str:
    fixed = FIXED_SOURCE_PATHS.get((image, repo))
    if fixed:
        return fixed
    return _python_package_dir_from_image(image, repo)


def _copy_repo_source(
    *,
    image: str,
    repo: str,
    temp_root: Path,
) -> dict[str, str]:
    source_path = _source_path_for_repo(image, repo)
    destination = temp_root / DEST_PACKAGE_ROOTS[repo]
    _docker_cp_python_tree_from_image(image, source_path, destination)

    package_source = {
        "image": image,
        "source_path": source_path,
        "registry_path": DEST_PACKAGE_ROOTS[repo].as_posix(),
    }
    package_version = _package_version_from_image(image, repo)
    if package_version:
        package_source["package_version"] = package_version
    git_root = FIXED_GIT_ROOTS.get((image, repo))
    if git_root:
        package_source.update(_git_info_from_image(image, git_root))
    return package_source


def _load_existing_registry(path: Path, image: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 2,
            "docker_image": image,
            "image_metadata": {"image": image},
            "package_sources": {},
            "kernels": [],
        }
    return _load_raw_registry(path)


def _registry_entries_as_dicts(entries: Iterable[TraceKernelEntry]) -> list[dict[str, str]]:
    return [entry.as_dict() for entry in entries]


def build_registry_data(
    *,
    docker_image: str,
    image_metadata: dict[str, str],
    package_sources: dict[str, dict[str, str]],
    discovered_entries: list[TraceKernelEntry],
) -> dict[str, Any]:
    selected_repos = set(package_sources)
    kernels = sorted(
        [
            entry.as_dict()
            for entry in discovered_entries
            if entry.repo in selected_repos
        ],
        key=REGISTRY_SORT_KEY,
    )
    data: dict[str, Any] = {
        "schema_version": 2,
        "docker_image": docker_image,
        "image_metadata": dict(image_metadata),
        "package_sources": {repo: package_sources[repo] for repo in sorted(package_sources)},
        "kernels": kernels,
    }
    _validate_raw_registry(
        data,
        path=TRACE_IMAGE_REGISTRY_PATHS[docker_image],
        docker_image=docker_image,
        repo_root=None,
        validate_files=False,
    )
    return data


def diff_registry_data(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_entries = {
        str(entry.get("id")): entry
        for entry in old.get("kernels", [])
        if isinstance(entry, dict) and entry.get("id")
    }
    new_entries = {
        str(entry.get("id")): entry
        for entry in new.get("kernels", [])
        if isinstance(entry, dict) and entry.get("id")
    }
    added = sorted(set(new_entries) - set(old_entries))
    removed = sorted(set(old_entries) - set(new_entries))
    changed = sorted(
        kernel_id
        for kernel_id in set(old_entries) & set(new_entries)
        if old_entries[kernel_id] != new_entries[kernel_id]
    )
    repos = sorted({
        str(entry.get("repo"))
        for entry in [*old_entries.values(), *new_entries.values()]
        if entry.get("repo")
    })
    counts_by_repo: dict[str, dict[str, int]] = {}
    for repo in repos:
        old_count = sum(1 for entry in old_entries.values() if entry.get("repo") == repo)
        new_count = sum(1 for entry in new_entries.values() if entry.get("repo") == repo)
        counts_by_repo[repo] = {
            "old": old_count,
            "new": new_count,
            "delta": new_count - old_count,
        }
    return {
        "old_count": len(old_entries),
        "new_count": len(new_entries),
        "delta": len(new_entries) - len(old_entries),
        "added": added,
        "removed": removed,
        "changed": changed,
        "counts_by_repo": counts_by_repo,
        "metadata_changed": {
            "image_metadata": old.get("image_metadata") != new.get("image_metadata"),
            "package_sources": old.get("package_sources") != new.get("package_sources"),
        },
    }


def format_registry_yaml(data: dict[str, Any]) -> str:
    image = data.get("docker_image", "")
    header_lines = [
        "# Generated by workload_optimizer.py update-trace-kernel-registry.",
        f"# docker_image: {image}",
    ]
    body = yaml.safe_dump(data, sort_keys=False, width=120)
    return "\n".join(header_lines) + "\n" + body


def _sample(items: list[str], limit: int = 30) -> list[str]:
    if len(items) <= limit:
        return items
    return items[:limit] + [f"... {len(items) - limit} more"]


def format_markdown_report(
    *,
    output_paths: dict[str, Path],
    registries: dict[str, dict[str, Any]],
    diffs: dict[str, dict[str, Any]],
) -> str:
    lines = ["# Trace Kernel Registry Diff", ""]
    for image in supported_trace_images():
        registry = registries[image]
        diff = diffs[image]
        lines.extend([
            f"## `{image}`",
            "",
            f"- output: `{output_paths[image]}`",
            f"- repos: `{', '.join(registry.get('package_sources', {}))}`",
            f"- kernels: {diff['old_count']} -> {diff['new_count']} ({diff['delta']:+d})",
            f"- added: {len(diff['added'])}",
            f"- removed: {len(diff['removed'])}",
            f"- changed: {len(diff['changed'])}",
            f"- image metadata changed: {diff['metadata_changed']['image_metadata']}",
            f"- package sources changed: {diff['metadata_changed']['package_sources']}",
            "",
            "| repo | old | new | delta |",
            "| --- | ---: | ---: | ---: |",
        ])
        for repo, counts in diff["counts_by_repo"].items():
            lines.append(f"| `{repo}` | {counts['old']} | {counts['new']} | {counts['delta']:+d} |")
        for label in ("added", "removed", "changed"):
            items = diff[label]
            if not items:
                continue
            lines.extend(["", f"### {label.title()} Kernel IDs", ""])
            for item in _sample(items):
                lines.append(f"- `{item}`")
        lines.append("")
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
        handle.write(text)
    os.replace(tmp, path)


def _generate_registry_for_image(
    *,
    image: str,
    temp_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    image_metadata = _docker_image_metadata(image)
    package_sources: dict[str, dict[str, str]] = {}
    copied_repos: list[str] = []
    for repo in FIXED_IMAGE_REPOS[image]:
        try:
            package_sources[repo] = _copy_repo_source(
                image=image,
                repo=repo,
                temp_root=temp_root,
            )
            copied_repos.append(repo)
        except RegistryUpdateError:
            if repo in {"vllm", "sglang"}:
                raise
            continue
    discovered = discover_trace_kernel_entries(temp_root)
    return build_registry_data(
        docker_image=image,
        image_metadata=image_metadata,
        package_sources=package_sources,
        discovered_entries=discovered,
    ), copied_repos


def update_trace_kernel_registry(
    *,
    repo_root: Path,
    report_path: Path | None = None,
    write: bool = False,
) -> RegistryUpdateResult:
    del repo_root
    output_paths = dict(TRACE_IMAGE_REGISTRY_PATHS)
    registries: dict[str, dict[str, Any]] = {}
    repos_by_image: dict[str, list[str]] = {}
    diffs: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="apex_trace_registry_sources_") as tmp:
        base_temp = Path(tmp)
        for image in supported_trace_images():
            temp_root = base_temp / image.replace("/", "_").replace(":", "_")
            registry, copied_repos = _generate_registry_for_image(
                image=image,
                temp_root=temp_root,
            )
            registries[image] = registry
            repos_by_image[image] = copied_repos
            old_registry = _load_existing_registry(output_paths[image], image)
            diffs[image] = diff_registry_data(old_registry, registry)

    report = format_markdown_report(
        output_paths=output_paths,
        registries=registries,
        diffs=diffs,
    )
    if report_path:
        _atomic_write(report_path, report)
    if write:
        for image, registry in registries.items():
            _atomic_write(output_paths[image], format_registry_yaml(registry))

    return RegistryUpdateResult(
        output_paths=output_paths,
        report_path=report_path,
        wrote_registry=write,
        repos_by_image=repos_by_image,
        diffs=diffs,
        report=report,
    )
