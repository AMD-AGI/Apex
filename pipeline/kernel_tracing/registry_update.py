"""Docker-driven maintenance for the trace-kernel registry."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from .discovery import discover_trace_kernel_entries
from .registry import (
    SUPPORTED_KERNELS_PATH,
    VALID_REPOS,
    TraceKernelEntry,
    _load_raw_registry,
    _validate_raw_registry,
)


VLLM_REPO_URL = "https://github.com/vllm-project/vllm"
SUPPORTED_FRAMEWORKS = {"sglang", "vllm"}
FRAMEWORK_REPOS = {
    "sglang": {"aiter", "sglang"},
    "vllm": {"vllm"},
}
SGLANG_IMAGE_SOURCES = {
    "aiter": "/sgl-workspace/aiter/aiter",
    "sglang": "/sgl-workspace/sglang/python/sglang",
}
SGLANG_GIT_ROOTS = {
    "aiter": "/sgl-workspace/aiter",
    "sglang": "/sgl-workspace/sglang",
}
DEST_PACKAGE_ROOTS = {
    "aiter": Path("tools/rocm/aiter/aiter"),
    "vllm": Path("tools/rocm/vllm/vllm"),
    "sglang": Path("tools/rocm/sglang/python/sglang"),
}
REGISTRY_SORT_KEY = lambda e: (e["repo"], e["kernel_type"], e["kernel_file"], e["kernel_name"], e["id"])


class RegistryUpdateError(RuntimeError):
    """Raised when the registry cannot be refreshed with reliable provenance."""


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str


@dataclass(frozen=True)
class RegistryUpdateResult:
    output_path: Path
    report_path: Path | None
    wrote_registry: bool
    selected_frameworks: list[str]
    selected_repos: list[str]
    images: dict[str, str]
    source_commits: dict[str, str]
    diff: dict[str, Any]
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
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        details = stderr or stdout or f"exit code {proc.returncode}"
        raise RegistryUpdateError(f"Command failed: {rendered}\n{details}")
    return CommandResult(stdout=proc.stdout, stderr=proc.stderr)


def parse_frameworks(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        parts = [part.strip().lower() for part in value.split(",")]
    else:
        parts = [str(part).strip().lower() for part in value]
    frameworks = [part for part in parts if part]
    unknown = sorted(set(frameworks) - SUPPORTED_FRAMEWORKS)
    if unknown:
        raise RegistryUpdateError(
            f"Unsupported framework(s): {', '.join(unknown)}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FRAMEWORKS))}"
        )
    deduped: list[str] = []
    for framework in frameworks:
        if framework not in deduped:
            deduped.append(framework)
    return deduped or ["sglang", "vllm"]


def repos_for_frameworks(frameworks: Iterable[str]) -> set[str]:
    repos: set[str] = set()
    for framework in frameworks:
        repos.update(FRAMEWORK_REPOS[framework])
    return repos


def _magpie_image_config_candidates(repo_root: Path) -> list[Path]:
    candidates = [repo_root / "tools" / "magpie" / "Magpie" / "benchmark_images.yaml"]
    magpie_root = os.environ.get("MAGPIE_ROOT", "").strip()
    if magpie_root:
        root = Path(magpie_root)
        candidates.extend([
            root / "Magpie" / "benchmark_images.yaml",
            root / "benchmark_images.yaml",
        ])
    return candidates


def resolve_framework_images(
    *,
    repo_root: Path,
    gpu_arch: str,
    frameworks: Iterable[str],
    sglang_image: str = "",
    vllm_image: str = "",
) -> dict[str, str]:
    """Resolve framework images using Magpie's benchmark image mapping."""
    overrides = {
        "sglang": sglang_image.strip(),
        "vllm": vllm_image.strip(),
    }
    mapping: dict[str, Any] = {}
    for path in _magpie_image_config_candidates(repo_root):
        if path.exists():
            mapping = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            break
    if not mapping:
        raise RegistryUpdateError("Could not find Magpie benchmark_images.yaml")

    images: dict[str, str] = {}
    for framework in frameworks:
        if overrides[framework]:
            images[framework] = overrides[framework]
            continue
        image = mapping.get(framework, {}).get(gpu_arch)
        if not image:
            available = sorted(mapping.get(framework, {}))
            raise RegistryUpdateError(
                f"No Magpie image mapping for {framework}/{gpu_arch}. "
                f"Available arches: {available}"
            )
        images[framework] = str(image)
    return images


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


def _docker_create_for_copy(image: str) -> str:
    return _run(
        ["docker", "create", "--entrypoint", "/bin/sh", image, "-c", "true"],
        timeout=120,
    ).stdout.strip()


def _docker_cp_from_image(image: str, container_path: str, destination: Path) -> None:
    container = _docker_create_for_copy(image)
    try:
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run(["docker", "cp", f"{container}:{container_path}", str(destination)], timeout=900)
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container],
            capture_output=True,
            text=True,
            timeout=120,
        )


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
        f"cd {shlex.quote(container_path)}\n"
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


def _git_commit_info_from_image(image: str, git_root: str) -> dict[str, str]:
    script = (
        f"git -C {git_root!r} rev-parse HEAD && "
        f"git -C {git_root!r} show -s --format=%cI HEAD"
    )
    lines = [line.strip() for line in _docker_run_shell(image, script).splitlines() if line.strip()]
    if len(lines) < 2 or not re.fullmatch(r"[0-9a-f]{40}", lines[0]):
        raise RegistryUpdateError(f"Could not read git commit metadata from {image}:{git_root}")
    return {"commit": lines[0], "commit_date": lines[1]}


def _vllm_package_dir_from_image(image: str) -> str:
    code = (
        "import importlib.util\n"
        "spec = importlib.util.find_spec('vllm')\n"
        "if spec is None or not spec.submodule_search_locations:\n"
        "    raise SystemExit('vllm package not found')\n"
        "print(next(iter(spec.submodule_search_locations)))\n"
    )
    return _docker_run_shell(image, f"python3 - <<'PY'\n{code}PY", timeout=300).strip()


def _vllm_package_version_from_image(image: str) -> str:
    code = "import importlib.metadata as m\nprint(m.version('vllm'))\n"
    return _docker_run_shell(image, f"python3 - <<'PY'\n{code}PY", timeout=300).strip()


def vllm_tag_from_version(version: str) -> str:
    """Map a vLLM package version like 0.19.1+rocm721 to tag v0.19.1."""
    base = version.split("+", 1)[0].strip()
    if not re.fullmatch(r"\d+(?:\.\d+)+(?:[a-zA-Z0-9_.-]*)?", base):
        raise RegistryUpdateError(f"Cannot map vLLM package version to tag: {version!r}")
    return f"v{base}"


def resolve_vllm_tag_commit(
    version: str,
    *,
    explicit_commit: str = "",
    run: Callable[[list[str]], CommandResult] | None = None,
) -> tuple[str, str, str]:
    """Return (commit, tag, resolution_method) for an installed vLLM package version."""
    tag = vllm_tag_from_version(version)
    if explicit_commit:
        commit = explicit_commit.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise RegistryUpdateError(f"Invalid --vllm-commit: {explicit_commit!r}")
        return commit, tag, "explicit --vllm-commit"

    runner = run or (lambda cmd: _run(cmd, timeout=120))
    result = runner([
        "git",
        "ls-remote",
        "--tags",
        VLLM_REPO_URL,
        f"refs/tags/{tag}",
        f"refs/tags/{tag}^{{}}",
    ])
    lines = [line.strip().split() for line in result.stdout.splitlines() if line.strip()]
    peeled = [parts[0] for parts in lines if len(parts) >= 2 and parts[1].endswith("^{}")]
    direct = [parts[0] for parts in lines if len(parts) >= 2 and parts[1] == f"refs/tags/{tag}"]
    commit = (peeled or direct or [""])[0]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RegistryUpdateError(
            f"Could not resolve vLLM version {version!r} to {VLLM_REPO_URL} tag {tag}. "
            "Pass --vllm-commit <sha> to override explicitly."
        )
    return commit, tag, f"git ls-remote {VLLM_REPO_URL} refs/tags/{tag}"


def _remote_commit_date(repo_url: str, commit: str, tag: str) -> str:
    with tempfile.TemporaryDirectory(prefix="apex_trace_registry_git_") as tmp:
        root = Path(tmp)
        _run(["git", "init", "-q"], cwd=root, timeout=120)
        _run(
            ["git", "fetch", "--depth=1", repo_url, f"refs/tags/{tag}:refs/tags/{tag}"],
            cwd=root,
            timeout=300,
        )
        return _run(["git", "show", "-s", "--format=%cI", commit], cwd=root, timeout=120).stdout.strip()


def _copy_sglang_sources(
    *,
    image: str,
    temp_root: Path,
    source_commits: dict[str, str],
    source_images: dict[str, dict[str, str]],
) -> None:
    image_meta = _docker_image_metadata(image)
    for repo, container_source in SGLANG_IMAGE_SOURCES.items():
        destination = temp_root / DEST_PACKAGE_ROOTS[repo]
        _docker_cp_python_tree_from_image(image, container_source, destination)
        git_info = _git_commit_info_from_image(image, SGLANG_GIT_ROOTS[repo])
        source_commits[repo] = git_info["commit"]
        source_images[repo] = {
            **image_meta,
            "framework": "sglang",
            "source_path": container_source,
            "commit": git_info["commit"],
            "commit_date": git_info["commit_date"],
            "commit_resolution": f"git:{SGLANG_GIT_ROOTS[repo]}",
        }


def _copy_vllm_sources(
    *,
    image: str,
    temp_root: Path,
    source_commits: dict[str, str],
    source_images: dict[str, dict[str, str]],
    vllm_commit: str = "",
) -> None:
    image_meta = _docker_image_metadata(image)
    package_dir = _vllm_package_dir_from_image(image)
    package_version = _vllm_package_version_from_image(image)
    commit, tag, method = resolve_vllm_tag_commit(
        package_version,
        explicit_commit=vllm_commit,
    )
    commit_date = _remote_commit_date(VLLM_REPO_URL, commit, tag)
    destination = temp_root / DEST_PACKAGE_ROOTS["vllm"]
    _docker_cp_python_tree_from_image(image, package_dir, destination)
    source_commits["vllm"] = commit
    source_images["vllm"] = {
        **image_meta,
        "framework": "vllm",
        "source_path": package_dir,
        "package_version": package_version,
        "package_tag": tag,
        "commit": commit,
        "commit_date": commit_date,
        "commit_resolution": method,
    }


def _registry_entries_as_dicts(entries: Iterable[TraceKernelEntry]) -> list[dict[str, str]]:
    return [entry.as_dict() for entry in entries]


def _load_existing_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 1,
            "source_commits": {},
            "source_images": {},
            "kernels": [],
        }
    return _load_raw_registry(path)


def build_registry_data(
    *,
    old_registry: dict[str, Any],
    discovered_entries: list[TraceKernelEntry],
    selected_repos: set[str],
    source_commits: dict[str, str],
    source_images: dict[str, dict[str, str]],
) -> dict[str, Any]:
    old_entries = [
        raw
        for raw in old_registry.get("kernels", [])
        if isinstance(raw, dict) and str(raw.get("repo", "")) not in selected_repos
    ]
    new_entries = [
        entry.as_dict()
        for entry in discovered_entries
        if entry.repo in selected_repos
    ]

    commits = dict(old_registry.get("source_commits") or {})
    commits.update(source_commits)
    missing_commits = sorted(VALID_REPOS - set(commits))
    if missing_commits:
        raise RegistryUpdateError(
            "Cannot build a partial trace-kernel registry without existing commits for: "
            f"{', '.join(missing_commits)}"
        )

    images = dict(old_registry.get("source_images") or {})
    images.update(source_images)

    kernels = sorted(old_entries + new_entries, key=REGISTRY_SORT_KEY)
    data: dict[str, Any] = {
        "schema_version": 1,
        "source_commits": {repo: commits[repo] for repo in sorted(VALID_REPOS)},
    }
    if images:
        data["source_images"] = {repo: images[repo] for repo in sorted(images)}
    data["kernels"] = kernels
    _validate_raw_registry(
        data,
        path=SUPPORTED_KERNELS_PATH,
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
    old_commits = old.get("source_commits") or {}
    new_commits = new.get("source_commits") or {}
    commit_changes = {
        repo: {
            "old": old_commits.get(repo, ""),
            "new": new_commits.get(repo, ""),
        }
        for repo in sorted(VALID_REPOS)
        if old_commits.get(repo) != new_commits.get(repo)
    }
    counts_by_repo: dict[str, dict[str, int]] = {}
    for repo in sorted(VALID_REPOS):
        old_count = sum(1 for entry in old_entries.values() if entry.get("repo") == repo)
        new_count = sum(1 for entry in new_entries.values() if entry.get("repo") == repo)
        counts_by_repo[repo] = {"old": old_count, "new": new_count, "delta": new_count - old_count}
    return {
        "old_count": len(old_entries),
        "new_count": len(new_entries),
        "delta": len(new_entries) - len(old_entries),
        "added": added,
        "removed": removed,
        "changed": changed,
        "source_commit_changes": commit_changes,
        "counts_by_repo": counts_by_repo,
    }


def format_registry_yaml(data: dict[str, Any]) -> str:
    header_lines = ["# Generated by workload_optimizer.py update-trace-kernel-registry."]
    commits = data.get("source_commits") or {}
    for repo in ("aiter", "vllm", "sglang"):
        if repo in commits:
            header_lines.append(f"# {repo}: {commits[repo]}")
    body = yaml.safe_dump(data, sort_keys=False, width=120)
    return "\n".join(header_lines) + "\n" + body


def _sample(items: list[str], limit: int = 30) -> list[str]:
    if len(items) <= limit:
        return items
    return items[:limit] + [f"... {len(items) - limit} more"]


def format_markdown_report(
    *,
    output_path: Path,
    frameworks: list[str],
    gpu_arch: str,
    images: dict[str, str],
    diff: dict[str, Any],
    new_registry: dict[str, Any],
) -> str:
    lines = [
        "# Trace Kernel Registry Diff",
        "",
        f"- output: `{output_path}`",
        f"- gpu_arch: `{gpu_arch}`",
        f"- frameworks: `{','.join(frameworks)}`",
        "",
        "## Images",
    ]
    for framework, image in images.items():
        lines.append(f"- `{framework}`: `{image}`")

    lines.extend([
        "",
        "## Source Commits",
    ])
    for repo, commit in (new_registry.get("source_commits") or {}).items():
        image_meta = (new_registry.get("source_images") or {}).get(repo, {})
        date = image_meta.get("commit_date", "")
        suffix = f" ({date})" if date else ""
        lines.append(f"- `{repo}`: `{commit}`{suffix}")

    lines.extend([
        "",
        "## Summary",
        "",
        f"- kernels: {diff['old_count']} -> {diff['new_count']} ({diff['delta']:+d})",
        f"- added: {len(diff['added'])}",
        f"- removed: {len(diff['removed'])}",
        f"- changed: {len(diff['changed'])}",
        "",
        "## Counts By Repo",
        "",
        "| repo | old | new | delta |",
        "| --- | ---: | ---: | ---: |",
    ])
    for repo, counts in diff["counts_by_repo"].items():
        lines.append(f"| `{repo}` | {counts['old']} | {counts['new']} | {counts['delta']:+d} |")

    if diff["source_commit_changes"]:
        lines.extend(["", "## Source Commit Changes", ""])
        for repo, change in diff["source_commit_changes"].items():
            lines.append(f"- `{repo}`: `{change['old']}` -> `{change['new']}`")
    for label in ("added", "removed", "changed"):
        items = diff[label]
        if not items:
            continue
        lines.extend(["", f"## {label.title()} Kernel IDs", ""])
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


def update_trace_kernel_registry(
    *,
    repo_root: Path,
    gpu_arch: str = "gfx950",
    frameworks: str | Iterable[str] = "sglang,vllm",
    output_path: Path = SUPPORTED_KERNELS_PATH,
    report_path: Path | None = None,
    write: bool = False,
    sglang_image: str = "",
    vllm_image: str = "",
    vllm_commit: str = "",
) -> RegistryUpdateResult:
    selected_frameworks = parse_frameworks(frameworks)
    selected_repos = repos_for_frameworks(selected_frameworks)
    images = resolve_framework_images(
        repo_root=repo_root,
        gpu_arch=gpu_arch,
        frameworks=selected_frameworks,
        sglang_image=sglang_image,
        vllm_image=vllm_image,
    )
    old_registry = _load_existing_registry(output_path)

    source_commits: dict[str, str] = {}
    source_images: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory(prefix="apex_trace_registry_sources_") as tmp:
        temp_root = Path(tmp)
        if "sglang" in selected_frameworks:
            _copy_sglang_sources(
                image=images["sglang"],
                temp_root=temp_root,
                source_commits=source_commits,
                source_images=source_images,
            )
        if "vllm" in selected_frameworks:
            _copy_vllm_sources(
                image=images["vllm"],
                temp_root=temp_root,
                source_commits=source_commits,
                source_images=source_images,
                vllm_commit=vllm_commit,
            )
        discovered = discover_trace_kernel_entries(temp_root)

    new_registry = build_registry_data(
        old_registry=old_registry,
        discovered_entries=discovered,
        selected_repos=selected_repos,
        source_commits=source_commits,
        source_images=source_images,
    )
    diff = diff_registry_data(old_registry, new_registry)
    report = format_markdown_report(
        output_path=output_path,
        frameworks=selected_frameworks,
        gpu_arch=gpu_arch,
        images=images,
        diff=diff,
        new_registry=new_registry,
    )

    if report_path:
        _atomic_write(report_path, report)
    if write:
        _atomic_write(output_path, format_registry_yaml(new_registry))

    return RegistryUpdateResult(
        output_path=output_path,
        report_path=report_path,
        wrote_registry=write,
        selected_frameworks=selected_frameworks,
        selected_repos=sorted(selected_repos),
        images=images,
        source_commits=dict(new_registry["source_commits"]),
        diff=diff,
        report=report,
    )
