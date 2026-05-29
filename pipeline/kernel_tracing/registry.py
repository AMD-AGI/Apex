"""Supported trace-kernel registry."""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_KERNELS_PATH = Path(__file__).with_name("supported_kernels.yaml")
VALID_REPOS = {"aiter", "vllm", "sglang"}
VALID_KERNEL_TYPES = {"triton", "hip"}
VALID_TRACE_MODES = {
    "triton-launch",
    "aiter-compile-ops",
    "vllm-custom-op",
    "sglang-custom-op",
}
VALID_PATCH_STRATEGIES = {"static"}
_HASH_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class TraceKernelEntry:
    id: str
    repo: str
    kernel_type: str
    kernel_name: str
    kernel_file: str
    trace_mode: str
    patch_strategy: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    def resolved_file(self, repo_root: Path) -> Path:
        path = Path(self.kernel_file)
        return path if path.is_absolute() else repo_root / path


def _load_raw_registry(path: Path = SUPPORTED_KERNELS_PATH) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Trace kernel registry must be a mapping: {path}")
    return data


def _validate_raw_registry(
    data: dict[str, Any],
    *,
    path: Path,
    repo_root: Path | None,
    validate_files: bool,
) -> list[TraceKernelEntry]:
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported trace kernel registry schema in {path}")

    commits = data.get("source_commits")
    if not isinstance(commits, dict):
        raise ValueError("Trace kernel registry is missing source_commits")
    for repo in sorted(VALID_REPOS):
        commit = commits.get(repo)
        if not isinstance(commit, str) or not _HASH_RE.match(commit):
            raise ValueError(f"Invalid source commit for {repo}: {commit!r}")

    raw_kernels = data.get("kernels")
    if not isinstance(raw_kernels, list):
        raise ValueError("Trace kernel registry is missing kernels list")

    entries: list[TraceKernelEntry] = []
    seen: set[str] = set()
    required = set(TraceKernelEntry.__dataclass_fields__)
    for idx, raw in enumerate(raw_kernels):
        if not isinstance(raw, dict):
            raise ValueError(f"Trace kernel entry #{idx} must be a mapping")
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"Trace kernel entry #{idx} missing fields: {missing}")
        entry = TraceKernelEntry(**{field: str(raw[field]) for field in required})
        if entry.id in seen:
            raise ValueError(f"Duplicate trace kernel id: {entry.id}")
        seen.add(entry.id)
        if entry.repo not in VALID_REPOS:
            raise ValueError(f"{entry.id}: unsupported repo {entry.repo!r}")
        if entry.kernel_type not in VALID_KERNEL_TYPES:
            raise ValueError(f"{entry.id}: unsupported kernel_type {entry.kernel_type!r}")
        if entry.trace_mode not in VALID_TRACE_MODES:
            raise ValueError(f"{entry.id}: unsupported trace_mode {entry.trace_mode!r}")
        if entry.patch_strategy not in VALID_PATCH_STRATEGIES:
            raise ValueError(
                f"{entry.id}: unsupported patch_strategy {entry.patch_strategy!r}"
            )
        if validate_files:
            if repo_root is None:
                raise ValueError("repo_root is required when validate_files=True")
            if not entry.resolved_file(repo_root).exists():
                raise ValueError(f"{entry.id}: missing kernel_file {entry.kernel_file}")
        entries.append(entry)
    return entries


def load_supported_kernels(
    *,
    path: Path = SUPPORTED_KERNELS_PATH,
    repo_root: Path | None = None,
    validate_files: bool = False,
) -> list[TraceKernelEntry]:
    """Load and validate the supported trace-kernel whitelist."""
    data = _load_raw_registry(path)
    return _validate_raw_registry(
        data,
        path=path,
        repo_root=repo_root,
        validate_files=validate_files,
    )


def find_supported_kernel(
    kernel_id: str,
    *,
    path: Path = SUPPORTED_KERNELS_PATH,
    repo_root: Path | None = None,
    validate_files: bool = False,
) -> TraceKernelEntry:
    entries = load_supported_kernels(
        path=path,
        repo_root=repo_root,
        validate_files=validate_files,
    )
    by_id = {entry.id: entry for entry in entries}
    if kernel_id in by_id:
        return by_id[kernel_id]

    suggestions = difflib.get_close_matches(kernel_id, sorted(by_id), n=3, cutoff=0.35)
    hint = ""
    if suggestions:
        hint = f" Did you mean: {', '.join(suggestions)}?"
    raise ValueError(
        f"Unsupported trace kernel id {kernel_id!r}.{hint} "
        "Run `python3 workload_optimizer.py list-trace-kernels` to see supported IDs."
    )
