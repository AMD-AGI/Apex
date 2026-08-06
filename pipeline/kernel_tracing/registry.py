"""Supported trace-kernel registries keyed by benchmark Docker image."""

from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


REGISTRY_DIR = Path(__file__).with_name("registries")
VLLM_TRACE_IMAGE = "vllm/vllm-openai-rocm:v0.23.0"
SGLANG_TRACE_IMAGE = "lmsysorg/sglang:v0.5.12-rocm720-mi35x"
SUPPORTED_TRACE_IMAGES = (VLLM_TRACE_IMAGE, SGLANG_TRACE_IMAGE)
TRACE_IMAGE_REGISTRY_PATHS = {
    VLLM_TRACE_IMAGE: REGISTRY_DIR / "vllm_v0_23_0.yaml",
    SGLANG_TRACE_IMAGE: REGISTRY_DIR / "sglang_v0_5_12_rocm720_mi35x.yaml",
}

# Kept as a compatibility constant for older imports. Runtime registry
# selection is image-aware and should use registry_path_for_image().
SUPPORTED_KERNELS_PATH = TRACE_IMAGE_REGISTRY_PATHS[VLLM_TRACE_IMAGE]

VALID_REPOS = {"aiter", "vllm", "sglang"}
VALID_KERNEL_TYPES = {"triton", "hip"}
VALID_TRACE_MODES = {
    "triton-launch",
    "aiter-compile-ops",
    "vllm-custom-op",
    "sglang-custom-op",
}
VALID_PATCH_STRATEGIES = {"static"}


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


def supported_trace_images() -> tuple[str, ...]:
    return SUPPORTED_TRACE_IMAGES


def registry_path_for_image(docker_image: str) -> Path:
    image = str(docker_image or "").strip()
    try:
        return TRACE_IMAGE_REGISTRY_PATHS[image]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_TRACE_IMAGES)
        raise ValueError(
            f"Unsupported trace-kernel Docker image {image!r}. "
            f"Supported images: {supported}."
        ) from exc


def _load_raw_registry(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Trace kernel registry must be a mapping: {path}")
    return data


def _validate_entry(
    raw: dict[str, Any],
    *,
    idx: int,
    seen: set[str],
    repo_root: Path | None,
    validate_files: bool,
) -> TraceKernelEntry:
    required = set(TraceKernelEntry.__dataclass_fields__)
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
    return entry


def _validate_raw_registry(
    data: dict[str, Any],
    *,
    path: Path,
    docker_image: str | None,
    repo_root: Path | None,
    validate_files: bool,
) -> list[TraceKernelEntry]:
    if data.get("schema_version") != 2:
        raise ValueError(f"Unsupported trace kernel registry schema in {path}")

    registry_image = data.get("docker_image")
    if registry_image not in SUPPORTED_TRACE_IMAGES:
        raise ValueError(f"{path}: unsupported docker_image {registry_image!r}")
    if docker_image and registry_image != docker_image:
        raise ValueError(
            f"{path}: registry is for {registry_image!r}, requested {docker_image!r}"
        )

    image_metadata = data.get("image_metadata")
    if not isinstance(image_metadata, dict) or image_metadata.get("image") != registry_image:
        raise ValueError(f"{path}: image_metadata.image must match docker_image")

    package_sources = data.get("package_sources")
    if not isinstance(package_sources, dict) or not package_sources:
        raise ValueError(f"{path}: registry is missing package_sources")
    unknown_sources = sorted(set(package_sources) - VALID_REPOS)
    if unknown_sources:
        raise ValueError(f"{path}: unsupported package_sources: {unknown_sources}")

    raw_kernels = data.get("kernels")
    if not isinstance(raw_kernels, list):
        raise ValueError(f"{path}: registry is missing kernels list")

    entries: list[TraceKernelEntry] = []
    seen: set[str] = set()
    for idx, raw in enumerate(raw_kernels):
        if not isinstance(raw, dict):
            raise ValueError(f"Trace kernel entry #{idx} must be a mapping")
        entry = _validate_entry(
            raw,
            idx=idx,
            seen=seen,
            repo_root=repo_root,
            validate_files=validate_files,
        )
        if entry.repo not in package_sources:
            raise ValueError(
                f"{entry.id}: repo {entry.repo!r} missing from package_sources"
            )
        entries.append(entry)
    return entries


def load_supported_kernels(
    *,
    docker_image: str = "",
    path: Path | None = None,
    repo_root: Path | None = None,
    validate_files: bool = False,
) -> list[TraceKernelEntry]:
    """Load and validate the registry for one supported Docker image."""
    if path is None:
        if not docker_image:
            raise ValueError(
                "Trace kernel registry selection requires --docker-image or "
                "-b/--benchmark-config."
            )
        path = registry_path_for_image(docker_image)
    data = _load_raw_registry(path)
    return _validate_raw_registry(
        data,
        path=path,
        docker_image=docker_image or None,
        repo_root=repo_root,
        validate_files=validate_files,
    )


def find_supported_kernel(
    kernel_id: str,
    *,
    docker_image: str,
    path: Path | None = None,
    repo_root: Path | None = None,
    validate_files: bool = False,
) -> TraceKernelEntry:
    entries = load_supported_kernels(
        docker_image=docker_image,
        path=path,
        repo_root=repo_root,
        validate_files=False,
    )
    by_id = {entry.id: entry for entry in entries}
    if kernel_id in by_id:
        entry = by_id[kernel_id]
        if validate_files:
            if repo_root is None:
                raise ValueError("repo_root is required when validate_files=True")
            if not entry.resolved_file(repo_root).exists():
                raise ValueError(f"{entry.id}: missing kernel_file {entry.kernel_file}")
        return entry

    suggestions = difflib.get_close_matches(kernel_id, sorted(by_id), n=3, cutoff=0.35)
    hint = ""
    if suggestions:
        hint = f" Did you mean: {', '.join(suggestions)}?"
    raise ValueError(
        f"Unsupported trace kernel id {kernel_id!r} for Docker image {docker_image!r}."
        f"{hint} Run `python3 workload_optimizer.py list-trace-kernels --docker-image "
        f"{docker_image}` to see supported IDs."
    )


def registry_summary(path: Path) -> dict[str, Any]:
    data = _load_raw_registry(path)
    entries = _validate_raw_registry(
        data,
        path=path,
        docker_image=None,
        repo_root=None,
        validate_files=False,
    )
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.repo] = counts.get(entry.repo, 0) + 1
    return {
        "docker_image": data["docker_image"],
        "path": str(path),
        "count": len(entries),
        "counts_by_repo": counts,
    }
