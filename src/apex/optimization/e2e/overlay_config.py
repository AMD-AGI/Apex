"""Derive immutable Magpie views by changing only ``benchmark.docker_image``."""

from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from apex.core import ConfigurationError, IntegrityError, sha256_json


@dataclass(frozen=True, slots=True)
class OverlayConfigSet:
    measurement: Path
    diagnostic: Path
    replay: Path


def derive_overlay_configs(
    *,
    measurement: Path,
    diagnostic: Path,
    replay: Path,
    output_dir: Path,
    image_id: str,
    workload_semantics_sha256: str,
) -> OverlayConfigSet:
    """Copy three views and replace exactly one scalar in each document."""

    if not image_id.startswith("sha256:") or len(image_id) != 71:
        raise ConfigurationError("Derived image identity is not immutable", "invalid_image_id")
    output = output_dir.resolve()
    if output_dir.exists() and output_dir.is_symlink():
        raise IntegrityError("Overlay config directory is a symlink", "unsafe_path")
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for kind, source in (
        ("measurement", measurement),
        ("diagnostic", diagnostic),
        ("replay", replay),
    ):
        before = _load(source)
        _validate_semantics(before, workload_semantics_sha256)
        after = copy.deepcopy(before)
        after["benchmark"]["docker_image"] = image_id
        _assert_only_image_changed(before, after)
        destination = output / f"{kind}.yaml"
        _write_once(destination, yaml.safe_dump(after, sort_keys=False).encode("utf-8"))
        reloaded = _load(destination)
        _assert_only_image_changed(before, reloaded)
        _validate_semantics(reloaded, workload_semantics_sha256)
        paths.append(destination.resolve())
    return OverlayConfigSet(*paths)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ConfigurationError("Benchmark view is missing or unsafe", "invalid_replay_config")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError("Benchmark view is invalid YAML", "invalid_replay_config") from error
    benchmark = value.get("benchmark") if isinstance(value, dict) else None
    if not isinstance(benchmark, dict) or not isinstance(benchmark.get("docker_image"), str):
        raise ConfigurationError("Benchmark view lacks docker_image", "invalid_replay_config")
    return value


def _validate_semantics(document: Mapping[str, Any], expected: str) -> None:
    benchmark = document["benchmark"]
    projected = copy.deepcopy(dict(benchmark))
    projected.pop("profiler", None)
    projected.pop("gap_analysis", None)
    projected.pop("docker_image", None)
    projected.pop("run_kind", None)
    if sha256_json(projected) != expected:
        raise IntegrityError(
            "Benchmark workload semantics changed before overlay deployment",
            "benchmark_semantics_changed",
        )
    apex = document.get("apex")
    metadata = apex.get("benchmark_view") if isinstance(apex, Mapping) else None
    if not isinstance(metadata, Mapping) or metadata.get("workload_semantics_sha256") != expected:
        raise IntegrityError(
            "Benchmark view metadata does not bind workload semantics",
            "benchmark_semantics_changed",
        )


def _assert_only_image_changed(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    expected = copy.deepcopy(dict(before))
    expected["benchmark"]["docker_image"] = after["benchmark"].get("docker_image")
    if after != expected:
        raise IntegrityError(
            "Overlay config changed fields other than benchmark.docker_image",
            "benchmark_config_mutation",
        )


def _write_once(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise IntegrityError("Overlay config path already exists", "immutable_benchmark_view")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


__all__ = ["OverlayConfigSet", "derive_overlay_configs"]
