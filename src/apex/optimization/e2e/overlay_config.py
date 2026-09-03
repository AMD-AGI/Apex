"""Derive immutable Magpie views by changing only ``benchmark.docker_image``."""

from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from apex.benchmark import validate_phase_set_contract
from apex.core import ConfigurationError, IntegrityError


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
    sources = (
        ("measurement", measurement),
        ("diagnostic", diagnostic),
        ("replay", replay),
    )
    originals = tuple(_load(source) for _, source in sources)
    validate_phase_set_contract(
        *originals,
        expected_semantics_sha256=workload_semantics_sha256,
    )
    derived: list[dict[str, Any]] = []
    for before in originals:
        after = copy.deepcopy(before)
        after["benchmark"]["docker_image"] = image_id
        _assert_only_image_changed(before, after)
        derived.append(after)
    serialized = tuple(
        yaml.safe_dump(after, sort_keys=False).encode("utf-8") for after in derived
    )
    round_tripped = tuple(_parse(content.decode("utf-8")) for content in serialized)
    for before, observed in zip(originals, round_tripped, strict=True):
        _assert_only_image_changed(before, observed)
    validate_phase_set_contract(
        *round_tripped,
        expected_semantics_sha256=workload_semantics_sha256,
    )

    destinations = tuple(output / f"{kind}.yaml" for kind, _ in sources)
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise IntegrityError(
            "Overlay config path already exists", "immutable_benchmark_view"
        )
    for destination, content in zip(destinations, serialized, strict=True):
        _write_once(destination, content)
    paths = tuple(path.resolve() for path in destinations)
    reloaded = tuple(_load(path) for path in paths)
    for before, observed in zip(originals, reloaded, strict=True):
        _assert_only_image_changed(before, observed)
    validate_phase_set_contract(
        *reloaded,
        expected_semantics_sha256=workload_semantics_sha256,
    )
    return OverlayConfigSet(*paths)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ConfigurationError("Benchmark view is missing or unsafe", "invalid_replay_config")
    try:
        return _parse(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError("Benchmark view is invalid YAML", "invalid_replay_config") from error


def _parse(content: str) -> dict[str, Any]:
    value = yaml.safe_load(content)
    benchmark = value.get("benchmark") if isinstance(value, dict) else None
    if not isinstance(benchmark, dict) or not isinstance(benchmark.get("docker_image"), str):
        raise ConfigurationError("Benchmark view lacks docker_image", "invalid_replay_config")
    return value


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
