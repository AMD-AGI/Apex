"""Authority-bound launch config that redirects only InferenceX execution state."""

from __future__ import annotations

import copy
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from apex.core import ConfigurationError, sha256_file, sha256_json


SCHEMA = "apex.magpie-launch-config-projection/v1"


@dataclass(frozen=True, slots=True)
class MagpieLaunchConfigReceipt:
    """Proof that only the runtime InferenceX locator changed."""

    canonical_config_sha256: str
    launch_config_sha256: str
    inferencex_source_root: str
    inferencex_projection_root: str
    inferencex_projection_receipt_sha256: str

    @property
    def sha256(self) -> str:
        return sha256_json(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "canonical_config_sha256": self.canonical_config_sha256,
            "launch_config_sha256": self.launch_config_sha256,
            "allowed_change": "/benchmark/inferencex_path",
            "inferencex_source_root": self.inferencex_source_root,
            "inferencex_projection_root": self.inferencex_projection_root,
            "inferencex_projection_receipt_sha256": (
                self.inferencex_projection_receipt_sha256
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_sha256": self.sha256}


def materialize_magpie_launch_config(
    canonical_path: Path,
    destination: Path,
    *,
    canonical_sha256: str,
    inferencex_source_root: Path,
    inferencex_projection_root: Path,
    inferencex_projection_receipt_sha256: str,
) -> MagpieLaunchConfigReceipt:
    """Write one immutable config with exactly one execution-locator change."""

    canonical = _load_config(canonical_path, canonical_sha256)
    benchmark = canonical.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise _invalid("Canonical Magpie benchmark config is invalid")
    source = inferencex_source_root.resolve(strict=True)
    projection = inferencex_projection_root.resolve(strict=True)
    existing = benchmark.get("inferencex_path")
    if existing and _config_relative_path(canonical_path, existing) != source:
        raise _invalid("Canonical InferenceX locator differs from dependencies")
    launch = copy.deepcopy(canonical)
    launch_benchmark = launch["benchmark"]
    assert isinstance(launch_benchmark, dict)
    launch_benchmark["inferencex_path"] = str(projection)
    _write_yaml(destination, launch)
    _verify_only_locator_changed(canonical, _load_config(destination, sha256_file(destination)))
    return MagpieLaunchConfigReceipt(
        canonical_config_sha256=canonical_sha256,
        launch_config_sha256=sha256_file(destination),
        inferencex_source_root=str(source),
        inferencex_projection_root=str(projection),
        inferencex_projection_receipt_sha256=inferencex_projection_receipt_sha256,
    )


def _config_relative_path(config_path: Path, value: object) -> Path:
    locator = Path(str(value))
    if not locator.is_absolute():
        locator = config_path.parent / locator
    return locator.resolve()


def _load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    try:
        observed = path.lstat()
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise _invalid("Cannot read Magpie launch config") from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or sha256_file(path) != expected_sha256
        or not isinstance(value, dict)
        or any(not isinstance(key, str) for key in value)
    ):
        raise _invalid("Magpie launch config identity is invalid")
    return value


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    payload = yaml.safe_dump(
        dict(value), allow_unicode=True, default_flow_style=False, sort_keys=True
    ).encode("utf-8")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o400
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise _invalid("Cannot write Magpie launch config")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_only_locator_changed(
    canonical: Mapping[str, Any], launch: Mapping[str, Any]
) -> None:
    left = copy.deepcopy(dict(canonical))
    right = copy.deepcopy(dict(launch))
    for value in (left, right):
        benchmark = value.get("benchmark")
        if not isinstance(benchmark, dict):
            raise _invalid("Magpie launch config benchmark is invalid")
        benchmark.pop("inferencex_path", None)
    if left != right:
        raise _invalid("Magpie launch config changed workload semantics")


def _invalid(message: str) -> ConfigurationError:
    return ConfigurationError(message, "magpie_launch_config_projection_invalid")


__all__ = [
    "MagpieLaunchConfigReceipt",
    "materialize_magpie_launch_config",
]
