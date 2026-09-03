"""Strict lock contract for the content-addressed lm-eval runtime."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from .repositories import BootstrapError


LOCK_SCHEMA = "apex.lm-eval-runtime-lock/v1"
RUNTIME_SCHEMA = "apex.lm-eval-runtime/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT = re.compile(r"[0-9a-f]{40}")
_ABI = re.compile(r"cpython-[0-9]{2,3}")


@dataclass(frozen=True, slots=True)
class DownloadLock:
    """One immutable downloadable artifact."""

    filename: str
    url: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class WheelLock:
    """One wheel installed into the isolated target tree."""

    name: str
    version: str
    filename: str
    sha256: str
    download: DownloadLock | None = None
    build_source: DownloadLock | None = None

    @property
    def requirement(self) -> str:
        return f"{self.name}=={self.version}"


@dataclass(frozen=True, slots=True)
class LmEvalRuntimeLock:
    """Validated runtime lock plus its exact evaluator identity."""

    path: Path
    source: DownloadLock
    source_repository: str
    source_date_epoch: int
    wheels: tuple[WheelLock, ...]
    base_distributions: Mapping[str, str]
    identity: Mapping[str, str]
    installed_tree_sha256: str
    runtime_sha256: str
    sha256: str

    @property
    def base_image(self) -> str:
        return self.identity["base_image_repo_digest"]


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BootstrapError(f"{field} must be a non-empty string")
    return value.strip()


def _digest(value: Any, field: str) -> str:
    result = _text(value, field)
    if not _SHA256.fullmatch(result):
        raise BootstrapError(f"{field} must be a lowercase SHA-256")
    return result


def _safe_filename(value: Any, field: str) -> str:
    result = _text(value, field)
    if Path(result).name != result or result in {".", ".."}:
        raise BootstrapError(f"{field} must be a single safe filename")
    return result


def _download(raw: Any, field: str) -> DownloadLock:
    if not isinstance(raw, Mapping) or set(raw) != {
        "filename", "url", "size_bytes", "sha256"
    }:
        raise BootstrapError(f"{field} must contain exact download fields")
    url = _text(raw["url"], f"{field}.url")
    if urlparse(url).scheme != "https":
        raise BootstrapError(f"{field}.url must use https")
    size = raw["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise BootstrapError(f"{field}.size_bytes must be a positive integer")
    return DownloadLock(
        filename=_safe_filename(raw["filename"], f"{field}.filename"),
        url=url,
        size_bytes=size,
        sha256=_digest(raw["sha256"], f"{field}.sha256"),
    )


def _wheel(raw: Any, index: int) -> WheelLock:
    field = f"wheels[{index}]"
    if not isinstance(raw, Mapping):
        raise BootstrapError(f"{field} must be an object")
    allowed = {"name", "version", "filename", "sha256", "download", "build_source"}
    if not set(raw).issubset(allowed) or not {"name", "version", "filename", "sha256"} <= set(raw):
        raise BootstrapError(f"{field} contains invalid fields")
    has_download = "download" in raw
    has_source = "build_source" in raw
    if has_download == has_source:
        raise BootstrapError(f"{field} requires exactly one artifact source")
    filename = _safe_filename(raw["filename"], f"{field}.filename")
    artifact = _download(
        raw["download"] if has_download else raw["build_source"],
        f"{field}.{'download' if has_download else 'build_source'}",
    )
    if has_download and artifact.filename != filename:
        raise BootstrapError(f"{field} download filename differs from wheel filename")
    return WheelLock(
        name=_text(raw["name"], f"{field}.name"),
        version=_text(raw["version"], f"{field}.version"),
        filename=filename,
        sha256=_digest(raw["sha256"], f"{field}.sha256"),
        download=artifact if has_download else None,
        build_source=artifact if has_source else None,
    )


def _identity(raw: Any) -> dict[str, str]:
    required = {
        "lm_eval_commit", "lm_eval_tree", "lm_eval_version", "python_abi",
        "python_soabi", "base_image_id", "base_image_repo_digest",
        "inferencex_commit", "inferencex_tree",
    }
    if not isinstance(raw, Mapping) or set(raw) != required:
        raise BootstrapError("identity must contain the exact evaluator identity fields")
    result = {key: _text(raw[key], f"identity.{key}") for key in sorted(required)}
    for key in ("lm_eval_commit", "lm_eval_tree", "inferencex_commit", "inferencex_tree"):
        if not _GIT.fullmatch(result[key]):
            raise BootstrapError(f"identity.{key} must be lowercase 40-hex")
    if not _ABI.fullmatch(result["python_abi"]):
        raise BootstrapError("identity.python_abi must be a CPython cache tag")
    if not result["base_image_id"].startswith("sha256:"):
        raise BootstrapError("identity.base_image_id must be a sha256 image ID")
    _digest(result["base_image_id"][7:], "identity.base_image_id")
    digest = result["base_image_repo_digest"].rsplit("@sha256:", 1)
    if len(digest) != 2:
        raise BootstrapError("identity.base_image_repo_digest must be immutable")
    _digest(digest[1], "identity.base_image_repo_digest")
    return result


def _base_distributions(raw: Any) -> dict[str, str]:
    if not isinstance(raw, Mapping) or not raw:
        raise BootstrapError("base_distributions must be a non-empty object")
    result = {
        _text(key, "base distribution name"): _text(value, f"base_distributions.{key}")
        for key, value in raw.items()
    }
    if list(result) != sorted(result, key=str.casefold):
        raise BootstrapError("base_distributions must be case-insensitively sorted")
    return result


def load_lm_eval_runtime_lock(path: Path) -> LmEvalRuntimeLock:
    """Load and strictly validate the reviewed lm-eval runtime lock."""

    try:
        payload = path.read_bytes()
        raw = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError(f"cannot read lm-eval runtime lock {path}: {error}") from error
    required = {
        "schema", "source", "source_repository", "source_date_epoch", "wheels",
        "base_distributions", "identity", "installed_tree_sha256", "runtime_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != required or raw.get("schema") != LOCK_SCHEMA:
        raise BootstrapError(f"lm-eval runtime lock must be exact {LOCK_SCHEMA}")
    wheels_raw = raw["wheels"]
    if not isinstance(wheels_raw, list) or not wheels_raw:
        raise BootstrapError("wheels must be a non-empty list")
    wheels = tuple(_wheel(value, index) for index, value in enumerate(wheels_raw))
    names = [item.name.casefold().replace("_", "-") for item in wheels]
    if names != sorted(names) or len(names) != len(set(names)):
        raise BootstrapError("wheels must be uniquely sorted by normalized name")
    epoch = raw["source_date_epoch"]
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
        raise BootstrapError("source_date_epoch must be a positive integer")
    identity = _identity(raw["identity"])
    base_distributions = _base_distributions(raw["base_distributions"])
    base_names = {
        name.casefold().replace("_", "-") for name in base_distributions
    }
    overlap = sorted(set(names) & base_names)
    if overlap:
        raise BootstrapError(
            "target wheels must not shadow base distributions: " + ", ".join(overlap)
        )
    lm_eval = [wheel for wheel in wheels if wheel.name.casefold().replace("_", "-") == "lm-eval"]
    source = _download(raw["source"], "source")
    if (
        len(lm_eval) != 1
        or lm_eval[0].version != identity["lm_eval_version"]
        or lm_eval[0].build_source != source
    ):
        raise BootstrapError(
            "lm_eval wheel must be built from the locked source at the locked version"
        )
    return LmEvalRuntimeLock(
        path=path.resolve(),
        source=source,
        source_repository=_text(raw["source_repository"], "source_repository"),
        source_date_epoch=epoch,
        wheels=wheels,
        base_distributions=base_distributions,
        identity=identity,
        installed_tree_sha256=_digest(raw["installed_tree_sha256"], "installed_tree_sha256"),
        runtime_sha256=_digest(raw["runtime_sha256"], "runtime_sha256"),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


__all__ = [
    "DownloadLock", "LmEvalRuntimeLock", "RUNTIME_SCHEMA", "WheelLock",
    "load_lm_eval_runtime_lock",
]
