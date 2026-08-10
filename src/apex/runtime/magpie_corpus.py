"""Exact Magpie benchmark-corpus manifest generation and verification."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .repositories import BootstrapError, canonical_repository, inspect_repository, run_command


SCHEMA = "apex.magpie-benchmark-corpus/v1"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_PREFIX = "examples/benchmarks/"


@dataclass(frozen=True, slots=True)
class CorpusFile:
    path: str
    sha256: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class MagpieCorpusManifest:
    repository: str
    commit: str
    repository_tree: str
    benchmark_tree: str
    files: tuple[CorpusFile, ...]
    manifest_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "repository": self.repository,
            "commit": self.commit,
            "repository_tree": self.repository_tree,
            "benchmark_tree": self.benchmark_tree,
            "files": [item.to_dict() for item in self.files],
            "summary": {"config_count": len(self.files)},
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "manifest_sha256": self.manifest_sha256}


def load_magpie_corpus_manifest(path: Path) -> MagpieCorpusManifest:
    """Load a strict, self-digested Magpie corpus manifest."""

    try:
        raw = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError(f"invalid Magpie corpus manifest {path}: {error}") from error
    if not isinstance(raw, Mapping) or raw.get("schema") != SCHEMA:
        raise BootstrapError("unsupported Magpie corpus manifest schema")
    expected_keys = {
        "schema", "repository", "commit", "repository_tree", "benchmark_tree",
        "files", "summary", "manifest_sha256",
    }
    if set(raw) != expected_keys:
        raise BootstrapError("Magpie corpus manifest field set differs")
    manifest = _parse_manifest(raw)
    if manifest.manifest_sha256 != _digest(manifest.payload()):
        raise BootstrapError("Magpie corpus manifest digest differs")
    return manifest


def build_magpie_corpus_manifest(root: Path) -> MagpieCorpusManifest:
    """Build a manifest from one exact clean Magpie Git checkout."""

    state = inspect_repository(root)
    if state.dirty_paths:
        raise BootstrapError("Magpie corpus checkout is dirty")
    benchmark_tree = _git(root, "rev-parse", "HEAD:examples/benchmarks")
    files = tuple(
        CorpusFile(path, hashlib.sha256((root / path).read_bytes()).hexdigest())
        for path in _config_paths(root)
    )
    placeholder = MagpieCorpusManifest(
        state.remote, state.commit, state.tree, benchmark_tree, files, ""
    )
    return MagpieCorpusManifest(
        state.remote,
        state.commit,
        state.tree,
        benchmark_tree,
        files,
        _digest(placeholder.payload()),
    )


def verify_magpie_corpus_manifest(
    manifest: MagpieCorpusManifest,
    root: Path,
    *,
    repository: str,
    commit: str,
) -> Mapping[str, Any]:
    """Verify lock identity, both Git trees, every path, and every byte hash."""

    if (
        canonical_repository(manifest.repository) != canonical_repository(repository)
        or manifest.commit != commit
    ):
        raise BootstrapError("Magpie corpus manifest differs from dependency lock")
    observed = build_magpie_corpus_manifest(root)
    expected = {**manifest.to_dict(), "repository": canonical_repository(manifest.repository)}
    actual = {**observed.to_dict(), "repository": canonical_repository(observed.repository)}
    if actual != expected:
        raise BootstrapError("Magpie benchmark corpus differs from frozen manifest")
    return manifest.to_dict()


def _parse_manifest(raw: Mapping[str, Any]) -> MagpieCorpusManifest:
    repository = _text(raw.get("repository"), "repository")
    commit = _hex(raw.get("commit"), _COMMIT, "commit")
    repository_tree = _hex(raw.get("repository_tree"), _COMMIT, "repository_tree")
    benchmark_tree = _hex(raw.get("benchmark_tree"), _COMMIT, "benchmark_tree")
    manifest_sha256 = _hex(raw.get("manifest_sha256"), _DIGEST, "manifest_sha256")
    files = _files(raw.get("files"))
    summary = raw.get("summary")
    if not isinstance(summary, Mapping) or set(summary) != {"config_count"}:
        raise BootstrapError("Magpie corpus summary is invalid")
    if summary.get("config_count") != len(files):
        raise BootstrapError("Magpie corpus count differs")
    return MagpieCorpusManifest(
        repository, commit, repository_tree, benchmark_tree, files, manifest_sha256
    )


def _files(value: object) -> tuple[CorpusFile, ...]:
    if not isinstance(value, list) or not value:
        raise BootstrapError("Magpie corpus files are invalid")
    parsed: list[CorpusFile] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise BootstrapError("Magpie corpus file entry is invalid")
        path = _text(item.get("path"), "path")
        if not path.startswith(_PREFIX) or Path(path).is_absolute() or ".." in Path(path).parts:
            raise BootstrapError("Magpie corpus path is unsafe")
        if Path(path).suffix.lower() not in {".yaml", ".yml"}:
            raise BootstrapError("Magpie corpus path is not YAML")
        parsed.append(CorpusFile(path, _hex(item.get("sha256"), _DIGEST, "sha256")))
    if tuple(item.path for item in parsed) != tuple(sorted({item.path for item in parsed})):
        raise BootstrapError("Magpie corpus paths are duplicated or unsorted")
    return tuple(parsed)


def _config_paths(root: Path) -> tuple[str, ...]:
    directory = root / "examples" / "benchmarks"
    if directory.is_symlink() or not directory.is_dir():
        raise BootstrapError("Magpie benchmark corpus directory is invalid")
    paths: list[str] = []
    for path in directory.rglob("*"):
        if path.suffix.lower() not in {".yaml", ".yml"}:
            continue
        if path.is_symlink() or not path.is_file():
            raise BootstrapError("Magpie benchmark corpus contains an unsafe path")
        paths.append(path.relative_to(root).as_posix())
    return tuple(sorted(paths))


def _digest(value: Mapping[str, Any]) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _git(root: Path, *args: str) -> str:
    return run_command(("git", "-C", str(root), *args)).stdout.strip()


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BootstrapError(f"Magpie corpus {field} is invalid")
    return value.strip()


def _hex(value: object, pattern: re.Pattern[str], field: str) -> str:
    text = _text(value, field)
    if not pattern.fullmatch(text):
        raise BootstrapError(f"Magpie corpus {field} is invalid")
    return text


__all__ = [
    "CorpusFile",
    "MagpieCorpusManifest",
    "build_magpie_corpus_manifest",
    "load_magpie_corpus_manifest",
    "verify_magpie_corpus_manifest",
]
