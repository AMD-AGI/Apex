"""Integrity-checked, instruction-only AMD kernel skill package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from apex.core import IntegrityError, sha256_file, sha256_json


_PLUGIN_NAME = "apex-amd-kernel"
_SKILL_NAMES = (
    "amd-hip-kernel-optimization",
    "amd-kernel-debugging",
    "amd-kernel-optimization",
)


@dataclass(frozen=True, slots=True)
class KernelSkillPackage:
    """Exact packaged instructions mounted into one native agent session."""

    root: Path
    skill_paths: Mapping[str, Path]
    digest: str


def load_kernel_skill_package(root: Path | None = None) -> KernelSkillPackage:
    selected = root or Path(__file__).resolve().parent / "plugins" / _PLUGIN_NAME
    selected = selected.resolve(strict=True)
    manifests = (
        selected / ".codex-plugin" / "plugin.json",
        selected / ".claude-plugin" / "plugin.json",
    )
    for path in manifests:
        _validate_manifest(path)
    skill_paths = {
        name: selected / "skills" / name / "SKILL.md" for name in _SKILL_NAMES
    }
    for name, path in skill_paths.items():
        _validate_skill(path, name)
    files = (*manifests, *_skill_asset_files(tuple(skill_paths.values())))
    digest = sha256_json(
        {
            path.relative_to(selected).as_posix(): sha256_file(path)
            for path in files
        }
    )
    return KernelSkillPackage(selected, skill_paths, digest)


def _skill_asset_files(skill_paths: tuple[Path, ...]) -> tuple[Path, ...]:
    files: list[Path] = []
    for skill_path in skill_paths:
        root = skill_path.parent
        if root.is_symlink() or not root.is_dir():
            raise IntegrityError("Kernel skill asset is unavailable", "skill_asset_invalid")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise IntegrityError(
                    "Kernel skill asset is unavailable", "skill_asset_invalid"
                )
            if path.is_file():
                files.append(path)
    return tuple(files)


def _validate_manifest(path: Path) -> None:
    document = _read_json_file(path)
    if document.get("name") != _PLUGIN_NAME or document.get("skills") != "./skills/":
        raise IntegrityError("Kernel skill manifest is invalid", "skill_asset_invalid")


def _validate_skill(path: Path, expected_name: str) -> None:
    text = _read_text_file(path)
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise IntegrityError("Kernel skill frontmatter is missing", "skill_asset_invalid")
    frontmatter, body = text[4:].split("\n---\n", 1)
    try:
        metadata = yaml.safe_load(frontmatter)
    except yaml.YAMLError as error:
        raise IntegrityError("Kernel skill frontmatter is invalid", "skill_asset_invalid") from error
    if (
        not isinstance(metadata, Mapping)
        or set(metadata) != {"name", "description"}
        or metadata.get("name") != expected_name
        or not str(metadata.get("description", "")).strip()
        or not body.strip()
        or "TODO" in text
    ):
        raise IntegrityError("Kernel skill content is invalid", "skill_asset_invalid")


def _read_json_file(path: Path) -> Mapping[str, object]:
    try:
        document = json.loads(_read_text_file(path))
    except json.JSONDecodeError as error:
        raise IntegrityError("Kernel skill manifest is invalid", "skill_asset_invalid") from error
    if not isinstance(document, Mapping):
        raise IntegrityError("Kernel skill manifest is invalid", "skill_asset_invalid")
    return document


def _read_text_file(path: Path) -> str:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("not a regular file")
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise IntegrityError("Kernel skill asset is unavailable", "skill_asset_invalid") from error


__all__ = ["KernelSkillPackage", "load_kernel_skill_package"]
