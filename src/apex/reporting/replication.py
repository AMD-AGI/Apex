"""Replication-guide projection from explicitly committed run receipts."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from apex.core import canonical_json_bytes, sha256_bytes
from apex.rl import EpisodeGraph


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_IMAGE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SECRET = re.compile(
    r"(?:sk-(?:ant-)?[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{12,}|"
    r"github_pat_[A-Za-z0-9_]{12,}|Bearer\s+\S+|https?://[^\s/@:]+:[^\s/@]+@)"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token)"
    r"\s*[:=]\s*[\"']?(?!\[REDACTED\])\S{4,}",
    re.I,
)
_CREDENTIAL_OPTION = re.compile(
    r"^--?(?:api[_-]?key|authorization|password|secret|access[_-]?token)$", re.I
)
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token)$", re.I
)


@dataclass(frozen=True, slots=True)
class ReplicationProjection:
    document: Mapping[str, Any]
    markdown: str

    @property
    def json_bytes(self) -> bytes:
        return canonical_json_bytes(self.document)

    @property
    def markdown_bytes(self) -> bytes:
        return self.markdown.encode("utf-8")

    @property
    def digest(self) -> str:
        return sha256_bytes(self.json_bytes)


def build_replication_guide(graph: EpisodeGraph) -> ReplicationProjection:
    """Never invent a command: only render argv committed in canonical events."""

    declarations = [
        event.payload["replication"]
        for event in graph.parent.events
        if isinstance(event.payload.get("replication"), Mapping)
    ]
    problems: list[str] = []
    if not declarations:
        declaration: Mapping[str, Any] = {}
        problems.append("replication_declaration_missing")
    elif len(declarations) > 1 and any(item != declarations[0] for item in declarations[1:]):
        declaration = {}
        problems.append("replication_declaration_conflict")
    else:
        declaration = declarations[-1]

    dependencies = _mapping_list(declaration.get("dependency_receipts"))
    sources = _mapping_list(declaration.get("source_commits"))
    commands, command_problems = _commands(declaration.get("commands"))
    problems.extend(command_problems)
    parent_image = _text(declaration.get("parent_image_digest"))
    derived_image = _text(declaration.get("derived_image_digest"))
    if not dependencies:
        problems.append("dependency_receipts_missing")
    for item in dependencies:
        commit = _text(item.get("commit"))
        digest = _text(item.get("digest"))
        if not (
            (commit is not None and _COMMIT.fullmatch(commit))
            or (digest is not None and re.fullmatch(r"[0-9a-f]{64}", digest))
        ):
            problems.append("dependency_receipt_invalid")
    if not sources:
        problems.append("source_commits_missing")
    for item in sources:
        commit = _text(item.get("commit"))
        if commit is None or not _COMMIT.fullmatch(commit):
            problems.append("source_commit_invalid")
    if parent_image is None or not _IMAGE.fullmatch(parent_image):
        problems.append("parent_image_digest_invalid")
    if not commands:
        problems.append("replication_commands_missing")
    kept = any(child.verdict == "keep" for child in graph.children)
    if kept:
        if derived_image is None or not _IMAGE.fullmatch(derived_image):
            problems.append("derived_image_digest_invalid")
        names = {item["name"] for item in commands}
        for required in ("apply_bundle", "build_image", "clean_replay"):
            if required not in names:
                problems.append(f"{required}_command_missing")
    problems = sorted(set(problems))
    document = {
        "schema_name": "apex.replication_guide",
        "schema_version": 1,
        "episode_graph_id": graph.graph_id,
        "run_id": graph.run_id,
        "reproducible": not problems,
        "validation_reasons": problems,
        "dependency_receipts": dependencies,
        "source_commits": sources,
        "parent_image_digest": parent_image,
        "derived_image_digest": derived_image,
        "commands": commands,
        "benchmark_config_receipts": _mapping_list(
            declaration.get("benchmark_config_receipts")
        ),
        "bundle_receipt": (
            _redact_mapping(declaration["bundle_receipt"])
            if isinstance(declaration.get("bundle_receipt"), Mapping)
            else None
        ),
    }
    return ReplicationProjection(document, _render(document))


def _commands(value: object) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(value, (list, tuple)):
        return [], []
    commands: list[dict[str, Any]] = []
    problems: list[str] = []
    names: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            problems.append("replication_command_invalid")
            continue
        name = _text(raw.get("name"))
        argv = raw.get("argv")
        if name is None or name in names or not isinstance(argv, (list, tuple)) or not argv:
            problems.append("replication_command_invalid")
            continue
        args = [str(item) for item in argv]
        if name is not None and _SECRET.search(name):
            problems.append("replication_command_contains_secret")
            name = _SECRET.sub("[REDACTED]", name)
        if any(not item or "\x00" in item for item in args):
            problems.append("replication_command_invalid")
            continue
        if _argv_contains_secret(args):
            problems.append("replication_command_contains_secret")
            args = _redact_argv(args)
        names.add(name)
        commands.append({"name": name, "argv": args})
    return commands, problems


def _render(document: Mapping[str, Any]) -> str:
    lines = [
        "# Apex replication guide",
        "",
        f"- Run: `{document['run_id']}`",
        f"- Episode graph: `{document['episode_graph_id']}`",
        f"- Reproducible from committed receipts: `{'yes' if document['reproducible'] else 'no'}`",
        "",
    ]
    if document["validation_reasons"]:
        lines.extend(["## Missing or invalid evidence", ""])
        lines.extend(f"- `{reason}`" for reason in document["validation_reasons"])
        lines.append("")
    lines.extend(["## Pinned inputs", ""])
    lines.append(f"- Parent image: `{document['parent_image_digest'] or 'missing'}`")
    lines.append(f"- Derived image: `{document['derived_image_digest'] or 'not produced'}`")
    for source in document["source_commits"]:
        lines.append(
            f"- Source `{_inline(source.get('name', 'repository'))}`: "
            f"`{_inline(source.get('commit', 'missing'))}`"
        )
    for dependency in document["dependency_receipts"]:
        lines.append(
            f"- Dependency `{_inline(dependency.get('name', 'dependency'))}`: "
            f"`{_inline(dependency.get('commit', dependency.get('digest', 'missing')))}`"
        )
    lines.extend(["", "## Exact commands", ""])
    if not document["commands"]:
        lines.append("No executable replication argv was committed; no command is inferred.")
    for command in document["commands"]:
        lines.extend(
            [
                f"### {_inline(command['name'])}",
                "",
            ]
        )
        rendered = shlex.join(command["argv"])
        lines.extend(f"    {line}" for line in rendered.splitlines() or [""])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _mapping_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [_redact_mapping(item) for item in value if isinstance(item, Mapping)]


def _redact_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
        name = str(key)
        if _SECRET_KEY.search(name):
            redacted[name] = "[REDACTED]"
        elif isinstance(item, Mapping):
            redacted[name] = _redact_mapping(item)
        elif isinstance(item, (list, tuple)):
            redacted[name] = [
                _redact_mapping(child) if isinstance(child, Mapping) else _redact_text(child)
                for child in item
            ]
        else:
            redacted[name] = _redact_text(item)
    return redacted


def _redact_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    return _SECRET_ASSIGNMENT.sub("[REDACTED]", _SECRET.sub("[REDACTED]", value))


def _argv_contains_secret(args: Sequence[str]) -> bool:
    return any(
        _SECRET.search(item)
        or _SECRET_ASSIGNMENT.search(item)
        or (_CREDENTIAL_OPTION.fullmatch(item) and index + 1 < len(args))
        for index, item in enumerate(args)
    )


def _redact_argv(args: Sequence[str]) -> list[str]:
    result: list[str] = []
    redact_next = False
    for item in args:
        if redact_next:
            result.append("[REDACTED]")
            redact_next = False
            continue
        result.append(str(_redact_text(item)))
        redact_next = bool(_CREDENTIAL_OPTION.fullmatch(item))
    return result


def _text(value: object) -> str | None:
    return None if value is None or not str(value).strip() else str(value)


def _inline(value: object) -> str:
    return str(value).replace("`", "\\`").replace("\r", " ").replace("\n", " ")


__all__ = ["ReplicationProjection", "build_replication_guide"]
