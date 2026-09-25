"""Bounded, race-detecting JSON/YAML loader for task descriptors."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.events import AliasEvent, CollectionEndEvent, CollectionStartEvent

from apex.core import ContractError


_MAX_DESCRIPTOR_BYTES = 1024 * 1024
_MAX_DOCUMENT_EVENTS = 10_000
_MAX_DOCUMENT_DEPTH = 32
_READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class _Snapshot:
    identity: tuple[int, ...]
    payload: bytes


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def load_mapping_document(
    path: Path,
    *,
    reason_code: str,
    document_name: str,
) -> Mapping[str, Any]:
    """Load one stable, bounded descriptor and require an object root."""

    selected = Path(path)
    try:
        before = _read_snapshot(selected)
        value = _decode_document(before.payload, selected.suffix.lower())
        _validate_tree(value)
        after = _read_snapshot(selected)
        if before != after:
            raise ValueError("descriptor bytes or file identity drifted while parsing")
        if not isinstance(value, Mapping):
            raise ValueError("document root must be an object")
        return value
    except ContractError:
        raise
    except (OSError, UnicodeDecodeError, ValueError, TypeError, RecursionError, yaml.YAMLError) as error:
        raise ContractError(
            f"Cannot load {document_name}: {selected}",
            reason_code,
            details={"cause": str(error)},
        ) from error


def _read_snapshot(path: Path) -> _Snapshot:
    before = path.lstat()
    _validate_file(before)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        _validate_file(opened)
        if _identity(before) != _identity(opened):
            raise ValueError("descriptor identity drifted before read")
        payload = _read_bounded(descriptor)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = path.lstat()
    if not (
        _identity(before) == _identity(after_fd) == _identity(after_path)
        and len(payload) == before.st_size
    ):
        raise ValueError("descriptor bytes or file identity drifted during read")
    return _Snapshot(_identity(before), payload)


def _validate_file(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or not 0 < metadata.st_size <= _MAX_DESCRIPTOR_BYTES
    ):
        raise ValueError("descriptor must be one bounded non-hardlinked regular file")


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_bounded(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, _MAX_DESCRIPTOR_BYTES + 1 - total))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > _MAX_DESCRIPTOR_BYTES:
            raise ValueError("descriptor exceeds the byte limit")


def _decode_document(payload: bytes, suffix: str) -> Any:
    text = payload.decode("utf-8")
    if suffix in {".yaml", ".yml"}:
        _validate_yaml_events(text)
        return yaml.load(text, Loader=_UniqueSafeLoader)
    return json.loads(
        text,
        object_pairs_hook=_unique_json_mapping,
        parse_constant=_reject_json_constant,
    )


def _validate_yaml_events(text: str) -> None:
    depth = 0
    for count, event in enumerate(yaml.parse(text), start=1):
        if count > _MAX_DOCUMENT_EVENTS or isinstance(event, AliasEvent):
            raise ValueError("YAML aliases or excessive events are forbidden")
        if isinstance(event, CollectionStartEvent):
            depth += 1
            if depth > _MAX_DOCUMENT_DEPTH:
                raise ValueError("document nesting exceeds the depth limit")
        elif isinstance(event, CollectionEndEvent):
            depth -= 1


def _unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


def _unique_json_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _validate_tree(value: Any, *, depth: int = 0, seen: list[int] | None = None) -> None:
    if depth > _MAX_DOCUMENT_DEPTH:
        raise ValueError("document nesting exceeds the depth limit")
    counter = seen if seen is not None else [0]
    counter[0] += 1
    if counter[0] > _MAX_DOCUMENT_EVENTS:
        raise ValueError("document contains too many values")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("document keys must be strings")
        for child in value.values():
            _validate_tree(child, depth=depth + 1, seen=counter)
    elif isinstance(value, list):
        for child in value:
            _validate_tree(child, depth=depth + 1, seen=counter)


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_mapping,
)


__all__ = ["load_mapping_document"]
