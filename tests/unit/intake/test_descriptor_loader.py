from __future__ import annotations

import os
from pathlib import Path

import pytest

from apex.core import ContractError
from apex.intake import descriptor_loader


def _load(path: Path) -> object:
    return descriptor_loader.load_mapping_document(
        path,
        reason_code="invalid_test_descriptor",
        document_name="test descriptor",
    )


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_loader_accepts_stable_json_and_yaml_mapping(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"task{suffix}"
    path.write_text('{"task_id": "rms", "values": [1, 2]}', encoding="utf-8")

    assert _load(path) == {"task_id": "rms", "values": [1, 2]}


@pytest.mark.parametrize(
    ("payload", "cause"),
    [
        ("first: &value [1]\nsecond: *value\n", "aliases"),
        ("task_id: first\ntask_id: second\n", "duplicate key"),
        ("1: value\n", "keys must be strings"),
        ("root: " + "[" * 33 + "0" + "]" * 33 + "\n", "depth limit"),
        ("root:\n" + "  - value\n" * 10_001, "excessive events"),
    ],
)
def test_yaml_structure_limits_fail_closed(
    tmp_path: Path,
    payload: str,
    cause: str,
) -> None:
    path = tmp_path / "task.yaml"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ContractError) as raised:
        _load(path)

    assert raised.value.reason_code == "invalid_test_descriptor"
    assert cause in raised.value.details["cause"]


@pytest.mark.parametrize(
    ("payload", "cause"),
    [
        ('{"task_id": "first", "task_id": "second"}', "duplicate key"),
        ('{"value": NaN}', "non-finite JSON"),
        ('{"root":' + "[" * 33 + "0" + "]" * 33 + "}", "depth limit"),
    ],
)
def test_json_structure_limits_fail_closed(
    tmp_path: Path,
    payload: str,
    cause: str,
) -> None:
    path = tmp_path / "task.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ContractError) as raised:
        _load(path)

    assert cause in raised.value.details["cause"]


def test_loader_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(source)

    with pytest.raises(ContractError) as symlink_error:
        _load(symlink)
    assert "non-hardlinked regular file" in symlink_error.value.details["cause"]

    hardlink = tmp_path / "hardlink.json"
    os.link(source, hardlink)
    with pytest.raises(ContractError) as hardlink_error:
        _load(hardlink)
    assert "non-hardlinked regular file" in hardlink_error.value.details["cause"]


@pytest.mark.parametrize("kind", ["empty", "directory", "fifo", "oversized"])
def test_loader_rejects_unbounded_or_nonregular_input(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "task.json"
    if kind == "empty":
        path.touch()
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.write_bytes(b" " * (1024 * 1024 + 1))

    with pytest.raises(ContractError) as raised:
        _load(path)

    assert "bounded non-hardlinked regular file" in raised.value.details["cause"]


def test_loader_detects_byte_drift_after_initial_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "task.json"
    payload = b'{"task_id": "rms"}'
    path.write_bytes(payload)
    original = descriptor_loader._decode_document

    def mutate(value: bytes, suffix: str) -> object:
        decoded = original(value, suffix)
        path.write_bytes(payload + b" ")
        return decoded

    monkeypatch.setattr(descriptor_loader, "_decode_document", mutate)

    with pytest.raises(ContractError) as raised:
        _load(path)

    assert "drifted" in raised.value.details["cause"]


def test_loader_detects_identity_drift_with_unchanged_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "task.json"
    replacement = tmp_path / "replacement.json"
    payload = b'{"task_id": "rms"}'
    path.write_bytes(payload)
    original = descriptor_loader._decode_document

    def replace(value: bytes, suffix: str) -> object:
        decoded = original(value, suffix)
        replacement.write_bytes(payload)
        replacement.replace(path)
        return decoded

    monkeypatch.setattr(descriptor_loader, "_decode_document", replace)

    with pytest.raises(ContractError) as raised:
        _load(path)

    assert "identity drifted" in raised.value.details["cause"]
