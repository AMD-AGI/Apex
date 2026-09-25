from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from apex.benchmark.quality_artifacts import load_declared_quality_artifacts
from apex.core import IntegrityError


def _receipt(path: Path, root: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_loads_only_exact_declared_bytes(tmp_path: Path) -> None:
    root = tmp_path / "authority"
    output = root / "lm_eval" / "results.json"
    output.parent.mkdir(parents=True)
    output.write_bytes(b'{"results": {}}')
    undeclared = root / "lm_eval" / "forged.json"
    undeclared.write_bytes(b"forged")

    loaded = load_declared_quality_artifacts(root, (_receipt(output, root),))

    assert len(loaded) == 1
    assert loaded[0].relative_path == "lm_eval/results.json"
    assert loaded[0].content == output.read_bytes()


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "a/../b", "./a"])
def test_rejects_unsafe_locator(tmp_path: Path, relative: str) -> None:
    tmp_path.mkdir(exist_ok=True)
    receipt = {"path": relative, "size_bytes": 1, "sha256": "1" * 64}

    with pytest.raises(IntegrityError) as caught:
        load_declared_quality_artifacts(tmp_path, (receipt,))

    assert caught.value.reason_code == "unsafe_quality_artifact"


def test_rejects_symlink_parent_and_file(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "results.json"
    target.write_bytes(b"{}")
    root = tmp_path / "root"
    root.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    receipt = {
        "path": "linked/results.json",
        "size_bytes": 2,
        "sha256": hashlib.sha256(b"{}").hexdigest(),
    }

    with pytest.raises(IntegrityError, match="unsafe"):
        load_declared_quality_artifacts(root, (receipt,))
    (root / "linked").unlink()
    (root / "linked").mkdir()
    (root / "linked" / "results.json").symlink_to(target)
    with pytest.raises(IntegrityError, match="unsafe"):
        load_declared_quality_artifacts(root, (receipt,))


def test_rejects_hardlink_and_digest_drift(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    original = root / "results.json"
    original.write_bytes(b"{}")
    linked = root / "linked.json"
    os.link(original, linked)

    with pytest.raises(IntegrityError, match="identity"):
        load_declared_quality_artifacts(root, (_receipt(linked, root),))

    linked.unlink()
    receipt = _receipt(original, root)
    original.write_bytes(b'{"changed": true}')
    with pytest.raises(IntegrityError) as caught:
        load_declared_quality_artifacts(root, (receipt,))
    assert caught.value.reason_code in {
        "unsafe_quality_artifact",
        "quality_artifact_receipt_mismatch",
    }


def test_rejects_duplicates_and_total_byte_limit(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    output = root / "results.json"
    output.write_bytes(b"1234")
    receipt = _receipt(output, root)

    with pytest.raises(IntegrityError, match="duplicated"):
        load_declared_quality_artifacts(root, (receipt, receipt))
    with pytest.raises(IntegrityError, match="identity"):
        load_declared_quality_artifacts(root, (receipt,), max_total_bytes=3)
