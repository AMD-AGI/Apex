from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apex.cli import app


def test_bundle_verify_dispatches_kernel_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "kernel-bundle"
    root.mkdir()
    monkeypatch.setattr(app, "detect_bundle_kind", lambda path: "kernel")
    monkeypatch.setattr(
        app,
        "load_and_verify_kernel_bundle",
        lambda path, expected_digest=None: SimpleNamespace(
            task_id="rms-norm",
            path=root,
            digest="a" * 64,
            changed_files=("kernel.py",),
        ),
    )

    assert app.main(["bundle", "verify", "--bundle", str(root), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["bundle_kind"] == "kernel"
    assert result["changed_files"] == ["kernel.py"]


def test_bundle_verify_dispatches_e2e_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "e2e-bundle"
    root.mkdir()
    monkeypatch.setattr(app, "detect_bundle_kind", lambda path: "e2e")
    monkeypatch.setattr(
        app,
        "load_and_verify_e2e_bundle",
        lambda path, expected_digest=None: SimpleNamespace(
            bundle_id="bundle-qwen",
            path=root,
            digest="b" * 64,
            verified=True,
            repositories=(SimpleNamespace(repository_id="vllm"),),
            derived_image=SimpleNamespace(reference="image@sha256:" + "c" * 64),
        ),
    )

    assert app.main(["bundle", "verify", "--bundle", str(root), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["bundle_kind"] == "e2e"
    assert result["terminal_verified"] is True
    assert result["repositories"] == ["vllm"]
