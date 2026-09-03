from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from apex.benchmark.evaluator_runtime_publication import (
    publish_lm_eval_runtime_evidence,
)
from apex.benchmark.lm_eval_runtime import parse_lm_eval_runtime_evidence
from apex.core import ConfigurationError, canonical_json_bytes
from apex.runtime import LmEvalRuntimeReceipt


def _identity() -> dict[str, str]:
    return {
        "lm_eval_commit": "1" * 40,
        "lm_eval_tree": "2" * 40,
        "lm_eval_version": "0.4.9.2",
        "python_abi": "cpython-312",
        "python_soabi": "cpython-312-x86_64-linux-gnu",
        "base_image_id": "sha256:" + "3" * 64,
        "base_image_repo_digest": "example/runtime@sha256:" + "4" * 64,
        "inferencex_commit": "5" * 40,
        "inferencex_tree": "6" * 40,
    }


def _write_read_only(path: Path, content: bytes) -> None:
    path.write_bytes(content)
    path.chmod(0o400)


def _probe(module_sha256: str) -> dict[str, object]:
    return {
        "schema": "apex.lm-eval-runtime-probe/v1",
        "python": {
            "implementation": "cpython",
            "version": [3, 12, 11],
            "executable": "/usr/local/bin/python3",
        },
        "lm_eval": {
            "version": "0.4.9.2",
            "module_path": "/evaluator/runtime/site-packages/lm_eval/__init__.py",
            "module_sha256": module_sha256,
        },
        "python_path": [
            "/evaluator/runtime/site-packages",
            "/usr/local/lib/python312.zip",
        ],
    }


def _setup(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    module = runtime_root / "site-packages" / "lm_eval" / "__init__.py"
    module.parent.mkdir(parents=True)
    module_bytes = b'__version__ = "0.4.9.2"\n'
    _write_read_only(module, module_bytes)
    identity = _identity()
    runtime_sha256 = "7" * 64
    record = {
        "path": "lm_eval/__init__.py",
        "size_bytes": len(module_bytes),
        "mode": 0o400,
        "sha256": hashlib.sha256(module_bytes).hexdigest(),
    }
    manifest_value = {
        "schema": "apex.lm-eval-runtime/v1",
        "runtime_sha256": runtime_sha256,
        "site_packages": "site-packages",
        "identity": identity,
        "files": [record],
    }
    manifest = runtime_root / "lm_eval_runtime_manifest.json"
    manifest_bytes = canonical_json_bytes(manifest_value) + b"\n"
    _write_read_only(manifest, manifest_bytes)
    receipt = LmEvalRuntimeReceipt(
        runtime_root.resolve(),
        runtime_sha256,
        hashlib.sha256(manifest_bytes).hexdigest(),
        identity,
        1,
        "8" * 64,
    )
    sidecar = tmp_path / "sidecar"
    sidecar.mkdir()
    probe_path = sidecar / "runtime_probe.json"
    probe_bytes = canonical_json_bytes(_probe(record["sha256"])) + b"\n"
    _write_read_only(probe_path, probe_bytes)
    contract = SimpleNamespace(
        runtime_sha256=receipt.runtime_sha256,
        runtime_manifest_sha256=receipt.manifest_sha256,
        runtime_lock_sha256=receipt.lock_sha256,
        image_id=identity["base_image_id"],
        image_repo_digest=identity["base_image_repo_digest"],
    )
    prepared = SimpleNamespace(
        runtime_receipt=receipt,
        runtime_mount=receipt.root,
        sidecar_root=sidecar.resolve(),
        contract=contract,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return prepared, workspace, module, manifest, probe_path


def _rewrite_json(path: Path, value: object) -> bytes:
    content = canonical_json_bytes(value) + b"\n"
    path.chmod(0o600)
    path.write_bytes(content)
    path.chmod(0o400)
    return content


def test_validates_probe_and_publishes_parser_compatible_read_only_evidence(
    tmp_path: Path,
) -> None:
    prepared, workspace, _, manifest, probe = _setup(tmp_path)

    published = publish_lm_eval_runtime_evidence(prepared, workspace)

    assert published.runtime_probe_sha256 == hashlib.sha256(probe.read_bytes()).hexdigest()
    assert published.manifest_path.read_bytes() == manifest.read_bytes()
    assert stat.S_IMODE(published.manifest_path.stat().st_mode) == 0o400
    assert stat.S_IMODE(published.receipt_path.stat().st_mode) == 0o400
    receipt = json.loads(published.receipt_path.read_bytes())
    assert receipt["schema"] == "magpie.lm-eval-runtime-receipt/v1"
    assert receipt["lm_eval_module"] == "site-packages/lm_eval/__init__.py"
    parsed = parse_lm_eval_runtime_evidence(
        {"lm_eval_runtime_receipt": published.evidence},
        workspace / "benchmark_report.json",
        expected=prepared.runtime_receipt,
        execution_mode="docker",
    )
    assert parsed.passed is True


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("python", "version", [3, 11, 11]),
        ("lm_eval", "version", "0.4.9.1"),
        ("lm_eval", "module_path", "/tmp/lm_eval/__init__.py"),
        ("lm_eval", "module_sha256", "f" * 64),
    ],
)
def test_rejects_runtime_probe_identity_drift(
    tmp_path: Path, section: str, field: str, replacement: object
) -> None:
    prepared, workspace, _, _, probe_path = _setup(tmp_path)
    value = json.loads(probe_path.read_bytes())
    value[section][field] = replacement
    _rewrite_json(probe_path, value)

    with pytest.raises(ConfigurationError, match="locked runtime"):
        publish_lm_eval_runtime_evidence(prepared, workspace)

    assert list(workspace.iterdir()) == []


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_rejects_linked_runtime_probe(tmp_path: Path, link_kind: str) -> None:
    prepared, workspace, _, _, probe = _setup(tmp_path)
    if link_kind == "symlink":
        content = probe.read_bytes()
        probe.unlink()
        target = probe.with_name("probe-target.json")
        _write_read_only(target, content)
        probe.symlink_to(target.name)
    else:
        os.link(probe, probe.with_name("probe-alias.json"))

    with pytest.raises(ConfigurationError, match="[Rr]untime artifact"):
        publish_lm_eval_runtime_evidence(prepared, workspace)

    assert list(workspace.iterdir()) == []


def test_rejects_oversized_runtime_probe(tmp_path: Path) -> None:
    prepared, workspace, _, _, probe = _setup(tmp_path)
    probe.chmod(0o600)
    probe.write_bytes(b"x" * (1024 * 1024 + 1))
    probe.chmod(0o400)

    with pytest.raises(ConfigurationError, match="identity or size"):
        publish_lm_eval_runtime_evidence(prepared, workspace)


def test_rejects_hardlinked_runtime_module(tmp_path: Path) -> None:
    prepared, workspace, module, _, _ = _setup(tmp_path)
    os.link(module, module.with_name("alias.py"))

    with pytest.raises(ConfigurationError, match="identity or size"):
        publish_lm_eval_runtime_evidence(prepared, workspace)


def test_rejects_manifest_identity_different_from_runtime_receipt(
    tmp_path: Path,
) -> None:
    prepared, workspace, _, manifest, _ = _setup(tmp_path)
    value = json.loads(manifest.read_bytes())
    value["identity"]["lm_eval_version"] = "0.4.9.1"
    manifest_bytes = _rewrite_json(manifest, value)
    receipt = replace(
        prepared.runtime_receipt,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    prepared.runtime_receipt = receipt
    prepared.runtime_mount = receipt.root
    prepared.contract.runtime_manifest_sha256 = receipt.manifest_sha256

    with pytest.raises(ConfigurationError, match="verified receipt"):
        publish_lm_eval_runtime_evidence(prepared, workspace)


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_destination_collision_preserves_existing_link_and_cleans_new_manifest(
    tmp_path: Path, link_kind: str
) -> None:
    prepared, workspace, _, _, _ = _setup(tmp_path)
    sentinel = tmp_path / "sentinel.json"
    _write_read_only(sentinel, b"sentinel\n")
    destination = workspace / "lm_eval_runtime_receipt.json"
    if link_kind == "symlink":
        destination.symlink_to(sentinel)
    else:
        os.link(sentinel, destination)

    with pytest.raises(ConfigurationError, match="Cannot publish"):
        publish_lm_eval_runtime_evidence(prepared, workspace)

    assert not (workspace / "lm_eval_runtime_manifest.json").exists()
    assert destination.exists()
    assert sentinel.read_bytes() == b"sentinel\n"


def test_rejects_symlinked_workspace(tmp_path: Path) -> None:
    prepared, workspace, _, _, _ = _setup(tmp_path)
    alias = tmp_path / "workspace-link"
    alias.symlink_to(workspace, target_is_directory=True)

    with pytest.raises(ConfigurationError, match="not canonical"):
        publish_lm_eval_runtime_evidence(prepared, alias)

    assert list(workspace.iterdir()) == []
