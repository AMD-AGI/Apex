"""CPU-only source lock materialization, verification, and receipt tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from apex.runtime import (
    BootstrapError,
    SourceLockManager,
    default_source_roots,
    load_source_lock,
)


ROOT = Path(__file__).resolve().parents[3]


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *args),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _init_repository(root: Path, remote: str) -> tuple[str, str]:
    root.mkdir(parents=True)
    subprocess.run(("git", "init", "-q", str(root)), check=True)
    _git(root, "config", "user.email", "apex-tests@example.invalid")
    _git(root, "config", "user.name", "Apex Tests")
    _git(root, "remote", "add", "origin", remote)
    (root / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "kernel.py")
    _git(root, "commit", "-q", "-m", "locked source")
    return _git(root, "rev-parse", "HEAD"), _git(root, "rev-parse", "HEAD^{tree}")


def _advance(root: Path) -> str:
    (root / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(root, "add", "kernel.py")
    _git(root, "commit", "-q", "-m", "advanced source")
    return _git(root, "rev-parse", "HEAD")


def _write_lock(
    path: Path,
    *,
    commit: str,
    tree: str,
    remote: str,
    managed_checkout: str = "vllm-locked",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "apex.e2e-source-locks",
                "version": 1,
                "receipt_schema": "apex.e2e-source-locks.receipt/v1",
                "sources": {
                    "vllm": {
                        "name": "vLLM",
                        "repository": remote,
                        "commit": commit,
                        "tree": tree,
                        "sibling": "vllm",
                        "managed_checkout": managed_checkout,
                        "root_env": "APEX_VLLM_SOURCE_ROOT",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _manager(
    lock_path: Path,
    *,
    sibling_root: Path,
    checkout_root: Path,
    explicit_roots: dict[str, Path] | None = None,
) -> SourceLockManager:
    return SourceLockManager(
        load_source_lock(lock_path),
        sibling_root=sibling_root,
        checkout_root=checkout_root,
        explicit_roots=explicit_roots or {},
        offline=True,
    )


def test_checked_in_source_lock_contains_reviewed_qwen_pins():
    lock = load_source_lock(ROOT / "scripts" / "e2e_source_locks.json")
    sources = {item.key: item for item in lock.sources}

    assert lock.receipt_schema == "apex.e2e-source-locks.receipt/v1"
    assert sources["vllm"].commit == "b1388b1fbf5aaef47937fabe98931211684666a6"
    assert sources["vllm"].tree == "33b782e425e42d42851a33f7876e97a8deeabb29"
    assert sources["aiter"].commit == "c3708fb7445899c14cdc6e8055953ee02ed78ddf"
    assert sources["aiter"].tree == "a30409ac03524781f175cbb03e82eefcafd52af1"
    assert default_source_roots(lock, Path("/cache")) == {
        "vllm": Path("/cache/vllm-v0.19.1"),
        "aiter": Path("/cache/aiter-v0.1.10.post2"),
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("commit", "abc", "40-hex"),
        ("tree", "f" * 39, "40-hex"),
        ("managed_checkout", "../escape", "safe relative"),
        ("root_env", "bad-env", "environment name"),
    ],
)
def test_source_lock_rejects_ambiguous_identity(tmp_path, field, value, message):
    source = tmp_path / "source"
    remote = "https://example.invalid/vllm.git"
    commit, tree = _init_repository(source, remote)
    lock_path = _write_lock(
        tmp_path / "lock.json", commit=commit, tree=tree, remote=remote
    )
    raw = json.loads(lock_path.read_text(encoding="utf-8"))
    raw["sources"]["vllm"][field] = value
    lock_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(BootstrapError, match=message):
        load_source_lock(lock_path)


def test_offline_materializer_clones_pin_without_mutating_advanced_sibling(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("APEX_VLLM_SOURCE_ROOT", raising=False)
    sibling = tmp_path / "siblings" / "vllm"
    remote = "https://example.invalid/vllm.git"
    locked_commit, locked_tree = _init_repository(sibling, remote)
    advanced_commit = _advance(sibling)
    lock_path = _write_lock(
        tmp_path / "lock.json",
        commit=locked_commit,
        tree=locked_tree,
        remote=remote,
    )
    managed = tmp_path / "managed"
    manager = _manager(
        lock_path, sibling_root=sibling.parent, checkout_root=managed
    )

    receipt = manager.materialize()
    repeated = manager.materialize()
    verified = manager.verify()

    selected = managed / "vllm-locked"
    assert receipt.root("vllm") == selected.resolve()
    assert receipt.sources["vllm"].resolution == "sibling-clone"
    assert repeated.sources["vllm"].resolution == "managed"
    assert verified.sources["vllm"].tree == locked_tree
    assert _git(selected, "rev-parse", "HEAD") == locked_commit
    assert _git(sibling, "rev-parse", "HEAD") == advanced_commit
    assert _git(sibling, "status", "--porcelain=v1") == ""


def test_tree_mismatch_fails_before_managed_checkout_is_published(tmp_path, monkeypatch):
    monkeypatch.delenv("APEX_VLLM_SOURCE_ROOT", raising=False)
    sibling = tmp_path / "siblings" / "vllm"
    remote = "https://example.invalid/vllm.git"
    commit, _tree = _init_repository(sibling, remote)
    lock_path = _write_lock(
        tmp_path / "lock.json", commit=commit, tree="f" * 40, remote=remote
    )
    managed = tmp_path / "managed"
    manager = _manager(
        lock_path, sibling_root=sibling.parent, checkout_root=managed
    )

    with pytest.raises(BootstrapError, match="tree="):
        manager.materialize()

    assert not (managed / "vllm-locked").exists()
    assert _git(sibling, "rev-parse", "HEAD") == commit


def test_verifier_rejects_dirty_managed_tree_without_changing_it(tmp_path, monkeypatch):
    monkeypatch.delenv("APEX_VLLM_SOURCE_ROOT", raising=False)
    managed = tmp_path / "managed" / "vllm-locked"
    remote = "https://example.invalid/vllm.git"
    commit, tree = _init_repository(managed, remote)
    lock_path = _write_lock(
        tmp_path / "lock.json", commit=commit, tree=tree, remote=remote
    )
    (managed / "untracked.txt").write_text("user data\n", encoding="utf-8")
    manager = _manager(
        lock_path, sibling_root=tmp_path / "siblings", checkout_root=managed.parent
    )

    with pytest.raises(BootstrapError, match="dirty worktree"):
        manager.verify()

    assert (managed / "untracked.txt").read_text(encoding="utf-8") == "user data\n"


def test_dry_run_plan_never_creates_source_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv("APEX_VLLM_SOURCE_ROOT", raising=False)
    sibling = tmp_path / "siblings" / "vllm"
    remote = "https://example.invalid/vllm.git"
    commit, tree = _init_repository(sibling, remote)
    lock_path = _write_lock(
        tmp_path / "lock.json", commit=commit, tree=tree, remote=remote
    )
    managed = tmp_path / "managed"
    plan = _manager(
        lock_path, sibling_root=sibling.parent, checkout_root=managed
    ).plan()

    assert plan["status"] == "planned"
    assert plan["sources"]["vllm"]["resolution"] == "planned-sibling-clone"
    assert plan["sources"]["vllm"]["action"] == "materialize"
    assert not managed.exists()
