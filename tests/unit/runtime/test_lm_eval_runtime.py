from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from apex.execution import ProcessResult
from apex.runtime import (
    BootstrapError,
    DownloadLock,
    LmEvalRuntimeLock,
    LmEvalRuntimePreparer,
    load_lm_eval_runtime_lock,
    verify_lm_eval_runtime,
)
from apex.runtime.dependency_cli import build_parser
from apex.runtime.lm_eval_runtime import canonical_json, collect_runtime_files


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = REPO_ROOT / "scripts" / "lm_eval_runtime.lock.json"


class _RecordingSupervisor:
    def __init__(self, *, fail_cleanup: bool = False) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.fail_cleanup = fail_cleanup

    def run(self, argv, **_kwargs) -> ProcessResult:
        command = tuple(argv)
        self.calls.append(command)
        cleanup = "-c" in command and "os.chown" in command[command.index("-c") + 1]
        exit_code = 9 if cleanup and self.fail_cleanup else 0
        return ProcessResult(command, exit_code, False, "", "cleanup failed", False, False, 0.1)


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


def _make_writable(root: Path) -> None:
    if root.is_symlink() or not root.exists():
        return
    root.chmod(0o755)
    for current, directories, filenames in os.walk(root):
        Path(current).chmod(0o755)
        for name in directories:
            path = Path(current) / name
            if not path.is_symlink():
                path.chmod(0o755)
        for name in filenames:
            path = Path(current) / name
            if not path.is_symlink():
                path.chmod(0o644)


@contextmanager
def _synthetic_runtime(tmp_path: Path):
    root = tmp_path / "runtime"
    site_packages = root / "site-packages" / "lm_eval"
    site_packages.mkdir(parents=True)
    module = site_packages / "__init__.py"
    module.write_text('__version__ = "0.4.9.2"\n', encoding="utf-8")
    module.chmod(0o444)
    site_packages.chmod(0o555)
    site_packages.parent.chmod(0o555)
    records = collect_runtime_files(site_packages.parent)
    identity = _identity()
    tree_sha256 = hashlib.sha256(canonical_json(records)).hexdigest()
    runtime_sha256 = hashlib.sha256(
        canonical_json({"identity": identity, "files": records})
    ).hexdigest()
    source = DownloadLock("source.tar.gz", "https://example.invalid/source", 1, "7" * 64)
    lock = LmEvalRuntimeLock(
        LOCK_PATH,
        source,
        "https://example.invalid/repository.git",
        1,
        (),
        {"pip": "1"},
        identity,
        tree_sha256,
        runtime_sha256,
        "8" * 64,
    )
    manifest = {
        "schema": "apex.lm-eval-runtime/v1",
        "runtime_sha256": runtime_sha256,
        "site_packages": "site-packages",
        "identity": identity,
        "files": records,
    }
    manifest_path = root / "lm_eval_runtime_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path.chmod(0o444)
    root.chmod(0o555)
    try:
        yield root, lock, module
    finally:
        _make_writable(root)


def test_reviewed_lock_pins_exact_runtime_without_shadowing_base_packages() -> None:
    lock = load_lm_eval_runtime_lock(LOCK_PATH)

    assert lock.identity["lm_eval_commit"] == "b315ef3b05176acc9732bb7fdec116abe1ecc476"
    assert lock.identity["lm_eval_tree"] == "6574cdae47205fcee11b76510fd09c5ae60a34c9"
    assert lock.identity["lm_eval_version"] == "0.4.9.2"
    assert lock.identity["python_abi"] == "cpython-312"
    assert lock.identity["base_image_id"] == "sha256:b599932816fe09f9ea2541655f5388457ac2494b87b551cefdbf2a207b0ed3a9"
    assert lock.installed_tree_sha256 == "23dc17079da4619a4cb37100f66f015dd9dd818df46e9f0ea16b541deaf27f60"
    assert lock.runtime_sha256 == "ca744a9e0ab994eba275a0fc0b01b762247f76f9cd0129b31b5dc2969b23732e"
    names = {wheel.name.casefold().replace("_", "-") for wheel in lock.wheels}
    assert len(lock.wheels) == 24
    assert not names & {"torch", "transformers", "datasets", "numpy", "scipy"}


def test_lock_rejects_a_target_wheel_that_shadows_the_base(tmp_path: Path) -> None:
    raw = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    wheel = dict(raw["wheels"][0])
    wheel["name"] = "datasets"
    raw["wheels"][0] = wheel
    raw["wheels"] = sorted(raw["wheels"], key=lambda value: value["name"].casefold())
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(BootstrapError, match="must not shadow"):
        load_lm_eval_runtime_lock(path)


def test_verifier_recomputes_a_valid_read_only_runtime(tmp_path: Path) -> None:
    with _synthetic_runtime(tmp_path) as (root, lock, _):
        receipt = verify_lm_eval_runtime(root, lock)

        assert receipt.root == root.resolve()
        assert receipt.runtime_sha256 == lock.runtime_sha256
        assert receipt.file_count == 1
        assert receipt.identity == lock.identity


def test_verifier_rejects_byte_tampering(tmp_path: Path) -> None:
    with _synthetic_runtime(tmp_path) as (root, lock, module):
        module.chmod(0o644)
        module.write_text("tampered\n", encoding="utf-8")
        module.chmod(0o444)

        with pytest.raises(BootstrapError, match="differs from disk"):
            verify_lm_eval_runtime(root, lock)


def test_verifier_rejects_extra_root_entries(tmp_path: Path) -> None:
    with _synthetic_runtime(tmp_path) as (root, lock, _):
        root.chmod(0o755)
        (root / "unexpected").write_text("extra", encoding="utf-8")
        root.chmod(0o555)

        with pytest.raises(BootstrapError, match="missing or extra"):
            verify_lm_eval_runtime(root, lock)


def test_verifier_rejects_writable_or_hardlinked_content(tmp_path: Path) -> None:
    with _synthetic_runtime(tmp_path) as (root, lock, module):
        module.chmod(0o644)
        with pytest.raises(BootstrapError, match="read-only"):
            verify_lm_eval_runtime(root, lock)
        module.chmod(0o444)
        site = root / "site-packages"
        site.chmod(0o755)
        os.link(module, site / "alias.py")
        site.chmod(0o555)
        with pytest.raises(BootstrapError, match="hardlinked"):
            verify_lm_eval_runtime(root, lock)


def test_verifier_rejects_a_symlinked_runtime_root(tmp_path: Path) -> None:
    with _synthetic_runtime(tmp_path) as (root, lock, _):
        alias = tmp_path / "runtime-link"
        alias.symlink_to(root, target_is_directory=True)

        with pytest.raises(BootstrapError, match="must not be a symlink"):
            verify_lm_eval_runtime(alias, lock)


def test_dependency_parser_exposes_runtime_prepare_and_verify_commands() -> None:
    parser = build_parser(REPO_ROOT)

    assert parser.parse_args(["prepare-runtime", "--offline"]).command == "prepare-runtime"
    assert parser.parse_args(["verify-runtime"]).command == "verify-runtime"


def test_source_wheel_build_is_root_networkless_and_restores_ownership(
    tmp_path: Path,
) -> None:
    supervisor = _RecordingSupervisor()
    preparer = LmEvalRuntimePreparer(
        load_lm_eval_runtime_lock(LOCK_PATH),
        apex_root=REPO_ROOT,
        inferencex_root=tmp_path,
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    source = tmp_path / "source"
    wheelhouse = tmp_path / "wheels"
    source.mkdir()
    wheelhouse.mkdir()

    preparer._build_source_wheel(source, wheelhouse)

    assert len(supervisor.calls) == 2
    build, cleanup = supervisor.calls
    assert "--network=none" in build and "--user" not in build
    assert "SOURCE_DATE_EPOCH=1764686356" in build
    assert "--no-index" in build and "--no-deps" in build
    assert "--network=none" in cleanup and "--user" not in cleanup
    assert "os.chown" in cleanup[cleanup.index("-c") + 1]


def test_root_build_fails_closed_when_ownership_cleanup_fails(tmp_path: Path) -> None:
    supervisor = _RecordingSupervisor(fail_cleanup=True)
    preparer = LmEvalRuntimePreparer(
        load_lm_eval_runtime_lock(LOCK_PATH),
        apex_root=REPO_ROOT,
        inferencex_root=tmp_path,
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    source = tmp_path / "source"
    wheelhouse = tmp_path / "wheels"
    source.mkdir()
    wheelhouse.mkdir()

    with pytest.raises(BootstrapError, match="cleanup failed"):
        preparer._build_source_wheel(source, wheelhouse)

    assert len(supervisor.calls) == 2
