"""CPU-only contract tests for the pinned dependency bootstrapper."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import apex.runtime as bootstrap


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "bootstrap_dependencies.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location(
        "apex_bootstrap_dependencies_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *args),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def init_repository(root: Path, remote: str) -> str:
    root.mkdir(parents=True)
    subprocess.run(("git", "init", "-q", str(root)), check=True)
    git(root, "config", "user.email", "apex-tests@example.invalid")
    git(root, "config", "user.name", "Apex Tests")
    git(root, "remote", "add", "origin", remote)
    (root / "package.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(root, "add", "package.py")
    git(root, "commit", "-q", "-m", "initial")
    return git(root, "rev-parse", "HEAD")


def add_commit(root: Path) -> str:
    (root / "package.py").write_text("VALUE = 2\n", encoding="utf-8")
    git(root, "add", "package.py")
    git(root, "commit", "-q", "-m", "advance")
    return git(root, "rev-parse", "HEAD")


def write_source_lock(path: Path, entries: dict[str, tuple[Path, str]]) -> Path:
    sources = {}
    for key, (root, remote) in entries.items():
        sources[key] = {
            "name": key,
            "repository": remote,
            "commit": git(root, "rev-parse", "HEAD"),
            "tree": git(root, "rev-parse", "HEAD^{tree}"),
            "sibling": key,
            "managed_checkout": f"{key}-locked",
            "root_env": f"APEX_{key.upper()}_SOURCE_ROOT",
        }
    path.write_text(
        json.dumps(
            {
                "schema": "apex.e2e-source-locks",
                "version": 1,
                "receipt_schema": "apex.e2e-source-locks.receipt/v1",
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )
    return path


def init_installable_repository(
    root: Path,
    *,
    remote: str,
    distribution: str,
    import_root: str,
    version: str,
) -> str:
    root.mkdir(parents=True)
    subprocess.run(("git", "init", "-q", str(root)), check=True)
    git(root, "config", "user.email", "apex-tests@example.invalid")
    git(root, "config", "user.name", "Apex Tests")
    git(root, "remote", "add", "origin", remote)
    package = root / import_root
    package.mkdir()
    (package / "__init__.py").write_text(
        f"__version__ = {version!r}\n", encoding="utf-8"
    )
    (root / "setup.py").write_text(
        "from setuptools import setup\n"
        f"setup(name={distribution!r}, version={version!r}, "
        f"packages={[import_root]!r})\n",
        encoding="utf-8",
    )
    (root / ".gitignore").write_text(
        "__pycache__/\n*.egg-info/\n", encoding="utf-8"
    )
    git(root, "add", ".gitignore", "setup.py", f"{import_root}/__init__.py")
    git(root, "commit", "-q", "-m", "installable package")
    return git(root, "rev-parse", "HEAD")


def dependency(commit: str, *, remote: str = "https://github.com/AMD-AGI/Magpie.git"):
    return bootstrap.LockedDependency(
        key="magpie",
        name="Magpie",
        repository=remote,
        commit=commit,
        sibling="Magpie",
        managed_checkout="magpie",
        root_env="MAGPIE_ROOT",
        distribution="magpie-eval",
        package_version="0.2.0",
        version_policy="exact",
        import_root="Magpie",
        extras=("mcp",),
    )


def test_repository_lock_contains_reviewed_exact_dependencies():
    lock = bootstrap.load_lock(REPO_ROOT / "scripts" / "dependencies.lock.json")

    assert lock.receipt_schema == "apex.dependencies.receipt/v1"
    assert len(lock.sha256) == 64
    observed = {item.key: item for item in lock.dependencies}
    assert observed["magpie"].commit == "210513b31b2f3607920be4000d37fc51f14c5711"
    assert observed["magpie"].import_root == "Magpie"
    assert observed["magpie"].package_version == "0.2.0"
    assert observed["tracelens"].commit == "4f25c1a6f03441e710a97d71a5de9cc5c2fc1555"
    assert observed["tracelens"].import_root == "TraceLens"
    assert observed["tracelens"].version_policy == "prefix"
    assert observed["inferencex"].commit == (
        "23f04b8baca7774f9c0bbcb7a31e9ad551a3b84b"
    )
    assert observed["inferencex"].repository_only
    assert observed["inferencex"].import_root == ""


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(commit="abc"), "40-hex"),
        (lambda value: value.update(sibling="../escape"), "safe relative"),
        (lambda value: value["python"].update(import_root="bad-name"), "import root"),
        (lambda value: value["python"].update(version_policy="compatible"), "exact or prefix"),
    ],
)
def test_lock_rejects_ambiguous_or_unsafe_entries(tmp_path, mutation, message):
    raw = json.loads((REPO_ROOT / "scripts" / "dependencies.lock.json").read_text())
    mutation(raw["dependencies"]["magpie"])
    lock_path = tmp_path / "dependencies.lock.json"
    lock_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(bootstrap.BootstrapError, match=message):
        bootstrap.load_lock(lock_path)


def test_repository_only_dependency_rejects_python_contract(tmp_path):
    raw = json.loads((REPO_ROOT / "scripts" / "dependencies.lock.json").read_text())
    raw["dependencies"]["inferencex"]["python"] = {
        "distribution": "unexpected",
        "version": "1",
        "version_policy": "exact",
        "import_root": "unexpected",
        "extras": [],
    }
    lock_path = tmp_path / "dependencies.lock.json"
    lock_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(bootstrap.BootstrapError, match="forbidden"):
        bootstrap.load_lock(lock_path)


def test_repository_identity_accepts_equivalent_ssh_and_https_urls():
    https = "https://github.com/AMD-AGI/Magpie.git"
    ssh = "git@github.com:AMD-AGI/Magpie.git"

    assert bootstrap.canonical_repository(https) == bootstrap.canonical_repository(ssh)


def test_resolver_prefers_clean_exact_sibling(tmp_path, monkeypatch):
    monkeypatch.delenv("MAGPIE_ROOT", raising=False)
    sibling = tmp_path / "siblings" / "Magpie"
    commit = init_repository(sibling, "git@github.com:AMD-AGI/Magpie.git")
    resolver = bootstrap.RepositoryResolver(
        sibling_root=sibling.parent,
        checkout_root=tmp_path / "managed",
        explicit_roots={},
        offline=True,
        dry_run=False,
    )

    resolved = resolver.resolve(dependency(commit))

    assert resolved.root == sibling.resolve()
    assert resolved.resolution == "sibling"
    assert not (tmp_path / "managed").exists()


def test_offline_resolver_clones_pin_without_mutating_advanced_sibling(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MAGPIE_ROOT", raising=False)
    sibling = tmp_path / "siblings" / "Magpie"
    locked_commit = init_repository(
        sibling, "git@github.com:AMD-AGI/Magpie.git"
    )
    advanced_commit = add_commit(sibling)
    resolver = bootstrap.RepositoryResolver(
        sibling_root=sibling.parent,
        checkout_root=tmp_path / "managed",
        explicit_roots={},
        offline=True,
        dry_run=False,
    )

    resolved = resolver.resolve(dependency(locked_commit))

    assert resolved.resolution == "sibling-clone"
    assert resolved.state and resolved.state.commit == locked_commit
    assert git(sibling, "rev-parse", "HEAD") == advanced_commit
    assert git(resolved.root, "remote", "get-url", "origin") == (
        "https://github.com/AMD-AGI/Magpie.git"
    )


def test_resolver_preserves_stale_managed_checkout_and_uses_versioned_target(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MAGPIE_ROOT", raising=False)
    sibling = tmp_path / "siblings" / "Magpie"
    locked_commit = init_repository(
        sibling, "https://github.com/AMD-AGI/Magpie.git"
    )
    add_commit(sibling)
    stale = tmp_path / "managed" / "magpie"
    init_repository(
        stale, "https://github.com/AMD-AGI/Magpie.git"
    )
    stale_commit = add_commit(stale)
    resolver = bootstrap.RepositoryResolver(
        sibling_root=sibling.parent,
        checkout_root=stale.parent,
        explicit_roots={},
        offline=True,
        dry_run=False,
    )

    resolved = resolver.resolve(dependency(locked_commit))

    assert resolved.root.name == f"magpie-{locked_commit[:12]}"
    assert resolved.state and resolved.state.commit == locked_commit
    assert git(stale, "rev-parse", "HEAD") == stale_commit


def test_explicit_mismatched_checkout_fails_instead_of_falling_back(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MAGPIE_ROOT", raising=False)
    explicit = tmp_path / "explicit"
    locked_commit = init_repository(
        explicit, "git@github.com:AMD-AGI/Magpie.git"
    )
    advanced_commit = add_commit(explicit)
    assert advanced_commit != locked_commit
    resolver = bootstrap.RepositoryResolver(
        sibling_root=tmp_path / "siblings",
        checkout_root=tmp_path / "managed",
        explicit_roots={"magpie": explicit},
        offline=False,
        dry_run=False,
    )

    with pytest.raises(bootstrap.BootstrapError, match="explicit checkout is not locked"):
        resolver.resolve(dependency(locked_commit))

    assert not (tmp_path / "managed").exists()


def test_offline_resolution_fails_when_no_local_source_contains_pin(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MAGPIE_ROOT", raising=False)
    sibling = tmp_path / "siblings" / "Magpie"
    observed_commit = init_repository(
        sibling, "git@github.com:AMD-AGI/Magpie.git"
    )
    missing_commit = "f" * 40
    assert observed_commit != missing_commit
    resolver = bootstrap.RepositoryResolver(
        sibling_root=sibling.parent,
        checkout_root=tmp_path / "managed",
        explicit_roots={},
        offline=True,
        dry_run=False,
    )

    with pytest.raises(bootstrap.BootstrapError, match="offline resolution failed"):
        resolver.resolve(dependency(missing_commit))


def test_probe_validation_detects_version_and_split_brain(tmp_path):
    expected_root = tmp_path / "expected"
    wrong_root = tmp_path / "old-magpie"
    probe = bootstrap.PythonProbe(
        ok=True,
        distribution_version="0.1.0",
        import_file=wrong_root / "Magpie" / "__init__.py",
        direct_url=None,
        error=None,
    )

    errors = bootstrap.probe_errors(
        dependency("a" * 40), probe, expected_root
    )

    assert any("distribution version=0.1.0" in error for error in errors)
    assert any("import resolved" in error for error in errors)


def test_prefix_version_accepts_tracelens_dynamic_build_version():
    assert bootstrap.version_matches(
        "0.1.0.dev20260807+g4f25c1a", "0.1.0", "prefix"
    )
    assert not bootstrap.version_matches("0.2.0", "0.1.0", "prefix")


def test_install_is_idempotent_when_exact_package_is_already_imported(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MAGPIE_ROOT", raising=False)
    source = tmp_path / "siblings" / "Magpie"
    commit = init_repository(source, "git@github.com:AMD-AGI/Magpie.git")
    locked = dependency(commit)
    lock = bootstrap.DependencyLock(
        path=tmp_path / "lock.json",
        receipt_schema="apex.dependencies.receipt/v1",
        dependencies=(locked,),
        sha256="0" * 64,
    )
    resolver = bootstrap.RepositoryResolver(
        sibling_root=source.parent,
        checkout_root=tmp_path / "managed",
        explicit_roots={},
        offline=True,
        dry_run=False,
    )

    class AlreadyInstalledEnvironment:
        def __init__(self):
            self.python = tmp_path / "venv" / "bin" / "python"
            self.python.parent.mkdir(parents=True)
            self.python.touch()
            self.install_calls = 0

        def ensure(self):
            return "existing"

        def probe(self, _dependency):
            return bootstrap.PythonProbe(
                ok=True,
                distribution_version="0.2.0",
                import_file=source / "Magpie" / "__init__.py",
                direct_url=None,
                error=None,
            )

        def install(self, _dependency, _root):
            self.install_calls += 1

    environment = AlreadyInstalledEnvironment()
    runner = bootstrap.DependencyBootstrapper(lock, resolver, environment)

    first = runner.install(dry_run=False)
    second = runner.install(dry_run=False)

    assert environment.install_calls == 0
    assert first["dependencies"]["magpie"]["action"] == "already-installed"
    assert second["dependencies"]["magpie"]["action"] == "already-installed"


def test_cli_performs_real_offline_install_verify_and_repeat_without_network(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MAGPIE_ROOT", raising=False)
    monkeypatch.delenv("TRACELENS_REPO_PATH", raising=False)
    siblings = tmp_path / "siblings"
    magpie_url = "https://example.invalid/AMD-AGI/Magpie.git"
    tracelens_url = "https://example.invalid/AMD-AGI/TraceLens.git"
    magpie_commit = init_installable_repository(
        siblings / "Magpie",
        remote=magpie_url,
        distribution="magpie-eval",
        import_root="Magpie",
        version="0.2.0",
    )
    tracelens_commit = init_installable_repository(
        siblings / "TraceLens",
        remote=tracelens_url,
        distribution="TraceLens",
        import_root="TraceLens",
        version="0.1.0.dev1",
    )
    vllm_url = "https://example.invalid/vllm-project/vllm.git"
    aiter_url = "https://example.invalid/ROCm/aiter.git"
    init_repository(siblings / "vllm", vllm_url)
    init_repository(siblings / "aiter", aiter_url)
    source_lock = write_source_lock(
        tmp_path / "e2e_source_locks.json",
        {
            "vllm": (siblings / "vllm", vllm_url),
            "aiter": (siblings / "aiter", aiter_url),
        },
    )
    lock = {
        "schema": "apex.dependencies.lock",
        "version": 1,
        "receipt_schema": "apex.dependencies.receipt/v1",
        "dependencies": {
            "magpie": {
                "name": "Magpie",
                "repository": magpie_url,
                "commit": magpie_commit,
                "sibling": "Magpie",
                "managed_checkout": "magpie",
                "root_env": "MAGPIE_ROOT",
                "python": {
                    "distribution": "magpie-eval",
                    "version": "0.2.0",
                    "version_policy": "exact",
                    "import_root": "Magpie",
                    "extras": [],
                },
            },
            "tracelens": {
                "name": "TraceLens",
                "repository": tracelens_url,
                "commit": tracelens_commit,
                "sibling": "TraceLens",
                "managed_checkout": "tracelens",
                "root_env": "TRACELENS_REPO_PATH",
                "python": {
                    "distribution": "TraceLens",
                    "version": "0.1.0",
                    "version_policy": "prefix",
                    "import_root": "TraceLens",
                    "extras": [],
                },
            },
        },
    }
    lock_path = tmp_path / "dependencies.lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    venv = tmp_path / "venv"
    common = (
        sys.executable,
        "-m",
        "apex.runtime.dependencies",
        "--lock",
        str(lock_path),
        "--sibling-root",
        str(siblings),
        "--checkout-root",
        str(tmp_path / "managed"),
        "--e2e-source-lock",
        str(source_lock),
        "--source-lock-root",
        str(tmp_path / "source-locks"),
        "--venv",
        str(venv),
        "--offline",
        "--json",
    )

    first = subprocess.run(
        (*common, "install"), capture_output=True, text=True, check=True
    )
    second = subprocess.run(
        (*common, "install"), capture_output=True, text=True, check=True
    )
    verified = subprocess.run(
        (*common, "verify"), capture_output=True, text=True, check=True
    )

    first_receipt = json.loads(first.stdout)
    second_receipt = json.loads(second.stdout)
    verify_receipt = json.loads(verified.stdout)
    assert first_receipt["status"] == "verified"
    assert {
        value["action"] for value in first_receipt["dependencies"].values()
    } == {"installed"}
    assert {
        value["action"] for value in second_receipt["dependencies"].values()
    } == {"already-installed"}
    assert verify_receipt["status"] == "verified"
    assert set(verify_receipt["e2e_source_locks"]["sources"]) == {"vllm", "aiter"}
    assert {
        value["action"] for value in verify_receipt["dependencies"].values()
    } == {"verified"}


def test_fresh_checkout_shim_executes_runtime_cli_for_dry_run(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MAGPIE_ROOT", raising=False)
    monkeypatch.delenv("TRACELENS_REPO_PATH", raising=False)
    source_root = tmp_path / "sources"
    vllm_url = "https://example.invalid/vllm-project/vllm.git"
    aiter_url = "https://example.invalid/ROCm/aiter.git"
    init_repository(source_root / "vllm", vllm_url)
    init_repository(source_root / "aiter", aiter_url)
    source_lock = write_source_lock(
        tmp_path / "e2e_source_locks.json",
        {
            "vllm": (source_root / "vllm", vllm_url),
            "aiter": (source_root / "aiter", aiter_url),
        },
    )
    result = subprocess.run(
        (
            sys.executable,
            str(SCRIPT_PATH),
            "install",
            "--dry-run",
            "--json",
            "--venv",
            str(REPO_ROOT / ".venv"),
            "--e2e-source-lock",
            str(source_lock),
            "--vllm-source-root",
            str(source_root / "vllm"),
            "--aiter-source-root",
            str(source_root / "aiter"),
        ),
        capture_output=True,
        text=True,
        check=True,
    )

    receipt = json.loads(result.stdout)
    assert receipt["status"] == "planned"
    assert set(receipt["dependencies"]) == {
        "magpie",
        "tracelens",
        "inferencex",
    }
    assert receipt["dependencies"]["inferencex"]["action"] == (
        "verify-repository"
    )


def test_launcher_installs_and_rechecks_exact_editable_build_tools(
    tmp_path, monkeypatch
):
    launcher = load_launcher()
    states = iter((False, True))
    commands = []
    monkeypatch.setattr(launcher, "_build_tools_ready", lambda _python: next(states))
    monkeypatch.setattr(
        launcher,
        "_run",
        lambda argv, **_kwargs: commands.append(tuple(argv)),
    )

    launcher._prepare_build_tools(tmp_path / "venv/bin/python", offline=False)

    assert commands == [
        (
            str(tmp_path / "venv/bin/python"),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "packaging==26.3",
            "setuptools==83.0.0",
            "wheel==0.47.0",
        )
    ]


def test_offline_launcher_fails_before_editable_install_without_build_tools(
    tmp_path, monkeypatch
):
    launcher = load_launcher()
    monkeypatch.setattr(launcher, "_build_tools_ready", lambda _python: False)

    with pytest.raises(launcher.LauncherError, match="offline setup requires"):
        launcher._prepare_build_tools(tmp_path / "venv/bin/python", offline=True)
