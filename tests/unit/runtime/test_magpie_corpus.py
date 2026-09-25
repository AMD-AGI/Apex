from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from apex.runtime import (
    BootstrapError,
    CorpusFile,
    build_magpie_corpus_manifest,
    load_magpie_corpus_manifest,
    verify_magpie_corpus_manifest,
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository(root: Path) -> Path:
    root.mkdir()
    subprocess.run(("git", "init", "-q", str(root)), check=True)
    _git(root, "config", "user.email", "apex-tests@example.invalid")
    _git(root, "config", "user.name", "Apex Tests")
    _git(root, "remote", "add", "origin", "https://example.invalid/Magpie.git")
    corpus = root / "examples" / "benchmarks"
    (corpus / "nested").mkdir(parents=True)
    (corpus / "a.yaml").write_text("benchmark: {framework: a}\n", encoding="utf-8")
    (corpus / "nested" / "b.yml").write_text(
        "benchmark: {framework: b}\n", encoding="utf-8"
    )
    (corpus / "README.md").write_text("not a config\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "corpus")
    return root


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_manifest_binds_sorted_yaml_paths_hashes_and_both_git_trees(tmp_path: Path) -> None:
    root = _repository(tmp_path / "magpie")
    generated = build_magpie_corpus_manifest(root)
    path = tmp_path / "manifest.json"
    _write(path, generated.to_dict())

    loaded = load_magpie_corpus_manifest(path)
    verified = verify_magpie_corpus_manifest(
        loaded,
        root,
        repository="https://example.invalid/Magpie.git",
        commit=_git(root, "rev-parse", "HEAD"),
    )

    assert [item.path for item in loaded.files] == [
        "examples/benchmarks/a.yaml",
        "examples/benchmarks/nested/b.yml",
    ]
    assert verified["repository_tree"] == _git(root, "rev-parse", "HEAD^{tree}")
    assert verified["benchmark_tree"] == _git(
        root, "rev-parse", "HEAD:examples/benchmarks"
    )
    assert verified["summary"] == {"config_count": 2}


def test_manifest_rejects_byte_path_and_lock_identity_drift(tmp_path: Path) -> None:
    root = _repository(tmp_path / "magpie")
    manifest = build_magpie_corpus_manifest(root)
    path = tmp_path / "manifest.json"
    tampered = manifest.to_dict()
    tampered["files"][0]["sha256"] = "0" * 64
    _write(path, tampered)

    with pytest.raises(BootstrapError, match="digest differs"):
        load_magpie_corpus_manifest(path)

    unsafe = replace(
        manifest,
        files=(CorpusFile("../escape.yaml", "0" * 64),),
    )
    with pytest.raises(BootstrapError, match="differs from frozen manifest"):
        verify_magpie_corpus_manifest(
            unsafe,
            root,
            repository=manifest.repository,
            commit=manifest.commit,
        )

    with pytest.raises(BootstrapError, match="differs from dependency lock"):
        verify_magpie_corpus_manifest(
            manifest,
            root,
            repository=manifest.repository,
            commit="0" * 40,
        )

    (root / manifest.files[0].path).write_text("changed: true\n", encoding="utf-8")
    with pytest.raises(BootstrapError, match="checkout is dirty"):
        verify_magpie_corpus_manifest(
            manifest,
            root,
            repository=manifest.repository,
            commit=manifest.commit,
        )
