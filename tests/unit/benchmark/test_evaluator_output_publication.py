from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from apex.benchmark import evaluator_output_publication as publication
from apex.benchmark.evaluator_output_publication import publish_evaluator_outputs
from apex.core import ConfigurationError


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "private-output"
    workspace = tmp_path / "magpie-workspace"
    source.mkdir()
    workspace.mkdir()
    results = source / "model" / "results_2026.json"
    samples = source / "model" / "samples_gsm8k_2026.jsonl"
    results.parent.mkdir()
    results.write_text(json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 1.0}}}))
    samples.write_text('{"doc_id": 1}\n')
    return source, workspace


def test_publishes_exact_outputs_and_seals_both_trees(tmp_path: Path) -> None:
    source, workspace = _roots(tmp_path)

    published = publish_evaluator_outputs(
        source, workspace, contract_sha256="1" * 64
    )

    assert published.root == workspace / "evaluator" / ("1" * 64)
    assert len(published.result_artifacts) == 1
    assert len(published.sample_artifacts) == 1
    assert published.result_artifacts[0].path.startswith("evaluator/")
    for root in (source, published.root):
        assert stat.S_IMODE(root.stat().st_mode) == 0o500
        assert all(
            stat.S_IMODE(path.stat().st_mode)
            == (0o400 if path.is_file() else 0o500)
            for path in root.rglob("*")
        )


@pytest.mark.parametrize("kind", ("unknown", "linked", "missing_samples"))
def test_rejects_incomplete_or_unsafe_output_set(
    tmp_path: Path, kind: str
) -> None:
    source, workspace = _roots(tmp_path)
    if kind == "unknown":
        (source / "debug.log").write_text("unexpected")
    elif kind == "linked":
        target = source / "model" / "samples_gsm8k_2026.jsonl"
        linked = source / "model" / "samples_linked.jsonl"
        os.link(target, linked)
    else:
        (source / "model" / "samples_gsm8k_2026.jsonl").unlink()

    with pytest.raises(ConfigurationError):
        publish_evaluator_outputs(source, workspace, contract_sha256="1" * 64)
    assert not (workspace / "evaluator" / ("1" * 64)).exists()


def test_never_overwrites_existing_publication(tmp_path: Path) -> None:
    source, workspace = _roots(tmp_path)
    destination = workspace / "evaluator" / ("1" * 64)
    destination.mkdir(parents=True)

    with pytest.raises(ConfigurationError, match="already exists"):
        publish_evaluator_outputs(source, workspace, contract_sha256="1" * 64)


@pytest.mark.parametrize("kind", ("file", "directory"))
def test_rejects_symlink_in_private_output_tree(tmp_path: Path, kind: str) -> None:
    source, workspace = _roots(tmp_path)
    if kind == "file":
        sample = source / "model" / "samples_gsm8k_2026.jsonl"
        external = tmp_path / "external-samples.jsonl"
        external.write_text(sample.read_text())
        sample.unlink()
        sample.symlink_to(external)
    else:
        model = source / "model"
        external = tmp_path / "external-model"
        model.rename(external)
        model.symlink_to(external, target_is_directory=True)

    with pytest.raises(ConfigurationError, match="unsafe"):
        publish_evaluator_outputs(source, workspace, contract_sha256="1" * 64)

    assert not (workspace / "evaluator").exists()


@pytest.mark.parametrize("kind", ("symlink", "file", "read_only"))
def test_rejects_unsafe_evaluator_parent(tmp_path: Path, kind: str) -> None:
    source, workspace = _roots(tmp_path)
    evaluator = workspace / "evaluator"
    if kind == "symlink":
        external = tmp_path / "external-evaluator"
        external.mkdir()
        evaluator.symlink_to(external, target_is_directory=True)
    elif kind == "file":
        evaluator.write_text("not a directory")
    else:
        evaluator.mkdir(mode=0o500)

    with pytest.raises(ConfigurationError, match="unsafe"):
        publish_evaluator_outputs(source, workspace, contract_sha256="1" * 64)

    assert not (evaluator / ("1" * 64)).exists()


def test_rejects_symlink_in_workspace_directory_chain(tmp_path: Path) -> None:
    source, _ = _roots(tmp_path)
    actual_parent = tmp_path / "actual-parent"
    actual_workspace = actual_parent / "workspace"
    actual_workspace.mkdir(parents=True)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(ConfigurationError, match="chain is unsafe"):
        publish_evaluator_outputs(
            source, linked_parent / "workspace", contract_sha256="1" * 64
        )

    assert not (actual_workspace / "evaluator").exists()


def test_failed_copy_removes_exclusive_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, workspace = _roots(tmp_path)

    def fail_copy(*_args: object) -> None:
        raise ConfigurationError("copy failed", "test_failure")

    monkeypatch.setattr(publication, "_publish_one", fail_copy)

    with pytest.raises(ConfigurationError, match="copy failed"):
        publish_evaluator_outputs(source, workspace, contract_sha256="1" * 64)

    assert not (workspace / "evaluator").exists()


def test_rejects_digest_that_could_escape_publication_root(tmp_path: Path) -> None:
    source, workspace = _roots(tmp_path)

    with pytest.raises(ConfigurationError, match="digest is invalid"):
        publish_evaluator_outputs(source, workspace, contract_sha256="../outside")

    assert not (workspace / "outside").exists()
