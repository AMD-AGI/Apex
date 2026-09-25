from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from apex.benchmark import evaluator_inferencex_runtime_publication as publication
from apex.benchmark.evaluator_execution import LmEvalExecutionContract
from apex.benchmark.evaluator_inferencex_projection import (
    materialize_inferencex_projection,
)
from apex.benchmark.evaluator_inferencex_runtime_publication import (
    publish_inferencex_projection_evidence,
)
from apex.benchmark.inferencex_runtime import parse_inferencex_runtime_evidence
from apex.benchmark.magpie_launch_projection import materialize_magpie_launch_config
from apex.core import ConfigurationError, canonical_json_bytes, sha256_file, sha256_json


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_immutable(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    path.chmod(0o400)


def _contract(config_sha256: str) -> LmEvalExecutionContract:
    return LmEvalExecutionContract(
        run_id="baseline-measurement",
        config_sha256=config_sha256,
        model="Qwen/example",
        endpoint_port=8888,
        policy_sha256="2" * 64,
        policy_lock_sha256="3" * 64,
        task_definition_sha256="4" * 64,
        effective_task_definition_sha256="5" * 64,
        task_materialization_receipt_sha256="6" * 64,
        dataset_receipt_sha256="7" * 64,
        dataset_revision="8" * 40,
        runtime_sha256="9" * 64,
        runtime_manifest_sha256="a" * 64,
        runtime_lock_sha256="b" * 64,
        launcher_sha256="0" * 64,
        image_repo_digest="example/eval@sha256:" + "c" * 64,
        image_id="sha256:" + "d" * 64,
        max_length=2248,
        max_gen_tokens=480,
        concurrent_requests=64,
        timeout_seconds=3600,
    )


def _setup(tmp_path: Path):
    source = tmp_path / "InferenceX"
    library = source / "benchmarks" / "benchmark_lib.sh"
    library.parent.mkdir(parents=True)
    library.write_text(
        "run_eval() { return 9; }\nappend_lm_eval_summary() { return 9; }\n"
    )
    subprocess.run(("git", "init", "-q", str(source)), check=True)
    _git(source, "config", "user.email", "tests@example.invalid")
    _git(source, "config", "user.name", "Tests")
    _git(source, "remote", "add", "origin", "https://example.invalid/InferenceX.git")
    _git(source, "add", ".")
    _git(source, "commit", "-q", "-m", "baseline")
    commit = _git(source, "rev-parse", "HEAD")
    tree = _git(source, "rev-parse", "HEAD^{tree}")
    magpie = tmp_path / "MagpieRoot"
    scripts = magpie / "Magpie" / "scripts" / "benchmark"
    scripts.mkdir(parents=True)
    (scripts / "vllm_mi300x.sh").write_text("run_eval --framework lm-eval\n")
    canonical = tmp_path / "canonical.yaml"
    canonical.write_text(f"benchmark:\n  inferencex_path: {source}\n")
    contract = _contract(sha256_file(canonical))
    run_root = tmp_path / "run"
    authority = run_root / "authority" / "lm_eval"
    authority.mkdir(parents=True)
    projection = materialize_inferencex_projection(
        source,
        magpie,
        authority / "inferencex",
        inferencex_commit=commit,
        inferencex_tree=tree,
        magpie_commit="3" * 40,
        magpie_tree="4" * 40,
        execution_contract=contract,
        nonce="5" * 64,
    )
    launch_path = authority / "magpie-launch.yaml"
    launch_receipt = materialize_magpie_launch_config(
        canonical,
        launch_path,
        canonical_sha256=sha256_file(canonical),
        inferencex_source_root=source,
        inferencex_projection_root=projection.root,
        inferencex_projection_receipt_sha256=projection.receipt.sha256,
    )
    projection_receipt_path = authority / "inferencex_projection_receipt.json"
    launch_receipt_path = authority / "magpie_launch_config_receipt.json"
    contract_path = authority / "execution_contract.json"
    _write_immutable(projection_receipt_path, projection.receipt.to_dict())
    _write_immutable(launch_receipt_path, launch_receipt.to_dict())
    _write_immutable(contract_path, contract.to_dict())
    handoff = {
        "schema": "apex.evaluator-handoff-receipt/v1",
        "verified": True,
        "request_sha256": "6" * 64,
        "execution_receipt_sha256": "7" * 64,
        "ordering_ns": {
            "listener_started": 1,
            "request_received": 2,
            "sidecar_started": 3,
            "sidecar_finished": 4,
            "handoff_released": 5,
        },
    }
    _write_immutable(authority / "handoff_receipt.json", handoff)
    workspace = run_root / "results" / "benchmark"
    workspace.mkdir(parents=True)
    prepared = SimpleNamespace(
        authority_root=authority,
        inferencex_projection=projection,
        inferencex_projection_receipt=projection.receipt,
        inferencex_projection_receipt_path=projection_receipt_path,
        launch_config_path=launch_path,
        launch_config_receipt=launch_receipt,
        launch_config_receipt_path=launch_receipt_path,
        contract=contract,
        contract_path=contract_path,
    )
    return prepared, workspace, source, commit, tree, sha256_json(handoff)


def test_publishes_honest_projection_receipt_without_legacy_runtime(
    tmp_path: Path,
) -> None:
    prepared, workspace, source, commit, tree, handoff_sha256 = _setup(tmp_path)

    receipt = publish_inferencex_projection_evidence(
        prepared,
        workspace,
        source_root=source,
        source_commit=commit,
        source_tree=tree,
        handoff_receipt_sha256=handoff_sha256,
    )

    path = workspace / "inferencex_runtime_receipt.json"
    assert receipt["schema"] == "apex.inferencex-runtime-receipt/v2"
    assert receipt["materialization_method"] == "apex_private_projection"
    assert receipt["runtime_path"] == "authority/lm_eval/inferencex"
    assert stat.S_IMODE(path.stat().st_mode) == 0o400
    assert path.read_bytes() == canonical_json_bytes(receipt) + b"\n"
    assert not (workspace / "inferencex_runtime").exists()
    evidence = parse_inferencex_runtime_evidence(
        {"inferencex_runtime_receipt": receipt},
        workspace / "benchmark_report.json",
        expected_source_root=source,
        expected_commit=commit,
        expected_tree=tree,
    )
    assert evidence.passed
    assert evidence.runtime_path == prepared.inferencex_projection.root


@pytest.mark.parametrize("unsafe", ("writable_launch", "linked_launch", "receipt_drift"))
def test_rejects_unsafe_prepared_authority(
    tmp_path: Path, unsafe: str
) -> None:
    prepared, workspace, source, commit, tree, handoff_sha256 = _setup(tmp_path)
    if unsafe == "writable_launch":
        prepared.launch_config_path.chmod(0o600)
    elif unsafe == "linked_launch":
        os.link(prepared.launch_config_path, prepared.launch_config_path.with_suffix(".link"))
    else:
        prepared.contract_path.chmod(0o600)
        prepared.contract_path.write_text(json.dumps({"tampered": True}))
        prepared.contract_path.chmod(0o400)

    with pytest.raises(ConfigurationError):
        publish_inferencex_projection_evidence(
            prepared,
            workspace,
            source_root=source,
            source_commit=commit,
            source_tree=tree,
            handoff_receipt_sha256=handoff_sha256,
        )

    assert not (workspace / "inferencex_runtime_receipt.json").exists()


def test_dirty_source_is_not_promoted_to_clean_runtime_evidence(tmp_path: Path) -> None:
    prepared, workspace, source, commit, tree, handoff_sha256 = _setup(tmp_path)
    (source / "untracked.txt").write_text("dirty\n")

    with pytest.raises(ConfigurationError, match="clean Git checkout"):
        publish_inferencex_projection_evidence(
            prepared,
            workspace,
            source_root=source,
            source_commit=commit,
            source_tree=tree,
            handoff_receipt_sha256=handoff_sha256,
        )


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    (
        ("runtime_path", "../authority/lm_eval/inferencex"),
        ("workspace_path", "../different-run"),
    ),
)
def test_v2_parser_rejects_run_root_locator_escape(
    tmp_path: Path, field: str, unsafe_value: str
) -> None:
    prepared, workspace, source, commit, tree, handoff_sha256 = _setup(tmp_path)
    receipt = dict(
        publish_inferencex_projection_evidence(
            prepared,
            workspace,
            source_root=source,
            source_commit=commit,
            source_tree=tree,
            handoff_receipt_sha256=handoff_sha256,
        )
    )
    receipt[field] = unsafe_value
    payload = dict(receipt)
    payload.pop("receipt_sha256")
    receipt["receipt_sha256"] = sha256_json(payload)
    receipt_path = workspace / "inferencex_runtime_receipt.json"
    receipt_path.chmod(0o600)
    _write_immutable(receipt_path, receipt)

    evidence = parse_inferencex_runtime_evidence(
        {"inferencex_runtime_receipt": receipt},
        workspace / "benchmark_report.json",
        expected_source_root=source,
        expected_commit=commit,
        expected_tree=tree,
    )

    assert not evidence.passed
    assert evidence.error == "invalid_inferencex_runtime_receipt"


def test_partial_publication_is_removed_but_collision_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, workspace, source, commit, tree, handoff_sha256 = _setup(tmp_path)

    def fail_after_partial_write(descriptor: int, _content: bytes) -> None:
        os.write(descriptor, b"{")
        raise OSError("injected write failure")

    monkeypatch.setattr(publication, "_write_all", fail_after_partial_write)
    with pytest.raises(ConfigurationError, match="Cannot publish"):
        publish_inferencex_projection_evidence(
            prepared,
            workspace,
            source_root=source,
            source_commit=commit,
            source_tree=tree,
            handoff_receipt_sha256=handoff_sha256,
        )
    assert not (workspace / "inferencex_runtime_receipt.json").exists()

    monkeypatch.undo()
    collision = workspace / "inferencex_runtime_receipt.json"
    collision.write_text("sentinel\n")
    with pytest.raises(ConfigurationError, match="Cannot publish"):
        publish_inferencex_projection_evidence(
            prepared,
            workspace,
            source_root=source,
            source_commit=commit,
            source_tree=tree,
            handoff_receipt_sha256=handoff_sha256,
        )
    assert collision.read_text() == "sentinel\n"
