from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from apex.benchmark.evaluator_execution import LmEvalExecutionContract
from apex.benchmark.evaluator_inferencex_projection import (
    materialize_inferencex_projection,
    verify_inferencex_projection,
)
from apex.core import ConfigurationError, sha256_file


def _contract() -> LmEvalExecutionContract:
    return LmEvalExecutionContract(
        run_id="baseline-measurement",
        config_sha256="1" * 64,
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


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    inferencex = tmp_path / "InferenceX"
    benchmarks = inferencex / "benchmarks"
    benchmarks.mkdir(parents=True)
    (benchmarks / "benchmark_lib.sh").write_text(
        "run_eval() { return 9; }\nappend_lm_eval_summary() { return 9; }\n",
        encoding="utf-8",
    )
    runner = inferencex / "runners" / "launch.sh"
    runner.parent.mkdir()
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    (runner.parent / "launch-link.sh").symlink_to("launch.sh")
    magpie = tmp_path / "MagpieRoot"
    scripts = magpie / "Magpie" / "scripts" / "benchmark"
    scripts.mkdir(parents=True)
    (scripts / "vllm_mi300x.sh").write_text(
        'source "$(dirname "$0")/benchmark_lib.sh"\nrun_eval --framework lm-eval\n',
        encoding="utf-8",
    )
    return inferencex, magpie


def test_materializes_private_projection_without_mutating_sources(tmp_path: Path) -> None:
    inferencex, magpie = _sources(tmp_path)
    source_library = inferencex / "benchmarks" / "benchmark_lib.sh"
    source_sha = sha256_file(source_library)

    projection = materialize_inferencex_projection(
        inferencex,
        magpie,
        tmp_path / "authority" / "inferencex",
        inferencex_commit="1" * 40,
        inferencex_tree="2" * 40,
        magpie_commit="3" * 40,
        magpie_tree="4" * 40,
        execution_contract=_contract(),
        nonce="5" * 64,
    )

    verify_inferencex_projection(projection)
    assert sha256_file(source_library) == source_sha
    assert (projection.root / "benchmarks" / "benchmark_lib.sh").stat().st_ino != source_library.stat().st_ino
    library = (projection.root / "benchmarks" / "benchmark_lib.sh").read_text()
    assert "Modified by Apex" in library
    assert "apex_evaluator_handoff.py" in library
    handoff = json.loads(projection.handoff_contract_path.read_text())
    assert handoff["nonce"] == "5" * 64
    assert handoff["execution_contract_sha256"] == _contract().sha256
    assert (projection.root / "runners" / "launch-link.sh").readlink() == Path("launch.sh")


def test_allows_idempotent_magpie_script_copy_but_rejects_drift(tmp_path: Path) -> None:
    inferencex, magpie = _sources(tmp_path)
    projection = materialize_inferencex_projection(
        inferencex,
        magpie,
        tmp_path / "projection",
        inferencex_commit="1" * 40,
        inferencex_tree="2" * 40,
        magpie_commit="3" * 40,
        magpie_tree="4" * 40,
        execution_contract=_contract(),
        nonce="5" * 64,
    )
    source = magpie / "Magpie" / "scripts" / "benchmark" / "vllm_mi300x.sh"
    target = projection.root / "benchmarks" / source.name
    shutil.copy2(source, target)
    target.chmod(0o755)
    verify_inferencex_projection(projection)

    target.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="changed during execution"):
        verify_inferencex_projection(projection)


def test_rejects_projection_symlink_that_escapes_source(tmp_path: Path) -> None:
    inferencex, magpie = _sources(tmp_path)
    (inferencex / "escape").symlink_to("../outside")

    with pytest.raises(ConfigurationError, match="unsafe symlink"):
        materialize_inferencex_projection(
            inferencex,
            magpie,
            tmp_path / "projection",
            inferencex_commit="1" * 40,
            inferencex_tree="2" * 40,
            magpie_commit="3" * 40,
            magpie_tree="4" * 40,
            execution_contract=_contract(),
            nonce="5" * 64,
        )
