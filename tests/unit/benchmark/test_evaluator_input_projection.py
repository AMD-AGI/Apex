from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from apex.benchmark.evaluator_artifact_receipt import EvaluatorArtifactReceipt
from apex.benchmark.evaluator_dataset import (
    EvaluatorDatasetFile,
    EvaluatorDatasetReceipt,
)
from apex.benchmark.evaluator_dataset_materialization import (
    EvaluatorDatasetMaterializationInput,
    materialize_evaluator_dataset_cas,
)
from apex.benchmark.evaluator_input_projection import (
    PROJECTION_SCHEMA,
    materialize_evaluator_sidecar_inputs,
)
from apex.core import ConfigurationError, sha256_file, sha256_json
from apex.runtime import LmEvalRuntimeReceipt
from apex.runtime.lm_eval_lock import RUNTIME_SCHEMA
from apex.runtime.lm_eval_runtime import canonical_json


_LAUNCHER = (
    Path(__file__).resolve().parents[3]
    / "src/apex/benchmark/evaluator_sidecar_entry.py"
)


def _dataset(tmp_path: Path) -> tuple[Path, EvaluatorDatasetReceipt]:
    source = tmp_path / "downloaded.jsonl"
    content = b'{"question":"1+1","answer":"2"}\n'
    source.write_bytes(content)
    artifact = EvaluatorArtifactReceipt(
        "test/data.jsonl", len(content), hashlib.sha256(content).hexdigest()
    )
    receipt = materialize_evaluator_dataset_cas(
        tmp_path / "dataset",
        repository="https://huggingface.co/datasets/openai/gsm8k",
        dataset_path="openai/gsm8k",
        dataset_name="main",
        revision="1" * 40,
        files=(EvaluatorDatasetMaterializationInput(
            "test", artifact.path, artifact.size_bytes, artifact.sha256, source
        ),),
    )
    return tmp_path / "dataset", receipt


def _runtime(tmp_path: Path) -> LmEvalRuntimeReceipt:
    root = tmp_path / "runtime"
    packages = root / "site-packages"
    packages.mkdir(parents=True)
    content = b"VALUE = 1\n"
    module = packages / "lm_eval.py"
    module.write_bytes(content)
    module.chmod(0o400)
    packages.chmod(0o500)
    identity = {"python_abi": "cpython-test"}
    files = [{
        "path": "lm_eval.py",
        "size_bytes": len(content),
        "mode": 0o400,
        "sha256": hashlib.sha256(content).hexdigest(),
    }]
    runtime_sha256 = hashlib.sha256(
        canonical_json({"identity": identity, "files": files})
    ).hexdigest()
    manifest = root / "lm_eval_runtime_manifest.json"
    manifest.write_bytes(json.dumps({
        "schema": RUNTIME_SCHEMA,
        "runtime_sha256": runtime_sha256,
        "site_packages": "site-packages",
        "identity": identity,
        "files": files,
    }, indent=2, sort_keys=True).encode() + b"\n")
    manifest_sha256 = sha256_file(manifest)
    manifest.chmod(0o400)
    root.chmod(0o500)
    return LmEvalRuntimeReceipt(
        root, runtime_sha256, manifest_sha256, identity, 1, "2" * 64
    )


def test_projects_and_reverifies_docker_inputs_under_run_authority(
    tmp_path: Path,
) -> None:
    dataset_root, dataset = _dataset(tmp_path)
    runtime = _runtime(tmp_path)
    authority = tmp_path / "run" / "authority" / "lm_eval"
    authority.mkdir(parents=True)

    projected = materialize_evaluator_sidecar_inputs(
        authority,
        dataset_root=dataset_root,
        dataset_receipt=dataset,
        runtime_receipt=runtime,
        launcher_source=_LAUNCHER,
        launcher_sha256=sha256_file(_LAUNCHER),
    )

    assert projected.dataset_mount.is_relative_to(authority)
    assert projected.runtime_mount.is_relative_to(authority)
    assert projected.runtime_mount != runtime.root
    assert projected.launcher_path.is_relative_to(authority)
    source = runtime.root / "site-packages/lm_eval.py"
    copied = projected.runtime_mount / "site-packages/lm_eval.py"
    assert source.stat().st_ino != copied.stat().st_ino
    assert stat.S_IMODE(projected.root.stat().st_mode) == 0o500
    value = json.loads(projected.receipt_path.read_text())
    assert value["schema"] == PROJECTION_SCHEMA
    assert value["verified"] is True
    assert value["dataset"]["receipt_sha256"] == dataset.sha256
    assert value["projection_sha256"] == sha256_json({
        key: item for key, item in value.items() if key != "projection_sha256"
    })


def test_rejects_runtime_receipt_drift_and_removes_partial_projection(
    tmp_path: Path,
) -> None:
    dataset_root, dataset = _dataset(tmp_path)
    runtime = replace(_runtime(tmp_path), runtime_sha256="f" * 64)
    authority = tmp_path / "run" / "authority" / "lm_eval"
    authority.mkdir(parents=True)

    with pytest.raises(ConfigurationError, match="differs from its receipt"):
        materialize_evaluator_sidecar_inputs(
            authority,
            dataset_root=dataset_root,
            dataset_receipt=dataset,
            runtime_receipt=runtime,
            launcher_source=_LAUNCHER,
            launcher_sha256=sha256_file(_LAUNCHER),
        )

    assert not (authority / "sidecar-inputs").exists()
