#!/usr/bin/env python3
"""Fetch the locked evaluator dataset once and publish a read-only local CAS."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from pathlib import Path

from apex.benchmark.evaluator_dataset_cas import verify_evaluator_dataset_root
from apex.benchmark.evaluator_dataset_materialization import (
    EvaluatorDatasetMaterializationInput,
    materialize_evaluator_dataset_cas,
    verify_dataset_receipt_against_inputs,
)
from apex.runtime import EvaluatorPolicyLock, verify_runtime_dependencies
from apex.runtime.evaluator_lock import EvaluatorDatasetLockFile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apex-root", type=Path, default=_default_apex_root())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args(argv)
    root = args.apex_root.resolve(strict=True)
    dependencies = verify_runtime_dependencies(apex_root=root)
    policy = dependencies.evaluator_policy
    if policy is None:
        parser.error("verified evaluator policy lock is unavailable")
    output = (args.output or _default_output(root, policy)).resolve()
    receipt = _existing(output, policy)
    if receipt is None:
        with tempfile.TemporaryDirectory(prefix="apex-evaluator-dataset-") as raw:
            sources = _download_all(policy, Path(raw), args.timeout_seconds)
            receipt = materialize_evaluator_dataset_cas(
                output,
                repository=policy.dataset_repository,
                dataset_path=policy.dataset_path,
                dataset_name=policy.dataset_name,
                revision=policy.dataset_revision,
                files=sources,
            )
    print(json.dumps({"root": str(output), **receipt.to_dict()}, sort_keys=True))
    return 0


def _existing(output: Path, policy: EvaluatorPolicyLock):
    if not output.exists():
        return None
    receipt = verify_evaluator_dataset_root(
        output,
        expected_repository=policy.dataset_repository,
        expected_path=policy.dataset_path,
        expected_name=policy.dataset_name,
        expected_revision=policy.dataset_revision,
        expected_splits=policy.dataset_splits,
    )
    verify_dataset_receipt_against_inputs(
        receipt, _inputs(policy, output / "files")
    )
    return receipt


def _download_all(
    policy: EvaluatorPolicyLock, root: Path, timeout_seconds: int
) -> tuple[EvaluatorDatasetMaterializationInput, ...]:
    if timeout_seconds <= 0:
        raise ValueError("download timeout must be positive")
    sources: list[EvaluatorDatasetMaterializationInput] = []
    for index, item in enumerate(policy.dataset_files):
        target = root / str(index)
        _download(policy, item, target, timeout_seconds)
        sources.append(_input(item, target))
    return tuple(sources)


def _download(
    policy: EvaluatorPolicyLock,
    item: EvaluatorDatasetLockFile,
    target: Path,
    timeout_seconds: int,
) -> None:
    url = (
        f"https://huggingface.co/datasets/{policy.dataset_path}/resolve/"
        f"{policy.dataset_revision}/{item.path}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "apex/0.1"})
    descriptor = os.open(
        target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if not str(response.geturl()).startswith("https://"):
                raise RuntimeError("dataset download redirected outside HTTPS")
            remaining = item.size_bytes + 1
            while remaining:
                chunk = response.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                _write_all(descriptor, chunk)
                remaining -= len(chunk)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _inputs(
    policy: EvaluatorPolicyLock, files_root: Path
) -> tuple[EvaluatorDatasetMaterializationInput, ...]:
    return tuple(
        _input(item, files_root.joinpath(*Path(item.path).parts))
        for item in policy.dataset_files
    )


def _input(
    item: EvaluatorDatasetLockFile, source: Path
) -> EvaluatorDatasetMaterializationInput:
    return EvaluatorDatasetMaterializationInput(
        item.split, item.path, item.size_bytes, item.sha256, source
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise RuntimeError("cannot write evaluator dataset download")
        view = view[written:]


def _default_output(root: Path, policy: EvaluatorPolicyLock) -> Path:
    return root / ".cache" / "apex-evaluator-datasets" / policy.policy_id / policy.dataset_revision


def _default_apex_root() -> Path:
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(main())
