"""Run-scoped preparation for a future exact-image lm-eval authority."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from apex.core import ConfigurationError, canonical_json_bytes
from apex.ports import MagpieAttestationRequest
from apex.runtime import DependencyReceipt, LmEvalRuntimeReceipt

from .evaluator_contract_factory import build_lm_eval_execution_contract
from .evaluator_dataset import EvaluatorDatasetReceipt
from .evaluator_dataset_cas import verify_evaluator_dataset_root
from .evaluator_execution import LmEvalExecutionContract
from .evaluator_inferencex_projection import (
    EvaluatorInferenceXProjectionReceipt,
    PreparedInferenceXProjection,
    materialize_inferencex_projection,
)
from .evaluator_input_projection import (
    EvaluatorSidecarInputProjection,
    materialize_evaluator_sidecar_inputs,
)
from .evaluator_task_materialization import (
    EvaluatorTaskMaterializationReceipt,
    materialize_evaluator_task,
)
from .magpie_launch_projection import (
    MagpieLaunchConfigReceipt,
    materialize_magpie_launch_config,
)


@dataclass(frozen=True, slots=True)
class PreparedLmEvalExecution:
    """Private paths and immutable identities ready for a sidecar launcher."""

    authority_root: Path
    task_mount: Path
    dataset_mount: Path
    runtime_mount: Path
    runtime_receipt: LmEvalRuntimeReceipt
    input_projection: EvaluatorSidecarInputProjection
    sidecar_root: Path
    output_root: Path
    task_receipt: EvaluatorTaskMaterializationReceipt
    dataset_receipt: EvaluatorDatasetReceipt
    contract: LmEvalExecutionContract
    task_receipt_path: Path
    contract_path: Path
    inferencex_projection: PreparedInferenceXProjection
    inferencex_projection_receipt: EvaluatorInferenceXProjectionReceipt
    inferencex_projection_receipt_path: Path
    launch_config_path: Path
    launch_config_receipt: MagpieLaunchConfigReceipt
    launch_config_receipt_path: Path


class LmEvalExecutionPreparer:
    """Prepare exact inputs without changing Magpie or InferenceX checkouts."""

    def __init__(self, receipt: DependencyReceipt, dataset_root: Path) -> None:
        self._receipt = receipt
        self._dataset_root = dataset_root

    def verify_dataset(self) -> EvaluatorDatasetReceipt:
        """Rehash the exact offline dataset before advertising formal support."""

        policy = self._receipt.evaluator_policy
        if policy is None:
            raise _invalid("Evaluator policy lock is unavailable")
        return verify_evaluator_dataset_root(
            self._dataset_root,
            expected_repository=policy.dataset_repository,
            expected_path=policy.dataset_path,
            expected_name=policy.dataset_name,
            expected_revision=policy.dataset_revision,
            expected_splits=policy.dataset_splits,
        )

    def prepare(self, request: MagpieAttestationRequest) -> PreparedLmEvalExecution:
        policy_lock = self._receipt.evaluator_policy
        runtime = self._receipt.lm_eval_runtime
        if policy_lock is None or runtime is None:
            raise _invalid("Evaluator dependency receipts are incomplete")
        if request.evaluator_policy_lock != policy_lock.to_dict():
            raise _invalid("Evaluator request policy lock differs from dependencies")
        if request.lm_eval_runtime != runtime.to_dict():
            raise _invalid("Evaluator request runtime differs from dependencies")
        dataset = self.verify_dataset()
        authority = _authority_root(request.run_root)
        task = materialize_evaluator_task(
            self._receipt.root("inferencex"),
            authority / "task-materialization",
            source_commit=self._receipt.commits["inferencex"],
            source_tree=_dependency_tree(self._receipt, "inferencex"),
            definition_path=policy_lock.task_definition_path,
            definition_sha256=policy_lock.task_definition_sha256,
            dataset_path=policy_lock.dataset_path,
            dataset_name=policy_lock.dataset_name,
            dataset_revision=policy_lock.dataset_revision,
            dataset_receipt_sha256=dataset.sha256,
            dataset_files=_container_dataset_files(dataset),
        )
        contract = build_lm_eval_execution_contract(
            request, task=task, dataset=dataset
        )
        projection, launch_path, launch_receipt = _projection_and_launch(
            self._receipt, request, authority, contract
        )
        return _finish_preparation(
            authority=authority,
            dataset_root=self._dataset_root,
            runtime_receipt=runtime,
            task=task,
            dataset=dataset,
            contract=contract,
            projection=projection,
            launch_path=launch_path,
            launch_receipt=launch_receipt,
        )


def _authority_root(run_root: Path) -> Path:
    run = run_root.resolve(strict=True)
    authority = run / "authority" / "lm_eval"
    try:
        authority.mkdir(mode=0o700, parents=True, exist_ok=False)
    except OSError as error:
        raise _invalid("Cannot create private evaluator authority root") from error
    return authority


def _projection_and_launch(
    receipt: DependencyReceipt,
    request: MagpieAttestationRequest,
    authority: Path,
    contract: LmEvalExecutionContract,
) -> tuple[PreparedInferenceXProjection, Path, MagpieLaunchConfigReceipt]:
    projection = materialize_inferencex_projection(
        receipt.root("inferencex"),
        receipt.root("magpie"),
        authority / "inferencex",
        inferencex_commit=receipt.commits["inferencex"],
        inferencex_tree=_dependency_tree(receipt, "inferencex"),
        magpie_commit=receipt.commits["magpie"],
        magpie_tree=_dependency_tree(receipt, "magpie"),
        execution_contract=contract,
    )
    launch_path = authority / "magpie-launch.yaml"
    launch_receipt = materialize_magpie_launch_config(
        request.config_path,
        launch_path,
        canonical_sha256=request.config_sha256,
        inferencex_source_root=receipt.root("inferencex"),
        inferencex_projection_root=projection.root,
        inferencex_projection_receipt_sha256=projection.receipt.sha256,
    )
    return projection, launch_path, launch_receipt


def _finish_preparation(
    *,
    authority: Path,
    dataset_root: Path,
    runtime_receipt: LmEvalRuntimeReceipt,
    task: EvaluatorTaskMaterializationReceipt,
    dataset: EvaluatorDatasetReceipt,
    contract: LmEvalExecutionContract,
    projection: PreparedInferenceXProjection,
    launch_path: Path,
    launch_receipt: MagpieLaunchConfigReceipt,
) -> PreparedLmEvalExecution:
    inputs = materialize_evaluator_sidecar_inputs(
        authority,
        dataset_root=dataset_root.resolve(strict=True),
        dataset_receipt=dataset,
        runtime_receipt=runtime_receipt,
        launcher_source=Path(__file__).with_name("evaluator_sidecar_entry.py"),
        launcher_sha256=contract.launcher_sha256,
    )
    projected_runtime = replace(runtime_receipt, root=inputs.runtime_mount)
    sidecar_root = authority / "sidecar"
    sidecar_root.mkdir(mode=0o700)
    output = sidecar_root / "output"
    output.mkdir(mode=0o700)
    receipts = _authority_receipts(
        authority, task, contract, projection.receipt, launch_receipt
    )
    return PreparedLmEvalExecution(
        authority_root=authority,
        task_mount=authority / "task-materialization" / "task",
        dataset_mount=inputs.dataset_mount,
        runtime_mount=inputs.runtime_mount,
        runtime_receipt=projected_runtime,
        input_projection=inputs,
        sidecar_root=sidecar_root,
        output_root=output,
        task_receipt=task,
        dataset_receipt=dataset,
        contract=contract,
        task_receipt_path=receipts[0],
        contract_path=receipts[1],
        inferencex_projection=projection,
        inferencex_projection_receipt=projection.receipt,
        inferencex_projection_receipt_path=receipts[2],
        launch_config_path=launch_path,
        launch_config_receipt=launch_receipt,
        launch_config_receipt_path=receipts[3],
    )


def _authority_receipts(
    authority: Path,
    task: EvaluatorTaskMaterializationReceipt,
    contract: LmEvalExecutionContract,
    projection: EvaluatorInferenceXProjectionReceipt,
    launch: MagpieLaunchConfigReceipt,
) -> tuple[Path, Path, Path, Path]:
    return (
        _write_immutable(authority / "task_materialization_receipt.json", task.to_dict()),
        _write_immutable(authority / "execution_contract.json", contract.to_dict()),
        _write_immutable(
            authority / "inferencex_projection_receipt.json", projection.to_dict()
        ),
        _write_immutable(
            authority / "magpie_launch_config_receipt.json", launch.to_dict()
        ),
    )


def _dependency_tree(receipt: DependencyReceipt, name: str) -> str:
    dependencies = receipt.raw.get("dependencies")
    value = dependencies.get(name) if isinstance(dependencies, Mapping) else None
    tree = value.get("tree") if isinstance(value, Mapping) else None
    if not isinstance(tree, str) or len(tree) != 40:
        raise _invalid(f"Evaluator dependency tree is missing: {name}")
    return tree


def _container_dataset_files(
    receipt: EvaluatorDatasetReceipt,
) -> Mapping[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for item in receipt.files:
        grouped.setdefault(item.split, []).append(
            f"/evaluator/dataset/{item.artifact.path}"
        )
    return {
        split: tuple(sorted(paths)) for split, paths in sorted(grouped.items())
    }


def _write_immutable(path: Path, value: Mapping[str, Any]) -> Path:
    payload = canonical_json_bytes(value) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise _invalid(f"Cannot write evaluator authority artifact: {path.name}") from error
    return path.resolve(strict=True)


def _invalid(message: str) -> ConfigurationError:
    return ConfigurationError(message, "evaluator_execution_preparation_invalid")


__all__ = ["LmEvalExecutionPreparer", "PreparedLmEvalExecution"]
