"""Docker formal-evaluator launch projection checks."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from apex.core import ContractError, sha256_file

from .evaluator_preparation import PreparedLmEvalExecution


def project_docker_evaluator_launch_argv(
    canonical: tuple[str, ...], launch_config: Path
) -> tuple[str, ...]:
    positions = tuple(
        index for index, value in enumerate(canonical)
        if value == "--benchmark-config"
    )
    if (
        positions != (4,)
        or len(canonical) != 8
        or canonical[5] == ""
        or Path(canonical[5]).resolve() == launch_config.resolve()
    ):
        raise ContractError(
            "Canonical Magpie argv cannot be projected", "magpie_launch_argv_invalid"
        )
    projected = list(canonical)
    projected[5] = str(launch_config.resolve(strict=True))
    return tuple(projected)


def inferencex_dependency_entry(
    dependencies: Mapping[str, object],
) -> Mapping[str, object]:
    values = dependencies.get("dependencies")
    selected = values.get("inferencex") if isinstance(values, Mapping) else None
    if not isinstance(selected, Mapping):
        raise ContractError(
            "InferenceX dependency evidence is missing",
            "magpie_dependency_observation_failed",
        )
    return selected


def validate_prepared_docker_evaluator_inputs(
    prepared: PreparedLmEvalExecution,
) -> None:
    launch = prepared.launch_config_path
    if (
        launch.is_symlink()
        or not launch.is_file()
        or launch.stat().st_nlink != 1
        or launch.stat().st_mode & 0o222
        or sha256_file(launch) != prepared.launch_config_receipt.launch_config_sha256
        or prepared.launch_config_receipt.canonical_config_sha256
        != prepared.contract.config_sha256
        or prepared.launch_config_receipt.inferencex_projection_root
        != str(prepared.inferencex_projection.root)
        or prepared.launch_config_receipt.inferencex_projection_receipt_sha256
        != prepared.inferencex_projection.receipt.sha256
        or prepared.inferencex_projection_receipt
        != prepared.inferencex_projection.receipt
    ):
        raise ContractError(
            "Evaluator launch projection changed", "evaluator_prepared_input_drift"
        )


def publish_inferencex_runtime(*args, **kwargs) -> Mapping[str, object]:
    from .evaluator_inferencex_runtime_publication import (
        publish_inferencex_projection_evidence,
    )

    return publish_inferencex_projection_evidence(*args, **kwargs)


__all__ = [
    "inferencex_dependency_entry",
    "project_docker_evaluator_launch_argv",
    "publish_inferencex_runtime",
    "validate_prepared_docker_evaluator_inputs",
]
