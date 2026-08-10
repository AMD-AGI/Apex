"""Static formal-quality support receipts for current Magpie observers."""

from __future__ import annotations

from apex.ports import MagpieFormalMeasurementSupport


def docker_formal_measurement_support(
    execution_mode: str, lifecycle: str
) -> MagpieFormalMeasurementSupport:
    del execution_mode, lifecycle
    return MagpieFormalMeasurementSupport(
        False,
        "magpie_quality_evaluator_authority_unavailable",
        None,
        (
            "magpie_inferencex_eval_argument_mismatch",
            "lm_eval_runtime_engagement_unproven",
            "quality_samples_unavailable",
        ),
    )


def local_formal_measurement_support(
    execution_mode: str, lifecycle: str
) -> MagpieFormalMeasurementSupport:
    del execution_mode, lifecycle
    return MagpieFormalMeasurementSupport(
        False,
        "magpie_local_quality_execution_unavailable",
        None,
        (
            "magpie_inferencex_eval_argument_mismatch",
            "local_lm_eval_interpreter_unbound",
            "local_lm_eval_python_abi_mismatch",
            "local_remote_eval_task_contract_mismatch",
            "local_remote_eval_samples_unavailable",
            "local_remote_eval_policy_unconsumed",
        ),
    )


def ray_formal_measurement_support(
    execution_mode: str, lifecycle: str
) -> MagpieFormalMeasurementSupport:
    del execution_mode, lifecycle
    return MagpieFormalMeasurementSupport(
        False,
        "magpie_quality_evaluator_authority_unavailable",
        None,
        ("ray_node_quality_execution_unproven",),
    )


__all__ = [
    "docker_formal_measurement_support",
    "local_formal_measurement_support",
    "ray_formal_measurement_support",
]
