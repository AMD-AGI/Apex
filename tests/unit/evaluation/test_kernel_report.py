from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from apex.core import ContractError, IntegrityError, TaskStatus
from apex.evaluation import (
    GateVerdict,
    MeasurementPolicy,
    MeasurementStatus,
    grade_kernel,
    load_kernel_measurement_report,
)


def _document(*, samples: int = 300) -> dict[str, object]:
    per_block = tuple(
        samples // 4 + (1 if index < samples % 4 else 0)
        for index in range(4)
    )
    health = {
        "device": "gfx950:0",
        "healthy": True,
        "temperature_c": 45.0,
        "clock_mhz": 2100.0,
    }
    implementations = (
        "reference", "optimized", "optimized", "reference",
        "optimized", "reference", "reference", "optimized",
    )
    seen = {"reference": 0, "optimized": 0}
    order = []
    for implementation in implementations:
        index = seen[implementation]
        seen[implementation] += 1
        latency = 10.0 if implementation == "reference" else 8.0
        order.append((implementation, latency, per_block[index]))
    return {
        "schema": "apex.kernel-measurement/v1",
        "policy_id": "kernel_invocation_nearest_rank_v1",
        "sample_unit": "kernel_invocation",
        "quantile_method": "nearest_rank_v1",
        "timer": "hip_event",
        "timer_resolution_ns": 1.0,
        "inner_repeats": 1,
        "measurement_method_sha256": "1" * 64,
        "abba_seed": 17,
        "warmup_samples": 20,
        "cases": [
            {
                "case_id": "m128-n4096",
                "workload_count": 2,
                "blocks": [
                    {
                        "block_id": index,
                        "order_position": index,
                        "implementation": implementation,
                        "samples_ms": [latency] * count,
                        "invalid_sample_counts": {},
                        "gpu_health_before": health,
                        "gpu_health_after": health,
                    }
                    for index, (implementation, latency, count) in enumerate(order)
                ],
            }
        ],
    }


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_report_preserves_raw_samples_and_recomputes_robust_grade(tmp_path: Path) -> None:
    report = load_kernel_measurement_report(_write(tmp_path / "timings.json", _document()))
    gates = GateVerdict(True, True, True, True)

    grade = grade_kernel(
        gates,
        report.cases,
        measurement_policy=report.policy,
        aggregation=report.aggregation,
    )

    assert grade.measurement_status is MeasurementStatus.VALID
    assert grade.s50 == pytest.approx(1.25)
    assert grade.s99 == pytest.approx(1.25)
    assert grade.srobust == pytest.approx(1.25)
    assert grade.promotion_eligible is True
    assert grade.srobust_ci_lower == pytest.approx(1.25)
    assert all(case.reference.artifact_sha256 == report.sha256 for case in grade.cases)
    assert [block.implementation for block in report.blocks] == [
        "reference",
        "optimized",
        "optimized",
        "reference",
        "optimized",
        "reference",
        "reference",
        "optimized",
    ]
    assert len(report.cases[0].paired_units) == 2


def test_frozen_task_policy_must_match_report_protocol(tmp_path: Path) -> None:
    path = _write(tmp_path / "timings.json", _document())

    with pytest.raises(ContractError) as raised:
        load_kernel_measurement_report(
            path,
            measurement_policy=MeasurementPolicy(warmup_samples=21),
        )

    assert raised.value.reason_code == "measurement_policy_mismatch"


def test_299_samples_parse_but_never_receive_p99_reward(tmp_path: Path) -> None:
    report = load_kernel_measurement_report(
        _write(tmp_path / "short.json", _document(samples=299))
    )
    grade = grade_kernel(GateVerdict(True, True, True, True), report.cases)

    assert grade.measurement_status is MeasurementStatus.INSUFFICIENT_SAMPLES
    assert grade.task_status is TaskStatus.NO_MEASUREMENT
    assert grade.reward is None


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(policy_id="untrusted_policy"),
        lambda value: value.update(sample_unit="batched_average"),
        lambda value: value["cases"][0].pop("blocks"),
        lambda value: value["cases"][0]["blocks"][0].update(samples_ms=[float("nan")]),
        lambda value: value["cases"][0].update(mean_ms=1.0),
    ],
)
def test_summaries_nonfinite_and_unknown_semantics_fail_closed(
    tmp_path: Path, mutation
) -> None:
    value = _document()
    mutation(value)
    with pytest.raises((ContractError, IntegrityError)):
        load_kernel_measurement_report(_write(tmp_path / "invalid.json", value))


def test_duplicate_keys_links_and_hardlinks_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":"apex.kernel-measurement/v1","schema":"other"}',
        encoding="utf-8",
    )
    with pytest.raises(IntegrityError):
        load_kernel_measurement_report(duplicate)

    original = _write(tmp_path / "original.json", _document())
    link = tmp_path / "link.json"
    link.symlink_to(original)
    with pytest.raises(IntegrityError) as symlink:
        load_kernel_measurement_report(link)
    assert symlink.value.reason_code == "unsafe_measurement_report"

    hardlink = tmp_path / "hardlink.json"
    os.link(original, hardlink)
    with pytest.raises(IntegrityError) as hard:
        load_kernel_measurement_report(hardlink)
    assert hard.value.reason_code == "unsafe_measurement_report"


def test_missing_report_is_a_typed_integrity_failure(tmp_path: Path) -> None:
    with pytest.raises(IntegrityError) as raised:
        load_kernel_measurement_report(tmp_path / "missing.json")

    assert raised.value.reason_code == "measurement_report_missing"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda value: value["cases"][0]["blocks"][1].update(
                implementation="reference"
            ),
            "invalid_abba_blocks",
        ),
        (
            lambda value: value["cases"][0]["blocks"][0][
                "gpu_health_before"
            ].update(healthy=False),
            "gpu_health_violation",
        ),
        (lambda value: value.update(inner_repeats=8), "unsupported_sample_unit"),
    ],
)
def test_abba_health_and_invocation_unit_fail_closed(
    tmp_path: Path, mutation, reason: str
) -> None:
    value = _document()
    mutation(value)
    with pytest.raises(ContractError) as raised:
        load_kernel_measurement_report(_write(tmp_path / "invalid-protocol.json", value))
    assert raised.value.reason_code == reason
