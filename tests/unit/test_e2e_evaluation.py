from __future__ import annotations

import pytest

from apex.core import ContractError
from apex.evaluation import E2EMeasurement, evaluate_current_anchor, evaluate_no_regression


def _measurement(
    *,
    throughput: float = 100,
    ttft: float = 100,
    tpot: float = 10,
    accuracy: float = 0.8,
    protocol: str = "a" * 64,
) -> E2EMeasurement:
    return E2EMeasurement(
        throughput,
        ttft,
        tpot,
        accuracy,
        100,
        protocol,
        "quality-receipt",
        "measurement-receipt",
    )


def test_current_anchor_keep_requires_gain_and_all_hard_gates() -> None:
    verdict = evaluate_current_anchor(
        _measurement(), _measurement(throughput=102, ttft=104.9, tpot=10.19)
    )
    assert verdict.keep
    assert verdict.reason_code == "accepted"


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (_measurement(throughput=102, accuracy=0.79), "accuracy_regression"),
        (_measurement(throughput=102, ttft=105.1), "ttft_p99_regression"),
        (_measurement(throughput=102, tpot=10.21), "tpot_p99_regression"),
        (_measurement(throughput=100.49), "insufficient_throughput_gain"),
    ],
)
def test_regressions_and_noise_are_reverted(candidate: E2EMeasurement, reason: str) -> None:
    verdict = evaluate_current_anchor(_measurement(), candidate)
    assert not verdict.keep
    assert verdict.reason_code == reason


def test_protocol_mismatch_is_not_comparable() -> None:
    with pytest.raises(ContractError) as failure:
        evaluate_current_anchor(_measurement(), _measurement(protocol="b" * 64))
    assert failure.value.reason_code == "measurement_protocol_mismatch"


def test_diagnostic_measurement_is_never_rewardable() -> None:
    with pytest.raises(ContractError) as failure:
        E2EMeasurement(100, 10, 2, 0.8, 100, "a" * 64, "q", "m", "diagnostic")
    assert failure.value.reason_code == "diagnostic_not_rewardable"


def test_unchanged_replay_allows_small_noise_but_no_quality_or_tail_regression() -> None:
    assert evaluate_no_regression(_measurement(), _measurement(throughput=99.1)).keep
    verdict = evaluate_no_regression(_measurement(), _measurement(throughput=98.9))
    assert not verdict.keep
    assert verdict.reason_code == "insufficient_throughput_gain"
