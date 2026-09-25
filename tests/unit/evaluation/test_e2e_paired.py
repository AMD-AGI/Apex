from __future__ import annotations

import pytest

from apex.core import ContractError
from apex.evaluation import (
    E2EAcceptancePolicy,
    E2EObservation,
    E2EPairedMeasurement,
    E2EPairedWindow,
    estimate_e2e_paired,
    evaluate_paired_current_anchor,
    load_e2e_paired_measurement,
)


def _observation(
    receipt: str,
    *,
    throughput: float = 100.0,
    ttft: float = 100.0,
    tpot: float = 10.0,
    accuracy: float = 0.8,
    protocol: str = "a" * 64,
) -> E2EObservation:
    return E2EObservation(
        throughput,
        ttft,
        tpot,
        accuracy,
        100,
        protocol,
        f"quality-{receipt}",
        f"measurement-{receipt}",
    )


def _window(
    index: int,
    ratio: float,
    *,
    candidate_ttft: float = 100.0,
    candidate_tpot: float = 10.0,
    candidate_accuracy: float = 0.8,
    protocol: str = "a" * 64,
) -> E2EPairedWindow:
    prefix = f"w{index}"
    return E2EPairedWindow(
        prefix,
        _observation(f"{prefix}-a0", protocol=protocol),
        _observation(
            f"{prefix}-c1",
            throughput=100.0 * ratio,
            ttft=candidate_ttft,
            tpot=candidate_tpot,
            accuracy=candidate_accuracy,
            protocol=protocol,
        ),
        _observation(
            f"{prefix}-c2",
            throughput=100.0 * ratio,
            ttft=candidate_ttft,
            tpot=candidate_tpot,
            accuracy=candidate_accuracy,
            protocol=protocol,
        ),
        _observation(f"{prefix}-a3", protocol=protocol),
    )


def _measurement(
    ratios: tuple[float, ...],
    *,
    policy: E2EAcceptancePolicy | None = None,
    **window_values: float,
) -> E2EPairedMeasurement:
    chosen = policy or E2EAcceptancePolicy()
    return E2EPairedMeasurement(
        tuple(_window(index, ratio, **window_values) for index, ratio in enumerate(ratios)),
        chosen.digest,
        chosen.min_paired_windows,
    )


def test_three_complete_abba_windows_produce_point_estimate_and_keep() -> None:
    measurement = _measurement((1.1, 1.1, 1.1))

    estimate = estimate_e2e_paired(measurement)
    verdict = evaluate_paired_current_anchor(measurement)

    assert estimate.throughput_ratio == pytest.approx(1.1)
    assert estimate.throughput_gain_pct == pytest.approx(10.0)
    assert estimate.throughput_confidence_lower_pct == pytest.approx(10.0)
    assert estimate.paired_window_count == 3
    assert estimate.paired_ratio_count == 6
    assert verdict.keep
    assert verdict.reason_code == "accepted"


def test_high_point_reward_does_not_replace_promotion_confidence() -> None:
    measurement = _measurement((1.2, 1.2, 0.9))

    verdict = evaluate_paired_current_anchor(measurement)

    assert verdict.throughput_gain_pct > 0.5
    assert verdict.estimate.throughput_confidence_lower_pct < 0.5
    assert not verdict.keep
    assert verdict.reason_code == "insufficient_throughput_confidence"


def test_any_accuracy_regression_is_a_hard_gate() -> None:
    verdict = evaluate_paired_current_anchor(
        _measurement((1.1, 1.1, 1.1), candidate_accuracy=0.799)
    )

    assert not verdict.keep
    assert verdict.reason_code == "accuracy_regression"


@pytest.mark.parametrize(
    ("candidate_ttft", "candidate_tpot", "keep", "reason"),
    (
        (105.0, 10.2, True, "accepted"),
        (105.00001, 10.2, False, "ttft_p99_regression"),
        (105.0, 10.20001, False, "tpot_p99_regression"),
    ),
)
def test_tail_latency_gate_boundaries(
    candidate_ttft: float, candidate_tpot: float, keep: bool, reason: str
) -> None:
    verdict = evaluate_paired_current_anchor(
        _measurement(
            (1.1, 1.1, 1.1),
            candidate_ttft=candidate_ttft,
            candidate_tpot=candidate_tpot,
        )
    )

    assert verdict.keep is keep
    assert verdict.reason_code == reason


def test_incomplete_window_set_is_not_a_formal_measurement() -> None:
    policy = E2EAcceptancePolicy()
    with pytest.raises(ContractError) as failure:
        E2EPairedMeasurement(
            (_window(0, 1.1), _window(1, 1.1)), policy.digest, policy.min_paired_windows
        )

    assert failure.value.reason_code == "insufficient_e2e_paired_windows"


def test_policy_minimum_and_digest_are_bound_before_estimation() -> None:
    report_policy = E2EAcceptancePolicy()
    report = _measurement((1.1, 1.1, 1.1), policy=report_policy)

    with pytest.raises(ContractError) as minimum_failure:
        estimate_e2e_paired(report, E2EAcceptancePolicy(min_paired_windows=4))
    assert minimum_failure.value.reason_code == "e2e_acceptance_policy_mismatch"

    changed = E2EAcceptancePolicy(bootstrap_seed=1)
    with pytest.raises(ContractError) as digest_failure:
        estimate_e2e_paired(report, changed)
    assert digest_failure.value.reason_code == "e2e_acceptance_policy_mismatch"


def test_protocol_drift_and_duplicate_raw_receipts_are_rejected() -> None:
    with pytest.raises(ContractError) as protocol_failure:
        E2EPairedWindow(
            "drift",
            _observation("a0"),
            _observation("c1"),
            _observation("c2", protocol="b" * 64),
            _observation("a3"),
        )
    assert protocol_failure.value.reason_code == "measurement_protocol_mismatch"

    policy = E2EAcceptancePolicy()
    duplicate = _window(1, 1.1)
    with pytest.raises(ContractError) as receipt_failure:
        E2EPairedMeasurement(
            (_window(0, 1.1), duplicate, duplicate),
            policy.digest,
            policy.min_paired_windows,
        )
    assert receipt_failure.value.reason_code == "invalid_e2e_paired_measurement"


def test_document_preserves_abba_order_and_all_raw_receipts() -> None:
    measurement = _measurement((1.1, 1.1, 1.1))
    document = measurement.to_dict()

    assert document["schema"] == "apex.e2e-paired-measurement/v1"
    assert document["acceptance_policy_digest"] == E2EAcceptancePolicy().digest
    assert all(window["order"] == ["anchor", "candidate", "candidate", "anchor"] for window in document["windows"])
    assert len(document["raw_measurement_receipts"]) == 12
    assert load_e2e_paired_measurement(document) == measurement


def test_loader_rejects_summary_only_or_tampered_raw_receipts() -> None:
    document = _measurement((1.1, 1.1, 1.1)).to_dict()
    document["raw_measurement_receipts"][0] = "forged"

    with pytest.raises(ContractError) as failure:
        load_e2e_paired_measurement(document)

    assert failure.value.reason_code == "invalid_e2e_paired_measurement"
