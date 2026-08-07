from __future__ import annotations

import random

import pytest

from apex.core import ContractError
from apex.evaluation import (
    MeasurementPolicy,
    PairedTimingUnit,
    SampleSeries,
    coefficient_of_variation,
    paired_block_bootstrap,
    quantiles,
)


def test_nearest_rank_p99_and_true_median_are_reproducible() -> None:
    values = [float(value) for value in range(1, 301)]
    random.Random(7).shuffle(values)

    result = quantiles(SampleSeries(tuple(values)), MeasurementPolicy())

    assert result.p50_ms == 150.5
    assert result.p99_ms == 297.0
    assert result.sample_count == 300
    assert result.quantile_method == "nearest_rank_v1"


def test_insufficient_samples_fail_closed() -> None:
    with pytest.raises(ContractError) as raised:
        quantiles(SampleSeries((1.0,) * 299), MeasurementPolicy())

    assert raised.value.reason_code == "insufficient_samples"


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_samples_are_invalid(value: float) -> None:
    with pytest.raises(ContractError) as raised:
        SampleSeries((value,))

    assert raised.value.reason_code == "invalid_latency_sample"


def test_batched_mean_is_not_reward_eligible() -> None:
    series = SampleSeries((1.0,) * 300, sample_unit="batch_per_launch_mean")

    with pytest.raises(ContractError) as raised:
        quantiles(series, MeasurementPolicy())

    assert raised.value.reason_code == "unsupported_sample_unit"


def test_policy_identity_is_exact_and_versioned() -> None:
    with pytest.raises(ContractError) as raised:
        MeasurementPolicy(policy_id="caller_defined")

    assert raised.value.reason_code == "unsupported_measurement_policy"


@pytest.mark.parametrize(
    "overrides",
    [
        {"keep_srobust_threshold": 1.049},
        {"confidence_srobust_floor": 0.99},
        {"worst_case_srobust_floor": 0.99},
        {"max_cv": 0.101},
        {"bootstrap_confidence_level": 0.90},
    ],
)
def test_policy_cannot_weaken_canonical_promotion_gates(
    overrides: dict[str, float],
) -> None:
    with pytest.raises(ContractError) as raised:
        MeasurementPolicy(**overrides)

    assert raised.value.reason_code == "invalid_measurement_policy"


def test_policy_rejects_fewer_than_300_valid_samples() -> None:
    with pytest.raises(ContractError) as raised:
        MeasurementPolicy(min_valid_samples=299)

    assert raised.value.reason_code == "invalid_measurement_policy"


def test_paired_block_bootstrap_is_seeded_and_resamples_whole_units() -> None:
    units = (
        PairedTimingUnit(0, (10.0,) * 150, (5.0,) * 150),
        PairedTimingUnit(1, (20.0,) * 150, (18.0,) * 150),
    )
    policy = MeasurementPolicy(bootstrap_repetitions=200)

    first = paired_block_bootstrap(units, policy, seed=17)
    replay = paired_block_bootstrap(units, policy, seed=17)
    other = paired_block_bootstrap(units, policy, seed=18)

    assert first == replay
    assert first is not None and other is not None
    assert first.srobust != other.srobust
    assert first.unit_count == 2


def test_population_cv_preserves_tail_variation() -> None:
    assert coefficient_of_variation((10.0,) * 300) == 0.0
    assert coefficient_of_variation((5.0, 15.0) * 150) == pytest.approx(0.5)
