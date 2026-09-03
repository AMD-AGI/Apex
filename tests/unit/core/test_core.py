from __future__ import annotations

import json

import pytest

from apex.core import (
    ContractError,
    TaskStatus,
    ValidationLevel,
    canonical_json_bytes,
    new_identifier,
    sha256_json,
    validate_identifier,
)


def test_canonical_json_is_order_independent() -> None:
    left = {"b": [2, 1], "a": {"value": "GPU"}}
    right = {"a": {"value": "GPU"}, "b": [2, 1]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert sha256_json(left) == sha256_json(right)
    assert json.loads(canonical_json_bytes(left)) == left


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"latency": float("nan")})


@pytest.mark.parametrize("value", ["../escape", " space", "", "a/b", "x" * 129])
def test_identifier_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(ContractError, match="Invalid"):
        validate_identifier(value)


def test_new_identifier_is_valid_and_unique() -> None:
    first = new_identifier("run")
    second = new_identifier("run")

    assert first != second
    assert validate_identifier(first) == first


def test_status_and_validation_level_are_separate_contracts() -> None:
    assert TaskStatus.SUCCEEDED.value == "succeeded"
    assert ValidationLevel.SOURCE_REBUILD_VERIFIED.value == "source_rebuild_verified"
    assert {item.value for item in TaskStatus}.isdisjoint({item.value for item in ValidationLevel})
