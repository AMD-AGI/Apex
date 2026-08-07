from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import ContractError
from apex.intake import E2EOptimizeSpec
from apex.optimization.e2e.use_case import _hf_offline


def _spec(tmp_path: Path, value: object) -> E2EOptimizeSpec:
    return E2EOptimizeSpec.from_mapping(
        {
            "config_path": str((tmp_path / "benchmark.yaml").resolve()),
            "results_dir": str((tmp_path / "results").resolve()),
            "deployment_hints": {"hf_offline": value},
        }
    )


def test_hf_offline_hint_is_explicit_boolean(tmp_path: Path) -> None:
    assert _hf_offline(_spec(tmp_path, True)) is True
    assert _hf_offline(_spec(tmp_path, False)) is False


@pytest.mark.parametrize("value", ["true", 1, None])
def test_hf_offline_hint_rejects_coercion(tmp_path: Path, value: object) -> None:
    with pytest.raises(ContractError) as caught:
        _hf_offline(_spec(tmp_path, value))
    assert caught.value.reason_code == "invalid_hf_offline"
