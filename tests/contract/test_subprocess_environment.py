from __future__ import annotations

import pytest

from apex.execution import (
    GPU_RUNTIME_ENVIRONMENT_KEYS,
    HF_CREDENTIAL_ENVIRONMENT_KEYS,
    build_subprocess_environment,
)


def test_runtime_and_credentials_are_opt_in_not_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("HF_TOKEN", "ambient-hf-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-agent-secret")

    default = build_subprocess_environment()
    selected = build_subprocess_environment(inherit=GPU_RUNTIME_ENVIRONMENT_KEYS)
    explicit_hf = build_subprocess_environment(
        {"HF_TOKEN": "caller-hf-secret"},
        allow_override_secrets=HF_CREDENTIAL_ENVIRONMENT_KEYS,
    )

    assert "ROCR_VISIBLE_DEVICES" not in default
    assert "HF_TOKEN" not in default
    assert "OPENAI_API_KEY" not in default
    assert selected["ROCR_VISIBLE_DEVICES"] == "1"
    assert "HF_TOKEN" not in selected
    assert explicit_hf["HF_TOKEN"] == "caller-hf-secret"
    assert "OPENAI_API_KEY" not in explicit_hf
