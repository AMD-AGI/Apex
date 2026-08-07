from __future__ import annotations

import pytest

from apex.core import ContractError
from apex.execution import (
    DOCKER_RUNTIME_ENVIRONMENT_KEYS,
    GPU_RUNTIME_ENVIRONMENT_KEYS,
    HF_RUNTIME_ENVIRONMENT_KEYS,
    build_subprocess_environment,
)


def test_environment_inherits_only_selected_runtime_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = {
        "ROCR_VISIBLE_DEVICES": "3",
        "HF_HOME": "/cache/huggingface",
        "DOCKER_HOST": "unix:///run/user/1000/docker.sock",
    }
    hostile = {
        "BASH_ENV": "/tmp/startup.sh",
        "ENV": "/tmp/sh-startup",
        "LD_PRELOAD": "/tmp/override.so",
        "PYTHONPATH": "/tmp/import-first",
        "DOCKER_AUTH_CONFIG": '{"auths":{"registry":{"auth":"secret"}}}',
        "AWS_SECRET_ACCESS_KEY": "secret",
        "UNRELATED_HOST_VALUE": "ambient",
    }
    for key, value in {**selected, **hostile}.items():
        monkeypatch.setenv(key, value)

    environment = build_subprocess_environment(
        inherit=(
            *GPU_RUNTIME_ENVIRONMENT_KEYS,
            *HF_RUNTIME_ENVIRONMENT_KEYS,
            *DOCKER_RUNTIME_ENVIRONMENT_KEYS,
        )
    )

    assert all(environment[key] == value for key, value in selected.items())
    assert all(key not in environment for key in hostile)
    assert environment["PYTHONNOUSERSITE"] == "1"


@pytest.mark.parametrize(
    "key",
    (
        "BASH_ENV",
        "ENV",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "DYLD_INSERT_LIBRARIES",
        "NODE_OPTIONS",
        "DOCKER_AUTH_CONFIG",
        "GIT_CONFIG_COUNT",
        "GITHUB_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
    ),
)
def test_explicit_unsafe_or_secret_override_fails_closed(key: str) -> None:
    with pytest.raises(ContractError) as failure:
        build_subprocess_environment({key: "injected"})

    assert failure.value.reason_code in {
        "unsafe_environment_variable",
        "secret_environment_variable",
    }


def test_explicit_normal_override_is_allowed_but_adapter_field_is_reserved() -> None:
    environment = build_subprocess_environment(
        {"TOKENIZERS_PARALLELISM": "false", "VLLM_ROCM_USE_AITER": "1"},
        fixed={"MAGPIE_ROOT": "/verified/magpie"},
    )
    assert environment["TOKENIZERS_PARALLELISM"] == "false"
    assert environment["VLLM_ROCM_USE_AITER"] == "1"

    with pytest.raises(ContractError) as failure:
        build_subprocess_environment(
            {"MAGPIE_ROOT": "/unverified/magpie"},
            fixed={"MAGPIE_ROOT": "/verified/magpie"},
        )
    assert failure.value.reason_code == "reserved_environment_variable"


def test_secret_inheritance_and_override_require_exact_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")

    environment = build_subprocess_environment(
        {"OPENAI_API_KEY": "explicit-openai-secret"},
        inherit_secrets=("OPENAI_API_KEY",),
        allow_override_secrets=("OPENAI_API_KEY",),
    )

    assert environment["OPENAI_API_KEY"] == "explicit-openai-secret"
    assert "ANTHROPIC_API_KEY" not in environment

    with pytest.raises(ContractError) as failure:
        build_subprocess_environment(
            {"ANTHROPIC_API_KEY": "cross-backend-secret"},
            inherit_secrets=("OPENAI_API_KEY",),
            allow_override_secrets=("OPENAI_API_KEY",),
        )
    assert failure.value.reason_code == "secret_environment_variable"
