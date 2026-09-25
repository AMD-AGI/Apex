"""Fail-closed subprocess environment construction."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping

from apex.core import ContractError


BASE_ENVIRONMENT_KEYS = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TERM",
    "COLORTERM",
    "NO_COLOR",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_DATA_HOME",
    "XDG_STATE_HOME",
    "VIRTUAL_ENV",
)

GPU_RUNTIME_ENVIRONMENT_KEYS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_OVERRIDE_GFX_VERSION",
    "HSA_XNACK",
    "HSA_ENABLE_SDMA",
    "ROCM_PATH",
    "ROCM_HOME",
    "HIP_PATH",
)

HF_RUNTIME_ENVIRONMENT_KEYS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_DATASETS_CACHE",
    "HF_HUB_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "HF_HUB_ENABLE_HF_TRANSFER",
)

HF_CREDENTIAL_ENVIRONMENT_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

DOCKER_RUNTIME_ENVIRONMENT_KEYS = (
    "DOCKER_HOST",
    "DOCKER_CONTEXT",
    "DOCKER_CONFIG",
    "DOCKER_TLS_VERIFY",
    "DOCKER_CERT_PATH",
    "DOCKER_API_VERSION",
)

AGENT_CONFIG_ENVIRONMENT_KEYS = (
    "CODEX_HOME",
    "CLAUDE_CONFIG_DIR",
    "CURSOR_CONFIG_DIR",
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
)

_BLOCKED_KEYS = frozenset(
    {
        "BASH_ENV",
        "ENV",
        "SHELLOPTS",
        "PS4",
        "CDPATH",
        "LD_PRELOAD",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_ORIGIN_PATH",
        "LD_PROFILE",
        "LD_DEBUG",
        "LD_DEBUG_OUTPUT",
        "GCONV_PATH",
        "PERL5OPT",
        "PERL5LIB",
        "RUBYOPT",
        "RUBYLIB",
        "NODE_OPTIONS",
        "NODE_PATH",
        "JAVA_TOOL_OPTIONS",
        "_JAVA_OPTIONS",
        "DOCKER_AUTH_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
    }
)
_BLOCKED_PREFIXES = (
    "PYTHON",
    "DYLD_",
    "GIT_CONFIG_KEY_",
    "GIT_CONFIG_VALUE_",
)
_TRUSTED_FIXED_KEYS = frozenset(
    {
        "PYTHONNOUSERSITE",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_TERMINAL_PROMPT",
        "GIT_OPTIONAL_LOCKS",
    }
)
_SECRET_NAME = re.compile(
    r"(?:^|_)(?:API_KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|PRIVATE_KEY|ACCESS_KEY)(?:_|$)"
)
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MAX_ENTRIES = 256
_MAX_VALUE_BYTES = 65_536
_MAX_TOTAL_BYTES = 262_144


def build_subprocess_environment(
    overrides: Mapping[str, str] | None = None,
    *,
    inherit: Iterable[str] = (),
    inherit_secrets: Iterable[str] = (),
    allow_override_secrets: Iterable[str] = (),
    fixed: Mapping[str, str] | None = None,
    reserved: Iterable[str] = (),
) -> dict[str, str]:
    """Build a bounded environment without inheriting arbitrary host state.

    Secret inheritance is opt-in by exact key. Caller overrides may add normal
    variables, but startup hooks, loader injection, Python injection, and
    unapproved credential-shaped keys fail closed.
    """

    inherited_secret_keys = frozenset(inherit_secrets)
    override_secret_keys = frozenset(allow_override_secrets)
    fixed_values = dict(fixed or {})
    reserved_keys = frozenset(reserved) | fixed_values.keys()
    environment: dict[str, str] = {}
    inherited_keys = (*BASE_ENVIRONMENT_KEYS, *tuple(inherit), *inherited_secret_keys)
    for key in dict.fromkeys(inherited_keys):
        value = os.environ.get(key)
        if value is None:
            continue
        _validate_entry(
            key,
            value,
            allowed_secret_keys=inherited_secret_keys,
            trusted_fixed=False,
        )
        environment[key] = value
    environment.setdefault("PATH", os.defpath)

    for key, value in dict(overrides or {}).items():
        if key in reserved_keys:
            raise ContractError(
                f"Environment variable is adapter-owned: {key}",
                "reserved_environment_variable",
                {"key": key},
            )
        _validate_entry(
            key,
            value,
            allowed_secret_keys=override_secret_keys,
            trusted_fixed=False,
        )
        environment[key] = value

    for key, value in fixed_values.items():
        _validate_entry(key, value, allowed_secret_keys=frozenset(), trusted_fixed=True)
        environment[key] = value
    environment["PYTHONNOUSERSITE"] = "1"
    _validate_bounds(environment)
    return environment


def _validate_entry(
    key: object,
    value: object,
    *,
    allowed_secret_keys: frozenset[str],
    trusted_fixed: bool,
) -> None:
    if not isinstance(key, str) or not _ENVIRONMENT_NAME.fullmatch(key):
        raise ContractError("Invalid subprocess environment name", "invalid_environment")
    if not isinstance(value, str) or "\x00" in value:
        raise ContractError(
            f"Invalid subprocess environment value: {key}", "invalid_environment"
        )
    normalized = key.upper()
    trusted_adapter_value = trusted_fixed and key in _TRUSTED_FIXED_KEYS
    if not trusted_adapter_value and (
        normalized in _BLOCKED_KEYS
        or any(normalized.startswith(prefix) for prefix in _BLOCKED_PREFIXES)
    ):
        raise ContractError(
            f"Unsafe subprocess environment variable: {key}",
            "unsafe_environment_variable",
            {"key": key},
        )
    if _looks_secret(normalized) and key not in allowed_secret_keys:
        raise ContractError(
            f"Unapproved secret environment variable: {key}",
            "secret_environment_variable",
            {"key": key},
        )


def _looks_secret(normalized: str) -> bool:
    return _SECRET_NAME.search(normalized) is not None


def _validate_bounds(environment: Mapping[str, str]) -> None:
    if len(environment) > _MAX_ENTRIES:
        raise ContractError(
            "Subprocess environment has too many entries", "environment_too_large"
        )
    total = 0
    for key, value in environment.items():
        encoded = len(key.encode("utf-8")) + len(value.encode("utf-8")) + 2
        if encoded > _MAX_VALUE_BYTES:
            raise ContractError(
                f"Subprocess environment value is too large: {key}",
                "environment_too_large",
            )
        total += encoded
    if total > _MAX_TOTAL_BYTES:
        raise ContractError(
            "Subprocess environment is too large", "environment_too_large"
        )


__all__ = [
    "AGENT_CONFIG_ENVIRONMENT_KEYS",
    "DOCKER_RUNTIME_ENVIRONMENT_KEYS",
    "GPU_RUNTIME_ENVIRONMENT_KEYS",
    "HF_CREDENTIAL_ENVIRONMENT_KEYS",
    "HF_RUNTIME_ENVIRONMENT_KEYS",
    "build_subprocess_environment",
]
