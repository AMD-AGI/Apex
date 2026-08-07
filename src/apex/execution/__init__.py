"""Concrete stateless Codex, Claude, and Cursor agent adapters."""

from .environment import (
    DOCKER_RUNTIME_ENVIRONMENT_KEYS,
    GPU_RUNTIME_ENVIRONMENT_KEYS,
    HF_CREDENTIAL_ENVIRONMENT_KEYS,
    HF_RUNTIME_ENVIRONMENT_KEYS,
    build_subprocess_environment,
)
from .registry import AgentRegistry, build_default_registry
from .supervisor import ProcessResult, SubprocessSupervisor
from .transcript import agent_transcript_document

__all__ = [
    "AgentRegistry",
    "DOCKER_RUNTIME_ENVIRONMENT_KEYS",
    "GPU_RUNTIME_ENVIRONMENT_KEYS",
    "HF_CREDENTIAL_ENVIRONMENT_KEYS",
    "HF_RUNTIME_ENVIRONMENT_KEYS",
    "ProcessResult",
    "SubprocessSupervisor",
    "build_default_registry",
    "build_subprocess_environment",
    "agent_transcript_document",
]
