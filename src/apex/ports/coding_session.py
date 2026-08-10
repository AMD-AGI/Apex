"""Native coding-session launch boundary, separate from formal campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from apex.core import AgentBackendName, ContractError


class CodingSessionOutput(str, Enum):
    INTERACTIVE = "interactive"
    TEXT = "text"
    JSONL = "jsonl"


class KernelEnhancement(str, Enum):
    AUTO = "auto"
    PLAIN = "plain"
    KERNEL = "kernel"


@dataclass(frozen=True, slots=True)
class CodingSessionRequest:
    """User-owned native session; it grants no evaluator authority."""

    workspace: Path
    results_dir: Path | None = None
    backend: AgentBackendName = AgentBackendName.CODEX
    prompt: str | None = None
    model: str | None = None
    effort: str | None = None
    output: CodingSessionOutput = CodingSessionOutput.INTERACTIVE
    enhancement: KernelEnhancement = KernelEnhancement.AUTO
    resume_session: str | None = None
    resume_latest: bool = False

    def __post_init__(self) -> None:
        if not self.workspace.is_absolute():
            raise ContractError("Session workspace must be absolute", "workspace_not_absolute")
        if self.results_dir is not None and not self.results_dir.is_absolute():
            raise ContractError("Session results must be absolute", "results_not_absolute")
        if self.prompt is not None and not self.prompt.strip():
            raise ContractError("Session prompt may not be empty", "empty_request")
        if self.output is not CodingSessionOutput.INTERACTIVE and self.prompt is None:
            raise ContractError("Headless session requires a prompt", "session_prompt_required")
        if self.resume_session is not None and not self.resume_session.strip():
            raise ContractError("Resume session ID may not be empty", "invalid_session_resume")
        if self.resume_session is not None and self.resume_latest:
            raise ContractError("Choose one resume target", "invalid_session_resume")
        if self.model is not None and not self.model.strip():
            raise ContractError("Session model may not be empty", "invalid_agent_options")
        if self.effort is not None and not self.effort.strip():
            raise ContractError("Session effort may not be empty", "invalid_agent_options")


class CodingSessionLauncher(Protocol):
    def launch(self, request: CodingSessionRequest) -> int: ...


__all__ = [
    "CodingSessionLauncher",
    "CodingSessionOutput",
    "CodingSessionRequest",
    "KernelEnhancement",
]
