"""Explicit agent backend registry with Codex as the product default."""

from __future__ import annotations

from typing import Iterable

from apex.core import AgentBackendName, ConfigurationError
from apex.ports import AgentBackend

from .claude import ClaudeBackend
from .codex import CodexBackend
from .cursor import CursorBackend


class AgentRegistry:
    def __init__(self, backends: Iterable[AgentBackend], *, default: AgentBackendName) -> None:
        self._backends = {backend.name: backend for backend in backends}
        if default not in self._backends:
            raise ConfigurationError("Default agent backend is not registered", "backend_not_registered")
        self.default = default

    def get(self, name: AgentBackendName | None = None) -> AgentBackend:
        selected = name or self.default
        try:
            return self._backends[selected]
        except KeyError as error:
            raise ConfigurationError(
                f"Agent backend is not registered: {selected.value}",
                "backend_not_registered",
            ) from error

    @property
    def names(self) -> tuple[AgentBackendName, ...]:
        return tuple(sorted(self._backends, key=lambda item: item.value))


def build_default_registry() -> AgentRegistry:
    return AgentRegistry(
        [CodexBackend(), ClaudeBackend(), CursorBackend()],
        default=AgentBackendName.CODEX,
    )
