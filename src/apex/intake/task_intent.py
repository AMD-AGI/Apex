"""Natural-language request accepted by the interactive kernel entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from apex.core import ContractError


@dataclass(frozen=True, slots=True)
class NaturalLanguageRequest:
    """A human optimization request plus the workspace it refers to."""

    text: str
    workspace: Path
    results_dir: Path

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ContractError("Optimization request is empty", "empty_request")
        if not self.workspace.is_absolute():
            raise ContractError("workspace must be absolute", "workspace_not_absolute")
        if not self.results_dir.is_absolute():
            raise ContractError("results_dir must be absolute", "results_not_absolute")
