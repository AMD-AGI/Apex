"""Resolve and freeze the evaluator-owned contract for a formal kernel run."""

from __future__ import annotations

from apex.evaluation import (
    EvaluationContractAuthorizer,
    EvaluationContractFreezer,
    EvaluationContractReceipt,
)
from apex.intake import ResolvedTaskSpec, TaskResolver, TaskSpec
from apex.ports import WorkspaceRepositoryIdentityPort
from apex.runtime import WorkspaceGitIdentityResolver


class KernelFormalContractResolver:
    """Turn discovered task inputs into one immutable authority-bound receipt."""

    def __init__(
        self,
        authorizer: EvaluationContractAuthorizer | None,
        repository_identities: WorkspaceRepositoryIdentityPort | None,
    ) -> None:
        self._freezer = EvaluationContractFreezer(authorizer)
        self._repositories = repository_identities or WorkspaceGitIdentityResolver()

    def freeze(
        self, task: TaskSpec
    ) -> tuple[ResolvedTaskSpec, EvaluationContractReceipt]:
        resolved = TaskResolver().resolve(task)
        repository = self._repositories.inspect(resolved.workspace)
        return resolved, self._freezer.freeze(resolved, repository)


__all__ = ["KernelFormalContractResolver"]
