"""Read-only delivery verification capability over scoped bundle paths."""

from __future__ import annotations

from pathlib import Path

from apex.core import ContractError
from apex.delivery import (
    detect_bundle_kind,
    load_and_verify_e2e_bundle,
    load_and_verify_kernel_bundle,
)
from apex.ports import CapabilityRequest, CapabilityResult

from .scope import CapabilityScope


class BundleVerifyHandler:
    """Run the official static loader without applying or rebuilding a bundle."""

    def __init__(self, scope: CapabilityScope) -> None:
        self._scope = scope

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        path = self._resolve(str(request.arguments["bundle_path"]))
        kind = detect_bundle_kind(path)
        if kind == "kernel":
            loaded = load_and_verify_kernel_bundle(path)
            verification = {
                "kind": kind,
                "digest": loaded.digest,
                "verified": True,
                "task_id": loaded.task_id,
                "changed_files": list(loaded.changed_files),
            }
        else:
            loaded = load_and_verify_e2e_bundle(path)
            verification = {
                "kind": kind,
                "digest": loaded.digest,
                "verified": loaded.verified,
                "bundle_id": loaded.bundle_id,
                "source_repositories": [
                    item.repository_id for item in loaded.repositories
                ],
            }
        return CapabilityResult(
            request.capability_id,
            {"verification": verification},
            reward_eligible=False,
        )

    def _resolve(self, relative: str) -> Path:
        matches: list[Path] = []
        for reader in (self._scope.read_results, self._scope.read_workspace):
            try:
                matches.append(reader(relative))
            except ContractError as error:
                if error.reason_code != "unsafe_capability_path":
                    raise
        unique = {item for item in matches}
        if len(unique) != 1:
            raise ContractError(
                "Bundle path is missing or ambiguous within capability scope",
                "unsafe_capability_path",
            )
        return unique.pop()


__all__ = ["BundleVerifyHandler"]
