"""Deterministic Apex execution identity fixtures."""

from __future__ import annotations

from apex.core import canonical_json_bytes, sha256_bytes
from apex.runtime import ApexExecutionIdentity


def execution_identity(
    *,
    apex_tree: str = "a" * 40,
    dependency_lock_sha256: str | None = None,
    dirty_paths: tuple[str, ...] = (),
) -> ApexExecutionIdentity:
    payload = {
        "schema": "apex.execution-identity/v1",
        "repository": {
            "root_sha256": "b" * 64,
            "status": "resolved",
            "remote": "https://github.com/AMD-AGI/Apex",
            "commit": "c" * 40,
            "tree": apex_tree,
            "dirty_paths": list(dirty_paths),
            "unavailable_reason": None,
        },
        "package": {
            "distribution": "amd-apex-optimizer",
            "version": "0.1.0",
            "source_manifest_sha256": "d" * 64,
            "file_count": 10,
        },
        "dependency_lock_sha256": dependency_lock_sha256,
    }
    canonical = canonical_json_bytes(payload)
    return ApexExecutionIdentity(canonical, sha256_bytes(canonical))


__all__ = ["execution_identity"]
