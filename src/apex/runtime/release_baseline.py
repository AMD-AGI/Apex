"""Remote-ref and source-identity policy for campaign and release baselines."""

from __future__ import annotations

from typing import Any, Mapping

from .release_evidence import BaselineAuditEvidence
from .repositories import canonical_repository


def assess_campaign_apex_baseline(
    value: BaselineAuditEvidence | None,
    expected: Mapping[str, Any],
    blockers: set[str],
) -> None:
    """Allow reviewed official main or codex refs to start qualification."""

    if value is None:
        blockers.add("apex_baseline_missing")
        return
    _assess_identity("apex", value, expected, blockers)
    if not _is_campaign_ref(value.branch):
        blockers.add("apex_campaign_branch_untrusted")


def assess_release_baseline(
    name: str,
    value: BaselineAuditEvidence | None,
    expected: Mapping[str, Any],
    blockers: set[str],
) -> None:
    """Require exact remote main for final release authorization."""

    if value is None:
        blockers.add(f"{name}_baseline_missing")
        return
    _assess_identity(name, value, expected, blockers)
    if not _is_release_main_ref(value.branch):
        blockers.add(f"{name}_release_branch_not_main")


def repository_key(value: object) -> str:
    """Normalize either a full repository URL or an already canonical key."""

    text = str(value).strip().rstrip("/")
    if (
        "://" not in text
        and ":" not in text
        and not text.startswith(("/", "."))
        and "/" in text
    ):
        return text.lower().removesuffix(".git")
    return canonical_repository(text)


def _assess_identity(
    name: str,
    value: BaselineAuditEvidence,
    expected: Mapping[str, Any],
    blockers: set[str],
) -> None:
    if value.component != name:
        blockers.add(f"{name}_baseline_component_mismatch")
    if repository_key(value.repository) != repository_key(expected["repository"]):
        blockers.add(f"{name}_baseline_repository_mismatch")
    expected_tree = expected.get("tree", expected.get("repository_tree"))
    if value.commit != expected["commit"] or value.tree != expected_tree:
        blockers.add(f"{name}_baseline_source_mismatch")
    if value.remote_tip != value.commit:
        blockers.add(f"{name}_baseline_not_remote_tip")
    if not value.fetched:
        blockers.add(f"{name}_fetch_unverified")
    if not value.ancestry_reviewed:
        blockers.add(f"{name}_ancestry_unreviewed")
    if not value.clean:
        blockers.add(f"{name}_baseline_dirty")


def _is_campaign_ref(value: str) -> bool:
    parts = _remote_tracking_ref_parts(value)
    return (len(parts) == 2 and parts[1] == "main") or (
        len(parts) >= 3 and parts[1] == "codex"
    )


def _is_release_main_ref(value: str) -> bool:
    parts = _remote_tracking_ref_parts(value)
    return len(parts) == 2 and parts[1] == "main"


def _remote_tracking_ref_parts(value: str) -> tuple[str, ...]:
    branch = value.removeprefix("refs/remotes/")
    if branch.startswith("refs/") or "\\" in branch or "@{" in branch:
        return ()
    parts = tuple(branch.split("/"))
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        return ()
    return parts

