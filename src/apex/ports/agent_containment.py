"""Immutable evidence contract for authoritative agent PID containment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from apex.core import ContractError


AGENT_PROCESS_CONTAINMENT_POLICY = "private_pid_namespace_init_pidfd_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class AgentProcessContainmentReceipt:
    """Kernel-backed proof that an agent's PID namespace is empty."""

    policy_id: str
    launcher_path: str
    launcher_sha256: str
    namespace_init_host_pid: int
    namespace_init_starttime: int
    namespace_init_inner_pid: int
    pid_namespace_inode: int
    mount_namespace_inode: int
    ipc_namespace_inode: int
    user_namespace_inode: int
    private_procfs_verified: bool
    pidfd_opened: bool
    termination_reason: str
    teardown_mode: str
    pidfd_sigkill_sent: bool
    namespace_init_exit_verified: bool
    wrapper_exit_verified: bool
    wrapper_force_killed: bool
    terminal_status_verified: bool
    terminal_status_absent_after_sigkill: bool
    status_eof_verified: bool
    namespace_membership_scan_complete: bool
    live_namespace_members_after: tuple[int, ...]

    def __post_init__(self) -> None:
        positive = (
            self.namespace_init_host_pid,
            self.namespace_init_starttime,
            self.namespace_init_inner_pid,
            self.pid_namespace_inode,
            self.mount_namespace_inode,
            self.ipc_namespace_inode,
            self.user_namespace_inode,
        )
        if (
            self.policy_id != AGENT_PROCESS_CONTAINMENT_POLICY
            or not Path(self.launcher_path).is_absolute()
            or not _SHA256.fullmatch(self.launcher_sha256)
            or any(type(value) is not int or value <= 0 for value in positive)
            or self.namespace_init_inner_pid != 1
        ):
            raise ContractError(
                "Agent process containment identity is invalid",
                "invalid_agent_process_containment",
            )
        if self.teardown_mode not in {"natural_exit", "pidfd_sigkill"} or (
            self.pidfd_sigkill_sent != (self.teardown_mode == "pidfd_sigkill")
        ):
            raise ContractError(
                "Agent process containment teardown is invalid",
                "invalid_agent_process_containment",
            )
        self._validate_evidence()

    def _validate_evidence(self) -> None:
        booleans = (
            self.private_procfs_verified,
            self.pidfd_opened,
            self.pidfd_sigkill_sent,
            self.namespace_init_exit_verified,
            self.wrapper_exit_verified,
            self.wrapper_force_killed,
            self.terminal_status_verified,
            self.terminal_status_absent_after_sigkill,
            self.status_eof_verified,
            self.namespace_membership_scan_complete,
        )
        invalid_members = any(
            type(pid) is not int or pid <= 0
            for pid in self.live_namespace_members_after
        )
        if (
            any(type(value) is not bool for value in booleans)
            or not isinstance(self.termination_reason, str)
            or not self.termination_reason
            or (self.terminal_status_absent_after_sigkill and not self.pidfd_sigkill_sent)
            or (
                self.terminal_status_verified
                and self.terminal_status_absent_after_sigkill
            )
            or invalid_members
        ):
            raise ContractError(
                "Agent process containment evidence is invalid",
                "invalid_agent_process_containment",
            )

    @property
    def namespace_empty_verified(self) -> bool:
        """Whether independent kernel and wrapper evidence proves quiescence."""

        terminal = self.terminal_status_verified or (
            self.pidfd_sigkill_sent and self.terminal_status_absent_after_sigkill
        )
        return (
            self.pidfd_opened
            and self.private_procfs_verified
            and self.namespace_init_exit_verified
            and self.wrapper_exit_verified
            and not self.wrapper_force_killed
            and terminal
            and self.status_eof_verified
            and self.namespace_membership_scan_complete
            and not self.live_namespace_members_after
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "apex.agent-process-containment/v1",
            "policy_id": self.policy_id,
            "launcher_path": self.launcher_path,
            "launcher_sha256": self.launcher_sha256,
            "namespace_init_host_pid": self.namespace_init_host_pid,
            "namespace_init_starttime": self.namespace_init_starttime,
            "namespace_init_inner_pid": self.namespace_init_inner_pid,
            "pid_namespace_inode": self.pid_namespace_inode,
            "mount_namespace_inode": self.mount_namespace_inode,
            "ipc_namespace_inode": self.ipc_namespace_inode,
            "user_namespace_inode": self.user_namespace_inode,
            "private_procfs_verified": self.private_procfs_verified,
            "pidfd_opened": self.pidfd_opened,
            "termination_reason": self.termination_reason,
            "teardown_mode": self.teardown_mode,
            "pidfd_sigkill_sent": self.pidfd_sigkill_sent,
            "namespace_init_exit_verified": self.namespace_init_exit_verified,
            "wrapper_exit_verified": self.wrapper_exit_verified,
            "wrapper_force_killed": self.wrapper_force_killed,
            "terminal_status_verified": self.terminal_status_verified,
            "terminal_status_absent_after_sigkill": (
                self.terminal_status_absent_after_sigkill
            ),
            "status_eof_verified": self.status_eof_verified,
            "namespace_membership_scan_complete": self.namespace_membership_scan_complete,
            "live_namespace_members_after": list(self.live_namespace_members_after),
            "namespace_empty_verified": self.namespace_empty_verified,
        }


__all__ = ["AGENT_PROCESS_CONTAINMENT_POLICY", "AgentProcessContainmentReceipt"]
