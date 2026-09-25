"""Ordinary process-group cleanup, without a namespace or sandbox claim."""

from dataclasses import asdict, dataclass

PROCESS_GROUP_POLICY = "process_group_v1"


@dataclass(frozen=True, slots=True)
class ProcessGroupReceipt:
    process_group: int
    termination_reason: str
    cleanup_verified: bool
    policy_id: str = PROCESS_GROUP_POLICY

    def __post_init__(self) -> None:
        if (type(self.process_group) is not int or self.process_group <= 0
                or type(self.cleanup_verified) is not bool
                or self.policy_id != PROCESS_GROUP_POLICY
                or self.termination_reason not in {"natural_exit", "timeout", "stdout_budget_boundary"}):
            raise ValueError("invalid process-group completion")

    def to_dict(self) -> dict[str, object]:
        return {"schema": "apex.process-group/v1", **asdict(self)}
