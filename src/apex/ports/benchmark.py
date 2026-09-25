"""Measurement boundary implemented only by the pinned Magpie adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol

from apex.core import sha256_bytes, sha256_json


class BenchmarkPass(str, Enum):
    """Profiler-off measurements and profiler-on diagnostics never mix."""

    MEASUREMENT = "measurement"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True, slots=True)
class BenchmarkRequest:
    run_id: str
    config_path: Path
    output_dir: Path
    pass_type: BenchmarkPass
    timeout_seconds: int = 5400
    environment: Mapping[str, str] = field(default_factory=dict)
    gpu_lease: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    run_id: str
    pass_type: BenchmarkPass
    succeeded: bool
    report_path: Path | None
    workspace_path: Path
    metrics: Mapping[str, float | int | str | None]
    artifact_paths: tuple[Path, ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RayExecutionContract:
    """Resolved per-run Ray address and shared-storage contract."""

    cluster_address: str
    shared_storage_path: Path
    ray_config_sha256: str
    multi_node: bool
    num_nodes: int
    total_num_gpus: int
    gpus_per_node: int

    def __post_init__(self) -> None:
        integers = (self.num_nodes, self.total_num_gpus, self.gpus_per_node)
        valid_digest = len(self.ray_config_sha256) == 64 and all(
            character in "0123456789abcdef" for character in self.ray_config_sha256
        )
        if (
            not self.cluster_address
            or len(self.cluster_address) > 2048
            or any(character.isspace() for character in self.cluster_address)
            or not self.shared_storage_path.is_absolute()
            or self.shared_storage_path == Path("/")
            or not valid_digest
            or not isinstance(self.multi_node, bool)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in integers
            )
            or (not self.multi_node and self.num_nodes != 1)
        ):
            raise ValueError("Ray execution contract is invalid")

    @property
    def results_path(self) -> Path:
        return self.shared_storage_path / "results"

    @property
    def address_sha256(self) -> str:
        return sha256_bytes(self.cluster_address.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class MagpieReportLocation:
    """One observer-owned local report locator or a fail-closed reason."""

    path: Path | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MagpieFormalMeasurementSupport:
    """Whether an observer can prove quality plus normal measurement."""

    available: bool
    reason_code: str | None
    evaluator_execution_mode: str | None
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        complete = (
            bool(self.evaluator_execution_mode)
            and self.reason_code is None
            and not self.blockers
        )
        if self.available is not complete:
            raise ValueError("Formal Magpie measurement support is inconsistent")
        if not self.available and (
            not self.reason_code
            or not self.blockers
            or any(not blocker for blocker in self.blockers)
        ):
            raise ValueError("Unavailable Magpie measurement support lacks blockers")


@dataclass(frozen=True, slots=True)
class RayArtifactClaim:
    """Trusted node claim for one immutable shared-workspace file."""

    role: str
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class RayNodeEvidenceBinding:
    """Exact Ray identity presented to an independent node-side authority."""

    run_id: str
    pass_type: BenchmarkPass
    config_sha256: str
    benchmark_argv_sha256: str
    gpu_lease_sha256: str
    ray_contract: RayExecutionContract
    cluster_identity_sha256: str
    job: Mapping[str, Any]
    driver_process: Mapping[str, Any]
    task: Mapping[str, Any]

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "run_id": self.run_id,
                "pass_type": self.pass_type.value,
                "config_sha256": self.config_sha256,
                "benchmark_argv_sha256": self.benchmark_argv_sha256,
                "gpu_lease_sha256": self.gpu_lease_sha256,
                "ray_config_sha256": self.ray_contract.ray_config_sha256,
                "ray_address_sha256": self.ray_contract.address_sha256,
                "cluster_identity_sha256": self.cluster_identity_sha256,
                "job": dict(self.job),
                "driver_process": dict(self.driver_process),
                "task": dict(self.task),
            }
        )


@dataclass(frozen=True, slots=True)
class RayNodeEvidenceReceipt:
    """Evidence returned by an injected node-side procfs/dependency/KFD authority."""

    schema: str
    authority_sha256: str
    binding_sha256: str
    magpie_task_id: str
    workspace_path: Path
    artifacts: tuple[RayArtifactClaim, ...]
    node_receipts: tuple[Mapping[str, Any], ...]
    dependencies: Mapping[str, Any]
    gpu_devices: tuple[Mapping[str, Any], ...]
    gpu_processes: tuple[Mapping[str, Any], ...]
    runtime: Mapping[str, Any]


class RayNodeEvidenceAuthority(Protocol):
    """Independent authority that observes worker nodes before and during a run."""

    @property
    def is_available(self) -> bool: ...

    def prepare(
        self,
        request: MagpieAttestationRequest,
        *,
        ray_contract: RayExecutionContract,
        cluster_identity_sha256: str,
    ) -> object: ...

    def complete(
        self,
        session: object,
        *,
        binding: RayNodeEvidenceBinding,
    ) -> RayNodeEvidenceReceipt: ...

    def abort(self, session: object, *, reason: str) -> None: ...


@dataclass(frozen=True, slots=True)
class MagpieAttestationRequest:
    """Inputs exposed to a trusted observer before Magpie is started."""

    run_id: str
    pass_type: BenchmarkPass
    config_path: Path
    run_root: Path
    benchmark_argv: tuple[str, ...]
    config_sha256: str
    execution_mode: str
    lifecycle: str
    requested_image: str | None
    gpu_lease: Mapping[str, object] | None
    ray_contract: RayExecutionContract | None = None
    evaluator_policy: Mapping[str, object] | None = None
    evaluator_policy_lock: Mapping[str, object] | None = None
    lm_eval_runtime: Mapping[str, object] | None = None
    model: str | None = None
    evaluator_endpoint_port: int | None = None
    evaluator_concurrent_requests: int | None = None
    evaluator_timeout_seconds: int | None = None


class MagpieExecutionAttestor(Protocol):
    """Observe one unchanged Magpie execution and mint Apex evidence."""

    @property
    def is_available(self) -> bool: ...

    def formal_measurement_support(
        self, execution_mode: str, lifecycle: str
    ) -> MagpieFormalMeasurementSupport: ...

    def prepare(self, request: MagpieAttestationRequest) -> object: ...

    def launch_argv(self, session: object) -> tuple[str, ...]: ...

    def abort(self, session: object, *, reason: str) -> None: ...

    def locate_report(self, session: object) -> MagpieReportLocation: ...

    def complete(
        self,
        session: object,
        *,
        report_path: Path | None,
        command_exit_code: int | None,
        timed_out: bool,
    ) -> Path | None: ...


class BenchmarkPort(Protocol):
    def run(self, request: BenchmarkRequest) -> BenchmarkResult: ...
