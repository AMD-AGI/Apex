"""Authoritative clean-process HSA GPU-agent enumeration."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from apex.core import ContractError, sha256_file, sha256_json
from apex.execution import ProcessResult, SubprocessSupervisor, build_subprocess_environment


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_GPU_UUID = re.compile(r"^GPU-[0-9a-f]{16}$")
_VISIBILITY_NAMES = frozenset(
    {
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    }
)


@dataclass(frozen=True, slots=True)
class HsaGpuIdentity:
    """One GPU in an unfiltered HSA runtime enumeration."""

    hsa_gpu_index: int
    node_id: int
    generic_node_id: int
    bdf_id: int
    domain: int
    unique_id: str

    def __post_init__(self) -> None:
        if (
            self.hsa_gpu_index < 0
            or self.node_id < 0
            or self.generic_node_id < 0
            or self.bdf_id < 0
            or self.bdf_id >= 2**32
            or self.domain < 0
            or self.domain >= 2**32
            or not _GPU_UUID.fullmatch(self.unique_id)
        ):
            raise ContractError(
                "The HSA GPU agent identity is invalid",
                "gpu_hsa_inventory_invalid",
            )
        if self.node_id != self.generic_node_id:
            raise ContractError(
                "HSA node identity APIs disagree",
                "gpu_hsa_inventory_invalid",
            )

    @property
    def pci_id(self) -> int:
        return (self.domain << 32) | self.bdf_id


@dataclass(frozen=True, slots=True)
class HsaInventoryEvidence:
    """Hash-bound helper and HSA library evidence for one clean enumeration."""

    schema_version: int
    policy_id: str
    helper_path: str
    helper_sha256: str
    library_path: str
    library_sha256: str
    devices: tuple[HsaGpuIdentity, ...]

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.policy_id != "clean_unfiltered_hsa_gpu_inventory_v1"
            or not Path(self.helper_path).is_absolute()
            or not Path(self.library_path).is_absolute()
            or not _DIGEST.fullmatch(self.helper_sha256)
            or not _DIGEST.fullmatch(self.library_sha256)
            or not self.devices
            or tuple(item.hsa_gpu_index for item in self.devices)
            != tuple(range(len(self.devices)))
        ):
            raise ContractError(
                "Clean HSA inventory evidence is incomplete",
                "gpu_hsa_inventory_invalid",
            )
        for field in ("node_id", "unique_id"):
            if len({getattr(item, field) for item in self.devices}) != len(self.devices):
                raise ContractError(
                    f"Clean HSA inventory contains duplicate {field} values",
                    "gpu_hsa_inventory_invalid",
                )
        if len({(item.domain, item.bdf_id) for item in self.devices}) != len(
            self.devices
        ):
            raise ContractError(
                "Clean HSA inventory contains duplicate PCI identities",
                "gpu_hsa_inventory_invalid",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "helper_path": self.helper_path,
            "helper_sha256": self.helper_sha256,
            "library_path": self.library_path,
            "library_sha256": self.library_sha256,
            "devices": [asdict(device) for device in self.devices],
        }

    @property
    def digest(self) -> str:
        return sha256_json(self.to_dict())


class HsaInventoryProvider(Protocol):
    def collect(self) -> HsaInventoryEvidence: ...


class CleanHsaInventoryProvider:
    """Run the bundled ctypes helper with no inherited GPU visibility mask."""

    def __init__(
        self,
        *,
        library_path: Path | None = None,
        helper_path: Path | None = None,
        python_path: Path | None = None,
        supervisor: SubprocessSupervisor | None = None,
    ) -> None:
        self._library_path = library_path
        self._helper_path = helper_path
        self._python_path = python_path
        self._supervisor = supervisor or SubprocessSupervisor(max_output_bytes=256 * 1024)

    def collect(self) -> HsaInventoryEvidence:
        library = _resolve_hsa_library(self._library_path)
        helper = _resolve_file(
            self._helper_path or Path(__file__).with_name("_hsa_inventory_helper.py"),
            reason="gpu_hsa_helper_unavailable",
        )
        python = _resolve_file(
            self._python_path or Path(sys.executable),
            reason="gpu_hsa_helper_unavailable",
        )
        helper_sha256 = sha256_file(helper)
        library_sha256 = sha256_file(library)
        environment = build_subprocess_environment()
        if _VISIBILITY_NAMES.intersection(environment):
            raise ContractError(
                "The clean HSA helper environment contains a GPU visibility mask",
                "gpu_hsa_helper_unavailable",
            )
        result = self._supervisor.run(
            (str(python), "-I", str(helper), str(library)),
            cwd=helper.parent,
            environment=environment,
            timeout_seconds=10,
        )
        devices = _parse_helper_result(result)
        if (
            sha256_file(helper) != helper_sha256
            or sha256_file(library) != library_sha256
        ):
            raise ContractError(
                "The HSA helper or library changed during enumeration",
                "gpu_hsa_helper_changed",
            )
        return HsaInventoryEvidence(
            schema_version=1,
            policy_id="clean_unfiltered_hsa_gpu_inventory_v1",
            helper_path=str(helper),
            helper_sha256=helper_sha256,
            library_path=str(library),
            library_sha256=library_sha256,
            devices=devices,
        )


def _parse_helper_result(result: ProcessResult) -> tuple[HsaGpuIdentity, ...]:
    if (
        result.exit_code != 0
        or result.timed_out
        or result.stdout_truncated
        or result.stderr_truncated
        or not result.cleanup_succeeded
    ):
        raise ContractError(
            "The clean HSA inventory helper failed",
            "gpu_hsa_helper_failed",
            {
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "stderr": result.stderr[-4096:],
            },
        )
    try:
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "devices"}:
            raise ValueError("invalid top-level keys")
        if payload["schema_version"] != 1 or not isinstance(payload["devices"], list):
            raise ValueError("invalid schema")
        devices = tuple(_parse_device(item) for item in payload["devices"])
    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
        raise ContractError(
            "The clean HSA inventory helper returned malformed output",
            "gpu_hsa_helper_failed",
        ) from error
    return devices


def _parse_device(raw: Any) -> HsaGpuIdentity:
    if not isinstance(raw, dict) or set(raw) != {
        "hsa_gpu_index",
        "node_id",
        "generic_node_id",
        "bdf_id",
        "domain",
        "unique_id",
    }:
        raise ValueError("invalid HSA device fields")
    if any(isinstance(raw[name], bool) for name in raw if name != "unique_id"):
        raise ValueError("boolean HSA identity")
    return HsaGpuIdentity(**raw)


def _resolve_hsa_library(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit is not None else [
        Path("/opt/rocm/lib/libhsa-runtime64.so.1"),
        Path("/opt/rocm/lib/libhsa-runtime64.so"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    raise ContractError(
        "A concrete HSA runtime library could not be resolved",
        "gpu_hsa_helper_unavailable",
    )


def _resolve_file(path: Path, *, reason: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ContractError("A trusted helper file is unavailable", reason) from error
    if not resolved.is_file():
        raise ContractError("A trusted helper file is unavailable", reason)
    return resolved


__all__ = [
    "CleanHsaInventoryProvider",
    "HsaGpuIdentity",
    "HsaInventoryEvidence",
    "HsaInventoryProvider",
]
