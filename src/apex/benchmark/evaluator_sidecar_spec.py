"""Exact Docker create contract for the network-isolated evaluator sidecar."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from apex.core import ConfigurationError, sha256_file, sha256_json

from .evaluator_preparation import PreparedLmEvalExecution


_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_REPO_DIGEST = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
_CONTAINER_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}")


@dataclass(frozen=True, slots=True)
class EvaluatorSidecarMount:
    """One exact host bind exposed to the sidecar."""

    role: str
    source: Path
    destination: str
    read_only: bool

    def __post_init__(self) -> None:
        if (
            not self.role
            or not self.source.is_absolute()
            or self.source.is_symlink()
            or not self.source.exists()
            or "," in str(self.source)
            or not self.destination.startswith("/")
            or "," in self.destination
        ):
            raise ValueError("Evaluator sidecar mount is invalid")

    @property
    def docker_value(self) -> str:
        mode = ",readonly" if self.read_only else ""
        return (
            f"type=bind,source={self.source},target={self.destination}{mode}"
        )


@dataclass(frozen=True, slots=True)
class EvaluatorSidecarDockerSpec:
    """Frozen create argv and expected daemon-visible sidecar identity."""

    container_name: str
    image_repo_digest: str
    image_id: str
    uid: int
    gid: int
    cwd: str
    environment: Mapping[str, str]
    mounts: tuple[EvaluatorSidecarMount, ...]
    sidecar_argv: tuple[str, ...]
    contract_sha256: str
    input_projection_sha256: str

    def __post_init__(self) -> None:
        if (
            not _CONTAINER_NAME.fullmatch(self.container_name)
            or not _REPO_DIGEST.fullmatch(self.image_repo_digest)
            or not _IMAGE_ID.fullmatch(self.image_id)
            or min(self.uid, self.gid) < 0
            or self.cwd != "/authority"
            or not self.environment
            or not self.sidecar_argv
            or self.sidecar_argv[0] != "python3"
            or len({item.role for item in self.mounts}) != len(self.mounts)
            or len({item.destination for item in self.mounts}) != len(self.mounts)
            or len(self.contract_sha256) != 64
            or len(self.input_projection_sha256) != 64
        ):
            raise ValueError("Evaluator sidecar Docker spec is invalid")

    @property
    def create_argv(self) -> tuple[str, ...]:
        argv = [
            "docker", "container", "create",
            "--name", self.container_name,
            "--network", "none",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            "--pids-limit", "512",
            "--user", f"{self.uid}:{self.gid}",
            "--workdir", self.cwd,
            "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=1073741824,mode=1777",
            "--label", f"apex.evaluator.contract={self.contract_sha256}",
        ]
        for mount in self.mounts:
            argv.extend(("--mount", mount.docker_value))
        for key, value in sorted(self.environment.items()):
            argv.extend(("--env", f"{key}={value}"))
        argv.extend(("--entrypoint", "python3", self.image_repo_digest))
        argv.extend(self.sidecar_argv[1:])
        return tuple(argv)

    @property
    def sha256(self) -> str:
        return sha256_json(
            {
                "create_argv": list(self.create_argv),
                "expected_image_id": self.image_id,
                "mount_roles": [item.role for item in self.mounts],
                "input_projection_sha256": self.input_projection_sha256,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "apex.evaluator-sidecar-docker-spec/v1",
            "container_name": self.container_name,
            "image_repo_digest": self.image_repo_digest,
            "image_id": self.image_id,
            "uid": self.uid,
            "gid": self.gid,
            "cwd": self.cwd,
            "environment": dict(sorted(self.environment.items())),
            "mounts": [
                {
                    "role": item.role,
                    "source": str(item.source),
                    "destination": item.destination,
                    "read_only": item.read_only,
                }
                for item in self.mounts
            ],
            "sidecar_argv": list(self.sidecar_argv),
            "contract_sha256": self.contract_sha256,
            "input_projection_sha256": self.input_projection_sha256,
            "create_argv_sha256": sha256_json(list(self.create_argv)),
            "spec_sha256": self.sha256,
        }


def build_evaluator_sidecar_spec(
    prepared: PreparedLmEvalExecution,
    broker_root: Path,
) -> EvaluatorSidecarDockerSpec:
    """Build a no-network/no-GPU create request from frozen authority paths."""

    launcher = prepared.input_projection.launcher_path.resolve(strict=True)
    if sha256_file(launcher) != prepared.contract.launcher_sha256:
        raise _invalid("Evaluator sidecar launcher changed after contract freeze")
    mounts = (
        EvaluatorSidecarMount("authority", prepared.sidecar_root, "/authority", False),
        EvaluatorSidecarMount("broker", broker_root, "/evaluator/broker", True),
        EvaluatorSidecarMount("contract", prepared.contract_path, "/evaluator/contract/execution_contract.json", True),
        EvaluatorSidecarMount("dataset", prepared.dataset_mount, "/evaluator/dataset", True),
        EvaluatorSidecarMount("launcher", launcher, "/evaluator/launcher/evaluator_sidecar_entry.py", True),
        EvaluatorSidecarMount("runtime", prepared.runtime_mount, "/evaluator/runtime", True),
        EvaluatorSidecarMount("task", prepared.task_mount, "/evaluator/task", True),
    )
    name = f"apex-lm-eval-{prepared.contract.run_id}-{prepared.contract.sha256[:12]}"
    return EvaluatorSidecarDockerSpec(
        container_name=name,
        image_repo_digest=prepared.contract.image_repo_digest,
        image_id=prepared.contract.image_id,
        uid=os.getuid(),
        gid=os.getgid(),
        cwd="/authority",
        environment=prepared.contract.environment,
        mounts=mounts,
        sidecar_argv=prepared.contract.sidecar_argv,
        contract_sha256=prepared.contract.sha256,
        input_projection_sha256=prepared.input_projection.receipt_sha256,
    )


def _invalid(message: str) -> ConfigurationError:
    return ConfigurationError(message, "evaluator_sidecar_spec_invalid")


__all__ = [
    "EvaluatorSidecarDockerSpec",
    "EvaluatorSidecarMount",
    "build_evaluator_sidecar_spec",
]
