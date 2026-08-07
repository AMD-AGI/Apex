"""Fixed-argv Docker operations for runtime-only Python source overlays."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol

from apex.core import ContractError, IntegrityError, sha256_file
from apex.execution import SubprocessSupervisor


_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_MARKER = "__APEX_PROBE_V1__"
_PACKAGE_PROBE = r"""
import hashlib, importlib.util, json, os, pathlib, stat, sys
package, relative = sys.argv[1], sys.argv[2]
spec = importlib.util.find_spec(package)
roots = tuple(spec.submodule_search_locations or ()) if spec else ()
if len(roots) != 1:
    raise SystemExit("package_root_unresolved")
root = pathlib.Path(roots[0]).resolve(strict=True)
parts = pathlib.PurePosixPath(relative).parts
target = root.joinpath(*parts)
metadata = target.lstat()
if target.is_symlink() or not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("installed_target_not_regular")
resolved = target.resolve(strict=True)
resolved.relative_to(root)
data = resolved.read_bytes()
print("__APEX_PROBE_V1__" + json.dumps({
    "package_root": str(root), "path": str(resolved),
    "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
    "mode": stat.S_IMODE(metadata.st_mode), "symlink": False,
}, sort_keys=True, separators=(",", ":")))
""".strip()
_READ_PROBE = r"""
import hashlib, json, pathlib, stat, sys
target = pathlib.Path(sys.argv[1])
metadata = target.lstat()
if target.is_symlink() or not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("loaded_target_not_regular")
data = target.read_bytes()
print("__APEX_PROBE_V1__" + json.dumps({
    "path": str(target.resolve(strict=True)),
    "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
    "mode": stat.S_IMODE(metadata.st_mode), "symlink": False,
}, sort_keys=True, separators=(",", ":")))
""".strip()


@dataclass(frozen=True, slots=True)
class ContainerImage:
    reference: str
    image_id: str

    def __post_init__(self) -> None:
        if not _IMAGE_ID.fullmatch(self.image_id):
            raise ContractError("Container image ID is not immutable", "invalid_image_id")


@dataclass(frozen=True, slots=True)
class InstalledPythonTarget:
    package: str
    repo_relative_path: str
    module_relative_path: str
    container_path: str
    sha256: str
    size: int
    mode: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoadedFileReceipt:
    container_path: str
    sha256: str
    size: int
    mode: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BuiltOverlay:
    image: ContainerImage
    dockerfile_sha256: str
    context_source_sha256: str


class ContainerEngine(Protocol):
    def inspect_image(self, reference: str, *, cwd: Path) -> ContainerImage: ...

    def resolve_python_target(
        self,
        image_id: str,
        *,
        library: str,
        repo_relative_path: str,
        cwd: Path,
    ) -> InstalledPythonTarget: ...

    def build_overlay(
        self,
        *,
        parent_image_id: str,
        candidate_source: Path,
        target: InstalledPythonTarget,
        build_root: Path,
        cwd: Path,
    ) -> BuiltOverlay: ...

    def read_file(
        self, image_id: str, *, container_path: str, cwd: Path
    ) -> LoadedFileReceipt: ...


class DockerEngine:
    """Use Docker without a shell, mounts, host-package edits, or mutable parents."""

    def __init__(self, supervisor: SubprocessSupervisor | None = None) -> None:
        self._supervisor = supervisor or SubprocessSupervisor(max_output_bytes=32 * 1024 * 1024)

    def inspect_image(self, reference: str, *, cwd: Path) -> ContainerImage:
        result = self._run(("docker", "image", "inspect", reference), cwd=cwd, timeout=60)
        try:
            values = json.loads(result.stdout)
            image_id = str(values[0]["Id"])
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise IntegrityError("Docker image inspection is invalid", "image_inspection_failed") from error
        return ContainerImage(reference, image_id)

    def resolve_python_target(
        self,
        image_id: str,
        *,
        library: str,
        repo_relative_path: str,
        cwd: Path,
    ) -> InstalledPythonTarget:
        module_relative = _module_relative(library, repo_relative_path)
        payload = self._python_probe(
            image_id,
            _PACKAGE_PROBE,
            (library, module_relative),
            cwd=cwd,
            timeout=120,
        )
        target = _loaded_receipt(payload)
        return InstalledPythonTarget(
            library,
            repo_relative_path,
            module_relative,
            target.container_path,
            target.sha256,
            target.size,
            target.mode,
        )

    def build_overlay(
        self,
        *,
        parent_image_id: str,
        candidate_source: Path,
        target: InstalledPythonTarget,
        build_root: Path,
        cwd: Path,
    ) -> BuiltOverlay:
        ContainerImage(parent_image_id, parent_image_id)
        if not candidate_source.is_file() or candidate_source.is_symlink():
            raise IntegrityError("Candidate source is not a regular file", "invalid_frozen_candidate")
        context = build_root.resolve() / "context"
        if context.exists() or context.is_symlink():
            raise IntegrityError("Docker overlay context already exists", "immutable_delivery_artifact")
        context.mkdir(parents=True)
        copied = context / "candidate.py"
        shutil.copyfile(candidate_source, copied)
        copied.chmod(0o444)
        if sha256_file(copied) != sha256_file(candidate_source):
            raise IntegrityError("Overlay context changed candidate bytes", "candidate_lineage_mismatch")
        dockerfile = context / "Dockerfile"
        dockerfile.write_text(
            f"FROM {parent_image_id}\n"
            f"COPY {json.dumps(['candidate.py', target.container_path])}\n",
            encoding="utf-8",
        )
        iidfile = build_root.resolve() / "derived-image.id"
        result = self._run(
            (
                "docker",
                "build",
                "--no-cache",
                "--pull=false",
                "--network=none",
                "--iidfile",
                str(iidfile),
                "--file",
                str(dockerfile),
                str(context),
            ),
            cwd=cwd,
            timeout=1800,
        )
        if not iidfile.is_file() or iidfile.is_symlink():
            raise IntegrityError("Docker did not emit an image ID", "overlay_build_failed")
        image_id = iidfile.read_text(encoding="utf-8").strip()
        inspected = self.inspect_image(image_id, cwd=cwd)
        if inspected.image_id != image_id:
            raise IntegrityError("Built image inspection changed identity", "image_identity_mismatch")
        return BuiltOverlay(
            inspected,
            sha256_file(dockerfile),
            sha256_file(copied),
        )

    def read_file(
        self, image_id: str, *, container_path: str, cwd: Path
    ) -> LoadedFileReceipt:
        payload = self._python_probe(
            image_id, _READ_PROBE, (container_path,), cwd=cwd, timeout=120
        )
        return _loaded_receipt(payload)

    def _python_probe(
        self,
        image_id: str,
        script: str,
        args: tuple[str, ...],
        *,
        cwd: Path,
        timeout: int,
    ) -> Mapping[str, Any]:
        ContainerImage(image_id, image_id)
        result = self._run(
            (
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--entrypoint",
                "python3",
                image_id,
                "-I",
                "-c",
                script,
                *args,
            ),
            cwd=cwd,
            timeout=timeout,
        )
        marked = [line[len(_MARKER) :] for line in result.stdout.splitlines() if line.startswith(_MARKER)]
        if len(marked) != 1:
            raise IntegrityError("Container byte probe emitted no unique receipt", "loaded_byte_probe_failed")
        try:
            payload = json.loads(marked[0])
        except json.JSONDecodeError as error:
            raise IntegrityError("Container byte receipt is invalid", "loaded_byte_probe_failed") from error
        if not isinstance(payload, Mapping):
            raise IntegrityError("Container byte receipt is not an object", "loaded_byte_probe_failed")
        return payload

    def _run(self, argv: tuple[str, ...], *, cwd: Path, timeout: int):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        result = self._supervisor.run(
            argv, cwd=cwd.resolve(), environment=environment, timeout_seconds=timeout
        )
        if result.timed_out or result.exit_code != 0 or result.stdout_truncated:
            raise IntegrityError(
                f"Container command failed: {argv[1]}", "container_command_failed"
            )
        return result


def _module_relative(library: str, repo_relative_path: str) -> str:
    if library not in {"vllm", "aiter"}:
        raise ContractError("Unknown overlay source library", "unsupported_source_library")
    path = PurePosixPath(repo_relative_path)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
        raise ContractError("Overlay supports one safe Python source", "unsupported_overlay_source")
    if len(path.parts) < 2 or path.parts[0] != library:
        raise ContractError(
            "Repository path does not map to installed package", "source_mapping_mismatch"
        )
    return PurePosixPath(*path.parts[1:]).as_posix()


def _loaded_receipt(payload: Mapping[str, Any]) -> LoadedFileReceipt:
    try:
        path = str(payload["path"])
        digest = str(payload["sha256"])
        size = int(payload["size"])
        mode = int(payload["mode"])
        symlink = payload["symlink"]
    except (KeyError, TypeError, ValueError) as error:
        raise IntegrityError("Loaded byte receipt fields are invalid", "loaded_byte_probe_failed") from error
    if (
        not path.startswith("/")
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or size < 0
        or mode < 0
        or symlink is not False
    ):
        raise IntegrityError("Loaded byte receipt is unsafe", "loaded_byte_probe_failed")
    return LoadedFileReceipt(path, digest, size, mode)


__all__ = [
    "BuiltOverlay",
    "ContainerEngine",
    "ContainerImage",
    "DockerEngine",
    "InstalledPythonTarget",
    "LoadedFileReceipt",
]
