"""Fixed-argv Docker operations for runtime-only Python source overlays."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol

from apex.core import ContractError, IntegrityError, sha256_bytes, sha256_file
from apex.execution import ProcessResult, SubprocessSupervisor


_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPO_DIGEST = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)((?:authorization|api[_-]?key|password|secret|token)\s*[=:]\s*)([^\s,;]+)"
)
_BEARER_VALUE = re.compile(r"(?i)(bearer\s+)([^\s,;]+)")
_FAILURE_OUTPUT_LIMIT = 64 * 1024
_OVERLAY_BUILD_ATTEMPT_LIMIT = 2
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
    repo_digests: tuple[str, ...] = ()
    verified_repo_digest: str | None = None

    def __post_init__(self) -> None:
        if not _IMAGE_ID.fullmatch(self.image_id):
            raise ContractError("Container image ID is not immutable", "invalid_image_id")
        if (
            self.repo_digests != tuple(sorted(set(self.repo_digests)))
            or any(not _REPO_DIGEST.fullmatch(item) for item in self.repo_digests)
        ):
            raise ContractError("Container repo digests are invalid", "invalid_image_digest")
        if self.verified_repo_digest is not None and (
            not _REPO_DIGEST.fullmatch(self.verified_repo_digest)
            or (
                self.verified_repo_digest != self.reference
                and self.verified_repo_digest not in self.repo_digests
            )
        ):
            raise ContractError(
                "Verified container repo digest is not inspection-bound",
                "invalid_image_digest",
            )


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
        parent: ContainerImage,
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
        self._supervisor = supervisor or SubprocessSupervisor(
            max_output_bytes=32 * 1024 * 1024
        )

    def inspect_image(self, reference: str, *, cwd: Path) -> ContainerImage:
        result = self._run(
            ("docker", "image", "inspect", reference),
            cwd=cwd,
            timeout=60,
            stage="image_inspect",
        )
        try:
            values = json.loads(result.stdout)
            if (
                not isinstance(values, list)
                or len(values) != 1
                or not isinstance(values[0], Mapping)
            ):
                raise TypeError("inspection shape")
            image_id = str(values[0]["Id"])
            raw_repo_digests = values[0].get("RepoDigests")
            if raw_repo_digests is None:
                raw_repo_digests = []
            if not isinstance(raw_repo_digests, list):
                raise TypeError("repo digests")
            repo_digests = tuple(sorted(set(str(item) for item in raw_repo_digests)))
            if not _IMAGE_ID.fullmatch(image_id) or any(
                not _REPO_DIGEST.fullmatch(item) for item in repo_digests
            ):
                raise TypeError("image identity")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
            raise IntegrityError(
                "Docker image inspection is invalid",
                "image_inspection_failed",
                _process_evidence(result, stage="image_inspect", cwd=cwd),
            ) from error
        verified = reference if _REPO_DIGEST.fullmatch(reference) else None
        return ContainerImage(reference, image_id, repo_digests, verified)

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
        parent: ContainerImage,
        candidate_source: Path,
        target: InstalledPythonTarget,
        build_root: Path,
        cwd: Path,
    ) -> BuiltOverlay:
        parent_locator = self._build_parent_locator(parent, cwd=cwd)
        context, copied, dockerfile, candidate_sha256, dockerfile_sha256 = (
            _materialize_build_context(
                candidate_source,
                target,
                build_root=build_root,
                parent_locator=parent_locator,
            )
        )
        iidfile = build_root.resolve() / "derived-image.id"
        result = self._build_image(
            iidfile=iidfile,
            dockerfile=dockerfile,
            dockerfile_sha256=dockerfile_sha256,
            context=context,
            context_source=copied,
            context_source_sha256=candidate_sha256,
            cwd=cwd,
        )
        if not iidfile.is_file() or iidfile.is_symlink():
            raise IntegrityError(
                "Docker did not emit an image ID",
                "overlay_build_failed",
                _process_evidence(result, stage="overlay_build", cwd=cwd),
            )
        image_id = iidfile.read_text(encoding="utf-8").strip()
        if not _IMAGE_ID.fullmatch(image_id):
            raise IntegrityError(
                "Docker emitted an invalid image ID",
                "overlay_build_failed",
                _process_evidence(result, stage="overlay_build", cwd=cwd),
            )
        inspected = self.inspect_image(image_id, cwd=cwd)
        if inspected.image_id != image_id:
            raise IntegrityError(
                "Built image inspection changed identity", "image_identity_mismatch"
            )
        self._verify_parent_locator(parent_locator, parent, cwd=cwd)
        return BuiltOverlay(
            inspected,
            dockerfile_sha256,
            candidate_sha256,
        )

    def _build_parent_locator(self, parent: ContainerImage, *, cwd: Path) -> str:
        locator = parent.verified_repo_digest
        if locator is None and parent.reference == parent.image_id:
            locator = f"apex-overlay-parent:sha256-{parent.image_id.removeprefix('sha256:')}"
            self._run(
                ("docker", "image", "tag", parent.image_id, locator),
                cwd=cwd,
                timeout=60,
                stage="derived_parent_alias",
            )
        if locator is None:
            raise IntegrityError(
                "Parent image has no immutable locator for Dockerfile FROM",
                "immutable_parent_locator_unresolved",
                {"parent_image_id": parent.image_id},
            )
        self._verify_parent_locator(locator, parent, cwd=cwd)
        return locator

    def _verify_parent_locator(
        self, locator: str, parent: ContainerImage, *, cwd: Path
    ) -> None:
        observed = self.inspect_image(locator, cwd=cwd)
        if observed.image_id != parent.image_id:
            raise IntegrityError(
                "Overlay parent locator changed image identity",
                "image_identity_mismatch",
                {
                    "parent_locator": locator,
                    "expected_image_id": parent.image_id,
                    "observed_image_id": observed.image_id,
                },
            )

    def _build_image(
        self,
        *,
        iidfile: Path,
        dockerfile: Path,
        dockerfile_sha256: str,
        context: Path,
        context_source: Path,
        context_source_sha256: str,
        cwd: Path,
    ) -> ProcessResult:
        argv = (
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
        )
        failures: list[Mapping[str, Any]] = []
        for attempt in range(1, _OVERLAY_BUILD_ATTEMPT_LIMIT + 1):
            _verify_build_context(
                dockerfile,
                dockerfile_sha256,
                context_source,
                context_source_sha256,
            )
            iidfile.unlink(missing_ok=True)
            try:
                return self._run(
                    argv, cwd=cwd, timeout=1800, stage="overlay_build"
                )
            except IntegrityError as error:
                if error.reason_code != "container_command_failed":
                    raise
                details = dict(error.details or {})
                failures.append({"attempt": attempt, **details})
                if not _retryable_build_failure(details):
                    raise
        raise IntegrityError(
            "Container overlay build retry budget was exhausted",
            "container_command_failed",
            {
                **dict(failures[-1]),
                "attempt_limit": _OVERLAY_BUILD_ATTEMPT_LIMIT,
                "attempts": failures,
            },
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
            stage="container_probe",
        )
        marked = [line[len(_MARKER) :] for line in result.stdout.splitlines() if line.startswith(_MARKER)]
        if len(marked) != 1:
            raise IntegrityError(
                "Container byte probe emitted no unique receipt",
                "loaded_byte_probe_failed",
                _process_evidence(result, stage="container_probe", cwd=cwd),
            )
        try:
            payload = json.loads(marked[0])
        except json.JSONDecodeError as error:
            raise IntegrityError(
                "Container byte receipt is invalid",
                "loaded_byte_probe_failed",
                _process_evidence(result, stage="container_probe", cwd=cwd),
            ) from error
        if not isinstance(payload, Mapping):
            raise IntegrityError("Container byte receipt is not an object", "loaded_byte_probe_failed")
        return payload

    def _run(
        self, argv: tuple[str, ...], *, cwd: Path, timeout: int, stage: str
    ) -> ProcessResult:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        result = self._supervisor.run(
            argv, cwd=cwd.resolve(), environment=environment, timeout_seconds=timeout
        )
        if (
            result.timed_out
            or result.exit_code != 0
            or result.stdout_truncated
            or result.stderr_truncated
            or not result.cleanup_succeeded
        ):
            raise IntegrityError(
                f"Container command failed during {stage}",
                "container_command_failed",
                _process_evidence(result, stage=stage, cwd=cwd),
            )
        return result


def _process_evidence(
    result: ProcessResult, *, stage: str, cwd: Path
) -> dict[str, Any]:
    return {
        "stage": stage,
        "argv": _safe_argv(result.argv),
        "cwd": str(cwd.resolve()),
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "cleanup_succeeded": result.cleanup_succeeded,
        **_stream_evidence("stdout", result.stdout, result.stdout_truncated),
        **_stream_evidence("stderr", result.stderr, result.stderr_truncated),
        "duration_seconds": result.duration_seconds,
    }


def _materialize_build_context(
    candidate_source: Path,
    target: InstalledPythonTarget,
    *,
    build_root: Path,
    parent_locator: str,
) -> tuple[Path, Path, Path, str, str]:
    if not candidate_source.is_file() or candidate_source.is_symlink():
        raise IntegrityError(
            "Candidate source is not a regular file", "invalid_frozen_candidate"
        )
    context = build_root.resolve() / "context"
    if context.exists() or context.is_symlink():
        raise IntegrityError(
            "Docker overlay context already exists", "immutable_delivery_artifact"
        )
    context.mkdir(parents=True)
    copied = context / "candidate.py"
    candidate_sha256 = sha256_file(candidate_source)
    shutil.copyfile(candidate_source, copied)
    copied.chmod(0o444)
    if sha256_file(copied) != candidate_sha256:
        raise IntegrityError(
            "Overlay context changed candidate bytes", "candidate_lineage_mismatch"
        )
    dockerfile = context / "Dockerfile"
    dockerfile.write_text(
        f"FROM {parent_locator}\n"
        f"COPY {json.dumps(['candidate.py', target.container_path])}\n",
        encoding="utf-8",
    )
    dockerfile.chmod(0o444)
    return context, copied, dockerfile, candidate_sha256, sha256_file(dockerfile)


def _verify_build_context(
    dockerfile: Path,
    dockerfile_sha256: str,
    context_source: Path,
    context_source_sha256: str,
) -> None:
    for path, expected in (
        (dockerfile, dockerfile_sha256),
        (context_source, context_source_sha256),
    ):
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise IntegrityError(
                "Immutable overlay context drifted between build attempts",
                "immutable_overlay_context_drift",
                {"path": path.name, "expected_sha256": expected},
            )


def _retryable_build_failure(details: Mapping[str, Any]) -> bool:
    return bool(
        details.get("timed_out") is False
        and details.get("cleanup_succeeded") is True
        and details.get("stdout_truncated") is False
        and details.get("stderr_truncated") is False
    )


def _stream_evidence(name: str, value: str, truncated: bool) -> dict[str, Any]:
    raw = value.encode("utf-8", errors="replace")
    redacted = _redact(value)
    encoded = redacted.encode("utf-8", errors="replace")
    excerpted = len(encoded) > _FAILURE_OUTPUT_LIMIT
    if excerpted:
        encoded = encoded[-_FAILURE_OUTPUT_LIMIT:]
        redacted = encoded.decode("utf-8", errors="replace")
    return {
        name: redacted,
        f"{name}_bytes": len(raw),
        f"{name}_sha256": sha256_bytes(raw),
        f"{name}_truncated": truncated,
        f"{name}_excerpted": excerpted,
    }


def _safe_argv(argv: tuple[str, ...]) -> list[str]:
    safe: list[str] = []
    redact_next = False
    for value in argv:
        if redact_next:
            safe.append(f"<redacted:sha256:{sha256_bytes(value.encode())}>")
            redact_next = False
            continue
        if value in {
            "--password",
            "--secret",
            "--token",
            "--build-arg",
            "-c",
        }:
            safe.append(value)
            redact_next = True
            continue
        safe.append(_redact(value))
    return safe


def _redact(value: str) -> str:
    redacted = _SECRET_ASSIGNMENT.sub(r"\1<redacted>", value)
    return _BEARER_VALUE.sub(r"\1<redacted>", redacted)


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
