"""Fail-closed tests-only Docker preflight for reviewed E2E source oracles."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

from apex.core import (
    ApexError,
    ContractError,
    IntegrityError,
    sha256_file,
    sha256_json,
)
from apex.execution import (
    DOCKER_RUNTIME_ENVIRONMENT_KEYS,
    ProcessResult,
    SubprocessSupervisor,
    build_subprocess_environment,
)

from .candidate import candidate_file_paths, make_candidate_read_only
from .kernel_lane import KernelOpportunity
from .oracle_container import (
    DockerOracleOverlayBuilder,
    OracleOverlayBuildPort,
    OracleTestExecution,
    materialize_runner,
    process_receipt,
    run_oracle_tests,
    runner_sha256,
)
from .oracles import CorrectnessOracleRegistry, ResolvedCorrectnessOracle
from .overlay_runtime import (
    BuiltOverlay,
    ContainerEngine,
    ContainerImage,
    DockerEngine,
    InstalledPythonTarget,
    LoadedFileReceipt,
)
from .services import MicroQualification, MicroQualificationRequest


_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_DEPENDENCY_MARKER = "__APEX_ORACLE_DEPENDENCIES_V1__"


class CommandPort(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: int,
        stdin_text: str | None = None,
    ) -> ProcessResult: ...


@dataclass(frozen=True, slots=True)
class OracleSourceLock:
    repository_id: str
    commit: str
    tree: str

    def __post_init__(self) -> None:
        if not self.repository_id or not re.fullmatch(r"[0-9a-f]{40}", self.commit):
            raise ContractError("Oracle source lock is invalid", "invalid_oracle_runtime")
        if not re.fullmatch(r"[0-9a-f]{40}", self.tree):
            raise ContractError("Oracle source tree is invalid", "invalid_oracle_runtime")


@dataclass(frozen=True, slots=True)
class OracleDependencyLock:
    distribution: str
    version: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.distribution) or not self.version:
            raise ContractError("Oracle dependency lock is invalid", "invalid_oracle_runtime")


@dataclass(frozen=True, slots=True)
class DockerOraclePolicy:
    parent_locator: str
    parent_image_id: str
    source_locks: tuple[OracleSourceLock, ...]
    dependencies: tuple[OracleDependencyLock, ...]
    timeout_seconds: int = 900

    def __post_init__(self) -> None:
        if "@sha256:" not in self.parent_locator or not _IMAGE_ID.fullmatch(
            self.parent_image_id
        ):
            raise ContractError("Oracle parent is mutable", "invalid_oracle_runtime")
        if not self.source_locks or self.timeout_seconds < 1:
            raise ContractError("Oracle runtime policy is incomplete", "invalid_oracle_runtime")
        names = tuple(item.repository_id for item in self.source_locks)
        if len(names) != len(set(names)):
            raise ContractError("Oracle source locks are duplicated", "invalid_oracle_runtime")

    @property
    def digest(self) -> str:
        return sha256_json(
            {
                "schema": "apex.qwen-oracle-runtime-policy/v1",
                "parent_locator": self.parent_locator,
                "parent_image_id": self.parent_image_id,
                "source_locks": [asdict(item) for item in self.source_locks],
                "dependencies": [asdict(item) for item in self.dependencies],
                "timeout_seconds": self.timeout_seconds,
                "runner_sha256": runner_sha256(),
            }
        )


class DockerOracleMicroQualifier:
    """Execute reviewed test subsets without claiming a canonical kernel grade.

    Passing this preflight only permits the existing deferred candidate to reach
    safety and full Magpie evaluation. It never emits compile, correctness,
    timing, or reward evidence.
    """

    qualification_mode = "e2e_quality_deferred"

    def __init__(
        self,
        *,
        oracles: CorrectnessOracleRegistry,
        policy: DockerOraclePolicy,
        engine: ContainerEngine | None = None,
        commands: CommandPort | None = None,
        overlay_builder: OracleOverlayBuildPort | None = None,
    ) -> None:
        self._oracles = oracles
        self._policy = policy
        self._engine = engine or DockerEngine()
        self._commands = commands or SubprocessSupervisor(max_output_bytes=8 * 1024 * 1024)
        self._overlay_builder = overlay_builder or DockerOracleOverlayBuilder(
            self._commands, self._engine
        )
        self._locks = {item.repository_id: item for item in policy.source_locks}

    def supports(self, opportunity: KernelOpportunity) -> bool:
        return _candidate_opportunity(opportunity)

    def verify(self, request: MicroQualificationRequest) -> MicroQualification:
        try:
            evidence = self._verify(request)
        except ApexError as error:
            return _qualification(request, False, error.reason_code, error.details or {})
        except (OSError, ValueError) as error:
            return _qualification(
                request,
                False,
                "oracle_preflight_failed",
                {"error_type": type(error).__name__},
            )
        return _qualification(request, True, "oracle_preflight_passed", evidence)

    def _verify(self, request: MicroQualificationRequest) -> Mapping[str, Any]:
        candidate = request.candidate
        opportunity = request.opportunity
        if not _candidate_integrity(request):
            raise IntegrityError(
                "Oracle preflight requires frozen source", "invalid_frozen_candidate"
            )
        make_candidate_read_only(candidate)
        oracle = self._resolve(opportunity)
        if oracle is None:
            raise ContractError("No reviewed oracle is bound", "oracle_preflight_unsupported")
        source_receipt = self._verify_source_lock(opportunity)
        root, tests, results = _prepare_artifacts(request.artifact_root)
        parent = self._engine.inspect_image(self._policy.parent_locator, cwd=root)
        if parent.image_id != self._policy.parent_image_id:
            raise IntegrityError("Oracle parent image drifted", "image_identity_mismatch")
        relative, candidate_path = _candidate_path(request)
        target = self._engine.resolve_python_target(
            parent.image_id,
            library=opportunity.origin_library,
            repo_relative_path=relative,
            cwd=root,
        )
        _validate_target(target, opportunity, relative, oracle)
        built = self._overlay_builder.build(
            parent_locator=self._policy.parent_locator,
            parent_image_id=parent.image_id,
            candidate_source=candidate_path,
            target=target,
            build_root=root / "overlay",
            cwd=root,
        )
        candidate_file_sha256 = sha256_file(candidate_path)
        if built.context_source_sha256 != candidate_file_sha256:
            raise IntegrityError("Oracle build changed candidate", "candidate_lineage_mismatch")
        installed = self._engine.read_file(
            built.image.image_id,
            container_path=target.container_path,
            cwd=root,
        )
        if installed.sha256 != candidate_file_sha256:
            raise IntegrityError(
                "Oracle image does not contain candidate bytes",
                "loaded_candidate_bytes_mismatch",
            )
        test_manifest = _materialize_tests(oracle, opportunity, tests)
        dependencies = self._probe_dependencies(built.image.image_id, root)
        execution = run_oracle_tests(
            self._commands,
            image_id=built.image.image_id,
            module_name=_module_name(target.package, target.module_relative_path),
            target=target,
            candidate_sha256=candidate_file_sha256,
            oracle=oracle,
            tests=tests,
            results=results,
            gpu_scope=request.gpu_device_scope,
            cwd=root,
            timeout_seconds=self._policy.timeout_seconds,
        )
        return _preflight_evidence(
            self._policy,
            request,
            oracle,
            source_receipt,
            parent,
            built,
            target,
            candidate_file_sha256,
            installed,
            execution,
            test_manifest,
            dependencies,
        )

    def _resolve(self, opportunity: KernelOpportunity) -> ResolvedCorrectnessOracle | None:
        assert opportunity.source_root is not None and opportunity.source_path is not None
        return self._oracles.resolve(
            repository_id=opportunity.origin_library,
            source_root=opportunity.source_root,
            source_path=opportunity.source_path,
        )

    def _verify_source_lock(self, opportunity: KernelOpportunity) -> Mapping[str, Any]:
        root = opportunity.source_root
        lock = self._locks.get(opportunity.origin_library)
        if root is None or lock is None:
            raise IntegrityError("Oracle source is not locked", "source_lock_unresolved")
        values = {
            "commit": _git(self._commands, root, ("git", "rev-parse", "HEAD")),
            "tree": _git(self._commands, root, ("git", "rev-parse", "HEAD^{tree}")),
            "status": _git(
                self._commands,
                root,
                ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            ),
        }
        if values != {"commit": lock.commit, "tree": lock.tree, "status": ""}:
            raise IntegrityError("Oracle source lock drifted", "source_lock_drift")
        return {"repository_id": lock.repository_id, **values}

    def _probe_dependencies(self, image_id: str, cwd: Path) -> Mapping[str, str]:
        expected = {item.distribution: item.version for item in self._policy.dependencies}
        script = (
            "import importlib.metadata as m,json,sys;"
            "names=json.loads(sys.argv[1]);"
            f"print('{_DEPENDENCY_MARKER}'+json.dumps({{n:m.version(n) for n in names}},"
            "sort_keys=True,separators=(',',':')))"
        )
        result = _run_checked(
            self._commands,
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
                json.dumps(sorted(expected)),
            ),
            cwd,
            120,
            "oracle_dependency_unavailable",
        )
        marked = [
            line.removeprefix(_DEPENDENCY_MARKER)
            for line in result.stdout.splitlines()
            if line.startswith(_DEPENDENCY_MARKER)
        ]
        if len(marked) != 1 or json.loads(marked[0]) != expected:
            raise IntegrityError("Oracle dependency lock drifted", "oracle_dependency_drift")
        return expected


def _preflight_evidence(
    policy: DockerOraclePolicy,
    request: MicroQualificationRequest,
    oracle: ResolvedCorrectnessOracle,
    source_receipt: Mapping[str, Any],
    parent: ContainerImage,
    built: BuiltOverlay,
    target: InstalledPythonTarget,
    candidate_file_sha256: str,
    installed: LoadedFileReceipt,
    execution: OracleTestExecution,
    test_manifest: Mapping[str, Any],
    dependencies: Mapping[str, str],
) -> Mapping[str, Any]:
    return {
        "schema": "apex.qwen-oracle-preflight/v1",
        "policy_sha256": policy.digest,
        "oracle_policy_sha256": oracle.policy_sha256,
        "oracle_binding_sha256": oracle.binding_sha256,
        "source_lock": dict(source_receipt),
        "parent_image": asdict(parent),
        "candidate_image": asdict(built.image),
        "overlay_build": {
            "dockerfile_sha256": built.dockerfile_sha256,
            "context_source_sha256": built.context_source_sha256,
        },
        "installed_target": target.to_dict(),
        "candidate_set_sha256": request.candidate.candidate_source_sha256,
        "candidate_file_sha256": candidate_file_sha256,
        "installed_candidate_file": installed.to_dict(),
        "loaded_candidate": dict(execution.loaded_candidate),
        "runner_sha256": execution.runner_sha256,
        "test_argv": list(oracle.test_argv),
        "test_manifest": dict(test_manifest),
        "dependencies": dict(dependencies),
        "gpu_device_scope": request.gpu_device_scope,
        "process": process_receipt(execution.process),
        "junit": dict(execution.junit),
        "preflight_execution_passed": True,
        "authority": "candidate_rejection_only_not_canonical_correctness",
    }


def _candidate_opportunity(opportunity: KernelOpportunity) -> bool:
    return bool(
        opportunity.eligible
        and opportunity.language in {"python", "triton"}
        and opportunity.origin_library == "vllm"
        and opportunity.source_root
        and opportunity.source_path
        and opportunity.correctness_oracle_sha256
    )


def _candidate_integrity(request: MicroQualificationRequest) -> bool:
    candidate = request.candidate
    return bool(
        candidate.succeeded
        and candidate.candidate_id
        and candidate.candidate_source_sha256
        and len(candidate.changed_files) == 1
        and candidate.changed_files == candidate.editable_files
        and candidate.changed_files[0].endswith(".py")
    )


def _candidate_path(request: MicroQualificationRequest) -> tuple[str, Path]:
    opportunity = request.opportunity
    assert opportunity.source_root is not None and opportunity.source_path is not None
    relative = opportunity.source_path.resolve(strict=True).relative_to(
        opportunity.source_root.resolve(strict=True)
    ).as_posix()
    paths = candidate_file_paths(request.candidate)
    if len(paths) != 1 or request.candidate.editable_files != (relative,):
        raise IntegrityError("Oracle candidate path drifted", "candidate_lineage_mismatch")
    path = paths[0]
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise IntegrityError("Oracle candidate is not regular", "invalid_frozen_candidate")
    return relative, path


def _validate_target(
    target: InstalledPythonTarget,
    opportunity: KernelOpportunity,
    relative: str,
    oracle: ResolvedCorrectnessOracle,
) -> None:
    expected_module = PurePosixPath(*PurePosixPath(relative).parts[1:]).as_posix()
    if (
        target.package != opportunity.origin_library
        or target.repo_relative_path != relative
        or target.module_relative_path != expected_module
        or not PurePosixPath(target.container_path).is_absolute()
        or PurePosixPath(target.container_path).suffix != ".py"
        or target.sha256 != oracle.source_sha256
    ):
        raise IntegrityError("Oracle installed source mapping drifted", "source_mapping_mismatch")


def _module_name(package: str, module_relative_path: str) -> str:
    path = PurePosixPath(module_relative_path)
    if path.suffix != ".py" or path.is_absolute() or ".." in path.parts:
        raise IntegrityError("Oracle module mapping is invalid", "source_mapping_mismatch")
    parts = [package, *path.with_suffix("").parts]
    if parts and parts[-1] == "__init__":
        parts.pop()
    if not parts or any(not part.isidentifier() for part in parts):
        raise IntegrityError("Oracle module mapping is invalid", "source_mapping_mismatch")
    return ".".join(parts)


def _prepare_artifacts(root: Path) -> tuple[Path, Path, Path]:
    if not root.is_absolute() or root.exists() or root.is_symlink():
        raise IntegrityError("Oracle artifact root must be new", "immutable_delivery_artifact")
    root.mkdir(parents=True, mode=0o700)
    tests = root / "tests-only"
    results = root / "results"
    tests.mkdir(mode=0o755)
    results.mkdir(mode=0o700)
    return root.resolve(), tests.resolve(), results.resolve()


def _materialize_tests(
    oracle: ResolvedCorrectnessOracle,
    opportunity: KernelOpportunity,
    destination: Path,
) -> Mapping[str, Any]:
    assert opportunity.source_root is not None
    source_root = opportunity.source_root.resolve(strict=True)
    files = (oracle.test_file, *oracle.support_files)
    manifest: dict[str, str] = {}
    for source in files:
        relative = source.resolve(strict=True).relative_to(source_root).as_posix()
        expected = oracle.test_files_sha256.get(relative)
        if expected is None or sha256_file(source) != expected:
            raise IntegrityError("Oracle test bytes drifted", "oracle_test_drift")
        target = destination.joinpath(*Path(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o444)
        if sha256_file(target) != expected:
            raise IntegrityError("Oracle test copy drifted", "oracle_test_drift")
        manifest[relative] = expected
    if (destination / "vllm").exists() or (destination / "aiter").exists():
        raise IntegrityError(
            "Tests-only bundle shadows installed source", "oracle_source_shadowed"
        )
    runner = materialize_runner(destination)
    bundle = {**manifest, runner["path"]: runner["sha256"]}
    return {
        "files": dict(sorted(manifest.items())),
        "runner": runner,
        "manifest_sha256": sha256_json(dict(sorted(bundle.items()))),
        "installed_source_shadowed": False,
    }


def _git(commands: CommandPort, cwd: Path, argv: tuple[str, ...]) -> str:
    result = commands.run(
        argv,
        cwd=cwd.resolve(strict=True),
        environment=build_subprocess_environment(
            fixed={
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_SYSTEM": "/dev/null",
                "GIT_TERMINAL_PROMPT": "0",
            }
        ),
        timeout_seconds=60,
    )
    if (
        result.exit_code != 0
        or result.timed_out
        or result.stdout_truncated
        or result.stderr_truncated
    ):
        raise IntegrityError("Oracle source inspection failed", "source_lock_inspection_failed")
    return result.stdout.strip()


def _run_checked(
    commands: CommandPort,
    argv: tuple[str, ...],
    cwd: Path,
    timeout: int,
    reason: str,
) -> ProcessResult:
    result = commands.run(
        argv,
        cwd=cwd.resolve(strict=True),
        environment=build_subprocess_environment(
            inherit=DOCKER_RUNTIME_ENVIRONMENT_KEYS,
        ),
        timeout_seconds=timeout,
    )
    if (
        result.exit_code != 0
        or result.timed_out
        or result.stdout_truncated
        or result.stderr_truncated
    ):
        raise IntegrityError("Oracle container command failed", reason, process_receipt(result))
    return result


def _qualification(
    request: MicroQualificationRequest,
    passed: bool,
    reason_code: str,
    details: Mapping[str, Any],
) -> MicroQualification:
    candidate = request.candidate
    return MicroQualification(
        candidate_id=candidate.candidate_id or candidate.attempt_id,
        grade=None,
        evidence={
            "schema": "apex.qwen-oracle-preflight/v1",
            "qualification_mode": "e2e_quality_deferred",
            "reason_code": reason_code,
            "oracle_preflight_passed": passed,
            "details": dict(details),
            "claims": {
                "compiled": "unmeasured",
                "correct": "unmeasured",
                "p50": "unmeasured",
                "p99": "unmeasured",
            },
            "kernel_reward": {"available": False, "reason_code": "no_raw_micro_harness"},
            "promotion_authority": {
                "correctness": "unchanged_magpie_quality_gate",
                "performance": "unchanged_magpie_e2e_measurement",
            },
            "anchor_generation": request.anchor_generation,
        },
        qualification_mode="e2e_quality_deferred",
        deferred_candidate_valid=passed,
    )


__all__ = [
    "DockerOracleMicroQualifier",
    "DockerOraclePolicy",
    "OracleDependencyLock",
    "OracleSourceLock",
]
