from __future__ import annotations

import json
from pathlib import Path

from apex.core import AgentBackendName, sha256_file
from apex.execution import ProcessResult
from apex.optimization.e2e.candidate import E2ECandidate
from apex.optimization.e2e.kernel_lane import KernelOpportunity
from apex.optimization.e2e.oracle_preflight import (
    DockerOracleMicroQualifier,
    DockerOraclePolicy,
    OracleDependencyLock,
    OracleSourceLock,
)
from apex.optimization.e2e.oracle_container import runner_sha256
from apex.optimization.e2e.oracles import (
    CorrectnessOracleBinding,
    CorrectnessOracleRegistry,
)
from apex.optimization.e2e.overlay_runtime import (
    BuiltOverlay,
    ContainerImage,
    InstalledPythonTarget,
    LoadedFileReceipt,
)
from apex.optimization.e2e.services import MicroQualificationRequest
from apex.ports import AgentResult


PARENT = "sha256:" + "a" * 64
DERIVED = "sha256:" + "b" * 64
COMMIT = "c" * 40
TREE = "d" * 40


class FakeEngine:
    def __init__(self, baseline: str, candidate: str) -> None:
        self.baseline = baseline
        self.candidate = candidate
        self.calls: list[str] = []

    def inspect_image(self, reference, *, cwd):
        self.calls.append("inspect")
        return ContainerImage(reference, PARENT)

    def resolve_python_target(self, image_id, *, library, repo_relative_path, cwd):
        self.calls.append("resolve")
        return InstalledPythonTarget(
            library,
            repo_relative_path,
            "kernel.py",
            "/usr/local/lib/python3.12/dist-packages/vllm/kernel.py",
            self.baseline,
            10,
            0o644,
        )

    def build(
        self,
        *,
        parent_locator,
        parent_image_id,
        candidate_source,
        target,
        build_root,
        cwd,
    ):
        self.calls.append("build")
        return BuiltOverlay(ContainerImage(DERIVED, DERIVED), "e" * 64, self.candidate)

    def read_file(self, image_id, *, container_path, cwd):
        self.calls.append("read")
        return LoadedFileReceipt(container_path, self.candidate, 12, 0o444)


class FakeCommands:
    def __init__(
        self,
        *,
        dependency_version: str = "9.0.2",
        loaded_module: str | None = None,
        loaded_sha256: str | None = None,
    ) -> None:
        self.dependency_version = dependency_version
        self.loaded_module = loaded_module
        self.loaded_sha256 = loaded_sha256
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv, *, cwd, environment, timeout_seconds, stdin_text=None):
        command = tuple(argv)
        self.calls.append(command)
        stdout = ""
        if command[:3] == ("git", "rev-parse", "HEAD"):
            stdout = COMMIT + "\n"
        elif command[:3] == ("git", "rev-parse", "HEAD^{tree}"):
            stdout = TREE + "\n"
        elif command[:2] == ("git", "status"):
            stdout = ""
        elif command[:3] == ("docker", "run", "--rm") and "-I" in command:
            stdout = (
                "__APEX_ORACLE_DEPENDENCIES_V1__"
                + json.dumps(
                    {"pytest": self.dependency_version},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        elif command[:3] == ("docker", "run", "--rm"):
            mount = next(
                item
                for item in command
                if item.startswith("type=bind,src=")
                and item.endswith(",dst=/opt/apex-result")
            )
            source_part = next(part for part in mount.split(",") if part.startswith("src="))
            result_root = Path(source_part.removeprefix("src="))
            (result_root / "junit.xml").write_text(
                '<testsuite tests="1" failures="0" errors="0" skipped="0"/>',
                encoding="utf-8",
            )
            runner_index = command.index("/opt/apex-oracle/apex_oracle_runner.py")
            module_name = command[runner_index + 1]
            expected_path = command[runner_index + 2]
            expected_sha256 = command[runner_index + 3]
            observed_module = self.loaded_module or module_name
            observed_sha256 = self.loaded_sha256 or expected_sha256
            receipt = result_root / "loaded-candidate.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema": "apex.qwen-oracle-loaded-candidate/v1",
                        "module_name": observed_module,
                        "expected_path": expected_path,
                        "expected_sha256": expected_sha256,
                        "status": "passed",
                        "reason_code": "same_process_import_and_tests_passed",
                        "pytest_exit_code": 0,
                        "before": {
                            "path": expected_path,
                            "sha256": observed_sha256,
                        },
                        "after": {
                            "path": expected_path,
                            "sha256": observed_sha256,
                            "same_module_object": True,
                        },
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            receipt.chmod(0o444)
            stdout = "1 passed in 0.01s\n"
        return ProcessResult(command, 0, False, stdout, "", False, False, 0.01)


def _fixture(
    tmp_path: Path,
    *,
    dependency_version: str = "9.0.2",
    loaded_module: str | None = None,
    loaded_sha256: str | None = None,
):
    source = tmp_path / "source"
    baseline = source / "vllm/kernel.py"
    test = source / "tests/test_kernel.py"
    baseline.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    baseline.write_text("VALUE = 1\n", encoding="utf-8")
    test.write_text("def test_kernel(): assert True\n", encoding="utf-8")
    registry = CorrectnessOracleRegistry(
        source_roots={"vllm": source},
        source_lock_sha256="f" * 64,
        bindings=(
            CorrectnessOracleBinding(
                "vllm",
                "vllm/kernel.py",
                "tests/test_kernel.py",
                (
                    "python3",
                    "-m",
                    "pytest",
                    "--junitxml=/opt/apex-result/junit.xml",
                    "tests/test_kernel.py::test_kernel",
                ),
            ),
        ),
    )
    resolved = registry.resolve(
        repository_id="vllm", source_root=source, source_path=baseline
    )
    assert resolved is not None
    workspace = tmp_path / "workspace"
    candidate_path = workspace / "vllm/kernel.py"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text("VALUE = 2\n", encoding="utf-8")
    opportunity = KernelOpportunity(
        "kernel-1",
        "evidence-1",
        "kernel",
        "operator",
        "decode",
        0,
        "triton",
        "vllm",
        (),
        (),
        "eager",
        "high",
        10.0,
        10.0,
        baseline,
        source,
        test,
        resolved.test_command,
        "eligible",
        "eligible",
        resolved.binding_sha256,
    )
    candidate = E2ECandidate(
        "attempt-1",
        "candidate-1",
        True,
        "candidate_frozen",
        workspace,
        ("vllm/kernel.py",),
        ("vllm/kernel.py",),
        "1" * 64,
        "2" * 64,
        AgentResult(AgentBackendName.CODEX, None, 0, False, (), "", "", 0.1),
    )
    engine = FakeEngine(sha256_file(baseline), sha256_file(candidate_path))
    commands = FakeCommands(
        dependency_version=dependency_version,
        loaded_module=loaded_module,
        loaded_sha256=loaded_sha256,
    )
    verifier = DockerOracleMicroQualifier(
        oracles=registry,
        policy=DockerOraclePolicy(
            "example.invalid/vllm@sha256:" + "9" * 64,
            PARENT,
            (OracleSourceLock("vllm", COMMIT, TREE),),
            (OracleDependencyLock("pytest", "9.0.2"),),
            timeout_seconds=30,
        ),
        engine=engine,
        commands=commands,
        overlay_builder=engine,
    )
    request = MicroQualificationRequest(
        "run-1",
        candidate,
        opportunity,
        tmp_path / "artifacts",
        0,
        "amd-gpu-set=3",
    )
    return verifier, request, engine, commands


def test_oracle_preflight_binds_loaded_bytes_exact_tests_and_gpu_scope(tmp_path: Path):
    verifier, request, engine, commands = _fixture(tmp_path)

    result = verifier.verify(request)

    assert result.qualified is True
    assert result.grade is None
    assert result.kernel_reward_available is False
    assert result.evidence["claims"]["correct"] == "unmeasured"
    details = result.evidence["details"]
    expected_sha256 = sha256_file(
        request.candidate.workspace / "vllm/kernel.py"
    )
    assert details["loaded_candidate"]["before"]["sha256"] == expected_sha256
    assert details["loaded_candidate"]["after"]["sha256"] == expected_sha256
    assert details["loaded_candidate"]["after"]["same_module_object"] is True
    assert details["loaded_candidate"]["read_only"] is True
    assert details["installed_candidate_file"]["sha256"] == expected_sha256
    assert details["runner_sha256"] == runner_sha256()
    assert details["gpu_device_scope"] == "amd-gpu-set=3"
    assert details["junit"]["tests"] == 1
    assert details["test_manifest"]["installed_source_shadowed"] is False
    assert details["test_manifest"]["runner"]["sha256"] == runner_sha256()
    test_call = commands.calls[-1]
    assert "ROCR_VISIBLE_DEVICES=3" in test_call
    assert "HIP_VISIBLE_DEVICES=0" in test_call
    assert "/opt/apex-oracle/apex_oracle_runner.py" in test_call
    assert "vllm.kernel" in test_call
    assert engine.calls == ["inspect", "resolve", "build", "read"]


def test_oracle_preflight_rejects_wrong_same_process_module(tmp_path: Path):
    verifier, request, _, _ = _fixture(tmp_path, loaded_module="vllm.old_kernel")

    result = verifier.verify(request)

    assert result.qualified is False
    assert result.grade is None
    assert result.evidence["reason_code"] == "loaded_byte_probe_failed"
    assert result.kernel_reward_available is False
    assert result.evidence["claims"]["correct"] == "unmeasured"


def test_oracle_preflight_rejects_old_loaded_bytes(tmp_path: Path):
    verifier, request, _, _ = _fixture(tmp_path, loaded_sha256="0" * 64)

    result = verifier.verify(request)

    assert result.qualified is False
    assert result.grade is None
    assert result.evidence["reason_code"] == "loaded_byte_probe_failed"
    assert result.kernel_reward_available is False
    assert result.evidence["claims"]["correct"] == "unmeasured"


def test_oracle_preflight_fails_closed_on_dependency_drift(tmp_path: Path):
    verifier, request, engine, _ = _fixture(tmp_path, dependency_version="8.0.0")

    result = verifier.verify(request)

    assert result.qualified is False
    assert result.grade is None
    assert result.evidence["reason_code"] == "oracle_dependency_drift"
    assert result.reason_code == "oracle_dependency_drift"
    assert result.evidence["claims"]["correct"] == "unmeasured"
    assert engine.calls == ["inspect", "resolve", "build", "read"]
