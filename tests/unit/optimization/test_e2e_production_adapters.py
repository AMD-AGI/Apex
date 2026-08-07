from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

import apex.bootstrap as application_bootstrap
from apex.core import AgentBackendName, ContractError, IntegrityError, TaskStatus, ValidationLevel
from apex.core import sha256_file, sha256_json
from apex.optimization.e2e.candidate import E2ECandidate
from apex.optimization.e2e.candidate import AgentCandidateWorker
from apex.optimization.e2e.deferred import E2EDeferredMicroQualifier
from apex.optimization.e2e.docker_overlay import (
    DockerOverlayDeployment,
    OverlayOnlyFinalDelivery,
)
from apex.optimization.e2e.source_delivery import SourceRebuildFinalDelivery
from apex.optimization.e2e.kernel_lane import KernelOpportunity
from apex.optimization.e2e.overlay_config import derive_overlay_configs
from apex.optimization.e2e.overlay_runtime import (
    BuiltOverlay,
    ContainerImage,
    DockerEngine,
    InstalledPythonTarget,
    LoadedFileReceipt,
)
from apex.execution import ProcessResult
from apex.optimization.e2e.services import (
    CandidateDeploymentRequest,
    FinalDeliveryRequest,
    MicroQualification,
    MicroQualificationRequest,
    SafetyQualification,
)
from apex.ports import AgentResult
from apex.runtime import (
    ContainerIdentity,
    DependencyReceipt,
    RepositoryLock,
    RunProvenance,
    SourceCheckoutReceipt,
    SourceLockReceipt,
)


PARENT_ID = "sha256:" + "a" * 64
DERIVED_ID = "sha256:" + "b" * 64


class FakeLockVerifier:
    def verify(self, lock, *, expected_root):
        assert Path(lock.path) == expected_root
        return {"name": lock.name, "commit": lock.commit, "tree": lock.tree, "clean": True}


class FakeEngine:
    def __init__(self, *, baseline_sha: str, candidate_sha: str) -> None:
        self.baseline_sha = baseline_sha
        self.candidate_sha = candidate_sha
        self.loaded_sha = candidate_sha
        self.inspected_parent_id = PARENT_ID
        self.mapping_path = "/usr/lib/python/site-packages/vllm/kernels/op.py"
        self.returned_repo_relative: str | None = None
        self.calls: list[str] = []

    def inspect_image(self, reference, *, cwd):
        self.calls.append("inspect")
        return ContainerImage(
            reference,
            reference if reference.startswith("sha256:") else self.inspected_parent_id,
        )

    def resolve_python_target(self, image_id, *, library, repo_relative_path, cwd):
        self.calls.append("resolve")
        return InstalledPythonTarget(
            library,
            self.returned_repo_relative or repo_relative_path,
            "kernels/op.py",
            self.mapping_path,
            self.baseline_sha,
            12,
            0o644,
        )

    def build_overlay(self, *, parent_image_id, candidate_source, target, build_root, cwd):
        self.calls.append("build")
        assert parent_image_id == PARENT_ID
        return BuiltOverlay(ContainerImage(DERIVED_ID, DERIVED_ID), "d" * 64, self.candidate_sha)

    def read_file(self, image_id, *, container_path, cwd):
        self.calls.append("read")
        return LoadedFileReceipt(container_path, self.loaded_sha, 16, 0o444)


class FakeDockerSupervisor:
    def __init__(self, candidate_sha: str) -> None:
        self.candidate_sha = candidate_sha
        self.argv: list[tuple[str, ...]] = []

    def run(self, argv, *, cwd, environment, timeout_seconds, stdin_text=None):
        command = tuple(argv)
        self.argv.append(command)
        stdout = ""
        if command[1:3] == ("image", "inspect"):
            reference = command[3]
            image_id = DERIVED_ID if reference == DERIVED_ID else PARENT_ID
            stdout = json.dumps([{"Id": image_id}])
        elif command[1] == "run":
            target = "/usr/lib/python/site-packages/vllm/kernels/op.py"
            digest = self.candidate_sha if command[6] == DERIVED_ID else "a" * 64
            stdout = (
                "__APEX_PROBE_V1__"
                + '{"mode":420,"package_root":"/usr/lib/python/site-packages/vllm",'
                + f'"path":"{target}","sha256":"{digest}","size":12,"symlink":false}}\n'
            )
        elif command[1] == "build":
            iidfile = Path(command[command.index("--iidfile") + 1])
            iidfile.write_text(DERIVED_ID, encoding="utf-8")
        return ProcessResult(command, 0, False, stdout, "", False, False, 0.01)


def _fixture(tmp_path: Path, *, library: str = "vllm", suffix: str = ".py"):
    source_root = tmp_path / "source"
    relative = Path(library) / "kernels" / f"op{suffix}"
    baseline = source_root / relative
    baseline.parent.mkdir(parents=True)
    baseline.write_text("BASELINE = True\n", encoding="utf-8")
    workspace = tmp_path / "candidate"
    candidate_path = workspace / relative
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text("BASELINE = False\n", encoding="utf-8")
    opportunity = _opportunity(source_root, baseline, library=library, language="triton")
    candidate = E2ECandidate(
        "attempt-1",
        "candidate-1",
        True,
        "candidate_frozen",
        workspace,
        (relative.as_posix(),),
        (relative.as_posix(),),
        "1" * 64,
        "2" * 64,
        AgentResult(AgentBackendName.CODEX, None, 0, False, (), "", "", 1.0),
    )
    views, semantics = _views(tmp_path)
    provenance = _provenance(source_root, library)
    safety = SafetyQualification("candidate-1", True, True, False, False, (), {})
    request = CandidateDeploymentRequest(
        run_id="run-1",
        candidate=candidate,
        opportunity=opportunity,
        provenance=provenance,
        benchmark_measurement=views[0],
        benchmark_diagnostic=views[1],
        workload_semantics_sha256=semantics,
        artifact_root=tmp_path / "artifacts",
        anchor_generation=0,
        safety=safety,
        benchmark_replay=views[2],
    )
    engine = FakeEngine(
        baseline_sha=sha256_file(baseline), candidate_sha=sha256_file(candidate_path)
    )
    return opportunity, candidate, request, engine


def _opportunity(root: Path, source: Path, *, library: str, language: str):
    return KernelOpportunity(
        "kernel-1",
        "evidence-1",
        "runtime_symbol",
        "operator",
        "decode",
        0,
        language,
        library,
        ("[16,128]",),
        ("fp8",),
        "eager",
        "exact",
        5.0,
        2.0,
        source,
        root,
        source,
        "pytest -q",
        "eligible",
        "eligible",
    )


def _provenance(source_root: Path, library: str) -> RunProvenance:
    lock = RepositoryLock(
        library,
        str(source_root),
        "https://example.invalid/repo.git",
        "c" * 40,
        "d" * 40,
        True,
    )
    return RunProvenance(
        1,
        "/benchmark.yaml",
        "e" * 64,
        "vllm",
        "Qwen/example",
        None,
        "gfx950",
        ContainerIdentity("example:v1", PARENT_ID, (), ()),
        (library,),
        (lock,),
        "partial",
        ("model_revision", "runtime_loaded_bytes"),
    )


def _views(tmp_path: Path):
    benchmark = {
        "framework": "vllm",
        "model": "Qwen/example",
        "docker_image": "example:v1",
        "envs": {"RUN_EVAL": "true", "TP": 1},
        "profiler": {"torch_profiler": {"enabled": False}},
        "gap_analysis": {"enabled": False},
    }
    projected = copy.deepcopy(benchmark)
    projected.pop("docker_image")
    projected.pop("profiler")
    projected.pop("gap_analysis")
    semantics = sha256_json(projected)
    paths = []
    for kind in ("measurement", "diagnostic", "replay"):
        document = {
            "benchmark": copy.deepcopy(benchmark),
            "apex": {
                "benchmark_view": {
                    "schema": "apex.benchmark-view.v1",
                    "kind": kind,
                    "workload_semantics_sha256": semantics,
                }
            },
        }
        path = tmp_path / f"{kind}.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        paths.append(path.resolve())
    return tuple(paths), semantics


def test_deferred_qualification_makes_no_micro_truth_or_reward_claim(tmp_path: Path):
    opportunity, candidate, request, _ = _fixture(tmp_path)
    result = E2EDeferredMicroQualifier().verify(
        MicroQualificationRequest("run-1", candidate, opportunity, tmp_path / "micro", 0)
    )

    assert result.qualified is True
    assert result.qualification_mode == "e2e_quality_deferred"
    assert result.compiled is None and result.correct is None
    assert result.performance_valid is False and result.sample_count == 0
    assert result.s50 is None and result.s99 is None and result.srobust is None
    assert result.kernel_reward_available is False
    assert result.evidence["promotion_authority"]["correctness"] == "unchanged_magpie_quality_gate"


def test_production_composition_injects_reviewed_qwen_source_delivery(
    monkeypatch, tmp_path
):
    magpie = tmp_path / "Magpie"
    tracelens = tmp_path / "TraceLens"
    inferencex = tmp_path / "InferenceX"
    magpie.mkdir()
    tracelens.mkdir()
    inferencex.mkdir()
    vllm = tmp_path / "verified-vllm"
    aiter = tmp_path / "verified-aiter"
    vllm.mkdir()
    aiter.mkdir()
    source_receipt = SourceLockReceipt(
        "apex.e2e-source-locks.receipt/v1",
        tmp_path / "source-lock.json",
        "d" * 64,
        {
            "vllm": SourceCheckoutReceipt(
                "vLLM", vllm, "https://github.com/vllm-project/vllm.git",
                "1" * 40, "2" * 40, "managed"
            ),
            "aiter": SourceCheckoutReceipt(
                "AITER", aiter, "https://github.com/ROCm/aiter.git",
                "3" * 40, "4" * 40, "managed"
            ),
        },
    )
    receipt = DependencyReceipt(
        "apex.dependency-receipt.v1",
        "f" * 64,
        Path("/verified/python"),
        {"magpie": magpie, "tracelens": tracelens, "inferencex": inferencex},
        {"magpie": "a" * 40, "tracelens": "b" * 40, "inferencex": "c" * 40},
        {},
        source_locks=source_receipt,
    )
    monkeypatch.setattr(
        application_bootstrap, "verify_runtime_dependencies", lambda: receipt
    )

    application = application_bootstrap.build_application(
        include_e2e=True, knowledge_enabled=False
    )

    assert application.e2e_optimizer is not None
    assert isinstance(application.e2e_optimizer._candidate_worker, AgentCandidateWorker)
    assert isinstance(application.e2e_optimizer._micro, E2EDeferredMicroQualifier)
    assert isinstance(application.e2e_optimizer._deployments, DockerOverlayDeployment)
    assert isinstance(application.e2e_optimizer._final_delivery, SourceRebuildFinalDelivery)
    for binding in application.e2e_optimizer._final_delivery._bindings:
        assert binding.verification_source_overrides == {
            name: {"vllm": vllm, "aiter": aiter}[name]
            for name in binding.profile.repository_ids
        }


def test_production_composition_uses_explicit_trusted_final_delivery(
    monkeypatch, tmp_path
):
    magpie = tmp_path / "Magpie"
    tracelens = tmp_path / "TraceLens"
    inferencex = tmp_path / "InferenceX"
    magpie.mkdir()
    tracelens.mkdir()
    inferencex.mkdir()
    receipt = DependencyReceipt(
        "apex.dependency-receipt.v1",
        "f" * 64,
        Path("/verified/python"),
        {"magpie": magpie, "tracelens": tracelens, "inferencex": inferencex},
        {"magpie": "a" * 40, "tracelens": "b" * 40, "inferencex": "c" * 40},
        {},
    )
    monkeypatch.setattr(
        application_bootstrap, "verify_runtime_dependencies", lambda: receipt
    )
    trusted = OverlayOnlyFinalDelivery()

    application = application_bootstrap.build_application(
        include_e2e=True,
        knowledge_enabled=False,
        e2e_final_delivery=trusted,
    )

    assert application.e2e_optimizer is not None
    assert application.e2e_optimizer._final_delivery is trusted


def test_deferred_contract_rejects_a_fabricated_kernel_grade():
    with pytest.raises(ContractError, match="cannot carry"):
        MicroQualification(
            candidate_id="candidate-1",
            grade=object(),  # type: ignore[arg-type]
            evidence={},
            qualification_mode="e2e_quality_deferred",
        )


def test_overlay_success_attests_loaded_bytes_and_changes_only_image(tmp_path: Path):
    opportunity, candidate, request, engine = _fixture(tmp_path)
    deployment = DockerOverlayDeployment(engine, FakeLockVerifier()).deploy(request)

    assert deployment.qualified is True
    assert deployment.validation_level is ValidationLevel.RUNTIME_OVERLAY_VERIFIED
    assert deployment.deployed_source_sha256 == candidate.candidate_source_sha256
    assert engine.calls == ["inspect", "resolve", "build", "read"]
    assert deployment.evidence["formal_source_rebuild"] is False
    for source, derived in zip(
        (request.benchmark_measurement, request.benchmark_diagnostic, request.benchmark_replay),
        (deployment.measurement_config, deployment.diagnostic_config, deployment.replay_config),
        strict=True,
    ):
        before = yaml.safe_load(source.read_text())
        after = yaml.safe_load(derived.read_text())
        assert after["benchmark"]["docker_image"] == DERIVED_ID
        after["benchmark"]["docker_image"] = before["benchmark"]["docker_image"]
        assert after == before


def test_docker_engine_uses_fixed_argv_and_immutable_parent(tmp_path: Path):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("VALUE = 2\n", encoding="utf-8")
    supervisor = FakeDockerSupervisor(sha256_file(candidate))
    engine = DockerEngine(supervisor)  # type: ignore[arg-type]

    parent = engine.inspect_image("example:v1", cwd=tmp_path)
    target = engine.resolve_python_target(
        parent.image_id,
        library="vllm",
        repo_relative_path="vllm/kernels/op.py",
        cwd=tmp_path,
    )
    built = engine.build_overlay(
        parent_image_id=parent.image_id,
        candidate_source=candidate,
        target=target,
        build_root=tmp_path / "build",
        cwd=tmp_path,
    )
    loaded = engine.read_file(
        built.image.image_id, container_path=target.container_path, cwd=tmp_path
    )

    assert loaded.sha256 == sha256_file(candidate)
    assert all(command[0] == "docker" for command in supervisor.argv)
    assert all("sh" not in command and "bash" not in command for command in supervisor.argv)
    assert all("--volume" not in command and "-v" not in command for command in supervisor.argv)
    dockerfile = (tmp_path / "build" / "context" / "Dockerfile").read_text()
    assert dockerfile.startswith(f"FROM {PARENT_ID}\n")
    assert target.container_path in dockerfile


def test_overlay_rejects_loaded_old_candidate_bytes(tmp_path: Path):
    _, _, request, engine = _fixture(tmp_path)
    engine.loaded_sha = engine.baseline_sha

    deployment = DockerOverlayDeployment(engine, FakeLockVerifier()).deploy(request)

    assert deployment.deployed is False
    assert deployment.reason_code == "loaded_candidate_bytes_mismatch"
    assert not (request.artifact_root / "configs").exists()


def test_overlay_rejects_installed_parent_that_does_not_match_source_lock(tmp_path: Path):
    _, _, request, engine = _fixture(tmp_path)
    engine.baseline_sha = "f" * 64

    deployment = DockerOverlayDeployment(engine, FakeLockVerifier()).deploy(request)

    assert deployment.deployed is False
    assert deployment.reason_code == "installed_baseline_source_mismatch"
    assert engine.calls == ["inspect", "resolve"]


def test_overlay_rejects_mutable_parent_identity_drift(tmp_path: Path):
    _, _, request, engine = _fixture(tmp_path)
    engine.inspected_parent_id = "sha256:" + "9" * 64

    deployment = DockerOverlayDeployment(engine, FakeLockVerifier()).deploy(request)

    assert deployment.reason_code == "image_identity_mismatch"
    assert engine.calls == ["inspect"]


@pytest.mark.parametrize(
    ("library", "suffix"),
    [("unknown", ".py"), ("vllm", ".hip")],
)
def test_overlay_does_not_support_unknown_library_or_non_python_source(
    tmp_path: Path, library: str, suffix: str
):
    opportunity, _, request, engine = _fixture(tmp_path, library=library, suffix=suffix)
    adapter = DockerOverlayDeployment(engine, FakeLockVerifier())

    assert adapter.supports(opportunity, request.provenance) is False


def test_overlay_rejects_wrong_repo_to_package_mapping(tmp_path: Path):
    opportunity, candidate, request, engine = _fixture(tmp_path)
    wrong = request.opportunity.source_root / "kernels" / "op.py"
    wrong.parent.mkdir(exist_ok=True)
    wrong.write_text("BASELINE = True\n", encoding="utf-8")
    wrong_opportunity = _opportunity(
        request.opportunity.source_root, wrong, library="vllm", language="triton"
    )
    changed = "kernels/op.py"
    wrong_candidate = E2ECandidate(
        candidate.attempt_id,
        candidate.candidate_id,
        True,
        candidate.reason_code,
        candidate.workspace,
        (changed,),
        (changed,),
        candidate.baseline_source_sha256,
        candidate.candidate_source_sha256,
        candidate.agent_result,
    )
    wrong_request = CandidateDeploymentRequest(
        request.run_id,
        wrong_candidate,
        wrong_opportunity,
        request.provenance,
        request.benchmark_measurement,
        request.benchmark_diagnostic,
        request.workload_semantics_sha256,
        request.artifact_root,
        request.anchor_generation,
        request.safety,
        request.benchmark_replay,
    )

    result = DockerOverlayDeployment(engine, FakeLockVerifier()).deploy(wrong_request)

    assert result.reason_code == "source_mapping_mismatch"
    assert engine.calls == []


def test_overlay_rejects_container_probe_mapping_for_another_file(tmp_path: Path):
    _, _, request, engine = _fixture(tmp_path)
    engine.returned_repo_relative = "vllm/kernels/other.py"

    result = DockerOverlayDeployment(engine, FakeLockVerifier()).deploy(request)

    assert result.reason_code == "source_mapping_mismatch"
    assert engine.calls == ["inspect", "resolve"]


def test_overlay_config_rejects_workload_mutation(tmp_path: Path):
    paths, semantics = _views(tmp_path)
    mutated = yaml.safe_load(paths[2].read_text())
    mutated["benchmark"]["envs"]["TP"] = 8
    paths[2].write_text(yaml.safe_dump(mutated), encoding="utf-8")

    with pytest.raises(IntegrityError, match="semantics") as caught:
        derive_overlay_configs(
            measurement=paths[0],
            diagnostic=paths[1],
            replay=paths[2],
            output_dir=tmp_path / "derived",
            image_id=DERIVED_ID,
            workload_semantics_sha256=semantics,
        )
    assert caught.value.reason_code == "benchmark_semantics_changed"


def test_overlay_final_delivery_never_claims_source_rebuild(tmp_path: Path):
    result = OverlayOnlyFinalDelivery().finalize(
        FinalDeliveryRequest(
            "run-1",
            (),
            _provenance(tmp_path, "vllm"),
            tmp_path / "original",
            tmp_path / "measurement",
            tmp_path / "diagnostic",
            tmp_path / "replay",
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            tmp_path / "artifacts",
        )
    )

    assert result.verified is False
    assert result.status is TaskStatus.PROVENANCE_UNRESOLVED
    assert result.validation_level is ValidationLevel.NONE
    assert result.clean_replay_verified is False
