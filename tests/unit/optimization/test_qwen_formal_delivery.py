from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import yaml
import pytest

import apex.optimization.e2e.qwen_profile as qwen_profile
from apex.benchmark import (
    InferenceXRuntimeEvidence,
    LatencyDistribution,
    LatencyMetrics,
    ModelRevisionEvidence,
    NormalizedBenchmarkResult,
    QualityEvidence,
    QualityMetric,
    ThroughputMetrics,
)
from apex.core import ContractError, sha256_file, sha256_json
from apex.delivery import (
    BuildRecipeLock,
    BuildStep,
    BuildStepReceipt,
    BuiltArtifact,
    DerivedImageIdentity,
    LoadedArtifact,
    LoadedByteEngagementReceipt,
    ReplayConfigInvariantReceipt,
    ReplayRequest,
    SourceBuildRequest,
)
from apex.delivery.git_patch import RepositoryApplyReceipt
from apex.evaluation import E2EMeasurement
from apex.execution import ProcessResult, SubprocessSupervisor
from apex.optimization.e2e.qwen_profile import (
    QWEN_CONFIG_SHA256,
    QWEN_MODEL_ID,
    QWEN_MODEL_REVISION,
    QWEN_PARENT_IMAGE_ID,
    QWEN_PARENT_LOCATOR,
    QWEN_PARENT_REFERENCE,
    QWEN_PARENT_REPO_DIGEST,
    QWEN_SOURCE_DATE_EPOCH,
    QwenAcceptanceProvenance,
    QwenAcceptanceProvenanceResolver,
    _profiles,
    build_qwen_correctness_oracles,
)
from apex.optimization.e2e.source_delivery_adapters import (
    QwenIndependentReplay,
    QwenIndependentSourceBuild,
    QwenPrimarySourceBuilder,
)
from apex.optimization.e2e.source_delivery_models import PrimarySourceBuildRequest
from apex.optimization.e2e.source_image_runtime import (
    DockerPythonSourceImageBuilder,
    SourceImageBuild,
)
from apex.ports import BenchmarkPass
from apex.runtime import ContainerIdentity, RepositoryLock, RunProvenance


DERIVED = "sha256:" + "d" * 64
STACK = "c" * 64
VLLM_COMMIT = "b1388b1fbf5aaef47937fabe98931211684666a6"
VLLM_TREE = "33b782e425e42d42851a33f7876e97a8deeabb29"
AITER_COMMIT = "c3708fb7445899c14cdc6e8055953ee02ed78ddf"
AITER_TREE = "a30409ac03524781f175cbb03e82eefcafd52af1"


def test_reviewed_qwen_config_hash_matches_pinned_magpie_checkout() -> None:
    apex_root = Path(__file__).resolve().parents[3]
    config = (
        apex_root.parent
        / "Magpie"
        / "examples"
        / "benchmarks"
        / "benchmark_vllm_qwen3_next_80b_fp8.yaml"
    )
    assert config.is_file(), "the pinned Magpie checkout is required for this contract test"
    assert sha256_file(config) == QWEN_CONFIG_SHA256


def _process(argv, *, stdout: str = "") -> ProcessResult:
    return ProcessResult(tuple(argv), 0, False, stdout, "", False, False, 0.01)


class HybridCommands:
    """Execute Git for real and emulate only the Docker process boundary."""

    def __init__(self) -> None:
        self.real = SubprocessSupervisor(max_output_bytes=4 * 1024 * 1024)
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        argv,
        *,
        cwd: Path,
        environment,
        timeout_seconds: int,
        stdin_text: str | None = None,
    ) -> ProcessResult:
        command = tuple(argv)
        self.calls.append(command)
        if command[0] == "git":
            return self.real.run(
                command,
                cwd=cwd,
                environment=environment,
                timeout_seconds=timeout_seconds,
                stdin_text=stdin_text,
            )
        if command[:3] == ("docker", "image", "inspect"):
            image = QWEN_PARENT_IMAGE_ID if command[-1] == QWEN_PARENT_LOCATOR else DERIVED
            return _process(command, stdout=json.dumps([{"Id": image}]))
        if command[:3] == ("docker", "buildx", "build"):
            iidfile = Path(command[command.index("--iidfile") + 1])
            iidfile.write_text(DERIVED + "\n", encoding="utf-8")
            return _process(command, stdout="deterministic build\n")
        if command[:2] == ("docker", "run"):
            script = command[command.index("-c") + 1]
            if "manifest_path" in script:
                expected = command[-2]
                roots = json.loads(command[-1])
                payload = {
                    "manifest_sha256": expected,
                    "packages": {
                        item: {"module_file": f"/opt/apex/python/{item}/__init__.py"}
                        for item in roots
                    },
                }
            else:
                payload = {
                    "artifacts": [
                        {
                            "module": item["module"],
                            "path": item["path"],
                            "sha256": item["sha256"],
                            "loaded": True,
                        }
                        for item in json.loads(command[-1])
                    ]
                }
            return _process(
                command,
                stdout="__APEX_FORMAL_SOURCE_V1__"
                + json.dumps(payload, sort_keys=True)
                + "\n",
            )
        raise AssertionError(f"unexpected command: {command}")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()


def _source_repository(tmp_path: Path) -> Path:
    root = tmp_path / "vllm-source"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "apex@example.invalid")
    _git(root, "config", "user.name", "Apex Test")
    package = root / "vllm"
    package.mkdir()
    (package / "__init__.py").write_text('VERSION = "base"\n', encoding="utf-8")
    (package / "kernel.py").write_text('VALUE = "base"\n', encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")
    (package / "kernel.py").write_text('VALUE = "optimized"\n', encoding="utf-8")
    return root.resolve()


def _recipe() -> BuildRecipeLock:
    return BuildRecipeLock(
        "qwen-test-vllm-source-v1",
        QWEN_PARENT_IMAGE_ID,
        "apex/qwen-test-source",
        (BuildStep(("git", "diff", "--check", "HEAD", "--", "vllm/"), "vllm", timeout_seconds=120),),
    )


def test_source_image_is_reproducible_and_uses_fixed_networkless_argv(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    commands = HybridCommands()
    builder = DockerPythonSourceImageBuilder(
        parent_locator=QWEN_PARENT_LOCATOR,
        parent_image_id=QWEN_PARENT_IMAGE_ID,
        source_date_epoch=QWEN_SOURCE_DATE_EPOCH,
        commands=commands,
    )

    first = builder.build(
        recipe=_recipe(),
        repository_roots={"vllm": source},
        source_stack_sha256=STACK,
        output_dir=(tmp_path / "build-one").resolve(),
    )
    second = builder.build(
        recipe=_recipe(),
        repository_roots={"vllm": source},
        source_stack_sha256=STACK,
        output_dir=(tmp_path / "build-two").resolve(),
    )

    assert first.image == second.image
    assert sha256_file(tmp_path / "build-one" / "source-layer.tar") == sha256_file(
        tmp_path / "build-two" / "source-layer.tar"
    )
    assert first.sbom_path.read_bytes() == second.sbom_path.read_bytes()
    assert first.artifacts[0].runtime_path == "/opt/apex/python/vllm/kernel.py"
    assert builder.engage(
        bundle_digest="b" * 64,
        image=first.image,
        source_stack_sha256=STACK,
        artifacts=first.artifacts,
        cwd=tmp_path.resolve(),
    ).verified
    builds = [item for item in commands.calls if item[:3] == ("docker", "buildx", "build")]
    assert len(builds) == 2
    for argv in builds:
        assert "--no-cache" in argv
        assert "--pull=false" in argv
        assert "--network=none" in argv
        assert "--provenance=false" in argv
        assert "--sbom=false" in argv
        assert "type=docker,rewrite-timestamp=true" in argv


def _run_provenance(roots: dict[str, Path]) -> RunProvenance:
    return RunProvenance(
        1,
        "/immutable/qwen.yaml",
        QWEN_CONFIG_SHA256,
        "vllm",
        QWEN_MODEL_ID,
        QWEN_MODEL_REVISION,
        "gfx950",
        ContainerIdentity(
            QWEN_PARENT_REFERENCE,
            QWEN_PARENT_IMAGE_ID,
            (f"vllm/vllm-openai-rocm@{QWEN_PARENT_REPO_DIGEST}",),
            (),
        ),
        ("vllm", "aiter"),
        (
            RepositoryLock(
                "vllm",
                str(roots["vllm"]),
                "https://github.com/vllm-project/vllm.git",
                VLLM_COMMIT,
                VLLM_TREE,
                True,
            ),
            RepositoryLock(
                "aiter",
                str(roots["aiter"]),
                "https://github.com/ROCm/aiter.git",
                AITER_COMMIT,
                AITER_TREE,
                True,
            ),
        ),
        "resolved",
        (),
    )


def test_qwen_profile_accepts_only_the_reviewed_identity(tmp_path: Path) -> None:
    roots = {"vllm": tmp_path / "vllm", "aiter": tmp_path / "aiter"}
    for path in roots.values():
        path.mkdir()
    provenance = _run_provenance(roots)
    guard = QwenAcceptanceProvenance(roots)

    guard._validate_run(provenance)
    for drifted in (
        replace(provenance, benchmark_config_sha256="0" * 64),
        replace(provenance, model_revision="other"),
        replace(provenance, gpu_arch="gfx942"),
        replace(
            provenance,
            container=replace(provenance.container, image_id="sha256:" + "0" * 64),
        ),
    ):
        try:
            guard._validate_run(drifted)
        except ContractError:
            pass
        else:
            raise AssertionError("reviewed Qwen provenance accepted identity drift")

    profiles = _profiles()
    assert [item.repository_ids for item in profiles] == [
        frozenset({"vllm"}),
        frozenset({"aiter"}),
        frozenset({"vllm", "aiter"}),
    ]
    assert all(item.recipe.parent_image_digest == QWEN_PARENT_IMAGE_ID for item in profiles)
    assert all(
        step.argv
        == ("git", "diff", "--check", "HEAD", "--", f"{step.repository_id}/")
        for profile in profiles
        for step in profile.recipe.steps
    )


def test_qwen_oracles_are_source_relative_and_source_lock_bound(tmp_path: Path) -> None:
    roots = {"vllm": tmp_path / "vllm", "aiter": tmp_path / "aiter"}
    roots["aiter"].mkdir()
    sources = {
        "vllm/v1/attention/ops/chunked_prefill_paged_decode.py": (
            "tests/kernels/attention/test_prefix_prefill.py"
        ),
        "vllm/model_executor/layers/fla/ops/fused_recurrent.py": (
            "tests/kernels/test_fused_recurrent_packed_decode.py"
        ),
        "vllm/model_executor/layers/mamba/ops/causal_conv1d.py": (
            "tests/kernels/mamba/test_causal_conv1d.py"
        ),
        "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py": (
            "tests/kernels/attention/test_cache.py"
        ),
        "vllm/v1/attention/ops/prefix_prefill.py": (
            "tests/kernels/attention/test_prefix_prefill.py"
        ),
    }
    extra_tests = {
        "tests/kernels/test_fused_sigmoid_gating_delta_rule.py",
        "tests/__init__.py",
        "tests/kernels/__init__.py",
        "tests/kernels/utils.py",
        "tests/kernels/quant_utils.py",
        "tests/kernels/attention/conftest.py",
    }
    for relative in set(sources).union(sources.values()).union(extra_tests):
        path = roots["vllm"] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("pass\n", encoding="utf-8")

    registry = build_qwen_correctness_oracles(source_roots=roots)

    assert len(registry.policy_sha256) == 64
    for source, test in sources.items():
        resolved = registry.resolve(
            repository_id="vllm",
            source_root=roots["vllm"],
            source_path=roots["vllm"] / source,
        )
        assert resolved is not None
        assert resolved.test_file == roots["vllm"] / test
        assert len(resolved.binding_sha256) == 64
        assert resolved.execution_mode == "routing_only"
        assert resolved.test_argv[:3] == ("python3", "-m", "pytest")
        assert "-k" not in resolved.test_argv
        assert any(item.startswith(f"{test}::") for item in resolved.test_argv)


class FakeProvenanceResolver:
    def __init__(self, observed: RunProvenance) -> None:
        self.observed = observed
        self.hints = None

    def resolve(self, config_path, *, gpu_arch, hints=None):
        self.hints = hints
        return replace(
            self.observed,
            benchmark_config_path=str(config_path),
            benchmark_config_sha256=sha256_file(config_path),
            gpu_arch=gpu_arch,
        )


def test_exact_config_injects_reviewed_source_and_model_locks(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "qwen.yaml"
    config.write_text("benchmark: exact-qwen-test\n", encoding="utf-8")
    roots = {"vllm": tmp_path / "vllm", "aiter": tmp_path / "aiter"}
    for path in roots.values():
        path.mkdir()
    delegate = FakeProvenanceResolver(_run_provenance(roots))
    monkeypatch.setattr(qwen_profile, "QWEN_CONFIG_SHA256", sha256_file(config))
    resolver = QwenAcceptanceProvenanceResolver(roots, delegate)

    observed = resolver.resolve(config, gpu_arch="gfx950")

    assert observed.model_revision == QWEN_MODEL_REVISION
    assert delegate.hints["model_revision"] == QWEN_MODEL_REVISION
    assert {
        item["name"]: (item["path"], item["commit"])
        for item in delegate.hints["source_repositories"]
    } == {
        "vllm": (str(roots["vllm"].resolve()), VLLM_COMMIT),
        "aiter": (str(roots["aiter"].resolve()), AITER_COMMIT),
    }
    with pytest.raises(ContractError, match="revision override"):
        resolver.resolve(
            config,
            gpu_arch="gfx950",
            hints={"model_revision": "mutable-main"},
        )


def _configs(tmp_path: Path) -> tuple[dict[str, Path], str]:
    tracelens = (tmp_path / "tracelens").resolve()
    tracelens.mkdir(exist_ok=True)
    benchmark = {
        "framework": "vllm",
        "model": QWEN_MODEL_ID,
        "docker_image": QWEN_PARENT_REFERENCE,
        "envs": {"RUN_EVAL": "true", "MAGPIE_EVAL_TASKS": "gsm8k"},
        "lm_eval_runtime": {
            "path": "/runtime/lm-eval",
            "sha256": "8" * 64,
            "identity": {"commit": "9" * 40},
        },
        "profiler": {
            "torch_profiler": {"enabled": False},
            "tracelens": {
                "enabled": False,
                "tracelens_repo_path": str(tracelens),
            },
            "targeted_trace": {"enabled": False, "targets": []},
        },
        "gap_analysis": {"enabled": False},
    }
    projected = copy.deepcopy(benchmark)
    for field in ("docker_image", "profiler", "gap_analysis"):
        projected.pop(field)
    semantics = sha256_json(projected)
    paths = {}
    for role in ("original", "measurement", "diagnostic", "replay"):
        benchmark_view = copy.deepcopy(benchmark)
        benchmark_view["run_kind"] = (
            "diagnostic" if role == "diagnostic" else "measurement"
        )
        quality = {
            "required": True,
            "kind": "lm_eval",
            "tasks": "gsm8k",
            "evaluator_policy": None,
        }
        if role == "diagnostic":
            benchmark_view["envs"]["RUN_EVAL"] = "false"
            benchmark_view.pop("lm_eval_runtime")
            benchmark_view["profiler"]["torch_profiler"]["enabled"] = True
            benchmark_view["profiler"]["tracelens"]["enabled"] = True
            benchmark_view["profiler"]["targeted_trace"] = {
                "enabled": True,
                "targets": [{"name_patterns": ["*"]}],
            }
            benchmark_view["gap_analysis"]["enabled"] = True
            quality = {
                "required": False,
                "kind": "trace_only",
                "tasks": "gsm8k",
                "evaluator_policy": None,
            }
        document = {
            "benchmark": benchmark_view,
            "apex": {
                "benchmark_view": {
                    "schema": "apex.benchmark-view.v1",
                    "kind": role,
                    "original_sha256": "a" * 64,
                    "workload_semantics_sha256": semantics,
                    "dependencies": {
                        "receipt_schema": "apex.dependency-receipt.v1",
                        "lock_sha256": "b" * 64,
                        "python": "/usr/bin/python3",
                        "magpie": {"root": "/magpie", "commit": "1" * 40},
                        "tracelens": {
                            "root": str(tracelens),
                            "commit": "2" * 40,
                        },
                        "inferencex": {
                            "root": "/inferencex",
                            "commit": "3" * 40,
                        },
                    },
                    "quality_contract": quality,
                }
            },
        }
        path = (tmp_path / "configs" / f"{role}.yaml").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        paths[role] = path
    return paths, semantics


class FakeBenchmark:
    def __init__(self, throughput: float) -> None:
        self.throughput = throughput
        self.calls = []

    def run_normalized(self, request) -> NormalizedBenchmarkResult:
        self.calls.append(request)
        workspace = request.output_dir / f"measurement-{len(self.calls)}"
        workspace.mkdir(parents=True)
        report = workspace / "benchmark_report.json"
        report.write_text('{"source":"magpie"}\n', encoding="utf-8")
        quality = workspace / "quality.json"
        quality.write_text('{"accuracy":1.0}\n', encoding="utf-8")
        distribution = LatencyDistribution(1.0, 1.0, 10.0, 0.0)
        return NormalizedBenchmarkResult(
            schema_version=1,
            run_id=request.run_id,
            pass_type=request.pass_type,
            succeeded=True,
            framework="vllm",
            model=QWEN_MODEL_ID,
            workspace_path=workspace,
            report_path=report,
            throughput=ThroughputMetrics(1.0, self.throughput, self.throughput, 32, 1.0),
            latency=LatencyMetrics(distribution, distribution, distribution, distribution),
            quality=QualityEvidence(
                True,
                "lm_eval",
                True,
                (QualityMetric("gsm8k", "exact_match,strict-match", 1.0, True),),
                (quality,),
            ),
            profiling_enabled=False,
            run_kind="measurement",
            reward_eligible=True,
            model_revision=ModelRevisionEvidence(
                True,
                True,
                QWEN_MODEL_REVISION,
                QWEN_MODEL_REVISION,
                None,
            ),
            inferencex_runtime=InferenceXRuntimeEvidence(
                True,
                True,
                Path("/data/viouyang/apex/runtime/InferenceX"),
                "3" * 40,
                workspace / "inferencex_runtime",
                None,
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            artifacts=(report, quality),
            errors=(),
            command_exit_code=0,
        )


class FakeImages:
    def __init__(self) -> None:
        self.builds = 0

    def build(self, *, recipe, repository_roots, source_stack_sha256, output_dir):
        self.builds += 1
        output_dir.mkdir(parents=True)
        sbom = output_dir / "image.spdx.json"
        sbom.write_text('{"spdxVersion":"SPDX-2.3"}\n', encoding="utf-8")
        artifact = BuiltArtifact(
            "vllm",
            "/opt/apex/python/vllm/kernel.py",
            "f" * 64,
            None,
            source_stack_sha256,
        )
        step = recipe.steps[0]
        receipt = BuildStepReceipt(
            0,
            step.repository_id,
            step.cwd,
            sha256_json(list(step.argv)),
            0,
            False,
            "1" * 64,
            "2" * 64,
        )
        image = DerivedImageIdentity(DERIVED, QWEN_PARENT_IMAGE_ID, DERIVED, sha256_file(sbom))
        return SourceImageBuild(
            image,
            sbom.resolve(),
            (artifact,),
            (receipt,),
            {"schema": "apex.qwen-source-build/v1"},
            {"verified": True},
        )

    def engage(self, *, bundle_digest, image, source_stack_sha256, artifacts, cwd):
        loaded = tuple(
            LoadedArtifact(
                item.component,
                item.runtime_path,
                item.sha256,
                item.sha256,
                None,
                None,
                "python_import",
                "vllm.kernel",
                True,
            )
            for item in artifacts
        )
        return LoadedByteEngagementReceipt(
            bundle_digest, image.image_digest, source_stack_sha256, True, loaded
        )


def _measurement(throughput: float, protocol: str, receipt: str) -> E2EMeasurement:
    return E2EMeasurement(throughput, 10.0, 10.0, 1.0, 32, protocol, receipt, receipt)


def _repository_receipt() -> RepositoryApplyReceipt:
    return RepositoryApplyReceipt(
        "vllm",
        "1" * 40,
        "2" * 40,
        "3" * 40,
        "4" * 64,
        True,
        True,
        True,
        True,
        True,
        True,
    )


def test_primary_build_and_independent_replay_require_real_receipts(
    tmp_path: Path,
) -> None:
    configs, protocol = _configs(tmp_path)
    images = FakeImages()
    primary_benchmark = FakeBenchmark(102.0)
    primary = QwenPrimarySourceBuilder(images, primary_benchmark)
    request = PrimarySourceBuildRequest(
        "formal-run",
        STACK,
        _recipe(),
        {"vllm": tmp_path},
        (),
        configs["original"],
        configs["measurement"],
        configs["diagnostic"],
        configs["replay"],
        _measurement(100.0, protocol, "baseline"),
        _measurement(101.0, protocol, "overlay"),
        (tmp_path / "primary").resolve(),
    )

    output = primary.build_and_validate(request)

    assert output.engagement_verified
    assert output.normal_runtime_measurement
    assert output.accuracy_passed
    assert output.latency_gates_passed
    assert output.objective_improved
    assert output.overlay_rebuild_parity_passed
    assert not output.safety_certified
    assert set(output.primary_receipts) == {
        "primary_build_receipt",
        "primary_engagement_receipt",
        "primary_benchmark_receipt",
        "primary_safety_receipt",
    }

    repository_receipt = _repository_receipt()
    independent = QwenIndependentSourceBuild(images).build(
        SourceBuildRequest(
            "b" * 64,
            STACK,
            _recipe(),
            output.derived_image,
            {"vllm": tmp_path},
            (repository_receipt,),
            (tmp_path / "independent-build").resolve(),
        )
    )
    assert independent.verified

    config_receipt = ReplayConfigInvariantReceipt(
        sha256_file(output.benchmark_measurement),
        sha256_file(output.benchmark_replay),
        protocol,
        output.derived_image.locator,
        True,
    )
    replay = QwenIndependentReplay(FakeBenchmark(102.1)).replay(
        ReplayRequest(
            "b" * 64,
            STACK,
            output.environment_id,
            output.derived_image,
            output.benchmark_replay,
            config_receipt,
            images.engage(
                bundle_digest="b" * 64,
                image=output.derived_image,
                source_stack_sha256=STACK,
                artifacts=independent.artifacts,
                cwd=tmp_path,
            ),
            (repository_receipt,),
            output.primary_receipts,
            (tmp_path / "independent-replay").resolve(),
        )
    )
    assert replay.verified
    assert replay.primary_environment_id != replay.replay_environment_id
