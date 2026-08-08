"""Exact formal-delivery composition for the Qwen3-Next acceptance workload."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

from apex.benchmark import MagpieBenchmarkAdapter, QWEN_CONFIG_SHA256
from apex.core import ContractError, IntegrityError, sha256_file
from apex.delivery import BuildRecipeLock, BuildStep, E2EBundleVerifier
from apex.runtime import (
    DependencyReceipt,
    ProvenanceResolver,
    RunProvenance,
    canonical_repository,
    default_source_lock_path,
    default_source_roots,
    load_source_lock,
)

from .oracles import CorrectnessOracleBinding, CorrectnessOracleRegistry
from .deferred import E2EDeferredMicroQualifier
from .oracle_preflight import (
    DockerOracleMicroQualifier,
    DockerOraclePolicy,
    OracleDependencyLock,
    OracleSourceLock,
)
from .source_delivery import FormalDeliveryBinding, SourceRebuildFinalDelivery
from .source_delivery_models import FormalRepositoryProfile, FormalSourceDeliveryProfile
from .source_delivery_provenance import ExactRequestProvenance
from .source_delivery_adapters import (
    QwenIndependentEngagement,
    QwenIndependentReplay,
    QwenIndependentSourceBuild,
    QwenPrimarySourceBuilder,
)
from .source_image_runtime import DockerPythonSourceImageBuilder
from .qwen_qualification import QwenCompositeMicroQualifier


QWEN_MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
QWEN_MODEL_REVISION = "c5f5f263bdd5cc134092897864e8905d8fe7b928"
QWEN_PARENT_REFERENCE = "vllm/vllm-openai-rocm:v0.19.1"
QWEN_PARENT_LOCATOR = (
    "vllm/vllm-openai-rocm@"
    "sha256:c3457ab4702a5bd665b06d7ba57e6105fe98adc4f5b3d4afcf98ec45551988e0"
)
QWEN_PARENT_REPO_DIGEST = (
    "sha256:c3457ab4702a5bd665b06d7ba57e6105fe98adc4f5b3d4afcf98ec45551988e0"
)
QWEN_PARENT_IMAGE_ID = "sha256:b599932816fe09f9ea2541655f5388457ac2494b87b551cefdbf2a207b0ed3a9"
QWEN_SOURCE_DATE_EPOCH = 1776474762


class ProvenanceResolverPort(Protocol):
    def resolve(
        self,
        config_path: Path,
        *,
        gpu_arch: str,
        hints: Mapping[str, Any] | None = None,
    ) -> RunProvenance: ...


class QwenAcceptanceProvenanceResolver:
    """Inject reviewed locks only when the exact acceptance config is selected."""

    def __init__(
        self,
        source_roots: Mapping[str, Path],
        delegate: ProvenanceResolverPort | None = None,
    ) -> None:
        identities = _source_identities()
        if set(source_roots) != set(identities):
            raise ContractError("Qwen source roots are incomplete", "source_lock_unresolved")
        self.source_roots = {
            name: path.expanduser().resolve() for name, path in source_roots.items()
        }
        self.delegate = delegate or ProvenanceResolver()

    def resolve(
        self,
        config_path: Path,
        *,
        gpu_arch: str,
        hints: Mapping[str, Any] | None = None,
    ) -> RunProvenance:
        chosen = dict(hints or {})
        if not config_path.is_file() or sha256_file(config_path) != QWEN_CONFIG_SHA256:
            return self.delegate.resolve(
                config_path, gpu_arch=gpu_arch, hints=chosen
            )
        _validate_hint_overrides(chosen, self.source_roots)
        chosen.update(_reviewed_provenance_hints(self.source_roots))
        observed = self.delegate.resolve(
            config_path, gpu_arch=gpu_arch, hints=chosen
        )
        QwenAcceptanceProvenance(self.source_roots)._validate_run(observed)
        return observed


class QwenAcceptanceProvenance(ExactRequestProvenance):
    """Reject every workload or source identity outside the reviewed profile."""

    def __init__(self, source_roots: Mapping[str, Path]) -> None:
        self.source_roots = {
            name: path.expanduser().resolve() for name, path in source_roots.items()
        }

    def lock(self, request):
        self._validate_run(request.provenance)
        return super().lock(request)

    def _validate_run(self, provenance: RunProvenance) -> None:
        container = provenance.container
        expected_repo = f"vllm/vllm-openai-rocm@{QWEN_PARENT_REPO_DIGEST}"
        if (
            provenance.benchmark_config_sha256 != QWEN_CONFIG_SHA256
            or provenance.framework != "vllm"
            or provenance.model_id != QWEN_MODEL_ID
            or provenance.model_revision != QWEN_MODEL_REVISION
            or provenance.gpu_arch != "gfx950"
            or container.requested_image != QWEN_PARENT_REFERENCE
            or container.image_id != QWEN_PARENT_IMAGE_ID
            or expected_repo not in container.repo_digests
            or provenance.active_components != ("vllm", "aiter")
        ):
            raise ContractError(
                "Run does not match the reviewed Qwen acceptance identity",
                "untrusted_build_recipe",
            )
        locks = {item.name: item for item in provenance.source_locks}
        identities = _source_identities()
        if set(locks) != set(identities):
            raise ContractError("Qwen source locks are incomplete", "source_lock_unresolved")
        for name, identity in identities.items():
            lock = locks[name]
            if (
                not lock.exact
                or lock.commit != identity["commit"]
                or lock.tree != identity["tree"]
                or canonical_repository(lock.url)
                != canonical_repository(identity["url"])
                or Path(lock.path).resolve() != self.source_roots[name]
            ):
                raise IntegrityError("Reviewed Qwen source lock drifted", "source_lock_drift")


def build_qwen_acceptance_delivery(
    dependency_receipt: DependencyReceipt,
    *,
    source_roots: Mapping[str, Path] | None = None,
) -> SourceRebuildFinalDelivery:
    """Compose the sole reviewed live profile; no tag or repository fallback."""

    roots = _validated_source_roots(source_roots)
    profiles = _profiles()
    primary_images = _image_builder()
    primary = QwenPrimarySourceBuilder(
        primary_images, MagpieBenchmarkAdapter(dependency_receipt)
    )
    verifier = build_qwen_acceptance_bundle_verifier(
        dependency_receipt, source_roots=roots
    )
    bindings = tuple(
        FormalDeliveryBinding(
            profile,
            primary,
            verifier,
            {
                item.repository_id: roots[item.repository_id]
                for item in profile.repositories
            },
        )
        for profile in profiles
    )
    return SourceRebuildFinalDelivery(
        bindings, provenance=QwenAcceptanceProvenance(roots)
    )


def build_qwen_acceptance_bundle_verifier(
    dependency_receipt: DependencyReceipt,
    *,
    source_roots: Mapping[str, Path] | None = None,
) -> E2EBundleVerifier:
    """Compose the reviewed recipes and independent Qwen verification backends."""

    roots = _validated_source_roots(source_roots)
    profiles = _profiles()
    images = _image_builder()
    identities = _source_identities()
    return E2EBundleVerifier(
        trusted_recipes={item.recipe.computed_sha256: item.recipe for item in profiles},
        trusted_source_urls={name: value["url"] for name, value in identities.items()},
        build_backend=QwenIndependentSourceBuild(images),
        engagement_backend=QwenIndependentEngagement(images, _project_root()),
        replay_backend=QwenIndependentReplay(MagpieBenchmarkAdapter(dependency_receipt)),
        default_source_overrides=roots,
        trusted_recipe_repositories={
            item.recipe.computed_sha256: item.repository_ids for item in profiles
        },
    )


def default_qwen_source_roots() -> Mapping[str, Path]:
    return default_source_roots(_qwen_source_lock())


def _validated_source_roots(
    source_roots: Mapping[str, Path] | None,
) -> dict[str, Path]:
    roots = dict(source_roots or default_qwen_source_roots())
    if set(roots) != set(_source_identities()):
        raise ContractError("Qwen source roots are incomplete", "source_lock_unresolved")
    return roots


def build_qwen_acceptance_provenance_resolver(
    *, source_roots: Mapping[str, Path] | None = None
) -> QwenAcceptanceProvenanceResolver:
    return QwenAcceptanceProvenanceResolver(
        source_roots or default_qwen_source_roots()
    )


def build_qwen_correctness_oracles(
    *, source_roots: Mapping[str, Path] | None = None
) -> CorrectnessOracleRegistry:
    """Bind dynamically discovered Qwen kernels to reviewed vLLM tests."""

    roots = dict(source_roots or default_qwen_source_roots())
    return CorrectnessOracleRegistry(
        source_roots=roots,
        bindings=_qwen_oracle_bindings(),
        source_lock_sha256=_qwen_source_lock().sha256,
    )


def _qwen_oracle_bindings() -> tuple[CorrectnessOracleBinding, ...]:
    prefix_test = "tests/kernels/attention/test_prefix_prefill.py"
    recurrent_test = "tests/kernels/test_fused_recurrent_packed_decode.py"
    causal_test = "tests/kernels/mamba/test_causal_conv1d.py"
    cache_test = "tests/kernels/attention/test_cache.py"
    return (
        CorrectnessOracleBinding(
            "vllm",
            "vllm/v1/attention/ops/chunked_prefill_paged_decode.py",
            prefix_test,
            _oracle_argv(
                f"{prefix_test}::test_qwen3_nonstandard_block_size"
                "[chunked_prefill_paged_decode-cuda:0-dtype0-128]"
            ),
        ),
        CorrectnessOracleBinding(
            "vllm",
            "vllm/model_executor/layers/fla/ops/fused_recurrent.py",
            recurrent_test,
            _oracle_argv(
                f"{recurrent_test}::test_fused_recurrent_packed_decode_matches_reference"
                "[False-dtype1]",
                f"{recurrent_test}::test_fused_recurrent_packed_decode_matches_reference"
                "[True-dtype1]",
                "tests/kernels/test_fused_sigmoid_gating_delta_rule.py::"
                "test_fused_sigmoid_gating_delta_rule_update_non_spec"
                "[dtype1-128-128-32-16-1-1]",
            ),
            ("tests/kernels/test_fused_sigmoid_gating_delta_rule.py",),
            3,
        ),
        CorrectnessOracleBinding(
            "vllm",
            "vllm/model_executor/layers/mamba/ops/causal_conv1d.py",
            causal_test,
            _oracle_argv(
                f"{causal_test}::test_causal_conv1d_update"
                "[4096-4-1-True-True-itype0]",
                f"{causal_test}::test_causal_conv1d_update"
                "[2064-4-1-False-False-itype0]",
            ),
            expected_test_count=2,
        ),
        CorrectnessOracleBinding(
            "vllm",
            "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py",
            cache_test,
            _oracle_argv(
                f"{cache_test}::test_reshape_and_cache_flash"
                "[triton-tensor-NHD-auto-cuda:0-0-dtype0-1024-16-256-8-42]",
                f"{cache_test}::test_reshape_and_cache_flash"
                "[triton-tensor-NHD-fp8-cuda:0-0-dtype0-1024-32-256-8-42]",
            ),
            (
                "tests/__init__.py",
                "tests/kernels/__init__.py",
                "tests/kernels/utils.py",
                "tests/kernels/quant_utils.py",
                "tests/kernels/attention/conftest.py",
            ),
            2,
        ),
        CorrectnessOracleBinding(
            "vllm",
            "vllm/v1/attention/ops/prefix_prefill.py",
            prefix_test,
            _oracle_argv(
                f"{prefix_test}::test_qwen3_nonstandard_block_size"
                "[context_attention_fwd-cuda:0-dtype0-128]"
            ),
        ),
    )


def build_qwen_oracle_micro_qualifier(
    oracles: CorrectnessOracleRegistry,
) -> QwenCompositeMicroQualifier:
    """Bind strict vLLM and deferred AITER lanes to the reviewed Qwen parent."""

    identities = _source_identities()
    return QwenCompositeMicroQualifier(
        vllm=DockerOracleMicroQualifier(
            oracles=oracles,
            policy=DockerOraclePolicy(
                QWEN_PARENT_LOCATOR,
                QWEN_PARENT_IMAGE_ID,
                tuple(
                    OracleSourceLock(name, identity["commit"], identity["tree"])
                    for name, identity in sorted(identities.items())
                ),
                (
                    OracleDependencyLock("pytest", "9.0.2"),
                    OracleDependencyLock("einops", "0.8.2"),
                ),
            ),
        ),
        aiter=E2EDeferredMicroQualifier(),
    )


def _oracle_argv(*node_ids: str) -> tuple[str, ...]:
    """Return a reviewed tests-only argv; parent ``tests/conftest.py`` is excluded."""

    return (
        "python3",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--rootdir=/opt/apex-oracle",
        "--confcutdir=/opt/apex-oracle/tests/kernels",
        "--junitxml=/opt/apex-result/junit.xml",
        "-q",
        "-x",
        *node_ids,
    )


def _reviewed_provenance_hints(
    roots: Mapping[str, Path],
) -> dict[str, Any]:
    identities = _source_identities()
    return {
        "model_revision": QWEN_MODEL_REVISION,
        "source_repositories": [
            {
                "name": name,
                "path": str(roots[name]),
                "commit": str(identity["commit"]),
            }
            for name, identity in identities.items()
        ],
    }


def _validate_hint_overrides(
    hints: Mapping[str, Any], roots: Mapping[str, Path]
) -> None:
    identities = _source_identities()
    revision = hints.get("model_revision")
    if revision is not None and revision != QWEN_MODEL_REVISION:
        raise ContractError(
            "Qwen model revision override differs from the reviewed lock",
            "source_lock_drift",
        )
    repositories = hints.get("source_repositories")
    if repositories is None:
        return
    if not isinstance(repositories, (list, tuple)):
        raise ContractError("Qwen source override is invalid", "source_lock_drift")
    observed = {}
    for item in repositories:
        if not isinstance(item, Mapping):
            raise ContractError("Qwen source override is invalid", "source_lock_drift")
        name = str(item.get("name", ""))
        observed[name] = item
    if (
        len(observed) != len(repositories)
        or set(observed) != set(identities)
    ):
        raise ContractError("Qwen source override is incomplete", "source_lock_drift")
    for name, identity in identities.items():
        item = observed[name]
        commit = item.get("commit")
        if (
            Path(str(item.get("path", ""))).expanduser().resolve() != roots[name]
            or (commit is not None and commit != identity["commit"])
        ):
            raise ContractError("Qwen source override drifted", "source_lock_drift")


def _profiles() -> tuple[FormalSourceDeliveryProfile, ...]:
    identities = _source_identities()
    return tuple(
        _profile(names, identities)
        for names in (("vllm",), ("aiter",), ("vllm", "aiter"))
    )


def _profile(
    names: tuple[str, ...], identities: Mapping[str, Mapping[str, str]]
) -> FormalSourceDeliveryProfile:
    repositories = tuple(
        FormalRepositoryProfile(
            name,
            name,
            identities[name]["url"],
            (f"{name}/",),
        )
        for name in names
    )
    suffix = "-".join(names)
    recipe = BuildRecipeLock(
        f"qwen3-next-80b-fp8-{suffix}-python-source-v1",
        QWEN_PARENT_IMAGE_ID,
        "apex/qwen3-next-80b-fp8-python-source",
        tuple(
            BuildStep(
                ("git", "diff", "--check", "HEAD", "--", f"{name}/"),
                name,
                timeout_seconds=120,
            )
            for name in names
        ),
    )
    return FormalSourceDeliveryProfile(
        f"qwen3-next-80b-fp8-{suffix}", repositories, recipe
    )


def _image_builder() -> DockerPythonSourceImageBuilder:
    return DockerPythonSourceImageBuilder(
        parent_locator=QWEN_PARENT_LOCATOR,
        parent_image_id=QWEN_PARENT_IMAGE_ID,
        source_date_epoch=QWEN_SOURCE_DATE_EPOCH,
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _qwen_source_lock():
    return load_source_lock(default_source_lock_path(_project_root()))


def _source_identities() -> Mapping[str, Mapping[str, str]]:
    identities = {
        item.key: {
            "url": item.repository,
            "commit": item.commit,
            "tree": item.tree,
            "checkout": item.managed_checkout,
        }
        for item in _qwen_source_lock().sources
    }
    if set(identities) != {"vllm", "aiter"}:
        raise ContractError(
            "Qwen source lock must contain exactly vllm and aiter",
            "source_lock_unresolved",
        )
    return identities


__all__ = [
    "QWEN_CONFIG_SHA256",
    "QWEN_MODEL_ID",
    "QWEN_MODEL_REVISION",
    "QWEN_PARENT_IMAGE_ID",
    "QWEN_PARENT_LOCATOR",
    "QWEN_PARENT_REFERENCE",
    "QWEN_PARENT_REPO_DIGEST",
    "QWEN_SOURCE_DATE_EPOCH",
    "QwenAcceptanceProvenance",
    "QwenAcceptanceProvenanceResolver",
    "build_qwen_acceptance_bundle_verifier",
    "build_qwen_acceptance_delivery",
    "build_qwen_acceptance_provenance_resolver",
    "build_qwen_correctness_oracles",
    "build_qwen_oracle_micro_qualifier",
    "default_qwen_source_roots",
]
