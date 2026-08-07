"""Exact formal-delivery composition for the Qwen3-Next acceptance workload."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

from apex.benchmark import MagpieBenchmarkAdapter
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


QWEN_CONFIG_SHA256 = "f97bda8e04655fbd1410bafb34072ec072de416ea7e24551d2618281e75deafb"
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

    roots = dict(source_roots or default_qwen_source_roots())
    if set(roots) != set(_source_identities()):
        raise ContractError("Qwen source roots are incomplete", "source_lock_unresolved")
    profiles = _profiles()
    primary_images = _image_builder()
    verifier_images = _image_builder()
    primary = QwenPrimarySourceBuilder(
        primary_images, MagpieBenchmarkAdapter(dependency_receipt)
    )
    independent_build = QwenIndependentSourceBuild(verifier_images)
    independent_engagement = QwenIndependentEngagement(
        verifier_images, _project_root()
    )
    independent_replay = QwenIndependentReplay(
        MagpieBenchmarkAdapter(dependency_receipt)
    )
    bindings = tuple(
        FormalDeliveryBinding(
            profile,
            primary,
            E2EBundleVerifier(
                trusted_recipes={profile.recipe.computed_sha256: profile.recipe},
                trusted_source_urls={
                    item.repository_id: item.trusted_url
                    for item in profile.repositories
                },
                build_backend=independent_build,
                engagement_backend=independent_engagement,
                replay_backend=independent_replay,
            ),
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


def default_qwen_source_roots() -> Mapping[str, Path]:
    return default_source_roots(_qwen_source_lock())


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
    bindings = tuple(
        CorrectnessOracleBinding("vllm", source, test, argv)
        for source, test, argv in (
            (
                "vllm/v1/attention/ops/chunked_prefill_paged_decode.py",
                "tests/kernels/attention/test_prefix_prefill.py",
                (
                    "python",
                    "-m",
                    "pytest",
                    "tests/kernels/attention/test_prefix_prefill.py",
                    "-q",
                    "-x",
                ),
            ),
            (
                "vllm/model_executor/layers/fla/ops/fused_recurrent.py",
                "tests/kernels/test_fused_recurrent_packed_decode.py",
                (
                    "python",
                    "-m",
                    "pytest",
                    "tests/kernels/test_fused_recurrent_packed_decode.py",
                    "tests/kernels/test_fused_sigmoid_gating_delta_rule.py",
                    "-q",
                    "-x",
                ),
            ),
            (
                "vllm/model_executor/layers/mamba/ops/causal_conv1d.py",
                "tests/kernels/mamba/test_causal_conv1d.py",
                (
                    "python",
                    "-m",
                    "pytest",
                    "tests/kernels/mamba/test_causal_conv1d.py",
                    "-q",
                    "-x",
                ),
            ),
            (
                "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py",
                "tests/kernels/attention/test_cache.py",
                (
                    "python",
                    "-m",
                    "pytest",
                    "tests/kernels/attention/test_cache.py",
                    "-q",
                    "-x",
                ),
            ),
            (
                "vllm/v1/attention/ops/prefix_prefill.py",
                "tests/kernels/attention/test_prefix_prefill.py",
                (
                    "python",
                    "-m",
                    "pytest",
                    "tests/kernels/attention/test_prefix_prefill.py",
                    "-q",
                    "-x",
                ),
            ),
        )
    )
    return CorrectnessOracleRegistry(
        source_roots=roots,
        bindings=bindings,
        source_lock_sha256=_qwen_source_lock().sha256,
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
    "build_qwen_acceptance_delivery",
    "build_qwen_acceptance_provenance_resolver",
    "build_qwen_correctness_oracles",
    "default_qwen_source_roots",
]
