"""Concrete stateless Codex, Claude, and Cursor agent adapters."""

from .environment import (
    DOCKER_RUNTIME_ENVIRONMENT_KEYS,
    GPU_RUNTIME_ENVIRONMENT_KEYS,
    HF_CREDENTIAL_ENVIRONMENT_KEYS,
    HF_RUNTIME_ENVIRONMENT_KEYS,
    build_subprocess_environment,
)
from .registry import AgentRegistry, build_default_registry
from .kernel_measurement import (
    STRUCTURED_KERNEL_MEASUREMENT_ADAPTER_ID,
    STRUCTURED_KERNEL_MEASUREMENT_METHOD_SHA256,
    StructuredKernelMeasurementAdapter,
)
from .template_materialization import (
    DockerTemplateImageSourceRuntime,
    KernelTemplateMaterializer,
    MaterializedKernelTemplate,
    TemplateImageSourceReceipt,
    TemplateImageSourceRuntime,
    TemplateMaterializationReceipt,
    template_source_tree_sha256,
)
from .native_session import (
    default_capability_results,
    NativeCodingSessionLauncher,
    NativeSessionInvocation,
    NativeSessionRunner,
    SubprocessSessionRunner,
)
from .backend_doctor import BackendDoctorReport, BackendFeature, NativeBackendDoctor
from .skill_assets import KernelSkillPackage, load_kernel_skill_package
from .supervisor import ProcessResult, SubprocessSupervisor
from .transcript import agent_transcript_document

__all__ = [
    "AgentRegistry",
    "BackendDoctorReport",
    "BackendFeature",
    "DOCKER_RUNTIME_ENVIRONMENT_KEYS",
    "GPU_RUNTIME_ENVIRONMENT_KEYS",
    "HF_CREDENTIAL_ENVIRONMENT_KEYS",
    "HF_RUNTIME_ENVIRONMENT_KEYS",
    "NativeCodingSessionLauncher",
    "NativeBackendDoctor",
    "NativeSessionInvocation",
    "NativeSessionRunner",
    "ProcessResult",
    "STRUCTURED_KERNEL_MEASUREMENT_ADAPTER_ID",
    "STRUCTURED_KERNEL_MEASUREMENT_METHOD_SHA256",
    "SubprocessSupervisor",
    "StructuredKernelMeasurementAdapter",
    "DockerTemplateImageSourceRuntime",
    "KernelTemplateMaterializer",
    "KernelSkillPackage",
    "MaterializedKernelTemplate",
    "TemplateImageSourceReceipt",
    "TemplateImageSourceRuntime",
    "TemplateMaterializationReceipt",
    "template_source_tree_sha256",
    "load_kernel_skill_package",
    "SubprocessSessionRunner",
    "build_default_registry",
    "build_subprocess_environment",
    "default_capability_results",
    "agent_transcript_document",
]
