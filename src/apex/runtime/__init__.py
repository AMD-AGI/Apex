"""Runtime dependency bootstrap and process-environment boundaries."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "BootstrapError": "repositories",
    "DependencyBootstrapper": "dependencies",
    "DependencyLock": "dependencies",
    "LockedDependency": "dependencies",
    "PythonEnvironment": "dependencies",
    "PythonProbe": "dependencies",
    "RepositoryResolver": "repositories",
    "RepositoryState": "repositories",
    "ResolvedRepository": "repositories",
    "canonical_repository": "repositories",
    "inspect_repository": "repositories",
    "load_lock": "dependencies",
    "probe_errors": "dependencies",
    "version_matches": "dependencies",
    "DependencyReceipt": "receipt",
    "verify_runtime_dependencies": "receipt",
    "DownloadLock": "lm_eval_lock",
    "LmEvalRuntimeLock": "lm_eval_lock",
    "WheelLock": "lm_eval_lock",
    "load_lm_eval_runtime_lock": "lm_eval_lock",
    "LmEvalRuntimeReceipt": "lm_eval_runtime",
    "default_lm_eval_runtime_root": "lm_eval_runtime",
    "verify_lm_eval_runtime": "lm_eval_runtime",
    "LmEvalRuntimePreparer": "lm_eval_prepare",
    "SourceCheckoutReceipt": "source_locks",
    "SourceLockManager": "source_locks",
    "SourceLockReceipt": "source_locks",
    "SourceLockSet": "source_locks",
    "SourceLockSpec": "source_locks",
    "default_source_checkout_root": "source_locks",
    "default_source_lock_path": "source_locks",
    "default_source_roots": "source_locks",
    "load_source_lock": "source_locks",
    "ContainerIdentity": "provenance",
    "ProvenanceResolver": "provenance",
    "RepositoryLock": "provenance",
    "RunProvenance": "provenance",
    "GpuLease": "gpu",
    "GpuLeaseManager": "gpu",
    "GpuLeaseReceipt": "gpu",
    "LocalGpuLease": "gpu",
    "LocalGpuLeaseManager": "gpu",
    "resolve_gpu_device_scope": "gpu",
    "GpuDeviceIdentity": "gpu_topology",
    "GpuSelectorRequest": "gpu_topology",
    "RsmiDeviceIdentity": "gpu_topology",
    "CleanHsaInventoryProvider": "hsa_inventory",
    "HsaGpuIdentity": "hsa_inventory",
    "HsaInventoryEvidence": "hsa_inventory",
    "GpuOwnershipInspector": "gpu_ownership",
    "GpuOwnershipReceipt": "gpu_ownership",
    "GpuProcessIdentity": "gpu_ownership",
    "RocmSmiGpuOwnershipInspector": "gpu_ownership",
    "collect_gpu_ownership": "gpu_ownership",
}


def __getattr__(name: str) -> Any:
    """Lazily expose the public API without preloading executable modules."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = sorted(_EXPORT_MODULES)
