"""Published Magpie main nested config fields that must not be silently dropped."""

from __future__ import annotations

import copy
from typing import Any, Mapping


_FIELDS: dict[tuple[str, ...], frozenset[str]] = {
    ("profiler",): frozenset(
        {"torch_profiler", "system_profiler", "tracelens", "gpu_monitor"}
    ),
    ("profiler", "torch_profiler"): frozenset({"enabled"}),
    ("profiler", "system_profiler"): frozenset({"enabled", "profile_args"}),
    ("profiler", "tracelens"): frozenset(
        {
            "enabled", "analysis_mode", "analysis_stages", "num_steps",
            "cli_timeout_seconds", "auto_patch_runtime", "tracelens_repo_path",
            "extension_wheel_path", "runtime_patch_image_tag",
            "runtime_patch_force_rebuild", "restore_patches", "export_format",
            "export_csv", "export_excel", "perf_report_enabled",
            "multi_rank_report_enabled", "gpu_arch_config",
        }
    ),
    ("profiler", "gpu_monitor"): frozenset(
        {"enabled", "interval_sec", "device_id"}
    ),
    ("gap_analysis",): frozenset(
        {
            "enabled", "trace_start_pct", "trace_end_pct", "top_k",
            "min_duration_us", "categories", "ignore_categories",
            "find_kernel_sources", "kernel_source_repos", "auto_clone_repos",
            "repos_base_dir",
        }
    ),
    ("gpu_selection",): frozenset(
        {"auto", "min_free_memory_gb", "count", "candidates"}
    ),
    ("ray_config",): frozenset(
        {
            "cluster_address", "shared_storage_path", "entrypoint_num_gpus",
            "entrypoint_num_cpus", "multi_node", "total_num_gpus", "num_nodes",
            "gpus_per_node", "pip_packages", "env_vars", "metadata",
            "install_magpie", "magpie_install_path",
        }
    ),
    ("server_lifecycle",): frozenset(
        {"enabled", "cleanup", "force_reuse", "pid_dir", "server_ready_timeout_s"}
    ),
}


def nested_unknown_fields(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze fields ignored by the exact public model at known semantic maps."""

    result: dict[str, Any] = {}
    for path, fields in _FIELDS.items():
        value: object = benchmark
        for part in path:
            value = value.get(part) if isinstance(value, Mapping) else None
        if not isinstance(value, Mapping):
            continue
        for key in sorted(set(value) - fields):
            result[".".join((*path, str(key)))] = copy.deepcopy(value[key])
    return result


__all__ = ["nested_unknown_fields"]
