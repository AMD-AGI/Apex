"""Trace-kernel orchestration."""

from __future__ import annotations

import json
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .agent_harness import AgentPatchRequest, run_agent_patch_fallback
from .mode_detection import detect_trace_mode, normalize_trace_mode
from .overlay import (
    ModuleMapping,
    infer_module_mapping,
    overlay_path_for,
    write_docker_wrapper,
    write_overlay_support,
)
from .patch_triton import PatchResult, patch_triton_launch_file
from .patch_wrapper import patch_aiter_compile_ops_file, patch_wrapper_entry_file
from .postprocess import postprocess_trace
from .runtime import write_runtime_file


@dataclass
class TraceKernelConfig:
    results_dir: Path
    kernel_name: str
    kernel_file: Path
    kernel_id: str = ""
    registry_entry: dict[str, Any] | None = None
    trace_mode: str = "auto"
    kernel_type: str = ""
    patch_strategy: str = "auto"
    benchmark_config: str = ""
    run_cmd: str = ""
    max_records: int = 100000
    sample_rate: float = 1.0
    small_tensor_stats: bool = False
    trace_all: bool = False
    agent_backend: str = "claude"
    agent_model: str | None = None
    agent_max_turns: int = 8
    benchmark_timeout: int = 5400
    docker_image: str = ""
    framework: str = ""
    dry_run: bool = False
    repo_root: Path = Path(__file__).resolve().parents[2]


def _trace_kind_for_mode(mode: str) -> str:
    if mode == "triton-launch":
        return "triton_launch"
    if mode == "aiter-compile-ops":
        return "hip_python_op"
    if mode == "vllm-custom-op":
        return "vllm_python_op"
    if mode == "sglang-custom-op":
        return "sglang_python_op"
    return mode.replace("-", "_")


def _prepare_static_patch(config: TraceKernelConfig, mode: str) -> PatchResult:
    fallback_source = config.kernel_file
    if mode == "aiter-compile-ops":
        module_name = "aiter.jit.core"
        package_rel_path = "aiter/jit/core.py"
        fallback_source = config.repo_root / "tools" / "rocm" / "aiter" / "aiter" / "jit" / "core.py"
    else:
        module_name, package_rel_path = infer_module_mapping(config.kernel_file, config.repo_root)
    source_path = _source_for_patch(config, module_name, package_rel_path, fallback_source)
    patched_files_dir = config.results_dir / "patched_files"
    output_path = overlay_path_for(patched_files_dir, package_rel_path)
    if mode == "triton-launch":
        return patch_triton_launch_file(
            source_path=source_path,
            output_path=output_path,
            kernel_name=config.kernel_name,
            module_name=module_name,
            package_rel_path=package_rel_path,
        )
    if mode == "aiter-compile-ops":
        return patch_aiter_compile_ops_file(
            source_path=source_path,
            output_path=output_path,
            trace_kind=_trace_kind_for_mode(mode),
            module_name=module_name,
            package_rel_path=package_rel_path,
        )
    return patch_wrapper_entry_file(
        source_path=source_path,
        output_path=output_path,
        kernel_name=config.kernel_name,
        trace_kind=_trace_kind_for_mode(mode),
        module_name=module_name,
        package_rel_path=package_rel_path,
        trace_all=config.trace_all,
    )


def _write_trace_config(config: TraceKernelConfig, mode: str, patch_result: PatchResult | None) -> None:
    data = asdict(config)
    data["results_dir"] = str(config.results_dir)
    data["kernel_file"] = str(config.kernel_file)
    data["repo_root"] = str(config.repo_root)
    data["kernel_id"] = config.kernel_id
    data["registry_entry"] = config.registry_entry
    data["resolved_trace_mode"] = mode
    if patch_result:
        data["patch_result"] = {
            "source_path": str(patch_result.source_path),
            "patched_path": str(patch_result.patched_path),
            "module_name": patch_result.module_name,
            "package_rel_path": patch_result.package_rel_path,
            "events": patch_result.events,
        }
    config.results_dir.mkdir(parents=True, exist_ok=True)
    (config.results_dir / "trace_config.json").write_text(
        json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
    )


def _compile_patched(path: Path) -> None:
    py_compile.compile(str(path), doraise=True)


def _base_trace_env(config: TraceKernelConfig, *, docker: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    root = "/apex_trace" if docker else str(config.results_dir)
    patched_files = f"{root}/patched_files" if docker else str(config.results_dir / "patched_files")
    trace_raw = f"{root}/trace_raw" if docker else str(config.results_dir / "trace_raw")
    target_kernel = "" if config.trace_all else config.kernel_name
    env.update({
        "APEX_TRACE_ENABLED": "1",
        "APEX_TRACE_PATCH_MANIFEST": f"{patched_files}/patch_manifest.json",
        "APEX_TRACE_OUTPUT_DIR": trace_raw,
        "APEX_TRACE_KERNEL_NAME": target_kernel,
        "APEX_TRACE_MAX_RECORDS": str(config.max_records),
        "APEX_TRACE_SAMPLE_RATE": str(config.sample_rate),
        "APEX_TRACE_SMALL_TENSOR_STATS": "1" if config.small_tensor_stats else "0",
        "PYTHONPATH": f"{patched_files}:{env.get('PYTHONPATH', '')}",
    })
    return env


def _merge_benchmark_envs(config_path: str, config: TraceKernelConfig, *, docker: bool) -> Path:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    bench = data.setdefault("benchmark", data)
    envs = bench.setdefault("envs", {})
    trace_env = _base_trace_env(config, docker=docker)
    for key in (
        "APEX_TRACE_ENABLED",
        "APEX_TRACE_PATCH_MANIFEST",
        "APEX_TRACE_OUTPUT_DIR",
        "APEX_TRACE_KERNEL_NAME",
        "APEX_TRACE_MAX_RECORDS",
        "APEX_TRACE_SAMPLE_RATE",
        "APEX_TRACE_SMALL_TENSOR_STATS",
    ):
        envs[key] = trace_env[key]
    old_py = str(envs.get("PYTHONPATH", ""))
    prefix = "/apex_trace/patched_files" if docker else str(config.results_dir / "patched_files")
    envs["PYTHONPATH"] = f"{prefix}:{old_py}" if old_py else prefix
    out = config.results_dir / "benchmark" / "trace_benchmark_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out


def _detect_magpie_run_mode() -> str:
    override = os.environ.get("MAGPIE_RUN_MODE", "").strip().lower()
    if override in ("local", "docker"):
        return override
    if shutil.which("docker") is None:
        return "local"
    try:
        res = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=5)
        return "docker" if res.returncode == 0 else "local"
    except Exception:
        return "local"


def _resolve_benchmark_docker_image(config: TraceKernelConfig) -> str:
    if config.docker_image:
        return config.docker_image
    if config.benchmark_config:
        try:
            data = yaml.safe_load(Path(config.benchmark_config).read_text(encoding="utf-8")) or {}
            bench = data.get("benchmark", data)
            if bench.get("docker_image"):
                return str(bench["docker_image"])
        except Exception:
            pass
    return os.environ.get("APEX_VLLM_ROCM_IMAGE", "vllm/vllm-openai-rocm:v0.19.0")


def _source_for_patch(
    config: TraceKernelConfig,
    module_name: str,
    package_rel_path: str,
    fallback_source: Path | None = None,
) -> Path:
    """Prefer the container-installed module source for Docker E2E tracing.

    Host checkouts under tools/rocm can drift from the benchmark image. Patching
    the host version and overlaying it into Docker can mismatch imported Triton
    kernel signatures. Container source extraction keeps the patched wrapper in
    lockstep with the image actually running the workload.
    """
    fallback_source = fallback_source or config.kernel_file
    if not config.benchmark_config or _detect_magpie_run_mode() != "docker":
        return fallback_source
    image = _resolve_benchmark_docker_image(config)
    if not image or shutil.which("docker") is None:
        return fallback_source
    out = config.results_dir / "container_sources" / package_rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    code = (
        "import pathlib, site, sysconfig\n"
        f"rel = {package_rel_path!r}\n"
        "roots = list(site.getsitepackages()) + [sysconfig.get_paths().get('purelib', '')]\n"
        "for root in roots:\n"
        "    p = pathlib.Path(root) / rel\n"
        "    if p.exists():\n"
        "        print(str(p))\n"
        "        print('__APEX_SOURCE_BEGIN__')\n"
        "        print(p.read_text(), end='')\n"
        "        raise SystemExit(0)\n"
        "raise SystemExit(2)\n"
    )
    try:
        proc = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception:
        return fallback_source
    if proc.returncode != 0 or "__APEX_SOURCE_BEGIN__" not in proc.stdout:
        return fallback_source
    _container_path, source = proc.stdout.split("__APEX_SOURCE_BEGIN__\n", 1)
    if "apex_trace_event" in source:
        return fallback_source
    out.write_text(source, encoding="utf-8")
    return out


@contextmanager
def _temporary_env(env: dict[str, str]):
    old = os.environ.copy()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def _run_trace_command(config: TraceKernelConfig) -> dict:
    env = _base_trace_env(config, docker=False)
    proc = subprocess.run(
        config.run_cmd,
        shell=True,
        cwd=str(config.repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=config.benchmark_timeout,
    )
    out = {
        "command": config.run_cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "success": proc.returncode == 0,
    }
    (config.results_dir / "benchmark").mkdir(parents=True, exist_ok=True)
    (config.results_dir / "benchmark" / "run_cmd_result.json").write_text(
        json.dumps(out, indent=2, sort_keys=True), encoding="utf-8"
    )
    return out


def _run_trace_benchmark(config: TraceKernelConfig) -> dict:
    from score import run_magpie_benchmark

    mode = _detect_magpie_run_mode()
    traced_cfg = _merge_benchmark_envs(config.benchmark_config, config, docker=(mode == "docker"))
    env = _base_trace_env(config, docker=False)
    if mode == "docker":
        wrapper_dir = write_docker_wrapper(config.results_dir)
        env["APEX_TRACE_HOST_RESULTS_DIR"] = str(config.results_dir)
        env["APEX_TRACE_REAL_DOCKER"] = shutil.which("docker") or "/usr/bin/docker"
        env["PATH"] = f"{wrapper_dir}:{env.get('PATH', '')}"
    with _temporary_env(env):
        result = run_magpie_benchmark(
            framework=config.framework or "vllm",
            model="",
            benchmark_config_path=str(traced_cfg),
            timeout=config.benchmark_timeout,
            docker_image=config.docker_image,
        )
    (config.results_dir / "benchmark").mkdir(parents=True, exist_ok=True)
    (config.results_dir / "benchmark" / "benchmark_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _trace_event_flags(results_dir: Path, kernel_name: str) -> dict[str, bool]:
    flags = {"any_event_found": False, "target_event_found": False}
    for path in (results_dir / "trace_raw").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") == "module_import":
                continue
            flags["any_event_found"] = True
            extra = event.get("extra") if isinstance(event.get("extra"), dict) else {}
            candidates = {
                str(value)
                for value in (
                    event.get("kernel_name"),
                    extra.get("load_name"),
                    extra.get("wrapper"),
                )
                if value is not None
            }
            if not kernel_name or kernel_name in candidates:
                flags["target_event_found"] = True
    return flags


def run_trace_kernel(config: TraceKernelConfig) -> dict[str, Any]:
    config.results_dir = Path(config.results_dir)
    config.kernel_file = Path(config.kernel_file)
    config.repo_root = Path(config.repo_root)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    (config.results_dir / "trace_raw").mkdir(parents=True, exist_ok=True)

    requested = normalize_trace_mode(config.trace_mode, config.kernel_type)
    mode = detect_trace_mode(config.kernel_file, config.kernel_name, requested)
    patch_result: PatchResult | None = None

    if config.patch_strategy == "agent" or mode == "agent":
        agent_manifest = run_agent_patch_fallback(AgentPatchRequest(
            results_dir=config.results_dir,
            apex_root=config.repo_root,
            kernel_name=config.kernel_name,
            kernel_file=config.kernel_file,
            trace_mode=mode,
            agent_backend=config.agent_backend,
            agent_model=config.agent_model,
            agent_max_turns=config.agent_max_turns,
        ))
        _write_trace_config(config, mode, None)
        result = {
            "success": True,
            "mode": mode,
            "kernel_id": config.kernel_id,
            "registry_entry": config.registry_entry,
            "agent_manifest": agent_manifest,
        }
    else:
        patch_result = _prepare_static_patch(config, mode)
        _compile_patched(patch_result.patched_path)
        mapping = ModuleMapping(
            module_name=patch_result.module_name,
            package_rel_path=patch_result.package_rel_path,
            source_path=patch_result.source_path,
            patched_path=patch_result.patched_path,
        )
        write_runtime_file(config.results_dir / "patched_files")
        write_overlay_support(results_dir=config.results_dir, mappings=[mapping])
        _write_trace_config(config, mode, patch_result)
        result = {
            "success": True,
            "mode": mode,
            "kernel_id": config.kernel_id,
            "registry_entry": config.registry_entry,
            "patched_file": str(patch_result.patched_path),
            "events": patch_result.events,
        }

    if config.dry_run:
        result["dry_run"] = True
        return result

    if config.run_cmd:
        run_result = _run_trace_command(config)
    elif config.benchmark_config:
        run_result = _run_trace_benchmark(config)
    else:
        raise ValueError("trace-kernel requires either --run-cmd or -b/--benchmark-config")

    ranges = postprocess_trace(config.results_dir)
    result["run_result"] = run_result
    result["workload_ranges"] = ranges
    event_flags = _trace_event_flags(config.results_dir, config.kernel_name)
    result.update(event_flags)
    result["event_found"] = (
        event_flags["any_event_found"] if config.trace_all else event_flags["target_event_found"]
    )
    result["success"] = bool(run_result.get("success", True)) and result["event_found"]
    (config.results_dir / "trace_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
