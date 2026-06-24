"""Trace-kernel orchestration."""

from __future__ import annotations

import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
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
from .postprocess import postprocess_trace, write_target_kernel_tensor_shapes
from .registry import SUPPORTED_TRACE_IMAGES, registry_path_for_image
from .runtime import write_runtime_file


@dataclass
class TraceKernelTarget:
    kernel_name: str
    kernel_file: Path
    kernel_id: str = ""
    registry_entry: dict[str, Any] | None = None
    trace_mode: str = "auto"
    kernel_type: str = ""
    patch_strategy: str = "auto"


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
    disable_benchmark_cuda_graph: bool = False
    dry_run: bool = False
    repo_root: Path = Path(__file__).resolve().parents[2]
    targets: list[TraceKernelTarget] = field(default_factory=list)


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


def _targets_from_config(config: TraceKernelConfig) -> list[TraceKernelTarget]:
    if config.targets:
        out = []
        for target in config.targets:
            out.append(TraceKernelTarget(
                kernel_name=target.kernel_name,
                kernel_file=Path(target.kernel_file),
                kernel_id=target.kernel_id,
                registry_entry=target.registry_entry,
                trace_mode=target.trace_mode,
                kernel_type=target.kernel_type,
                patch_strategy=target.patch_strategy,
            ))
        return out
    return [TraceKernelTarget(
        kernel_name=config.kernel_name,
        kernel_file=Path(config.kernel_file),
        kernel_id=config.kernel_id,
        registry_entry=config.registry_entry,
        trace_mode=config.trace_mode,
        kernel_type=config.kernel_type,
        patch_strategy=config.patch_strategy,
    )]


def _target_dict(target: TraceKernelTarget) -> dict[str, Any]:
    data = asdict(target)
    data["kernel_file"] = str(target.kernel_file)
    return data


def _target_kernel_names(config: TraceKernelConfig) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for target in _targets_from_config(config):
        if target.kernel_name and target.kernel_name not in seen:
            names.append(target.kernel_name)
            seen.add(target.kernel_name)
    return names


def _patch_location(
    config: TraceKernelConfig,
    target: TraceKernelTarget,
    mode: str,
) -> tuple[Path, str, str, Path]:
    fallback_source = target.kernel_file
    if mode == "aiter-compile-ops":
        module_name = "aiter.jit.core"
        package_rel_path = "aiter/jit/core.py"
        fallback_source = config.repo_root / "tools" / "rocm" / "aiter" / "aiter" / "jit" / "core.py"
    else:
        module_name, package_rel_path = infer_module_mapping(target.kernel_file, config.repo_root)
    source_path = _source_for_patch(config, module_name, package_rel_path, fallback_source)
    output_path = overlay_path_for(config.results_dir / "patched_files", package_rel_path)
    return source_path, module_name, package_rel_path, output_path


def _prepare_static_patch(
    config: TraceKernelConfig,
    mode: str,
    target: TraceKernelTarget | None = None,
    source_override: Path | None = None,
) -> PatchResult:
    target = target or _targets_from_config(config)[0]
    source_path, module_name, package_rel_path, output_path = _patch_location(
        config, target, mode,
    )
    source_path = source_override or source_path
    if mode == "triton-launch":
        return patch_triton_launch_file(
            source_path=source_path,
            output_path=output_path,
            kernel_name=target.kernel_name,
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
        kernel_name=target.kernel_name,
        trace_kind=_trace_kind_for_mode(mode),
        module_name=module_name,
        package_rel_path=package_rel_path,
        trace_all=config.trace_all,
    )


def _write_trace_config(
    config: TraceKernelConfig,
    mode: str,
    patch_result: PatchResult | None,
    *,
    targets: list[TraceKernelTarget] | None = None,
    target_modes: dict[str, str] | None = None,
    patch_results: list[PatchResult] | None = None,
) -> None:
    data = asdict(config)
    data["results_dir"] = str(config.results_dir)
    data["kernel_file"] = str(config.kernel_file)
    data["repo_root"] = str(config.repo_root)
    data["kernel_id"] = config.kernel_id
    data["registry_entry"] = config.registry_entry
    data["resolved_trace_mode"] = mode
    if targets is not None:
        data["targets"] = [_target_dict(target) for target in targets]
        data["kernel_ids"] = [target.kernel_id for target in targets]
    if target_modes is not None:
        data["resolved_trace_modes"] = target_modes
    if patch_result:
        data["patch_result"] = {
            "source_path": str(patch_result.source_path),
            "patched_path": str(patch_result.patched_path),
            "module_name": patch_result.module_name,
            "package_rel_path": patch_result.package_rel_path,
            "events": patch_result.events,
        }
    if patch_results is not None:
        data["patch_results"] = [
            {
                "source_path": str(result.source_path),
                "patched_path": str(result.patched_path),
                "module_name": result.module_name,
                "package_rel_path": result.package_rel_path,
                "events": result.events,
            }
            for result in patch_results
        ]
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
    target_kernel = "" if config.trace_all else ",".join(_target_kernel_names(config))
    env.update({
        "APEX_TRACE_ENABLED": "1",
        "APEX_TRACE_PATCH_MANIFEST": f"{patched_files}/patch_manifest.json",
        "APEX_TRACE_OUTPUT_DIR": trace_raw,
        "APEX_TRACE_KERNEL_NAME": target_kernel,
        "APEX_TRACE_KERNEL_NAMES": target_kernel,
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
        "APEX_TRACE_KERNEL_NAMES",
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


_CUDA_GRAPH_ARG_RE = re.compile(
    r"(?<!\S)--cuda-graph-max-bs(?:=\S+|\s+(?:\"[^\"]+\"|'[^']+'|[^\s\\]+))"
)


def _benchmark_section(config_path: str) -> dict[str, Any]:
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return data.get("benchmark", data)


def _inferencex_benchmark_roots(config: TraceKernelConfig) -> list[Path]:
    roots: list[Path] = []
    candidates: list[Path] = []
    magpie_root = os.environ.get("MAGPIE_ROOT", "").strip()
    if magpie_root:
        candidates.append(Path(magpie_root))
    candidates.append(config.repo_root / "tools" / "magpie")

    seen: set[Path] = set()
    for candidate in candidates:
        benchmarks = candidate / "InferenceX" / "benchmarks"
        try:
            resolved = benchmarks.resolve()
        except OSError:
            resolved = benchmarks
        if resolved in seen or not benchmarks.exists():
            continue
        roots.append(benchmarks)
        seen.add(resolved)
    return roots


def _find_inferencex_benchmark_script(
    benchmarks_root: Path,
    script_name: str,
) -> Path | None:
    script_name = script_name.strip().lstrip("/")
    if not script_name:
        return None

    direct = benchmarks_root / script_name
    if direct.is_file():
        return direct

    basename = Path(script_name).name
    for match in benchmarks_root.rglob(basename):
        if not match.is_file():
            continue
        rel = match.relative_to(benchmarks_root).as_posix()
        if rel == script_name or rel.endswith(f"/{script_name}") or basename == script_name:
            return match
    return None


def _resolve_inferencex_benchmark_script(
    config: TraceKernelConfig,
) -> tuple[Path, Path]:
    if not config.benchmark_config:
        raise ValueError("--disable-benchmark-cuda-graph requires -b/--benchmark-config")
    bench = _benchmark_section(config.benchmark_config)
    script_name = str(bench.get("benchmark_script") or "").strip()
    if not script_name:
        raise ValueError(
            "--disable-benchmark-cuda-graph requires benchmark.benchmark_script "
            "in the benchmark config"
        )

    searched: list[str] = []
    for root in _inferencex_benchmark_roots(config):
        searched.append(str(root))
        script = _find_inferencex_benchmark_script(root, script_name)
        if script is not None:
            return script, script.relative_to(root)

    raise ValueError(
        "Could not resolve benchmark_script for --disable-benchmark-cuda-graph: "
        f"{script_name!r}. Searched: {', '.join(searched) or '<none>'}"
    )


def _remove_cuda_graph_max_bs(lines: list[str]) -> tuple[list[str], bool]:
    out: list[str] = []
    removed = False
    for line in lines:
        new_line, count = _CUDA_GRAPH_ARG_RE.subn("", line)
        if count:
            removed = True
            new_line = re.sub(r"[ \t]+\\$", r" \\", new_line.rstrip())
            if new_line.strip() in {"", "\\"}:
                continue
        out.append(new_line)
    return out, removed


def _insert_launch_flags(
    lines: list[str],
    *,
    marker: re.Pattern[str],
    flags: list[str],
    source_path: Path,
) -> bool:
    text = "\n".join(lines)
    missing_flags = [flag for flag in flags if flag not in text]
    if not missing_flags:
        return True

    for idx, line in enumerate(lines):
        if not marker.search(line):
            continue
        if not line.rstrip().endswith("\\"):
            raise ValueError(
                "Cannot inject cuda graph disable flags into one-line benchmark "
                f"launch command in {source_path}"
            )
        indent = "    "
        for next_line in lines[idx + 1:]:
            if next_line.strip():
                match = re.match(r"^(\s*)", next_line)
                indent = match.group(1) if match else indent
                break
        additions = [f"{indent}{flag} \\" for flag in missing_flags]
        lines[idx + 1:idx + 1] = additions
        return True
    return False


def _rewrite_benchmark_script_disable_cuda_graph(
    text: str,
    *,
    source_path: Path,
) -> str:
    lines, _removed = _remove_cuda_graph_max_bs(text.splitlines())
    joined = "\n".join(lines)

    if "sglang.launch_server" in joined:
        found = _insert_launch_flags(
            lines,
            marker=re.compile(r"\bsglang\.launch_server\b"),
            flags=["--disable-cuda-graph", "--disable-piecewise-cuda-graph"],
            source_path=source_path,
        )
    elif re.search(r"\bvllm\s+serve\b", joined):
        found = _insert_launch_flags(
            lines,
            marker=re.compile(r"\bvllm\s+serve\b"),
            flags=["--enforce-eager"],
            source_path=source_path,
        )
    else:
        found = False

    if not found:
        raise ValueError(
            "Cannot disable benchmark cuda graph because no recognizable "
            f"SGLang or vLLM launch command was found in {source_path}"
        )

    trailing_newline = "\n" if text.endswith("\n") else ""
    return "\n".join(lines) + trailing_newline


def _prepare_no_cudagraph_benchmark_script(
    config: TraceKernelConfig,
) -> tuple[Path, str]:
    source_path, rel_path = _resolve_inferencex_benchmark_script(config)
    output_path = config.results_dir / "no_cudagraph_benchmarks" / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rewritten = _rewrite_benchmark_script_disable_cuda_graph(
        source_path.read_text(encoding="utf-8"),
        source_path=source_path,
    )
    output_path.write_text(rewritten, encoding="utf-8")
    output_path.chmod(0o755)
    container_path = f"/opt/InferenceX/benchmarks/{rel_path.as_posix()}"
    return output_path, container_path


def resolve_trace_docker_image(
    *,
    benchmark_config: str = "",
    docker_image: str = "",
    framework: str = "",
    repo_root: Path | None = None,
) -> str:
    """Resolve and validate the Docker image for image-bound trace registries."""
    if docker_image:
        resolved = docker_image.strip()
    else:
        bench: dict[str, Any] = {}
        if not benchmark_config:
            raise ValueError(
                "Trace kernel registry selection requires --docker-image or "
                "-b/--benchmark-config"
            )
        data = yaml.safe_load(Path(benchmark_config).read_text(encoding="utf-8")) or {}
        bench = data.get("benchmark", data)
        if bench.get("docker_image"):
            resolved = str(bench["docker_image"]).strip()
        else:
            selected_framework = (framework or str(bench.get("framework") or "vllm")).lower()
            gpu_arch = str(bench.get("gpu_arch") or "gfx950").strip()
            try:
                from Magpie.modes.benchmark.image_selector import ImageSelector

                resolved = ImageSelector().select_image(
                    framework=selected_framework,
                    gpu_arch=gpu_arch,
                )
            except Exception:
                image_config = (
                    Path(os.environ.get("MAGPIE_ROOT", ""))
                    / "Magpie"
                    / "benchmark_images.yaml"
                )
                if not image_config.exists() and repo_root is not None:
                    image_config = (
                        repo_root
                        / "tools"
                        / "magpie"
                        / "Magpie"
                        / "benchmark_images.yaml"
                    )
                mapping = yaml.safe_load(image_config.read_text(encoding="utf-8")) or {}
                resolved = str(mapping.get(selected_framework, {}).get(gpu_arch) or "")
    registry_path_for_image(resolved)
    return resolved


def _resolve_benchmark_docker_image(config: TraceKernelConfig) -> str:
    return resolve_trace_docker_image(
        benchmark_config=config.benchmark_config,
        docker_image=config.docker_image,
        framework=config.framework,
        repo_root=config.repo_root,
    )


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
    if shutil.which("docker") is None:
        raise RuntimeError(
            "Docker benchmark tracing requires docker to extract the exact "
            f"source for {module_name} from {image}."
        )
    out = config.results_dir / "container_sources" / package_rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    code = (
        "import importlib.util, pathlib, site, sys, sysconfig\n"
        f"module_name = {module_name!r}\n"
        f"rel = {package_rel_path!r}\n"
        "roots = []\n"
        "def add_root(root):\n"
        "    if root and root not in roots:\n"
        "        roots.append(root)\n"
        "for root in (\n"
        "    '/sgl-workspace/sglang/python',\n"
        "    '/sgl-workspace/aiter',\n"
        "    '/sgl-workspace/vllm',\n"
        "    '/workspace/sglang/python',\n"
        "    '/workspace/aiter',\n"
        "    '/workspace/vllm',\n"
        "    '/workspace',\n"
        "    '/app',\n"
        "):\n"
        "    add_root(root)\n"
        "for root in list(sys.path) + list(site.getsitepackages()) + [sysconfig.get_paths().get('purelib', '')]:\n"
        "    add_root(root)\n"
        "def emit(path):\n"
        "    print(str(path))\n"
        "    print('__APEX_SOURCE_BEGIN__')\n"
        "    print(path.read_text(), end='')\n"
        "    raise SystemExit(0)\n"
        "for root in roots:\n"
        "    p = pathlib.Path(root) / rel\n"
        "    if p.exists() and p.is_file():\n"
        "        emit(p)\n"
        "try:\n"
        "    spec = importlib.util.find_spec(module_name)\n"
        "    origin = getattr(spec, 'origin', None) if spec else None\n"
        "    if origin and origin not in {'built-in', 'namespace'}:\n"
        "        p = pathlib.Path(origin)\n"
        "        if p.exists() and p.is_file():\n"
        "            emit(p)\n"
        "except Exception:\n"
        "    pass\n"
        "raise SystemExit(2)\n"
    )
    try:
        proc = subprocess.run(
            ["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        supported = ", ".join(SUPPORTED_TRACE_IMAGES)
        raise RuntimeError(
            f"Failed to extract {module_name} ({package_rel_path}) from {image}. "
            f"Supported trace images: {supported}."
        ) from exc
    if proc.returncode != 0 or "__APEX_SOURCE_BEGIN__" not in proc.stdout:
        stderr = proc.stderr.strip()
        raise RuntimeError(
            f"Could not find {module_name} ({package_rel_path}) in Docker image "
            f"{image}. {stderr}"
        )
    container_path_text, source = proc.stdout.split("__APEX_SOURCE_BEGIN__\n", 1)
    if "apex_trace_event" in source:
        raise RuntimeError(
            f"Refusing to patch {module_name} from {image}: extracted source "
            "already contains apex_trace_event."
        )
    out.write_text(source, encoding="utf-8")
    container_path = container_path_text.strip().splitlines()[-1]
    if container_path:
        out.with_suffix(out.suffix + ".container_path").write_text(
            container_path, encoding="utf-8"
        )
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
    extra_mounts: list[tuple[Path, str]] = []
    no_cudagraph_overlay: dict[str, str] | None = None
    if config.disable_benchmark_cuda_graph:
        if mode != "docker":
            raise ValueError(
                "--disable-benchmark-cuda-graph currently requires Docker benchmark mode"
            )
        host_script, container_script = _prepare_no_cudagraph_benchmark_script(config)
        extra_mounts.append((host_script, container_script))
        no_cudagraph_overlay = {
            "host_script": str(host_script),
            "container_script": container_script,
        }

    traced_cfg = _merge_benchmark_envs(config.benchmark_config, config, docker=(mode == "docker"))
    env = _base_trace_env(config, docker=False)
    if mode == "docker":
        wrapper_dir = write_docker_wrapper(config.results_dir, extra_mounts=extra_mounts)
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
    if no_cudagraph_overlay is not None:
        result.setdefault("trace_kernel_overlays", {})["no_cudagraph_benchmark_script"] = (
            no_cudagraph_overlay
        )
    (config.results_dir / "benchmark").mkdir(parents=True, exist_ok=True)
    (config.results_dir / "benchmark" / "benchmark_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _trace_event_flags(
    results_dir: Path,
    kernel_name: str | list[str],
    *,
    include_details: bool = False,
) -> dict[str, Any]:
    flags = {"any_event_found": False, "target_event_found": False}
    if isinstance(kernel_name, str):
        target_names = [kernel_name] if kernel_name else []
    else:
        target_names = [name for name in kernel_name if name]
    target_set = set(target_names)
    found_by_target = {name: False for name in target_names}
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
            if not target_set or candidates & target_set:
                flags["target_event_found"] = True
                for name in candidates & target_set:
                    found_by_target[name] = True
    if target_set:
        flags["target_event_found"] = all(found_by_target.values())
    if include_details:
        flags["target_events_found"] = found_by_target
        flags["missing_kernel_names"] = [
            name for name, found in found_by_target.items() if not found
        ]
    return flags


def _compile_unique_patches(patch_results: list[PatchResult]) -> None:
    seen: set[Path] = set()
    for result in patch_results:
        path = result.patched_path
        if path in seen:
            continue
        seen.add(path)
        _compile_patched(path)


def _module_mappings_from_patches(patch_results: list[PatchResult]) -> list[ModuleMapping]:
    by_module: dict[str, ModuleMapping] = {}
    for result in patch_results:
        if result.module_name in by_module:
            continue
        by_module[result.module_name] = ModuleMapping(
            module_name=result.module_name,
            package_rel_path=result.package_rel_path,
            source_path=result.source_path,
            patched_path=result.patched_path,
        )
    return list(by_module.values())


def _prepare_static_patches(
    config: TraceKernelConfig,
    targets: list[TraceKernelTarget],
    target_modes: dict[str, str],
) -> list[PatchResult]:
    patch_results: list[PatchResult] = []
    patched_outputs: dict[Path, PatchResult] = {}

    for target in targets:
        mode = target_modes[target.kernel_id or target.kernel_name]
        requested = target.patch_strategy
        if requested == "agent" or mode == "agent":
            raise ValueError(
                "multi-kernel trace currently supports static registry entries only; "
                f"{target.kernel_id or target.kernel_name} resolved to agent mode"
            )

        _, _, _, output_path = _patch_location(config, target, mode)
        if (
            output_path in patched_outputs
            and (
                mode == "aiter-compile-ops"
                or (config.trace_all and mode in {"vllm-custom-op", "sglang-custom-op"})
            )
        ):
            continue

        if output_path in patched_outputs:
            patch_result = _prepare_static_patch(
                config,
                mode,
                target,
                source_override=patched_outputs[output_path].patched_path,
            )
        else:
            patch_result = _prepare_static_patch(config, mode, target)
        patched_outputs[output_path] = patch_result
        patch_results.append(patch_result)

    return patch_results


def run_trace_kernel(config: TraceKernelConfig) -> dict[str, Any]:
    config.results_dir = Path(config.results_dir)
    config.kernel_file = Path(config.kernel_file)
    config.repo_root = Path(config.repo_root)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    trace_raw_dir = config.results_dir / "trace_raw"
    trace_raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        trace_raw_dir.chmod(0o777)
    except OSError:
        pass

    targets = _targets_from_config(config)
    for target in targets:
        target.kernel_file = Path(target.kernel_file)

    target_modes: dict[str, str] = {}
    for target in targets:
        requested = normalize_trace_mode(target.trace_mode, target.kernel_type)
        target_modes[target.kernel_id or target.kernel_name] = detect_trace_mode(
            target.kernel_file, target.kernel_name, requested,
        )

    first_target = targets[0]
    mode = target_modes[first_target.kernel_id or first_target.kernel_name]
    multi_target = len(targets) > 1
    patch_result: PatchResult | None = None

    if not multi_target and (first_target.patch_strategy == "agent" or mode == "agent"):
        agent_manifest = run_agent_patch_fallback(AgentPatchRequest(
            results_dir=config.results_dir,
            apex_root=config.repo_root,
            kernel_name=first_target.kernel_name,
            kernel_file=first_target.kernel_file,
            trace_mode=mode,
            agent_backend=config.agent_backend,
            agent_model=config.agent_model,
            agent_max_turns=config.agent_max_turns,
        ))
        _write_trace_config(
            config,
            mode,
            None,
            targets=targets,
            target_modes=target_modes,
        )
        result = {
            "success": True,
            "mode": mode,
            "kernel_id": first_target.kernel_id,
            "registry_entry": first_target.registry_entry,
            "agent_manifest": agent_manifest,
        }
    else:
        patch_results = _prepare_static_patches(config, targets, target_modes)
        patch_result = patch_results[0]
        _compile_unique_patches(patch_results)
        write_runtime_file(config.results_dir / "patched_files")
        write_overlay_support(
            results_dir=config.results_dir,
            mappings=_module_mappings_from_patches(patch_results),
        )
        _write_trace_config(
            config,
            mode,
            patch_result,
            targets=targets,
            target_modes=target_modes,
            patch_results=patch_results,
        )
        result = {
            "success": True,
            "mode": mode,
            "kernel_id": first_target.kernel_id,
            "registry_entry": first_target.registry_entry,
            "patched_file": str(patch_result.patched_path),
            "events": patch_result.events,
        }
        if multi_target:
            result.update({
                "kernel_ids": [target.kernel_id for target in targets],
                "targets": [
                    {
                        **_target_dict(target),
                        "resolved_trace_mode": target_modes[target.kernel_id or target.kernel_name],
                    }
                    for target in targets
                ],
                "patched_files": sorted({
                    str(result.patched_path) for result in patch_results
                }),
                "events": [
                    event
                    for result in patch_results
                    for event in result.events
                ],
            })

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
    target_names = _target_kernel_names(config)
    event_flags = _trace_event_flags(
        config.results_dir,
        target_names if multi_target else first_target.kernel_name,
        include_details=multi_target,
    )
    result.update(event_flags)
    result["event_found"] = (
        event_flags["any_event_found"] if config.trace_all else event_flags["target_event_found"]
    )
    result["success"] = bool(run_result.get("success", True)) and result["event_found"]
    write_target_kernel_tensor_shapes(config.results_dir, ranges, result)
    (config.results_dir / "trace_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
