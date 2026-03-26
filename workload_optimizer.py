#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
workload_optimizer.py — Modular workload optimization trajectory pipeline.

Each step can be run independently via subcommands, or all at once via 'run'.
State is persisted in pipeline_state.json so steps can be resumed/rerun.

Subcommands:
    benchmark       Run initial E2E benchmark (or load existing results)
    identify        Identify & classify bottleneck kernels from benchmark
    list-kernels    Show identified kernels (for interactive selection)
    optimize        Optimize selected kernels (agent + grading loop)
    grade           Re-grade existing solutions without re-running agent
    integrate       Re-inject optimized kernels for final benchmark
    benchmark-final Run final E2E benchmark with optimized kernels
    score           Compute trajectory reward and push to leaderboard
    report          Generate markdown report and replication guide
    run             Full pipeline (all steps sequentially)
    export-rl       Export trajectories to RL training dataset format
    optimize-kernel Optimize a standalone kernel (no full pipeline needed)
    grade-kernel    Grade an existing baseline + solution pair

Usage:
    # Step-by-step (each step resumes from previous state):
    python workload_optimizer.py benchmark   -b config.yaml -r /results --skip-benchmark report.json
    python workload_optimizer.py identify    -r /results --kernel-types triton --top-k 20
    python workload_optimizer.py list-kernels -r /results
    python workload_optimizer.py optimize    -r /results --kernels fused_moe,gemm_bf16
    python workload_optimizer.py integrate   -r /results --kernels fused_moe,gemm_bf16
    python workload_optimizer.py benchmark-final -r /results
    python workload_optimizer.py score       -r /results --leaderboard
    python workload_optimizer.py report      -r /results

    # Full pipeline in one command:
    python workload_optimizer.py run -b config.yaml -r /results --kernel-types triton --leaderboard
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import importlib.util
import json
import os
import re as _re_mod
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "graders"))
sys.path.insert(0, str(REPO_ROOT / "prompts"))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from score import (
    KernelResult,
    cleanup_inference_server,
    run_magpie_benchmark,
    run_magpie_compare,
    parse_compare_result,
    extract_tps,
    workload_kernel_reward,
    workload_model_reward,
    trajectory_reward,
)
from kernel_grader import grade_task, find_solution
from reflector import reflect, should_continue
from ground_truth import get_spec as get_ground_truth_spec, build_correctness_config
from trajectory import WorkloadTrajectoryRecord, get_store
from leaderboard import Leaderboard, LeaderboardEntry
from kernel_bottleneck import (
    BottleneckKernel,
    extract_bottlenecks,
    classify_kernel,
    match_to_kernel_spec,
    detect_origin_library,
    filter_by_types,
    filter_by_names,
    deduplicate_by_spec,
    format_bottleneck_table,
)
from kernel_prompt import (
    KERNEL_SPECS,
    KERNEL_MAP,
    KernelSpec,
    applicable_kernels as _applicable_kernels,
    build_kernel_prompt as _build_rich_kernel_prompt,
    _format_sources_block,
    ARCH_MAP,
    DEFAULT_TARGET,
)
from models import MODELS, ModelConfig


MAGPIE_ROOT = Path(os.environ.get(
    "MAGPIE_ROOT",
    str(REPO_ROOT.parent / "Magpie"),
))
os.environ.setdefault("MAGPIE_ROOT", str(MAGPIE_ROOT))

_GLOBAL_GAP_CACHE_DIR = Path.home() / ".cache" / "apex" / "gap_analysis"


def _gap_cache_key(benchmark_config: str) -> str:
    """Stable cache key derived from benchmark config file content."""
    import hashlib
    cfg_path = Path(benchmark_config)
    if cfg_path.exists():
        content = cfg_path.read_bytes()
    else:
        content = benchmark_config.encode()
    return hashlib.sha256(content).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Kernel patching infrastructure (Fixes 1, 9, 10, 11)
# ---------------------------------------------------------------------------

_KERNEL_SPEC_TO_MODULE: dict[str, dict[str, str]] = {
    "paged_attn_decode": {
        "aiter": "aiter.ops.triton.pa_decode",
        "vllm": "vllm.v1.attention.ops.triton_unified_attention",
    },
    "paged_attn_decode_gluon": {
        "aiter": "aiter.ops.triton.pa_decode_gluon",
    },
    "flash_attn_prefill": {
        "aiter": "aiter.ops.triton.attention.pa_prefill",
        "vllm": "vllm.v1.attention.ops.triton_prefill_attention",
    },
    "mla_attn": {
        "aiter": "aiter.mla",
        "vllm": "vllm.v1.attention.ops.flashmla",
    },
    "gemm_w8a8": {
        "aiter": "aiter.ops.gemm_op_a8w8",
        "vllm": "vllm.model_executor.layers.quantization.utils.fp8_utils",
    },
    "gemm_bf16": {
        "aiter": "aiter.ops.gemm_op_a16w16",
    },
    "fused_moe": {
        "aiter": "aiter.fused_moe",
        "vllm": "vllm.model_executor.layers.fused_moe.fused_moe",
    },
    "rms_norm": {
        "aiter": "aiter.ops.triton.normalization.rmsnorm",
        "vllm": "vllm.model_executor.layers.layernorm",
    },
    "silu_mul": {
        "aiter": "aiter.ops.activation",
        "vllm": "vllm.model_executor.layers.activation",
    },
    "act_quant_fp8": {
        "aiter": "aiter.ops.quant",
        "vllm": "vllm.model_executor.layers.quantization.utils.fp8_utils",
    },
    "rope_embedding": {
        "aiter": "aiter.ops.triton.rope.rope",
        "vllm": "vllm.model_executor.layers.rotary_embedding",
    },
    "kv_cache_ops": {
        "aiter": "aiter.ops.cache",
        "vllm": "vllm.v1.attention.ops.triton_reshape_and_cache_flash",
    },
    "all_reduce": {
        "aiter": "aiter.dist.device_communicators.custom_all_reduce",
        "vllm": "vllm.distributed.device_communicators.custom_all_reduce",
    },
}


def _get_module_for_spec(kernel_spec: str, library: str = "aiter") -> Optional[str]:
    """Get the Python module path for a kernel spec in a given library."""
    lib_map = _KERNEL_SPEC_TO_MODULE.get(kernel_spec, {})
    return lib_map.get(library)


_LIBRARY_PREFIXES: dict[str, list[str]] = {
    "aiter": ["aiter.ops.triton", "aiter.ops", "aiter"],
    "vllm": ["vllm.v1.attention.ops", "vllm.model_executor.layers",
             "vllm.model_executor.layers.fused_moe", "vllm.distributed"],
    "sglang": ["sglang.srt.layers", "sglang.srt.layers.attention"],
    "pytorch": ["torch.nn.modules", "torch.nn.functional"],
}

_HIP_SPEC_TO_SO: dict[str, dict[str, str]] = {
    "all_reduce": {
        "aiter": "custom_all_reduce",
        "vllm": "_C",
    },
    "kv_cache_ops": {
        "vllm": "_C",
    },
    "silu_mul": {
        "aiter": "activation_kernels",
        "vllm": "_C",
    },
}

_MONOLITHIC_SO_LIBRARIES = {"vllm", "pytorch", "sglang"}

PATCH_LOCK_PATH = Path("/tmp/magpie_kernel_patch.lock")
PATCH_MANIFEST_PATH = Path("/tmp/magpie_kernel_patch_manifest.json")

MIN_SPEEDUP_FOR_REINJECTION = 1.05
SMOKE_TEST_REGRESSION_THRESHOLD = 0.5

_SESSION_BACKUPS: dict[str, str] = {}


def _resolve_installed_module_path(kernel_spec: str, library: str = "aiter") -> Optional[Path]:
    """Resolve a kernel spec to its installed Python module file path.

    Tries the static map for the given library first, then auto-discovers
    by scanning library-specific module prefixes.
    """
    module_name = _get_module_for_spec(kernel_spec, library)
    if module_name:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin:
                return Path(spec.origin)
        except (ModuleNotFoundError, ValueError):
            pass

    prefixes = _LIBRARY_PREFIXES.get(library, _LIBRARY_PREFIXES["aiter"])
    for prefix in prefixes:
        candidate = f"{prefix}.{kernel_spec}"
        try:
            spec = importlib.util.find_spec(candidate)
            if spec and spec.origin:
                print(f"    Auto-discovered module for {kernel_spec}: {candidate}")
                return Path(spec.origin)
        except (ModuleNotFoundError, ValueError):
            continue

    # Fallback: try aiter if the requested library didn't resolve
    if library != "aiter":
        return _resolve_installed_module_path(kernel_spec, library="aiter")

    return None


def _clear_pycache(module_path: Path) -> None:
    """Remove __pycache__ for the directory containing a patched module."""
    cache_dir = module_path.parent / "__pycache__"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"    Cleared bytecode cache: {cache_dir}")


def _is_hip_patchable(kernel_spec: str, library: str) -> bool:
    """Check if a HIP kernel can be patched as a standalone .so."""
    if library in _MONOLITHIC_SO_LIBRARIES:
        so_map = _HIP_SPEC_TO_SO.get(kernel_spec, {})
        so_name = so_map.get(library, "")
        if so_name in ("_C", "_custom_ops", ""):
            return False
    return True


def _find_installed_so(kernel_spec: str, library: str = "aiter") -> Optional[Path]:
    """Find the installed shared library for a HIP kernel spec."""
    so_map = _HIP_SPEC_TO_SO.get(kernel_spec, {})
    so_name = so_map.get(library)
    if not so_name:
        return None
    if so_name in ("_C", "_custom_ops"):
        return None

    pkg_name = {"aiter": "aiter", "vllm": "vllm", "sglang": "sglang",
                "pytorch": "torch"}.get(library, library)
    try:
        pkg = importlib.import_module(pkg_name)
        pkg_dir = Path(pkg.__file__).parent
        for so in pkg_dir.rglob(f"*{so_name}*.so"):
            return so
    except (ImportError, Exception):
        pass
    return None


def _hipcc_compile(source: Path, build_dir: str, kernel_spec: str, gpu_arch: str = "gfx950") -> Optional[Path]:
    """Compile a .hip/.cu source into a shared library."""
    output = Path(build_dir) / f"{kernel_spec}.so"
    cmd = [
        "hipcc", "-shared", "-fPIC", "-O3",
        f"--offload-arch={gpu_arch}",
        "-o", str(output),
        str(source),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"    hipcc error: {result.stderr[:500]}")
            return None
        return output
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"    hipcc failed: {e}")
        return None


def _compile_and_patch_hip(
    solution_path: Path, kernel_spec: str, gpu_arch: str = "gfx950",
    library: str = "aiter",
) -> tuple[Optional[Path], Optional[Path]]:
    """Compile a .hip solution and patch the installed .so library."""
    installed_so = _find_installed_so(kernel_spec, library=library)
    if not installed_so:
        print(f"    WARNING: Cannot find installed .so for {kernel_spec}, skipping HIP patch")
        return None, None

    backup = Path(str(installed_so) + ".bak")
    shutil.copy2(installed_so, backup)

    with tempfile.TemporaryDirectory(prefix="hip_build_") as build_dir:
        compiled = _hipcc_compile(solution_path, build_dir, kernel_spec, gpu_arch)
        if compiled and compiled.exists():
            shutil.copy2(compiled, installed_so)
            print(f"    Patched HIP kernel: {installed_so}")
        else:
            shutil.move(str(backup), str(installed_so))
            print(f"    WARNING: HIP compilation failed for {kernel_spec}")
            return None, None

    return installed_so, backup


def _acquire_patch_lock(timeout_s: float = 60.0) -> "IO":
    """Acquire an exclusive file lock with stale-PID detection."""
    lock_fd = open(PATCH_LOCK_PATH, "w+")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_fd.seek(0)
            lock_fd.truncate()
            lock_fd.write(str(os.getpid()))
            lock_fd.flush()
            return lock_fd
        except BlockingIOError:
            # Check if the holding PID is still alive
            try:
                lock_fd.seek(0)
                holder_pid = int(lock_fd.read().strip())
                os.kill(holder_pid, 0)
            except (ValueError, ProcessLookupError, PermissionError):
                print(f"  WARNING: Breaking stale patch lock (holder PID gone)")
                lock_fd.close()
                PATCH_LOCK_PATH.unlink(missing_ok=True)
                return _acquire_patch_lock(timeout_s=5.0)

            if time.monotonic() > deadline:
                lock_fd.close()
                raise RuntimeError(
                    f"Patch lock held by PID {holder_pid} for >{timeout_s}s. "
                    f"Kill it or delete {PATCH_LOCK_PATH}"
                )
            time.sleep(2.0)


def _release_patch_lock(lock_fd) -> None:
    """Release the patch lock and clean up lock/manifest files."""
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
    except Exception:
        pass
    PATCH_LOCK_PATH.unlink(missing_ok=True)
    PATCH_MANIFEST_PATH.unlink(missing_ok=True)


def _recover_orphaned_patches() -> None:
    """Restore any orphaned patches from a previous crashed run."""
    if not PATCH_MANIFEST_PATH.exists():
        return
    print("  WARNING: Found orphaned kernel patches from a previous crashed run")
    try:
        manifest = json.loads(PATCH_MANIFEST_PATH.read_text())
        for installed, backup in manifest.items():
            if Path(backup).exists():
                shutil.move(backup, installed)
                print(f"    Restored: {installed}")
            else:
                print(f"    WARNING: Backup missing, cannot restore: {backup}")
        PATCH_MANIFEST_PATH.unlink(missing_ok=True)
        print("  Orphaned patches recovered successfully")
    except Exception as e:
        print(f"  WARNING: Failed to recover orphaned patches: {e}")


def _ensure_clean_baseline() -> None:
    """Guarantee all patchable library modules are in original state.
    Called once at pipeline startup to prevent cross-session contamination.
    """
    _recover_orphaned_patches()

    for library in ("aiter", "vllm", "sglang"):
        try:
            pkg = importlib.import_module(library)
            pkg_dir = Path(pkg.__file__).parent
            restored = 0
            for bak in sorted(pkg_dir.rglob("*.bak")):
                original = bak.with_suffix("")
                if original.suffix in (".py", ".so") and original.exists():
                    shutil.copy2(bak, original)
                    _clear_pycache(original)
                    restored += 1
                bak.unlink()
            if restored:
                print(f"  Restored {restored} leftover patch(es) in {library}")
        except ImportError:
            continue

    PATCH_LOCK_PATH.unlink(missing_ok=True)
    PATCH_MANIFEST_PATH.unlink(missing_ok=True)
    print("  Baseline libraries verified clean")


def _session_cleanup() -> None:
    """atexit handler: restore any patches made during this session."""
    if _SESSION_BACKUPS:
        print(f"\n  Session cleanup: restoring {len(_SESSION_BACKUPS)} patched module(s)...")
        for installed, backup in list(_SESSION_BACKUPS.items()):
            try:
                if Path(backup).exists():
                    shutil.copy2(backup, installed)
                    _clear_pycache(Path(installed))
                Path(backup).unlink(missing_ok=True)
            except Exception as e:
                print(f"    WARNING: Failed to restore {installed}: {e}")
        _SESSION_BACKUPS.clear()
    PATCH_LOCK_PATH.unlink(missing_ok=True)
    PATCH_MANIFEST_PATH.unlink(missing_ok=True)


def _register_session_handlers() -> None:
    """Register atexit + signal handlers for guaranteed cleanup."""
    import atexit
    import signal as _sig
    atexit.register(_session_cleanup)
    for sig in (_sig.SIGTERM, _sig.SIGINT):
        old = _sig.getsignal(sig)
        def _handler(signum, frame, _old=old):
            _session_cleanup()
            if callable(_old) and _old not in (_sig.SIG_DFL, _sig.SIG_IGN):
                _old(signum, frame)
            else:
                raise SystemExit(128 + signum)
        _sig.signal(sig, _handler)


def _apply_multi_file_patch(
    solution_dir: Path, spec: str, library: str, gpu_arch: str,
    backups: dict[Path, Path], write_manifest_fn, ast_module,
) -> None:
    """Apply a multi-file solution from a directory with a manifest.json.

    The manifest maps relative file names to their install targets:
      {"rmsnorm.py": "aiter.ops.triton.normalization.rmsnorm",
       "__init__.py": "aiter.ops.triton.normalization.__init__"}

    All files in the manifest are backed up and patched atomically -- if any
    file fails, the entire group is rolled back.
    """
    manifest_path = solution_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"    WARNING: Multi-file solution dir {solution_dir} has no manifest.json, skipping")
        return

    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"    REJECTED multi-file {spec}: bad manifest.json: {e}")
        return

    group_backups: list[tuple[Path, Path]] = []
    group_ok = True

    for filename, target_module in manifest.items():
        src = solution_dir / filename
        if not src.exists():
            print(f"    REJECTED multi-file {spec}: missing file {filename}")
            group_ok = False
            break

        if src.suffix == ".py":
            try:
                ast_module.parse(src.read_text())
            except SyntaxError as e:
                print(f"    REJECTED multi-file {spec}/{filename}: syntax error: {e}")
                group_ok = False
                break

            installed_path = _resolve_module_to_path(target_module)
            if not installed_path:
                print(f"    WARNING: Cannot resolve {target_module} for {spec}/{filename}, skipping group")
                group_ok = False
                break

            backup = Path(str(installed_path) + ".bak")
            shutil.copy2(installed_path, backup)
            shutil.copy2(src, installed_path)
            _fixup_patched_imports(installed_path, library=library)
            _clear_pycache(installed_path)
            group_backups.append((installed_path, backup))
            _SESSION_BACKUPS[str(installed_path)] = str(backup)
            print(f"    Patched (multi): {installed_path} <- {filename}")
        else:
            print(f"    WARNING: Multi-file patch skipping non-.py file {filename}")

    if not group_ok:
        for installed_path, backup in group_backups:
            shutil.copy2(backup, installed_path)
            _clear_pycache(installed_path)
            _SESSION_BACKUPS.pop(str(installed_path), None)
        print(f"    Rolled back multi-file group for {spec}")
        return

    for installed_path, backup in group_backups:
        backups[installed_path] = backup
    write_manifest_fn()


def _resolve_module_to_path(module_name: str) -> Optional[Path]:
    """Resolve a dotted module name to its installed file path."""
    import importlib
    try:
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin:
            return Path(spec.origin)
    except (ModuleNotFoundError, ValueError):
        pass
    return None


def _apply_kernel_patches(
    reinjected_dir: Path, gpu_arch: str = "gfx950",
) -> tuple[dict[Path, Path], "IO"]:
    """Patch installed kernel modules with optimized solutions.

    Supports both single-file solutions (*_solution.py) and multi-file
    solution directories (*_solution/ with manifest.json).

    Reads per-solution .library metadata files to determine which library
    to target. The manifest is written incrementally after each patch so
    that ``_recover_orphaned_patches()`` can restore even if the process
    crashes mid-way.

    Returns (backups_dict, lock_fd) for use with _restore_kernel_patches.
    """
    import ast as _ast

    _recover_orphaned_patches()
    lock_fd = _acquire_patch_lock()
    backups: dict[Path, Path] = {}

    def _write_manifest() -> None:
        PATCH_MANIFEST_PATH.write_text(json.dumps(
            {str(k): str(v) for k, v in backups.items()}
        ))

    try:
        if not reinjected_dir.exists():
            print("  No reinjected directory found -- nothing to patch")
            return backups, lock_fd

        # Single-file solutions
        for solution_file in sorted(reinjected_dir.glob("*_solution.*")):
            if solution_file.is_dir():
                continue
            spec = solution_file.name.split("_solution")[0]
            suffix = solution_file.suffix

            lib_meta = solution_file.parent / f"{solution_file.name}.library"
            library = "aiter"
            if lib_meta.exists():
                library = lib_meta.read_text().strip() or "aiter"

            if suffix == ".py":
                try:
                    _ast.parse(solution_file.read_text())
                except SyntaxError as e:
                    print(f"    REJECTED {spec}: solution has syntax error: {e}")
                    continue

                installed_path = _resolve_installed_module_path(spec, library=library)
                if not installed_path:
                    print(f"    WARNING: No installed module for {spec} (library={library}), skipping")
                    continue
                backup = Path(str(installed_path) + ".bak")
                shutil.copy2(installed_path, backup)
                shutil.copy2(solution_file, installed_path)
                _fixup_patched_imports(installed_path, library=library)
                _clear_pycache(installed_path)
                backups[installed_path] = backup
                _SESSION_BACKUPS[str(installed_path)] = str(backup)
                _write_manifest()
                print(f"    Patched: {installed_path} (library={library})")

            elif suffix in (".hip", ".cu"):
                if not _is_hip_patchable(spec, library):
                    print(f"    Skipping HIP patch for {spec}: {library} compiles into "
                          f"monolithic _C.so (not individually patchable)")
                    continue
                installed, backup = _compile_and_patch_hip(
                    solution_file, spec, gpu_arch, library=library,
                )
                if installed and backup:
                    backups[installed] = backup
                    _SESSION_BACKUPS[str(installed)] = str(backup)
                    _write_manifest()
            else:
                print(f"    WARNING: Unknown solution type {suffix} for {spec}")

        # Multi-file solution directories
        for solution_dir in sorted(reinjected_dir.glob("*_solution")):
            if not solution_dir.is_dir():
                continue
            spec = solution_dir.name.split("_solution")[0]

            lib_meta = solution_dir.parent / f"{solution_dir.name}.library"
            library = "aiter"
            if lib_meta.exists():
                library = lib_meta.read_text().strip() or "aiter"

            _apply_multi_file_patch(
                solution_dir, spec, library, gpu_arch,
                backups, _write_manifest, _ast,
            )

    except Exception:
        _restore_kernel_patches(backups, lock_fd)
        raise

    return backups, lock_fd


def _restore_kernel_patches(
    backups: dict[Path, Path], lock_fd=None,
) -> None:
    """Restore all backed-up modules and release the patch lock."""
    for installed, backup in backups.items():
        try:
            if backup.exists():
                shutil.move(str(backup), str(installed))
                _clear_pycache(installed)
                _SESSION_BACKUPS.pop(str(installed), None)
                print(f"    Restored: {installed}")
            else:
                print(f"    WARNING: Backup missing for {installed}")
        except Exception as e:
            print(f"    ERROR restoring {installed}: {e}")
    if lock_fd:
        _release_patch_lock(lock_fd)


def _verify_patched_kernels(
    backups: dict[Path, Path],
    config: "WorkloadConfig",
) -> list[Path]:
    """Run correctness checks on patched installed modules.

    For each patched module:
      1. Verify it imports without error
      2. If a testcase.py exists in the corresponding task dir, run it
      3. Run the library's own test suite via ground_truth MANUAL_REGISTRY

    Returns list of installed paths that failed verification (already rolled back).
    """
    import importlib

    _MODULE_TO_SPEC: dict[str, str] = {}
    for kspec, lib_map in _KERNEL_SPEC_TO_MODULE.items():
        for _lib, mod_name in lib_map.items():
            _MODULE_TO_SPEC[mod_name] = kspec

    failed: list[Path] = []
    for installed_path, backup_path in list(backups.items()):
        spec_name = installed_path.stem
        module_name = None
        for mod, kspec in _MODULE_TO_SPEC.items():
            try:
                s = importlib.util.find_spec(mod)
                if s and s.origin and Path(s.origin) == installed_path:
                    module_name = mod
                    spec_name = kspec
                    break
            except (ModuleNotFoundError, ValueError):
                continue

        # Fallback: derive module name from filesystem path when static map misses
        if module_name is None:
            parts = installed_path.parts
            sp_idx = next(
                (i for i, p in enumerate(parts) if p == "site-packages"), None
            )
            if sp_idx is not None and installed_path.suffix == ".py":
                mod_parts = list(parts[sp_idx + 1:])
                if mod_parts:
                    last = mod_parts[-1]
                    if last.endswith(".py"):
                        mod_parts[-1] = last[:-3]
                    candidate = ".".join(mod_parts)
                    if candidate.endswith(".__init__"):
                        candidate = candidate[: -len(".__init__")]
                    module_name = candidate
                    print(f"    Derived module name: {module_name} (from path)")

        # 1. Import check
        if module_name:
            try:
                importlib.invalidate_caches()
                mod = importlib.import_module(module_name)
                importlib.reload(mod)
                print(f"    VERIFIED import: {module_name}")
            except Exception as e:
                print(f"    FAILED import for {spec_name}: {e}")
                shutil.copy2(backup_path, installed_path)
                _clear_pycache(installed_path)
                print(f"    Rolled back: {installed_path}")
                failed.append(installed_path)
                continue

        # 2. Testcase check (from task dir)
        testcase_ran = False
        if hasattr(config, "output_dir"):
            task_id_patterns = [
                f"workload__vllm__{spec_name}",
                f"workload__sglang__{spec_name}",
            ]
            for pattern in task_id_patterns:
                task_dir = config.output_dir / pattern
                testcase = task_dir / "testcase.py"
                if testcase.exists():
                    try:
                        result = subprocess.run(
                            [sys.executable, str(testcase)],
                            capture_output=True, text=True, timeout=120,
                            cwd=str(task_dir),
                        )
                        if result.returncode != 0:
                            print(f"    FAILED testcase for {spec_name}: {result.stderr[:200]}")
                            shutil.copy2(backup_path, installed_path)
                            _clear_pycache(installed_path)
                            print(f"    Rolled back: {installed_path}")
                            failed.append(installed_path)
                            testcase_ran = True
                        else:
                            print(f"    VERIFIED testcase: {spec_name}")
                            testcase_ran = True
                    except subprocess.TimeoutExpired:
                        print(f"    TIMEOUT testcase for {spec_name}")
                        shutil.copy2(backup_path, installed_path)
                        _clear_pycache(installed_path)
                        failed.append(installed_path)
                        testcase_ran = True
                    break

        if installed_path in failed:
            continue

        # 3. Library test verification via ground_truth MANUAL_REGISTRY
        gt_spec = get_ground_truth_spec(spec_name)
        if gt_spec and gt_spec.unit_test_command and gt_spec.source_library:
            working_dir = REPO_ROOT / "tools" / "rocm" / gt_spec.source_library
            if working_dir.is_dir():
                print(f"    Running library test for {spec_name}: {gt_spec.unit_test_command}")
                try:
                    lib_result = subprocess.run(
                        gt_spec.unit_test_command,
                        shell=True,
                        cwd=str(working_dir),
                        capture_output=True,
                        text=True,
                        timeout=180,
                    )
                    if lib_result.returncode != 0:
                        print(f"    FAILED library test for {spec_name} "
                              f"(exit {lib_result.returncode}): {lib_result.stderr[:300]}")
                        shutil.copy2(backup_path, installed_path)
                        _clear_pycache(installed_path)
                        print(f"    Rolled back: {installed_path}")
                        failed.append(installed_path)
                    else:
                        print(f"    VERIFIED library test: {spec_name}")
                except subprocess.TimeoutExpired:
                    print(f"    TIMEOUT library test for {spec_name} (180s)")
                    shutil.copy2(backup_path, installed_path)
                    _clear_pycache(installed_path)
                    failed.append(installed_path)
                except Exception as e:
                    print(f"    WARNING: library test error for {spec_name}: {e}")

    return failed


# ---------------------------------------------------------------------------
# Dispatch path validation (Fix 7)
# ---------------------------------------------------------------------------

def _validate_optimization_relevance(
    solution_path: Path,
    baseline_path: Optional[Path],
    benchmark_config: dict,
    model_config: dict,
    kernel_spec: str,
) -> Optional[str]:
    """Check if the optimization targets a code path actually used at runtime.

    Returns a warning string if the optimization is likely a no-op, else None.
    """
    if kernel_spec != "paged_attn_decode":
        return None

    if not baseline_path or not baseline_path.exists() or not solution_path.exists():
        return None

    try:
        baseline_text = baseline_path.read_text()
        solution_text = solution_path.read_text()
    except OSError:
        return None

    baseline_partition = _re_mod.search(r'_SEQ_PARTITION_SIZE\s*=\s*(\d+)', baseline_text)
    solution_partition = _re_mod.search(r'_SEQ_PARTITION_SIZE\s*=\s*(\d+)', solution_text)

    if not baseline_partition or not solution_partition:
        return None
    if baseline_partition.group(1) == solution_partition.group(1):
        return None

    bench_section = benchmark_config.get("benchmark", {})
    envs = bench_section.get("envs", {})
    conc = int(envs.get("CONC", 64))
    osl = int(envs.get("OSL", 1024))

    num_q_heads = model_config.get("num_q_heads", 0)
    if num_q_heads == 0:
        return None

    num_seqs_times_heads = conc * num_q_heads
    max_seq_len = osl
    use_v1 = (num_seqs_times_heads > 512) and (max_seq_len <= 8192)

    if use_v1:
        return (
            f"WARNING: Optimization modifies V2 path (_SEQ_PARTITION_SIZE "
            f"{baseline_partition.group(1)}->{solution_partition.group(1)}) but workload "
            f"uses V1 path (max_seq_len={max_seq_len} <= 8192 and "
            f"num_seqs*num_q_heads={num_seqs_times_heads} > 512). "
            f"This optimization may have NO EFFECT on the actual workload."
        )
    return None


# ---------------------------------------------------------------------------
# Shape validation (Fix 8)
# ---------------------------------------------------------------------------

def _load_model_config(benchmark_config: dict) -> dict:
    """Extract model architecture parameters from the target model's config.json.

    Returns a dict with GQA params (for attention kernels) and dimension params
    (for GEMM / normalization / activation kernels).
    """
    bench_section = benchmark_config.get("benchmark", {})
    model_path = bench_section.get("model", "")
    if not model_path:
        return {}
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return {}
    try:
        cfg = json.loads(config_path.read_text())
        num_heads = cfg.get("num_attention_heads", 0)
        hidden = cfg.get("hidden_size", 0)
        return {
            "num_q_heads": num_heads,
            "num_kv_heads": cfg.get("num_key_value_heads", num_heads),
            "head_dim": hidden // max(num_heads, 1),
            "hidden_size": hidden,
            "intermediate_size": cfg.get("intermediate_size", 0),
            "num_experts": cfg.get("num_local_experts", cfg.get("num_experts", 0)),
            "vocab_size": cfg.get("vocab_size", 0),
        }
    except Exception as e:
        print(f"  [warn] Could not load model config from {benchmark_config}: {e}")
        return {}


# Keep backward-compatible alias
_load_model_gqa_config = _load_model_config


_ATTENTION_KERNEL_SPECS = frozenset({
    "flash_attn_prefill", "paged_attn_decode", "mla_attn",
})

_GEMM_KERNEL_SPECS = frozenset({
    "gemm_bf16", "gemm_w8a8",
})

_NORM_ACT_KERNEL_SPECS = frozenset({
    "rms_norm", "silu_mul", "act_quant_fp8", "rope_embedding",
})

_MOE_KERNEL_SPECS = frozenset({
    "fused_moe",
})


def _validate_solution_shapes(
    solution_path: Path, model_config: dict,
    kernel_spec: str = "",
) -> tuple[bool, list[str]]:
    """Check if solution.py test shapes match the target model config.

    Uses per-kernel-spec shape patterns:
      - Attention kernels: num_q_heads, num_kv_heads, head_dim
      - GEMM kernels: M/N/K dimensions vs hidden_size / intermediate_size
      - Norm/activation: hidden_size / dim
      - MoE: num_experts, hidden_size
      - All others: skipped (no validation)

    Returns (has_mismatch, list_of_warnings).
    """
    if not model_config or not solution_path.exists():
        return False, []

    try:
        text = solution_path.read_text()
    except OSError:
        return False, []

    shape_checks: dict[str, list[tuple[str, int]]] = {}

    if kernel_spec in _ATTENTION_KERNEL_SPECS or not kernel_spec:
        shape_checks.update({
            "num_q_heads": [
                (r'num_q(?:uery)?_heads\s*=\s*(\d+)', model_config.get("num_q_heads", 0)),
                (r'num_heads\s*=\s*(\d+)', model_config.get("num_q_heads", 0)),
                (r'NUM_Q(?:UERY)?_HEADS\s*=\s*(\d+)', model_config.get("num_q_heads", 0)),
            ],
            "num_kv_heads": [
                (r'num_kv_heads\s*=\s*(\d+)', model_config.get("num_kv_heads", 0)),
                (r'NUM_KV_HEADS\s*=\s*(\d+)', model_config.get("num_kv_heads", 0)),
            ],
            "head_dim": [
                (r'head_(?:sz|dim|size)\s*=\s*(\d+)', model_config.get("head_dim", 0)),
                (r'HEAD_(?:DIM|SIZE)\s*=\s*(\d+)', model_config.get("head_dim", 0)),
            ],
        })

    if kernel_spec in _GEMM_KERNEL_SPECS:
        hidden = model_config.get("hidden_size", 0)
        inter = model_config.get("intermediate_size", 0)
        expected_dims = [d for d in (hidden, inter) if d > 0]
        if expected_dims:
            shape_checks["gemm_N_or_K"] = [
                (r'[NK]\s*=\s*(\d+)', 0),
            ]
            shape_checks["_gemm_dims_raw"] = expected_dims  # type: ignore[assignment]

    if kernel_spec in _NORM_ACT_KERNEL_SPECS:
        hidden = model_config.get("hidden_size", 0)
        if hidden:
            shape_checks["hidden_size"] = [
                (r'hidden_size\s*=\s*(\d+)', hidden),
                (r'dim\s*=\s*(\d+)', hidden),
                (r'D\s*=\s*(\d+)', hidden),
            ]

    if kernel_spec in _MOE_KERNEL_SPECS:
        n_experts = model_config.get("num_experts", 0)
        if n_experts:
            shape_checks["num_experts"] = [
                (r'num_experts\s*=\s*(\d+)', n_experts),
                (r'NUM_EXPERTS\s*=\s*(\d+)', n_experts),
                (r'E\s*=\s*(\d+)', n_experts),
            ]

    if not shape_checks:
        return False, []

    warnings: list[str] = []

    # Special handling for GEMM dimension checks
    if "_gemm_dims_raw" in shape_checks:
        expected_dims = shape_checks.pop("_gemm_dims_raw")
        gemm_patterns = shape_checks.pop("gemm_N_or_K", [])
        for pattern_str, _ in gemm_patterns:
            for m in _re_mod.finditer(pattern_str, text):
                val = int(m.group(1))
                if val > 64 and val not in expected_dims:
                    warnings.append(
                        f"solution.py uses GEMM dim {m.group(0).strip()}={val} "
                        f"which doesn't match model dims {expected_dims}"
                    )

    for param, patterns_expected in shape_checks.items():
        for pattern_str, expected in patterns_expected:
            if expected == 0:
                continue
            match = _re_mod.search(pattern_str, text, _re_mod.IGNORECASE)
            if match:
                found_val = int(match.group(1))
                if found_val != expected:
                    warnings.append(
                        f"solution.py tests with {param}={found_val} "
                        f"but target model has {param}={expected}"
                    )
                break

    return len(warnings) > 0, warnings


# ---------------------------------------------------------------------------
# Multi-run benchmark averaging (Fix 12)
# ---------------------------------------------------------------------------

MAX_CV_PCT = 20.0
MIN_COMPLETION_RATIO = 0.9


def _check_gpu_health() -> dict:
    """Validate GPU state before benchmarking. Returns health report."""
    report: dict = {"healthy": True, "warnings": []}
    try:
        out = subprocess.check_output(
            ["rocm-smi", "--showtemp", "--showclocks", "--showmeminfo", "vram"],
            text=True, timeout=10,
        )
        for line in out.splitlines():
            if "Temperature" in line and "edge" in line.lower():
                m = _re_mod.search(r"(\d+\.?\d*)", line)
                if m:
                    temp = float(m.group(1))
                    if temp > 85:
                        report["warnings"].append(f"GPU temp {temp}C > 85C (throttling likely)")
                        report["healthy"] = False
                    report["temperature_c"] = temp
        if "throttle" in out.lower() or "limited" in out.lower():
            report["warnings"].append("GPU clock throttling detected")
            report["healthy"] = False
    except FileNotFoundError:
        report["warnings"].append("rocm-smi not found — cannot check GPU health")
    except Exception as e:
        report["warnings"].append(f"Could not query GPU health: {e}")
    if report["warnings"]:
        for w in report["warnings"]:
            print(f"  WARNING: {w}")
    return report


def _cleanup_stale_tmp(max_age_hours: float = 4.0, prefixes: tuple = ("magpie_bench_", "magpie_")):
    """Remove stale temporary benchmark directories from /tmp to prevent disk exhaustion."""
    import time as _time
    cutoff = _time.time() - max_age_hours * 3600
    tmp = Path("/tmp")
    freed = 0
    try:
        for entry in tmp.iterdir():
            if not entry.is_dir():
                continue
            if not any(entry.name.startswith(p) for p in prefixes):
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff:
                import shutil as _shutil
                _shutil.rmtree(entry, ignore_errors=True)
                freed += 1
        if freed:
            print(f"  [cleanup] Removed {freed} stale benchmark dir(s) from /tmp")
    except OSError as e:
        print(f"  [cleanup] Could not scan /tmp: {e}")


_INFERENCEX_CHECKED = False


def _ensure_inferencex_path(benchmark_config_path: str) -> None:
    """Ensure the benchmark YAML has a valid inferencex_path.

    Magpie's default falls back to /root/workspace/InferenceX (a Docker-only
    path) when the config omits inferencex_path.  This auto-detects InferenceX
    relative to MAGPIE_ROOT and injects it into the config YAML so users don't
    have to configure it manually.  Runs at most once per process.
    """
    global _INFERENCEX_CHECKED
    if _INFERENCEX_CHECKED:
        return
    _INFERENCEX_CHECKED = True

    cfg_path = Path(benchmark_config_path)
    if not cfg_path.exists():
        return

    with open(cfg_path) as f:
        data = yaml.safe_load(f) or {}
    bench = data.get("benchmark", data)

    existing = bench.get("inferencex_path", "")
    if existing and existing != "/root/workspace/InferenceX" and Path(existing).exists():
        return

    magpie_root = os.environ.get("MAGPIE_ROOT", "")
    candidates = [
        Path(magpie_root) / "InferenceX" if magpie_root else None,
        REPO_ROOT.parent / "Magpie" / "InferenceX",
        Path.home() / "InferenceX",
    ]
    resolved: Optional[Path] = None
    for cand in candidates:
        if cand and cand.is_dir():
            resolved = cand
            break

    if not resolved:
        if magpie_root:
            print(f"  [auto-config] InferenceX not found. Clone it manually:\n"
                  f"    git clone https://github.com/SemiAnalysisAI/InferenceX.git "
                  f"{Path(magpie_root) / 'InferenceX'}")
        return

    # Update the in-memory YAML data and write it back properly
    if "benchmark" in data:
        data["benchmark"]["inferencex_path"] = str(resolved)
    else:
        data["inferencex_path"] = str(resolved)

    with open(cfg_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"  [auto-config] Set inferencex_path={resolved}")


def _run_benchmark_multi(config: "WorkloadConfig", label: str = "benchmark") -> dict:
    """Run benchmark N times and return result with averaged throughput + statistics.

    Includes warmup run, CV-based outlier rejection, and minimum completion ratio
    filtering for stable measurements.
    """
    import copy as _copy

    _ensure_inferencex_path(config.benchmark_config)
    _cleanup_stale_tmp()
    gpu_health = _check_gpu_health()

    n = getattr(config, "num_benchmark_runs", 1)
    if n <= 1:
        cleanup_inference_server()
        result = run_magpie_benchmark(
            framework=config.framework or "vllm",
            model="",
            benchmark_config_path=config.benchmark_config,
            timeout=config.benchmark_timeout,
        )
        cleanup_inference_server()
        tps = extract_tps(result)
        result["_multi_run"] = {
            "num_runs": 1,
            "individual_tps": [round(tps, 2)],
            "mean_tps": round(tps, 2),
            "std_tps": 0.0,
            "cv_pct": 0.0,
            "note": "single-run -- no statistical confidence",
            "gpu_health": gpu_health,
        }
        return result

    # Warmup run (discarded): clears JIT, page faults, first-request overhead
    print(f"    Warmup run (discarded)...")
    cleanup_inference_server()
    run_magpie_benchmark(
        framework=config.framework or "vllm",
        model="",
        benchmark_config_path=config.benchmark_config,
        timeout=config.benchmark_timeout,
    )
    cleanup_inference_server()

    all_tps: list[float] = []
    all_results: list[dict] = []
    for i in range(n):
        print(f"    Run {i+1}/{n}...")
        cleanup_inference_server()
        result = run_magpie_benchmark(
            framework=config.framework or "vllm",
            model="",
            benchmark_config_path=config.benchmark_config,
            timeout=config.benchmark_timeout,
        )
        tps = extract_tps(result)
        run_failed = (
            result.get("success") is False or
            bool(result.get("error")) or
            (tps <= 0 and bool(result.get("errors")))
        )
        if run_failed:
            err = result.get("error")
            if not err:
                errs = result.get("errors") or []
                err = errs[0] if errs else "benchmark produced no throughput"
            print(f"    Run {i+1}/{n}: FAILED ({str(err)[:100]})")
            continue

        # Minimum completion ratio check
        completed = result.get("completed_requests", result.get("completed", 0))
        total = result.get("total_requests", result.get("total", 0))
        if total > 0 and completed > 0:
            ratio = completed / total
            if ratio < MIN_COMPLETION_RATIO:
                print(f"    Run {i+1}/{n}: REJECTED (completion ratio {ratio:.1%} < {MIN_COMPLETION_RATIO:.0%})")
                continue

        all_tps.append(tps)
        all_results.append(result)
        print(f"    Run {i+1}/{n}: {tps:.1f} tok/s")

        # Save per-run report
        if hasattr(config, "output_dir") and config.output_dir:
            runs_dir = config.output_dir / "benchmark_runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            run_file = runs_dir / f"{label.replace(' ', '_')}_{i+1}.json"
            try:
                with open(run_file, "w") as f:
                    json.dump(result, f, indent=2, default=str)
            except Exception as e:
                print(f"  [warn] Could not write benchmark run file {run_file}: {e}")

    cleanup_inference_server()

    if not all_results:
        return {"error": "All benchmark runs failed", "success": False}

    mean_tps = sum(all_tps) / len(all_tps)
    std_tps = (sum((t - mean_tps) ** 2 for t in all_tps) / len(all_tps)) ** 0.5
    cv_pct = std_tps / mean_tps * 100 if mean_tps > 0 else 0

    # CV gate: iteratively drop worst outlier while variance is too high
    dropped_values: list[float] = []
    while cv_pct > MAX_CV_PCT and len(all_tps) > 2:
        sorted_tps = sorted(all_tps)
        median_tps = sorted_tps[len(sorted_tps) // 2]
        worst_idx = max(range(len(all_tps)), key=lambda i: abs(all_tps[i] - median_tps))
        dropped_val = all_tps.pop(worst_idx)
        all_results.pop(worst_idx)
        dropped_values.append(dropped_val)
        print(f"    CV={cv_pct:.1f}% > {MAX_CV_PCT}% — dropped outlier {dropped_val:.1f} tok/s")
        mean_tps = sum(all_tps) / len(all_tps)
        std_tps = (sum((t - mean_tps) ** 2 for t in all_tps) / len(all_tps)) ** 0.5
        cv_pct = std_tps / mean_tps * 100 if mean_tps > 0 else 0

    if cv_pct > MAX_CV_PCT:
        print(f"    WARNING: CV={cv_pct:.1f}% still above {MAX_CV_PCT}% after outlier removal")
        print(f"    Baseline may be unreliable — consider more runs or checking GPU state")

    avg_result = _copy.deepcopy(all_results[0])

    if "throughput" in avg_result and isinstance(avg_result["throughput"], dict):
        avg_result["throughput"]["output_throughput"] = mean_tps
    avg_result["_multi_run"] = {
        "num_runs": len(all_tps),
        "individual_tps": [round(t, 2) for t in all_tps],
        "mean_tps": round(mean_tps, 2),
        "std_tps": round(std_tps, 2),
        "cv_pct": round(cv_pct, 2),
        "cv_warning": cv_pct > MAX_CV_PCT,
        "warmup_run": True,
        "outliers_dropped": [round(d, 2) for d in dropped_values] if dropped_values else None,
        "gpu_health": gpu_health,
    }

    print(f"    {label}: {mean_tps:.1f} +/- {std_tps:.1f} tok/s "
          f"(CV={cv_pct:.1f}%, n={len(all_tps)})")
    return avg_result


def _is_improvement_significant(baseline_result: dict, final_result: dict) -> bool:
    """Check if throughput improvement exceeds measurement noise."""
    b_multi = baseline_result.get("_multi_run", {})
    f_multi = final_result.get("_multi_run", {})

    if not b_multi or not f_multi:
        return True

    b_mean = b_multi.get("mean_tps", 0)
    b_std = b_multi.get("std_tps", 0)
    f_mean = f_multi.get("mean_tps", 0)
    f_std = f_multi.get("std_tps", 0)

    if b_mean <= 0 or f_mean <= 0:
        return True

    improvement = f_mean - b_mean
    combined_std = (b_std ** 2 + f_std ** 2) ** 0.5

    if combined_std > 0 and improvement < 2 * combined_std:
        print(f"  WARNING: Improvement {improvement:.1f} tok/s < 2*std {2 * combined_std:.1f} tok/s")
        print(f"  The throughput change is NOT statistically significant")
        return False
    return True


# ---------------------------------------------------------------------------
# Pipeline state management
# ---------------------------------------------------------------------------

STATE_FILE = "pipeline_state.json"


class PipelineState:
    """Persistent state across pipeline steps, stored as pipeline_state.json."""

    def __init__(self, results_dir: Path):
        self.path = results_dir / STATE_FILE
        self._data: dict = {}
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2, default=str)
        os.replace(tmp, self.path)

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value):
        self._data[key] = value
        self.save()

    def update(self, d: dict):
        self._data.update(d)
        self.save()

    def require(self, key: str, step_name: str):
        """Raise if a prerequisite step hasn't been run."""
        v = self._data.get(key)
        if v is None:
            raise SystemExit(
                f"[error] '{key}' not found in state. Run the '{step_name}' step first."
            )
        return v

    @property
    def data(self) -> dict:
        return dict(self._data)

    @property
    def completed_steps(self) -> list[str]:
        return self._data.get("completed_steps", [])

    def mark_step(self, step: str):
        steps = self._data.setdefault("completed_steps", [])
        if step not in steps:
            steps.append(step)
        self.save()

    def record_step_time(self, step: str, elapsed_s: float):
        """Record wall-clock time for a pipeline step."""
        timings = self._data.setdefault("step_timings", {})
        timings[step] = round(elapsed_s, 2)
        self.save()

    @property
    def step_timings(self) -> dict[str, float]:
        return self._data.get("step_timings", {})


def _detect_kernel_python() -> str:
    """Auto-detect python with torch+triton for kernel execution.

    Uses the currently active python (sys.executable) since the venv should
    already be activated before running the pipeline.
    """
    import sys as _sys
    return _sys.executable or "python3"


SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer specializing in AMD ROCm optimization.
Your task is to optimize GPU kernels for maximum performance on AMD Instinct GPUs.

You have access to the filesystem AND the following MCP tools:

MAGPIE (kernel evaluation framework):
- mcp__magpie__analyze: Analyze a kernel for correctness and performance profiling
- mcp__magpie__compare: Compare baseline vs optimized kernel (correctness + speedup)
- mcp__magpie__hardware_spec: Get GPU hardware specifications
- mcp__magpie__suggest_optimizations: Get optimization suggestions from analysis results
- mcp__magpie__benchmark: Run E2E LLM inference benchmarks

GPU INFO:
- mcp__gpu-info__get_gpu_info: Detect GPU and get hardware specs
- mcp__gpu-info__get_arch_optimization_hints: Architecture-specific optimization hints

SOURCE FINDER:
- mcp__source-finder__find_kernel_source: Find source code for a kernel type
- mcp__source-finder__classify_kernel: Classify kernel by name
- mcp__source-finder__find_ck_template: Find CK templates for an operation
- mcp__source-finder__identify_kernel_origin: Trace which library a kernel comes from

RAG SERVER:
- mcp__rag-server__search_kernel_optimization: Search optimization patterns
- mcp__rag-server__search_gpu_documentation: Search AMD GPU docs
- mcp__rag-server__get_optimization_snippet: Get code snippets for a pattern
- mcp__rag-server__analyze_kernel_for_optimization: Analyze kernel and suggest optimizations
- mcp__rag-server__get_optimization_playbook: Get complete optimization playbook

KERNEL PERF:
- mcp__kernel-perf__profile_kernel: Profile kernel with rocprof
- mcp__kernel-perf__roofline_analysis: Roofline model analysis
- mcp__kernel-perf__statistical_test: Statistical comparison of measurements

FUSION ADVISOR:
- mcp__fusion-advisor__detect_fusion_opportunities: Find kernel fusion opportunities
- mcp__fusion-advisor__generate_fused_kernel: Generate fused kernel implementations
- mcp__fusion-advisor__estimate_fusion_benefit: Estimate fusion benefit

ASM TOOLS:
- mcp__asm-tools__disassemble_kernel: Disassemble kernel to ISA
- mcp__asm-tools__analyze_isa: Analyze instruction mix and register usage
- mcp__asm-tools__count_instructions: Count instruction types

SKILLS (read these files BEFORE starting optimization):
- For Triton kernels: tools/skills/triton-kernel-optimization/SKILL.md
- For HIP/C++ kernels: tools/skills/hip-kernel-optimization/SKILL.md
- Architecture context: tools/skills/gpu-architecture-fundamentals/SKILL.md
- MI300/MI355 specifics: tools/skills/mi300-cdna3-architecture/SKILL.md
- AMD aiter patterns: tools/skills/aiter-reflection/SKILL.md
- Prior experiments: tools/skills/kernel-exp-history/SKILL.md
- Profiling guide: tools/skills/rocprof-compute/SKILL.md

WORKFLOW:
1. Read relevant skill files from tools/skills/ for domain knowledge
2. Read the baseline kernel code, understand it thoroughly
3. Use mcp__gpu-info__get_gpu_info to understand the target GPU
4. Use mcp__source-finder__find_kernel_source to find all implementations
5. Use mcp__rag-server__search_kernel_optimization for relevant patterns
6. Use mcp__fusion-advisor__detect_fusion_opportunities for fusion chances
7. Write an optimized version to solution.py
8. Use mcp__magpie__analyze to profile your solution
9. Use mcp__magpie__compare to compare baseline vs solution
10. Use mcp__asm-tools__analyze_isa for ISA-level analysis if needed
11. Iterate until speedup is substantial

Focus on: memory coalescing, LDS usage, MFMA utilization, register pressure,
bank conflicts, optimal block/tile sizes for the target architecture.

IMPORTANT:
- Write your optimized kernel to solution.py in the task directory
- The solution must be a self-contained Python file with a __main__ block that
  runs the kernel and prints PASS/FAIL
- Do NOT modify files outside the task directory
- Do NOT create new scripts — all evaluation uses Magpie MCP (analyze, compare)
- Do NOT hardcode kernel names — they are provided dynamically by the pipeline
- Only solutions with >5% speedup will be integrated into the final benchmark
- Use Magpie compare to verify correctness AND measure speedup every iteration
- Do ALL work directly yourself — read files, write code, call MCP tools directly.
  Do NOT delegate to sub-agents via the Agent tool. Work hands-on.
"""


@dataclass
class WorkloadConfig:
    benchmark_config: str = ""
    skip_benchmark: Optional[str] = None
    kernel_types: list[str] = field(default_factory=lambda: ["all"])
    kernels: list[str] = field(default_factory=lambda: ["all"])
    top_k: int = 10
    top_k_mode: str = "post-filter"
    max_iterations: int = 5
    max_turns_per_iter: int = 25
    score_threshold: float = 300.0
    agent_model: str = ""
    agent_version: str = "v1.0"
    agent_backend: str = "claude"
    framework: str = ""
    gpu_arch: str = "gfx950"
    docker_image: str = ""
    kernel_python: str = ""
    output_dir: Path = field(default_factory=lambda: REPO_ROOT / "output")
    results_dir: Optional[Path] = None
    trajectory_store: str = "file"
    push_leaderboard: bool = False
    dry_run: bool = False
    num_benchmark_runs: int = 5
    benchmark_timeout: int = 5400

    def __post_init__(self):
        if not self.agent_model:
            try:
                from agents.backends import resolve_default_model
                self.agent_model = resolve_default_model(self.agent_backend)
            except Exception:
                from agents.backends import DEFAULT_CLAUDE_MODEL, DEFAULT_CODEX_MODEL
                self.agent_model = (
                    DEFAULT_CODEX_MODEL if self.agent_backend == "codex"
                    else DEFAULT_CLAUDE_MODEL
                )

    @property
    def effective_results_dir(self) -> Path:
        return self.results_dir or self.output_dir


@dataclass
class KernelOptResult:
    kernel_name: str = ""
    kernel_spec: str = ""
    category: str = ""
    origin_library: str = "unknown"
    compiled: bool = False
    correct: bool = False
    baseline_ms: float = 0.0
    optimized_ms: float = 0.0
    speedup: float = 0.0
    score: float = 0.0
    iterations_used: int = 0
    reinjected: bool = False
    agent_turns: int = 0
    shape_mismatch: bool = False
    error: Optional[str] = None
    agent_trace: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "agent_trace"}
        if self.agent_trace:
            d["agent_trace"] = self.agent_trace[-50:]
        return d


# ---------------------------------------------------------------------------
# Step 1: E2E Benchmark
# ---------------------------------------------------------------------------

def _run_initial_benchmark(config: WorkloadConfig) -> dict:
    if config.skip_benchmark:
        skip_path = Path(config.skip_benchmark)
        if not skip_path.exists():
            print(f"  [error] --skip-benchmark path does not exist: {skip_path}")
            return {"error": f"skip-benchmark file not found: {skip_path}"}
        print(f"  Loading existing benchmark: {skip_path}")
        with open(skip_path) as f:
            return json.load(f)

    if config.dry_run:
        return _dry_run_benchmark_result()

    print(f"  Running Magpie benchmark with config: {config.benchmark_config}")
    print(f"  This may take 10-90 minutes for large models...")
    result = _run_benchmark_multi(config, label="baseline")
    return result


def _dry_run_benchmark_result() -> dict:
    return {
        "success": True, "dry_run": True, "framework": "vllm",
        "model": "dry-run-model",
        "throughput": {"output_throughput": 100.0, "total_token_throughput": 200.0},
        "kernel_summary": [], "top_bottlenecks": [],
        "gap_analysis": {
            "top_kernels": [
                {"name": "triton_poi_fused_constant_pad_nd_moe_forward_0",
                 "calls": 1000, "self_cuda_total_us": 5000000, "avg_time_us": 5000, "pct_total": 25.0},
                {"name": "kernel_unified_attention_2d",
                 "calls": 500, "self_cuda_total_us": 3000000, "avg_time_us": 6000, "pct_total": 15.0},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Step 2-4: Bottleneck extraction, classification, and selection
# ---------------------------------------------------------------------------

def _select_kernels(benchmark_result: dict, config: WorkloadConfig, state: "PipelineState | None" = None) -> list[BottleneckKernel]:
    top_k_mode = getattr(config, "top_k_mode", "post-filter")

    config_envs = None
    if state:
        bench_cfg = state._data.get("benchmark_config", {})
        config_envs = bench_cfg.get("benchmark", {}).get("envs")
    if not config_envs:
        bench_cfg = getattr(config, "_benchmark_config", None) or {}
        config_envs = bench_cfg.get("benchmark", {}).get("envs")

    per_run_dir = None
    if config.output_dir:
        runs_path = config.output_dir / "benchmark_runs"
        if runs_path.exists():
            per_run_dir = str(runs_path)

    if top_k_mode == "post-filter" and config.kernel_types:
        all_bottlenecks = extract_bottlenecks(benchmark_result, top_k=200, config_envs=config_envs, per_run_dir=per_run_dir)
        print(f"\n  All bottleneck kernels ({len(all_bottlenecks)} total):")
        print(format_bottleneck_table(all_bottlenecks[:20]))
        if len(all_bottlenecks) > 20:
            print(f"  ... and {len(all_bottlenecks) - 20} more")

        filtered = filter_by_types(all_bottlenecks, config.kernel_types)
        type_str = ",".join(config.kernel_types)
        print(f"\n  After type filter ({type_str}): {len(filtered)} kernels")
    else:
        all_bottlenecks = extract_bottlenecks(benchmark_result, top_k=config.top_k, config_envs=config_envs, per_run_dir=per_run_dir)
        print(f"\n  All bottleneck kernels ({len(all_bottlenecks)}):")
        print(format_bottleneck_table(all_bottlenecks))

        filtered = filter_by_types(all_bottlenecks, config.kernel_types)
        if len(filtered) != len(all_bottlenecks):
            type_str = ",".join(config.kernel_types)
            print(f"\n  After type filter ({type_str}): {len(filtered)} kernels")

    if config.kernels and "all" not in config.kernels:
        filtered = filter_by_names(filtered, config.kernels)
        print(f"  After name filter: {len(filtered)} kernels")

    deduped = deduplicate_by_spec(filtered)
    if len(deduped) != len(filtered):
        print(f"  After dedup by spec: {len(deduped)} unique kernel specs")

    with_spec = [k for k in deduped if k.matched_kernel_spec]
    if len(with_spec) != len(deduped):
        skipped = len(deduped) - len(with_spec)
        print(f"  Removed {skipped} kernel(s) without known spec mapping")

    # Apply top-k truncation after filtering in post-filter mode
    if top_k_mode == "post-filter" and len(with_spec) > config.top_k:
        print(f"  Truncating to top-{config.top_k} from {len(with_spec)} candidates")
        with_spec = with_spec[:config.top_k]

    # Cache successful extraction for future fallback
    if with_spec and config.effective_results_dir:
        _save_gap_analysis_cache(config.effective_results_dir, all_bottlenecks)
    elif not with_spec and config.effective_results_dir:
        cached = _load_gap_analysis_cache(config.effective_results_dir, config_envs)
        if cached:
            print(f"  Loaded {len(cached)} kernel(s) from gap_analysis_cache.json")
            with_spec = cached
            if config.kernel_types:
                with_spec = filter_by_types(with_spec, config.kernel_types)
            with_spec = [k for k in with_spec if k.matched_kernel_spec]
            with_spec = deduplicate_by_spec(with_spec)
            if top_k_mode == "post-filter" and len(with_spec) > config.top_k:
                with_spec = with_spec[:config.top_k]

    if with_spec:
        print(f"\n  Selected kernels for optimization:")
        print(format_bottleneck_table(with_spec))

    return with_spec


def _save_gap_analysis_cache(results_dir: Path, kernels: list[BottleneckKernel]) -> None:
    """Persist successful kernel extraction so future runs can use it as fallback."""
    cache_path = results_dir / "gap_analysis_cache.json"
    entries = []
    for k in kernels:
        entries.append({
            "name": k.name,
            "total_time_us": k.total_time_us,
            "calls": k.calls,
            "avg_time_us": k.avg_time_us,
            "percent_total": k.percent_total,
            "category": k.category,
            "matched_kernel_spec": k.matched_kernel_spec,
            "origin_library": k.origin_library,
        })
    try:
        with open(cache_path, "w") as f:
            json.dump(entries, f, indent=2)
    except Exception as e:
        print(f"  [warn] Could not write gap_analysis_cache: {e}")


def _load_gap_analysis_cache(
    results_dir: Path, config_envs: dict | None = None,
) -> list[BottleneckKernel]:
    """Load kernel list from a previous successful extraction."""
    cache_path = results_dir / "gap_analysis_cache.json"
    if not cache_path.exists():
        return []
    try:
        entries = json.loads(cache_path.read_text())
    except Exception:
        return []
    kernels = []
    for e in entries:
        bk = BottleneckKernel(
            name=e["name"],
            total_time_us=e.get("total_time_us", 0),
            calls=e.get("calls", 0),
            avg_time_us=e.get("avg_time_us", 0),
            percent_total=e.get("percent_total", 0),
        )
        bk.category = e.get("category") or classify_kernel(bk.name)
        bk.matched_kernel_spec = e.get("matched_kernel_spec") or match_to_kernel_spec(bk.name)
        bk.origin_library = e.get("origin_library") or detect_origin_library(
            bk.name, bk.matched_kernel_spec, config_envs)
        kernels.append(bk)
    return kernels


# ---------------------------------------------------------------------------
# Kernel source resolution
# ---------------------------------------------------------------------------

def _find_baseline_sources(kernel_spec: str, library: str = "aiter") -> list[str]:
    """Find actual source file paths for a kernel spec from KERNEL_SPECS.

    For aiter: prefers role="impl" sources.
    For vllm/sglang: includes sources whose library matches, regardless of role.
    """
    try:
        from kernel_prompt import KERNEL_SPECS
        for ks in KERNEL_SPECS:
            if ks.kernel_type == kernel_spec:
                paths = []
                if library == "aiter":
                    for source in ks.sources:
                        if source.role != "impl":
                            continue
                        for p in source.paths:
                            full = REPO_ROOT / "tools" / "rocm" / p
                            if full.exists():
                                paths.append(str(full))
                else:
                    for source in ks.sources:
                        src_lib = getattr(source, "library", "")
                        if src_lib == library or source.role == "wrapper":
                            for p in source.paths:
                                full = REPO_ROOT / "tools" / "rocm" / p
                                if full.exists():
                                    paths.append(str(full))
                if not paths:
                    for source in ks.sources:
                        for p in source.paths:
                            full = REPO_ROOT / "tools" / "rocm" / p
                            if full.exists():
                                paths.append(str(full))
                return paths
    except Exception as e:
        print(f"  [warn] Could not resolve baseline sources for {kernel_spec}: {e}")
    return []


def _fixup_aiter_imports(filepath: Path) -> None:
    """Backward-compat wrapper — delegates to _fixup_patched_imports."""
    _fixup_patched_imports(filepath, library="aiter")


def _fixup_patched_imports(filepath: Path, library: str = "aiter") -> None:
    """Fix import paths in a patched module file for the given library.

    For aiter: resolves relative imports, flattens _triton_kernels,
    stubs AiterTritonLogger, guards top-level aiter imports.
    For vllm/sglang: resolves relative imports to absolute package paths.
    For pytorch: minimal fixup (typically stable imports).
    """
    import re as _re
    try:
        text = filepath.read_text()
    except OSError:
        return
    original = text

    if library == "aiter":
        text = _fixup_aiter_imports_impl(text, filepath, _re)
    elif library == "vllm":
        text = _fixup_vllm_imports_impl(text, filepath, _re)
    elif library == "sglang":
        text = _fixup_sglang_imports_impl(text, filepath, _re)
    # pytorch: no fixup needed

    if text != original:
        filepath.write_text(text)


_AITER_OPTIONAL_IMPORTS = {
    "aiter.jit_config",
    "aiter.aiter_device",
    "aiter.test_common",
    "aiter.test_utils",
    "aiter.ops.triton.utils.logger",
    "aiter.ops.triton.utils.testing",
}


def _fixup_aiter_imports_impl(text: str, filepath: Path, _re) -> str:
    """Aiter-specific import fixups."""
    fpath_str = str(filepath.resolve())
    aiter_idx = fpath_str.find("/aiter/aiter/")
    if aiter_idx >= 0:
        rel = fpath_str[aiter_idx + len("/aiter/aiter/"):]
        parts = Path(rel).parent.parts
    else:
        parts = ()

    def _pkg_prefix(ndots: int) -> str:
        """Map dot count to package prefix: . = current, .. = parent, etc.
        Aiter parts are relative to the inner aiter/ so we prepend 'aiter'.
        """
        up = ndots - 1
        keep = len(parts) - up
        if keep >= 1:
            return ".".join(["aiter"] + list(parts[:keep]))
        return "aiter"

    # Handle bare `from . import foo` FIRST
    text = _re.sub(
        r'^from\s+(\.{1,})\s+import',
        lambda m: f"from {_pkg_prefix(len(m.group(1)))} import",
        text,
        flags=_re.MULTILINE,
    )

    # Handle `from .foo import bar` (require word char after dots)
    text = _re.sub(
        r'^from\s+(\.{1,})([\w][\w.]*)\s+import',
        lambda m: f"from {_pkg_prefix(len(m.group(1)))}.{m.group(2)} import",
        text,
        flags=_re.MULTILINE,
    )

    import importlib as _il
    _flatten_needed = False
    try:
        _il.import_module("aiter.ops.triton._triton_kernels.attention.pa_decode")
    except (ImportError, ModuleNotFoundError):
        _flatten_needed = True
    if _flatten_needed:
        text = _re.sub(
            r'(aiter\.ops\.triton\._triton_kernels)\.\w+\.([\w.]+)',
            r'\1.\2',
            text,
        )
    if "AiterTritonLogger" in text:
        text = _re.sub(
            r'^from\s+aiter\.ops\.triton\.utils\.logger\s+import\s+AiterTritonLogger\s*$',
            'class AiterTritonLogger:\n    def info(self, *a, **kw): pass',
            text,
            flags=_re.MULTILINE,
        )

    # Only wrap known-optional aiter imports in try/except, not all of them
    def _wrap_if_optional(match: _re.Match) -> str:
        line = match.group(0)
        for opt_mod in _AITER_OPTIONAL_IMPORTS:
            if opt_mod in line:
                return f"try:\n    {line}\nexcept ImportError:\n    pass"
        return line

    text = _re.sub(
        r'^(?:from\s+aiter\s+import\s+.+|import\s+aiter\S*)$',
        _wrap_if_optional,
        text,
        flags=_re.MULTILINE,
    )
    return text


def _fixup_vllm_imports_impl(text: str, filepath: Path, _re) -> str:
    """vLLM-specific import fixups: resolve relative imports to absolute."""
    fpath_str = str(filepath.resolve())
    vllm_idx = fpath_str.find("/vllm/")
    if vllm_idx >= 0:
        after = fpath_str[vllm_idx + 1:]
        parts = Path(after).parent.parts  # e.g. ('vllm', 'v1', 'attention')
    else:
        parts = ()

    def _pkg_prefix(ndots: int) -> str:
        """Map dot count to package prefix: . = current, .. = parent, etc."""
        up = ndots - 1
        keep = len(parts) - up
        if keep >= 1:
            return ".".join(parts[:keep])
        return "vllm"

    # Handle bare `from . import foo` FIRST
    text = _re.sub(
        r'^from\s+(\.{1,})\s+import',
        lambda m: f"from {_pkg_prefix(len(m.group(1)))} import",
        text,
        flags=_re.MULTILINE,
    )

    # Handle `from .foo import bar` (require word char after dots)
    text = _re.sub(
        r'^from\s+(\.{1,})([\w][\w.]*)\s+import',
        lambda m: f"from {_pkg_prefix(len(m.group(1)))}.{m.group(2)} import",
        text,
        flags=_re.MULTILINE,
    )
    return text


def _fixup_sglang_imports_impl(text: str, filepath: Path, _re) -> str:
    """SGLang-specific import fixups: resolve relative imports to absolute."""
    fpath_str = str(filepath.resolve())
    sg_idx = fpath_str.find("/sglang/")
    if sg_idx >= 0:
        after = fpath_str[sg_idx + 1:]
        parts = Path(after).parent.parts
    else:
        parts = ()

    def _pkg_prefix(ndots: int) -> str:
        up = ndots - 1
        keep = len(parts) - up
        if keep >= 1:
            return ".".join(parts[:keep])
        return "sglang"

    # Handle bare `from . import foo` FIRST
    text = _re.sub(
        r'^from\s+(\.{1,})\s+import',
        lambda m: f"from {_pkg_prefix(len(m.group(1)))} import",
        text,
        flags=_re.MULTILINE,
    )

    # Handle `from .foo import bar` (require word char after dots)
    text = _re.sub(
        r'^from\s+(\.{1,})([\w][\w.]*)\s+import',
        lambda m: f"from {_pkg_prefix(len(m.group(1)))}.{m.group(2)} import",
        text,
        flags=_re.MULTILINE,
    )
    return text


def _read_source_code(path: str, max_lines: int = 500) -> str:
    """Read source file content, truncating if very long."""
    try:
        text = Path(path).read_text()
        lines = text.splitlines()
        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n\n... ({len(lines) - max_lines} more lines truncated)"
        return text
    except Exception as e:
        return f"(could not read: {e})"


def _snapshot_environment() -> dict:
    """Capture env vars and package versions for reproducibility."""
    import importlib.metadata as _meta
    env_vars = {k: v for k, v in os.environ.items()
                if k.startswith("VLLM_ROCM_USE_AITER") or k.startswith("TRITON_")}
    versions: dict[str, str] = {}
    for pkg in ("aiter", "vllm", "torch", "triton", "sglang"):
        try:
            versions[pkg] = _meta.version(pkg)
        except Exception:
            versions[pkg] = "not installed"
    return {"env_vars": env_vars, "package_versions": versions}


def _extract_timing_from_raw(raw: dict) -> tuple[float, float]:
    """Extract (baseline_ms, optimized_ms) from a Magpie compare raw result.

    Checks multiple places in the result dict:
    1. Top-level baseline_ms / optimized_ms (set by _measure_speedup fallback)
    2. kernel_results list from Magpie compare (performance_result.summary.Duration)
    3. _benchmark_speedup if present, combined with optimized time
    """
    if not raw:
        return 0.0, 0.0

    b_ms = float(raw.get("baseline_ms", 0) or 0)
    o_ms = float(raw.get("optimized_ms", 0) or 0)
    if b_ms > 0 and o_ms > 0:
        return b_ms, o_ms

    results = raw.get("results", raw)
    kernel_results = results.get("kernel_results", [])
    if len(kernel_results) >= 2:
        b_ms = _time_from_kr(kernel_results[0])
        o_ms = _time_from_kr(kernel_results[-1])
        if b_ms > 0 and o_ms > 0:
            return b_ms, o_ms

    return b_ms, o_ms


def _time_from_kr(kr: dict) -> float:
    """Extract kernel duration in ms from a Magpie kernel_result entry."""
    perf = kr.get("performance_result") or kr.get("performance", {})
    if isinstance(perf, dict):
        for key in ("avg_time_ms", "mean_ms", "avg_ms", "time_ms", "median_ms"):
            v = perf.get(key)
            if v is not None and float(v) > 0:
                return float(v)
        summary = perf.get("summary", {})
        if isinstance(summary, dict):
            dur = summary.get("Duration")
            if dur is not None and float(dur) > 0:
                return float(dur)
        metrics = perf.get("metrics", {})
        for key in ("avg_time_ms", "mean_ms", "avg_ms"):
            v = metrics.get(key)
            if v is not None and float(v) > 0:
                return float(v)
        kernels = perf.get("kernels", [])
        if kernels and isinstance(kernels, list):
            total_dur = sum(float(k.get("duration", 0) or 0) for k in kernels)
            if total_dur > 0:
                return total_dur
    return 0.0


def _create_task_config(
    task_dir: Path,
    kernel: BottleneckKernel,
    config: WorkloadConfig,
    baseline_paths: list[str],
    benchmark_config: Optional[dict] = None,
) -> Path:
    """Pre-create config.yaml in the task directory with baseline paths.

    If no local baseline.py exists, creates a minimal one that copies the
    original source and adds a __main__ block for standalone execution.
    """
    spec = kernel.matched_kernel_spec or "unknown"
    ext = ".py" if kernel.category == "triton" else ".hip"

    local_baseline_ref = task_dir / f"baseline_ref{ext}"
    local_baseline = task_dir / f"baseline{ext}"
    if local_baseline_ref.exists():
        baseline_path = f"./baseline_ref{ext}"
    elif local_baseline.exists():
        baseline_path = f"./baseline{ext}"
    elif baseline_paths:
        import shutil
        src = Path(baseline_paths[0])
        if src.exists():
            shutil.copy2(src, local_baseline)
            origin_lib = getattr(kernel, "origin_library", "aiter") or "aiter"
            _fixup_patched_imports(local_baseline, library=origin_lib)
            baseline_path = f"./baseline{ext}"
        else:
            baseline_path = baseline_paths[0]
    else:
        baseline_path = ""

    kernel_python = getattr(config, "kernel_python", "") or _detect_kernel_python()

    # Ground truth lookup via shared helper
    gt_spec = get_ground_truth_spec(spec)
    correctness_cfg, gt_mode = build_correctness_config(
        gt_spec, rocm_root=REPO_ROOT / "tools" / "rocm",
    )

    if gt_mode == "pytorch":
        correctness_cmd = f"{kernel_python} solution{ext}"
        if gt_spec and gt_spec.pytorch_reference_code:
            ref_path = task_dir / "reference.py"
            ref_path.write_text(gt_spec.pytorch_reference_code)
            if gt_spec.test_shapes_code:
                shapes_path = task_dir / "test_shapes.py"
                shapes_path.write_text(gt_spec.test_shapes_code)
            print(f"    Correctness mode: pytorch (real reference from {gt_spec.source_library}/{gt_spec.source_file})")
        else:
            print(f"    Correctness mode: pytorch (baseline-vs-solution, no library reference found)")
        correctness_cfg["command"] = correctness_cmd
    else:
        print(f"    Correctness mode: {gt_mode}")

    cfg = {
        "gpu": {"device": 0, "arch": config.gpu_arch},
        "baseline": {"path": baseline_path},
        "optimized": {"path": f"./solution{ext}"},
        "correctness": correctness_cfg,
        "performance": {
            "command": kernel_python,
            "warmup_iterations": 10,
            "iterations": 100,
        },
        "_pipeline_metadata": {
            "kernel_type": spec,
            "framework": config.framework or "vllm",
            "generated_by": "workload_optimizer.py",
            "tamper_protected": True,
            "correctness_mode": correctness_cfg.get("mode", "pytorch"),
        },
    }

    config_path = task_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return config_path


# ---------------------------------------------------------------------------
# Step 5: Per-kernel optimization loop
# ---------------------------------------------------------------------------

# Kernel type classification for optimization strategy hints
_KERNEL_TYPE_TO_MCP_CLASS = {
    "flash_attn_prefill": "attention", "paged_attn_decode": "attention",
    "paged_attn_decode_gluon": "attention", "mla_attn": "attention",
    "gemm_bf16": "gemm", "gemm_w8a8": "gemm",
    "fused_moe": "moe", "rms_norm": "normalization",
    "silu_mul": "elementwise", "act_quant_fp8": "quantization",
    "rope_embedding": "elementwise", "kv_cache_ops": "elementwise",
    "all_reduce": "reduction",
}

# Library alternatives by kernel class (embedded from source-finder knowledge)
_LIBRARY_ALTERNATIVES = {
    "gemm": [
        "hipBLASLt: Tuned GEMM with epilogue fusion — best for standard shapes, use torch.addmm/torch.mm",
        "rocBLAS: General-purpose BLAS — good baseline, auto-tuned via Tensile",
        "CK (composable_kernel): Tile-based C++ templates — maximum control, complex setup",
        "Triton: Custom tiled GEMM — good for non-standard shapes or fused epilogues",
    ],
    "attention": [
        "aiter Flash Attention: Triton-based flash attn — primary path for ROCm",
        "CK FMHA: Composable Kernel fused MHA — alternative high-perf path",
        "Custom Triton: Write custom attention with online softmax and tiling",
    ],
    "moe": [
        "aiter fused_moe: Fused gate+topk+expert GEMM — primary ROCm implementation",
        "Custom Triton: Token sorting + batched GEMM — good for unbalanced expert loads",
    ],
    "normalization": [
        "aiter Triton RMSNorm: Memory-bound kernel — maximize bandwidth with float4 loads",
        "CK normalization: Template-based alternative",
        "Fused approach: Combine with subsequent activation (SwiGLU, SiLU) to save memory round-trip",
    ],
    "elementwise": [
        "Fuse with adjacent ops: Eliminate intermediate memory traffic",
        "Vectorized loads/stores: Use float4 for 128-byte coalescing on AMD",
    ],
    "quantization": [
        "aiter FP8 quant: Per-token dynamic quantization kernels",
        "Fused quant+GEMM: Combine quantization with the subsequent GEMM to avoid extra memory pass",
    ],
}


def _profile_baseline_kernel(
    baseline_path: str,
    task_dir: Path,
    gpu_arch: str = "gfx950",
) -> str:
    """Run rocprof on the baseline kernel and return a formatted profiling summary.

    Returns empty string if profiling fails.
    """
    try:
        cmd = [
            "rocprof", "--stats",
            sys.executable, str(baseline_path),
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=str(task_dir),
        )
        if proc.returncode != 0:
            return ""

        lines = []
        output = proc.stdout + proc.stderr
        for line in output.splitlines():
            if any(kw in line.lower() for kw in (
                "kernel", "duration", "occupancy", "lds", "vgpr", "sgpr",
                "memory", "bandwidth", "cache", "flop",
            )):
                lines.append(line.strip())

        if not lines:
            return ""

        section = "## Baseline Profiling Results (rocprof)\n\n"
        section += "```\n"
        section += "\n".join(lines[:20])
        section += "\n```\n"
        section += "\nUse this data to identify whether the kernel is compute-bound or memory-bound.\n"
        return section

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return ""


def _build_multi_strategy_block(kernel_spec: str) -> str:
    """Build the multi-strategy exploration instruction block."""
    kernel_class = _KERNEL_TYPE_TO_MCP_CLASS.get(kernel_spec, "elementwise")
    alternatives = _LIBRARY_ALTERNATIVES.get(kernel_class, [])

    alt_text = "\n".join(f"  - {a}" for a in alternatives)

    return textwrap.dedent(f"""\
## Optimization Strategy (explore multiple approaches, keep the best)

You MUST explore at least TWO of these approaches, benchmark both with
mcp__magpie__compare, and keep the fastest correct one as your final solution.py.

### Strategy 1: Library Dispatch
- Replace the kernel with optimized library calls (hipBLASLt, rocBLAS, torch.mm/addmm)
- Best for standard ops (GEMM, normalization) where libraries are heavily tuned
- Use source-finder `find_library_alternative` to find candidates
- Known alternatives for {kernel_class}:
{alt_text}

### Strategy 2: Custom Triton/HIP Kernel
- Write a purpose-built kernel with optimal tiling, MFMA usage, and memory access
- Best for fused ops, non-standard shapes, or when library dispatch has overhead
- Use rag-server `get_optimization_snippet` for AMD-specific patterns

### Strategy 3: Kernel Fusion
- Fuse this kernel with adjacent operations to eliminate intermediate memory traffic
- Best for element-wise ops, activation+norm, attention components
- Use fusion-advisor `detect_fusion_opportunities` to find candidates

Write each attempt as a separate function, compare using mcp__magpie__compare,
and export the fastest correct implementation as your final solution.py.
""")


def _build_reference_section(kernel_spec: str, baseline_sources: list[str]) -> str:
    """Build a reference implementations section from kernel spec sources."""
    kernel_def = KERNEL_MAP.get(kernel_spec)
    if not kernel_def or not kernel_def.sources:
        return ""

    lines = ["## Reference Implementations (from ROCm libraries)\n"]
    for src in kernel_def.sources:
        role_tag = f" ({src.role})" if src.role != "impl" else " (primary)"
        lines.append(f"### {src.library}{role_tag}")
        for p in src.paths:
            full = REPO_ROOT / "tools" / "rocm" / p
            if full.exists():
                lines.append(f"  - `tools/rocm/{p}` (exists on disk)")
            else:
                lines.append(f"  - `tools/rocm/{p}`")
        lines.append("")

    lines.append(
        "Use `source-finder find_kernel_source` to search for additional implementations "
        "not listed above.\n"
    )
    return "\n".join(lines)


def _build_kernel_prompt(
    kernel: BottleneckKernel,
    config: WorkloadConfig,
    benchmark_config: dict,
    task_dir: Path,
) -> str:
    """Build a prompt using the rich kernel_prompt.py template + actual source code.

    Includes profiling data, multi-strategy instructions, and reference implementations.
    """
    spec = kernel.matched_kernel_spec or "unknown"
    model_id = benchmark_config.get("benchmark", {}).get("model", config.framework)
    framework = config.framework or "vllm"
    gpu_arch = config.gpu_arch or DEFAULT_TARGET
    origin_lib = kernel.origin_library if kernel.origin_library != "unknown" else "aiter"
    baseline_sources = _find_baseline_sources(spec, library=origin_lib)

    rich_prompt = _try_build_rich_prompt(spec, model_id, framework, gpu_arch, origin_lib)

    source_sections = []
    for src_path in baseline_sources[:3]:
        code = _read_source_code(src_path)
        source_sections.append(f"### Source: {src_path}\n```python\n{code}\n```")

    sources_text = "\n\n".join(source_sections) if source_sections else (
        "No source files found on disk. Use source-finder MCP to search for kernel "
        "implementations under tools/rocm/aiter/ and tools/rocm/composable_kernel/."
    )

    # Profiler-guided: run rocprof on baseline before building prompt
    profiling_section = ""
    local_baseline = task_dir / "baseline.py"
    if local_baseline.exists():
        profiling_section = _profile_baseline_kernel(
            str(local_baseline), task_dir, gpu_arch,
        )

    multi_strategy = _build_multi_strategy_block(spec)
    reference_section = _build_reference_section(spec, baseline_sources)

    profiling_context = textwrap.dedent(f"""\
## Profiling Context

**Profiler kernel name:** `{kernel.name}`
**Category:** {kernel.category}
**Current GPU time:** {kernel.percent_total:.1f}% of total ({kernel.total_time_us/1000:.1f} ms over {kernel.calls} calls)
**Task directory:** `{task_dir}`

{profiling_section}

{reference_section}

## Baseline Source Code (from disk)

{sources_text}

{multi_strategy}

## Your Task

1. Read the skill files listed above for domain-specific optimization knowledge.
2. Read and understand the baseline kernel implementation above.
3. Use MCP tools: source-finder to find all implementations, rag-server for patterns,
   gpu-info for arch hints, fusion-advisor for fusion opportunities.
4. Identify performance bottlenecks (memory access patterns, compute utilization, occupancy).
5. Explore MULTIPLE optimization strategies (see above) and benchmark each.
6. Write the best optimized version to: `{task_dir}/solution.py`
7. Use mcp__magpie__compare to validate correctness and measure speedup.
8. The config.yaml at `{task_dir}/config.yaml` already has the baseline path set.

## IMPORTANT Constraints
- Your solution must be functionally equivalent to the baseline (same inputs → same outputs).
- Do NOT modify files outside `{task_dir}/`.
- Focus on real performance improvements, not just code style changes.
- Include the kernel function with the same signature as the baseline.
""")

    if rich_prompt:
        return rich_prompt + "\n\n" + profiling_context
    return profiling_context


def _try_build_rich_prompt(
    kernel_spec: str, model_id: str, framework: str, gpu_arch: str,
    origin_library: str = "aiter",
) -> str:
    """Try to build a rich prompt using kernel_prompt.py templates.

    Returns the rich prompt text, or empty string if the model/kernel isn't found.
    """
    try:
        kernel = KERNEL_MAP.get(kernel_spec)
        if not kernel:
            return ""

        model = None
        for m in MODELS:
            if m.hf_id == model_id:
                model = m
                break
        if not model:
            for m in MODELS:
                if m.hf_id.split("/")[-1].lower() in model_id.lower():
                    model = m
                    break
        if not model:
            model = MODELS[0]

        result = _build_rich_kernel_prompt(
            model=model,
            kernel=kernel,
            framework=framework,
            gpu_arch=gpu_arch,
            origin_library=origin_library,
        )
        return result.get("prompt", "")
    except Exception as e:
        print(f"    [warn] Could not build rich prompt: {e}")
        return ""


def _make_kernel_task_id(kernel: BottleneckKernel, config: WorkloadConfig) -> str:
    spec = kernel.matched_kernel_spec or kernel.name[:30].replace(" ", "_")
    spec = spec.replace("::", "_").replace("<", "").replace(">", "")
    framework = config.framework or "vllm"
    return f"workload__{framework}__{spec}"


def _serialize_agent_messages(messages: list) -> list[dict]:
    """Serialize Claude/Codex SDK message objects into JSON-safe dicts."""
    serialized = []
    for msg in messages:
        if isinstance(msg, dict):
            serialized.append(msg)
            continue
        entry: dict = {}
        if hasattr(msg, "role"):
            entry["role"] = str(msg.role)
        if hasattr(msg, "type"):
            entry["type"] = str(msg.type)
        if hasattr(msg, "content"):
            content = msg.content
            if isinstance(content, str):
                entry["content"] = content[:2000]
            elif isinstance(content, list):
                blocks = []
                for block in content:
                    if hasattr(block, "text"):
                        blocks.append({"type": "text", "text": str(block.text)[:1000]})
                    elif hasattr(block, "name"):
                        blocks.append({
                            "type": "tool_use",
                            "name": str(block.name),
                            "input_keys": list(block.input.keys()) if hasattr(block, "input") else [],
                        })
                    else:
                        blocks.append({"type": str(type(block).__name__)})
                entry["content"] = blocks
        if hasattr(msg, "num_turns"):
            entry["num_turns"] = msg.num_turns
        if hasattr(msg, "total_cost_usd"):
            entry["cost_usd"] = msg.total_cost_usd
        serialized.append(entry)
    return serialized


def _run_agent_iteration(
    task_dir: Path,
    prompt: str,
    config: WorkloadConfig,
    iteration: int,
    previous_reflection: str = "",
) -> tuple[list[dict], bool]:
    full_prompt = prompt
    if previous_reflection:
        full_prompt = previous_reflection + "\n\n---\n\n" + prompt

    if config.dry_run:
        solution = task_dir / "solution.py"
        if not solution.exists():
            solution.write_text(textwrap.dedent("""\
                import numpy as np

                def kernel_fn(*args, **kwargs):
                    return args[0] if args else None
            """))
        return [{"role": "assistant", "content": f"[dry-run] iteration {iteration}"}], True

    try:
        from agents.backends import run_agent_task
        messages, solution_written = run_agent_task(
            prompt=full_prompt,
            cwd=task_dir,
            model=config.agent_model,
            max_turns=config.max_turns_per_iter,
            agent=config.agent_backend,
            system_prompt=SYSTEM_PROMPT,
            solution_path=task_dir / "solution.py",
        )
        return messages, solution_written
    except Exception as e:
        print(f"    [agent] Error: {e}")
        import traceback
        traceback.print_exc()
        return [{"type": "error", "error": str(e)[:500]}], False


def _optimize_kernel(
    kernel: BottleneckKernel,
    config: WorkloadConfig,
    benchmark_config: dict,
) -> KernelOptResult:
    task_id = _make_kernel_task_id(kernel, config)
    task_dir = config.output_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    spec = kernel.matched_kernel_spec or "unknown"
    print(f"\n    {'='*55}")
    print(f"    Optimizing: {spec} ({kernel.category})")
    print(f"    Task dir:   {task_dir}")
    print(f"    Profiler:   {kernel.name[:70]}")
    print(f"    GPU time:   {kernel.percent_total:.1f}%")
    print(f"    {'='*55}")

    origin_lib = kernel.origin_library if kernel.origin_library != "unknown" else "aiter"
    opt_result = KernelOptResult(
        kernel_name=kernel.name,
        kernel_spec=spec,
        category=kernel.category,
        origin_library=origin_lib,
    )

    baseline_sources = _find_baseline_sources(spec, library=origin_lib)
    if baseline_sources:
        print(f"    Baseline sources: {[Path(p).name for p in baseline_sources]}")
    else:
        print(f"    [warn] No baseline sources found for {spec}")

    _create_task_config(task_dir, kernel, config, baseline_sources, benchmark_config)
    prompt = _build_kernel_prompt(kernel, config, benchmark_config, task_dir)

    best_kr: Optional[KernelResult] = None
    reflection_prompt = ""
    total_agent_turns = 0

    pre_existing_solution = find_solution(task_dir)
    if pre_existing_solution:
        print(f"    Pre-existing solution found: {pre_existing_solution.name}")
        print(f"    Skipping agent, going straight to grading.")

    for iteration in range(1, config.max_iterations + 1):
        print(f"\n    --- Iteration {iteration}/{config.max_iterations} ---")

        if pre_existing_solution and iteration == 1:
            solution_written = True
        else:
            print("    Running agent...")
            t0 = time.monotonic()
            messages, solution_written = _run_agent_iteration(
                task_dir, prompt, config, iteration, reflection_prompt,
            )
            agent_time = time.monotonic() - t0
            for msg in messages:
                if hasattr(msg, "num_turns"):
                    total_agent_turns += getattr(msg, "num_turns", 0)
            # Capture agent trace
            opt_result.agent_trace.extend(_serialize_agent_messages(messages))
            print(f"    Agent completed in {agent_time:.1f}s")

            # Agent may have written a solution before crashing
            if not solution_written and find_solution(task_dir):
                print("    Agent crashed but solution file exists — grading it.")
                solution_written = True

        if not solution_written:
            print("    Agent did not write a solution.")
            opt_result.error = "Agent did not produce solution.py"
            if iteration < config.max_iterations:
                delay = min(5 * iteration, 15)
                print(f"    Retrying in {delay}s...")
                time.sleep(delay)
            continue

        print("    Grading with Magpie...")
        kr = grade_task(task_dir, isolate_caches=True, gpu_device=0)
        print(f"      compiled={kr.compiled} correct={kr.correct} "
              f"speedup={kr.speedup:.2f}x score={kr.score:.0f}")
        if kr.error:
            print(f"      error: {kr.error[:200]}")

        if best_kr is None or kr.score > best_kr.score:
            best_kr = kr
            best_solution = find_solution(task_dir)
            if best_solution:
                shutil.copy2(best_solution, task_dir / "solution_best.py")
                shutil.copy2(best_solution, task_dir / f"solution_iter{iteration}.py")

        opt_result.iterations_used = iteration

        if kr.score >= config.score_threshold:
            print(f"    Target reached: score={kr.score:.0f} >= {config.score_threshold}")
            break

        # Iterative profiling: profile solution if compiled and correct
        solution_profile = ""
        if kr.compiled and kr.correct:
            sol = find_solution(task_dir)
            if sol:
                solution_profile = _profile_baseline_kernel(
                    str(sol), task_dir, config.gpu_arch or DEFAULT_TARGET,
                )

        reflection_prompt = reflect(
            kr, task_dir, iteration,
            kernel_type=spec,
            target_speedup=config.score_threshold / 100.0,
            min_speedup=MIN_SPEEDUP_FOR_REINJECTION,
            profile_data=solution_profile,
        )

        if not should_continue(kr, iteration, config.max_iterations, config.score_threshold):
            print(f"    Stopping: score={kr.score:.0f}")
            break

    # Restore best solution if a later iteration overwrote it
    best_snapshot = task_dir / "solution_best.py"
    if best_snapshot.exists():
        target = task_dir / "solution.py"
        shutil.copy2(best_snapshot, target)
        print(f"    Restored best solution (score: {best_kr.score:.0f})" if best_kr else "")

    if best_kr:
        raw = best_kr.raw or {}
        opt_result.compiled = best_kr.compiled
        opt_result.correct = best_kr.correct
        opt_result.speedup = best_kr.speedup
        opt_result.score = best_kr.score
        b_ms, o_ms = _extract_timing_from_raw(raw)
        if b_ms == 0 and o_ms == 0 and best_kr.speedup > 0:
            o_ms = 1.0
            b_ms = best_kr.speedup * o_ms
        opt_result.baseline_ms = b_ms
        opt_result.optimized_ms = o_ms
        opt_result.agent_turns = total_agent_turns
        opt_result.error = best_kr.error

        model_cfg = _load_model_config(benchmark_config)
        solution = find_solution(task_dir)
        if solution and model_cfg:
            has_mismatch, shape_warnings = _validate_solution_shapes(
                solution, model_cfg, kernel_spec=spec,
            )
            opt_result.shape_mismatch = has_mismatch
            for w in shape_warnings:
                print(f"    WARNING: {w}")
    elif config.dry_run:
        opt_result.compiled = True
        opt_result.correct = True
        opt_result.speedup = 1.5
        opt_result.score = 270.0
        opt_result.baseline_ms = 10.0
        opt_result.optimized_ms = 6.67
        opt_result.iterations_used = 1

    return opt_result


# ---------------------------------------------------------------------------
# Step 6: Kernel re-injection
# ---------------------------------------------------------------------------

def _reinject_kernel(
    opt_result: KernelOptResult,
    task_dir: Path,
    config: WorkloadConfig,
    origin_library: str = "aiter",
) -> bool:
    if not opt_result.compiled or not opt_result.correct:
        print(f"    Skipping re-injection for {opt_result.kernel_spec}: "
              f"compiled={opt_result.compiled} correct={opt_result.correct}")
        return False

    if opt_result.speedup < MIN_SPEEDUP_FOR_REINJECTION:
        print(f"    Skipping re-injection for {opt_result.kernel_spec}: "
              f"speedup={opt_result.speedup:.3f}x < {MIN_SPEEDUP_FOR_REINJECTION}x threshold")
        return False

    # Guard: HIP kernels in monolithic .so cannot be reinjected
    if opt_result.category == "hip" and not _is_hip_patchable(opt_result.kernel_spec, origin_library):
        print(f"    Skipping re-injection for {opt_result.kernel_spec}: "
              f"{origin_library} compiles into monolithic _C.so (not individually patchable)")
        return False

    solution = find_solution(task_dir)
    if solution is None:
        print(f"    No solution file found in {task_dir}")
        return False

    inject_dir = config.output_dir / "reinjected"
    inject_dir.mkdir(parents=True, exist_ok=True)

    dest = inject_dir / f"{opt_result.kernel_spec}_{solution.name}"
    shutil.copy2(solution, dest)

    # Write library metadata so _apply_kernel_patches knows which library to target
    lib_meta = inject_dir / f"{dest.name}.library"
    lib_meta.write_text(origin_library)

    print(f"    Re-injected: {solution.name} -> {dest} (library={origin_library})")
    opt_result.reinjected = True
    return True


# ---------------------------------------------------------------------------
# Step 7: Final E2E benchmark
# ---------------------------------------------------------------------------


def _smoke_test_e2e(config: WorkloadConfig, baseline_tps: float) -> bool:
    """Run a single-shot E2E benchmark to catch catastrophic regressions.

    Returns True if the smoke test passes (no major regression).
    """
    if baseline_tps <= 0:
        return True
    print(f"    Smoke-test: running single E2E benchmark with patches active...")
    cleanup_inference_server()
    result = run_magpie_benchmark(
        framework=config.framework or "vllm",
        model="",
        benchmark_config_path=config.benchmark_config,
        timeout=config.benchmark_timeout,
    )
    cleanup_inference_server()
    smoke_tps = extract_tps(result)
    ratio = smoke_tps / baseline_tps if baseline_tps > 0 else 0
    print(f"    Smoke-test: {smoke_tps:.1f} tok/s vs baseline {baseline_tps:.1f} tok/s ({ratio:.2f}x)")
    if ratio < SMOKE_TEST_REGRESSION_THRESHOLD:
        print(f"    SMOKE-TEST FAILED: {ratio:.2f}x < {SMOKE_TEST_REGRESSION_THRESHOLD}x "
              f"-- rolling back all patches")
        return False
    return True


def _bisect_bad_patches(
    backups: dict[Path, Path], config: WorkloadConfig, baseline_tps: float,
) -> dict[Path, Path]:
    """Binary search for the patch(es) causing E2E regression.
    Returns the subset of backups that are safe to keep.
    """
    items = list(backups.items())
    if len(items) <= 1:
        # Single patch — test it alone
        if _smoke_test_e2e(config, baseline_tps):
            return dict(items)
        print(f"    Bisect: single patch {items[0][0]} is the culprit")
        shutil.copy2(items[0][1], items[0][0])
        _clear_pycache(items[0][0])
        _SESSION_BACKUPS.pop(str(items[0][0]), None)
        return {}

    mid = len(items) // 2
    left_items = items[:mid]
    right_items = items[mid:]

    # Temporarily restore right half to test left half in isolation
    for installed, backup in right_items:
        shutil.copy2(backup, installed)
        _clear_pycache(installed)

    left_ok = _smoke_test_e2e(config, baseline_tps)

    # Restore right half patches
    for installed, backup in right_items:
        sol_bak = Path(str(installed) + ".bak")
        if sol_bak.exists():
            shutil.copy2(sol_bak, installed)
            _clear_pycache(installed)

    good: dict[Path, Path] = {}
    if left_ok:
        good.update(dict(left_items))
        # Left is fine, problem is in right half — recurse
        right_good = _bisect_bad_patches(dict(right_items), config, baseline_tps)
        good.update(right_good)
    else:
        # Left also has problems — recurse both halves
        left_good = _bisect_bad_patches(dict(left_items), config, baseline_tps)
        good.update(left_good)
        # Re-apply left good before testing right
        right_good = _bisect_bad_patches(dict(right_items), config, baseline_tps)
        good.update(right_good)

    return good


def _check_baseline_drift(config: WorkloadConfig, original_baseline_tps: float) -> float:
    """Quick single-run baseline to detect thermal drift since original measurement."""
    if original_baseline_tps <= 0:
        return original_baseline_tps
    print(f"  Baseline drift check: running single unpatched benchmark...")
    cleanup_inference_server()
    result = run_magpie_benchmark(
        framework=config.framework or "vllm",
        model="",
        benchmark_config_path=config.benchmark_config,
        timeout=config.benchmark_timeout,
    )
    cleanup_inference_server()
    current_tps = extract_tps(result)
    if current_tps <= 0:
        print(f"  Baseline drift check: could not extract TPS, using original")
        return original_baseline_tps
    drift_ratio = current_tps / original_baseline_tps
    print(f"  Baseline drift check: {current_tps:.1f} tok/s "
          f"(original: {original_baseline_tps:.1f}, drift: {drift_ratio:.2f}x)")
    if drift_ratio < 0.85 or drift_ratio > 1.15:
        print(f"  WARNING: >15% baseline drift detected — GPU state may have changed")
    return current_tps


def _run_final_benchmark(config: WorkloadConfig, baseline_tps: float = 0.0) -> dict:
    if config.dry_run:
        return {
            "success": True, "dry_run": True,
            "throughput": {"output_throughput": 150.0, "total_token_throughput": 300.0},
            "_kernel_patches_applied": False,
        }

    if config.skip_benchmark:
        print(f"  WARNING: Loading cached result -- kernel patches NOT applied to this run")
        print(f"  WARNING: E2E throughput comparison will NOT reflect kernel optimizations")
        result = json.loads(Path(config.skip_benchmark).read_text())
        result["_kernel_patches_applied"] = False
        return result

    reinjected_dir = config.output_dir / "reinjected"
    has_patches = reinjected_dir.exists() and any(reinjected_dir.glob("*_solution.*"))

    if not has_patches:
        print(f"  No reinjected kernels found — skipping final benchmark")
        print(f"  Using baseline TPS ({baseline_tps:.1f}) as final TPS (no optimizations applied)")
        baseline_result = {}
        if config.effective_results_dir:
            state_path = config.effective_results_dir / "pipeline_state.json"
            try:
                baseline_result = json.loads(state_path.read_text()).get("baseline_result", {})
            except Exception:
                pass
        result = dict(baseline_result) if baseline_result else {
            "success": True,
            "throughput": {"output_throughput": baseline_tps, "total_token_throughput": baseline_tps * 2},
        }
        result["_kernel_patches_applied"] = False
        result["_patched_kernels"] = []
        result["_skipped_reason"] = "no_reinjected_kernels"
        return result

    _recover_orphaned_patches()
    _cleanup_stale_tmp()

    # Quick drift check before applying patches
    drift_baseline = _check_baseline_drift(config, baseline_tps)

    with tempfile.TemporaryDirectory(prefix="triton_cache_") as triton_cache:
        old_triton_cache = os.environ.get("TRITON_CACHE_DIR")
        os.environ["TRITON_CACHE_DIR"] = triton_cache

        backups, lock_fd = _apply_kernel_patches(reinjected_dir, config.gpu_arch)
        try:
            # Post-reinjection correctness gate
            failed = _verify_patched_kernels(backups, config)
            if failed:
                for fp in failed:
                    backups.pop(fp, None)

            if not backups:
                print(f"  All patches failed verification — running unpatched benchmark")
                _restore_kernel_patches({}, lock_fd)
                result = _run_benchmark_multi(config, label="final (unpatched)")
                result["_kernel_patches_applied"] = False
                result["_patched_kernels"] = []
                return result

            # E2E smoke test: catch catastrophic regressions before full benchmark
            result_meta: dict = {}
            if not _smoke_test_e2e(config, baseline_tps):
                good_patches = _bisect_bad_patches(backups, config, baseline_tps)
                if good_patches:
                    bad_patches = set(backups) - set(good_patches)
                    for fp in bad_patches:
                        shutil.copy2(backups[fp], fp)
                        _clear_pycache(fp)
                        _SESSION_BACKUPS.pop(str(fp), None)
                    print(f"    Bisect: kept {len(good_patches)} good patch(es), "
                          f"removed {len(bad_patches)} bad patch(es)")
                    result_meta = {"_bisect_removed": [str(p) for p in bad_patches]}
                    backups = good_patches

                    # Re-verify surviving patches together after bisection
                    post_bisect_failed = _verify_patched_kernels(backups, config)
                    if post_bisect_failed:
                        for fp in post_bisect_failed:
                            backups.pop(fp, None)
                        result_meta["_post_bisect_verify_failed"] = [
                            str(p) for p in post_bisect_failed
                        ]
                else:
                    for fp in list(backups.keys()):
                        shutil.copy2(backups[fp], fp)
                        _clear_pycache(fp)
                        _SESSION_BACKUPS.pop(str(fp), None)
                    _restore_kernel_patches({}, lock_fd)
                    result = _run_benchmark_multi(config, label="final (unpatched, smoke-test rollback)")
                    result["_kernel_patches_applied"] = False
                    result["_patched_kernels"] = []
                    result["_smoke_test_failed"] = True
                    return result

            print(f"  Running final E2E benchmark with {len(backups)} verified patched kernel(s)...")
            result = _run_benchmark_multi(config, label="final (patched)")
            result["_kernel_patches_applied"] = len(backups) > 0
            result["_patched_kernels"] = [str(p) for p in backups.keys()]
            result["_baseline_drift"] = {
                "original_tps": baseline_tps,
                "drift_checked_tps": drift_baseline,
                "drift_ratio": round(drift_baseline / baseline_tps, 3) if baseline_tps > 0 else 1.0,
            }
            result["_verification_failures"] = [str(p) for p in failed]
            if result_meta:
                result.update(result_meta)
        finally:
            _restore_kernel_patches(backups, lock_fd)
            if old_triton_cache is not None:
                os.environ["TRITON_CACHE_DIR"] = old_triton_cache
            else:
                os.environ.pop("TRITON_CACHE_DIR", None)

    return result


# ---------------------------------------------------------------------------
# Leaderboard management
# ---------------------------------------------------------------------------

def _save_leaderboard(
    entry: LeaderboardEntry,
    results_dir: Path,
) -> None:
    """Append entry to leaderboard.jsonl and regenerate leaderboard.json."""
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / "leaderboard.jsonl"
    json_path = results_dir / "leaderboard.json"

    with open(jsonl_path, "a") as f:
        f.write(json.dumps(entry.to_dict(), default=str) + "\n")

    entries = []
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(LeaderboardEntry.from_dict(json.loads(line)))

    agents: dict[str, list[LeaderboardEntry]] = {}
    for e in entries:
        agents.setdefault(e.agent_version, []).append(e)

    agent_summaries = []
    for version, runs in sorted(agents.items()):
        avg_arena = sum(r.arena_score for r in runs) / len(runs)
        avg_sp = sum(r.speedup for r in runs) / len(runs)
        agent_summaries.append({
            "agent_version": version,
            "runs": len(runs),
            "avg_arena_score": round(avg_arena, 2),
            "avg_speedup": round(avg_sp, 4),
            "best_arena_score": round(max(r.arena_score for r in runs), 2),
        })

    top = sorted(entries, key=lambda e: e.arena_score, reverse=True)[:20]

    summary = {
        "total_runs": len(entries),
        "agents": agent_summaries,
        "top_scores": [e.to_dict() for e in top],
        "latest_entry": entry.to_dict(),
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Leaderboard saved: {json_path}")
    print(f"  Total runs: {len(entries)}, arena_score: {entry.arena_score:.2f}")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _generate_report(
    trajectory: WorkloadTrajectoryRecord,
    kernel_results: list[KernelOptResult],
    config: WorkloadConfig,
    baseline_result: dict,
    final_result: dict,
    reward: dict,
    results_dir: Path,
    step_timings: dict[str, float] | None = None,
) -> Path:
    """Generate a comprehensive markdown report."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    baseline_tps = trajectory.baseline_tps
    final_tps = trajectory.final_tps
    tps_ratio = final_tps / baseline_tps if baseline_tps > 0 else 0.0

    latency = baseline_result.get("latency", {})
    ttft = latency.get("ttft", {})
    tpot = latency.get("tpot", {})
    e2el = latency.get("e2el", {})

    lines = [
        f"# Workload Optimization Report",
        f"",
        f"**Generated:** {ts}",
        f"**Trajectory ID:** `{trajectory.trajectory_id}`",
        f"",
        f"## Configuration",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Workload | `{trajectory.workload_id}` |",
        f"| Model | `{trajectory.model_id}` |",
        f"| Framework | `{trajectory.framework}` |",
        f"| GPU | `{config.gpu_arch}` |",
        f"| Agent | `{config.agent_model}` ({config.agent_version}) |",
        f"| Benchmark config | `{config.benchmark_config}` |",
        f"| Max iterations/kernel | {config.max_iterations} |",
        f"| Max turns/iteration | {config.max_turns_per_iter} |",
        f"| Kernel type filter | {','.join(config.kernel_types)} |",
        f"| Skip benchmark | {'Yes' if config.skip_benchmark else 'No'} |",
        f"",
        f"## Baseline E2E Performance",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Output throughput | {baseline_tps:.2f} tok/s |",
    ]

    if ttft:
        lines += [
            f"| TTFT mean | {ttft.get('mean_ms', 0):.2f} ms |",
            f"| TTFT p99 | {ttft.get('p99_ms', 0):.2f} ms |",
        ]
    if tpot:
        lines += [
            f"| TPOT mean | {tpot.get('mean_ms', 0):.2f} ms |",
            f"| TPOT p99 | {tpot.get('p99_ms', 0):.2f} ms |",
        ]
    if e2el:
        lines += [
            f"| E2E latency mean | {e2el.get('mean_ms', 0):.2f} ms |",
            f"| E2E latency p99 | {e2el.get('p99_ms', 0):.2f} ms |",
        ]

    lines += [
        f"",
        f"## Bottleneck Kernels Identified",
        f"",
        f"| # | Category | Spec | Time% | Calls | Name |",
        f"|---|----------|------|-------|-------|------|",
    ]

    for i, bk in enumerate(trajectory.bottleneck_kernels or [], 1):
        name = bk.get("name", "")[:60]
        lines.append(
            f"| {i} | {bk.get('category', '')} | {bk.get('matched_kernel_spec', '-')} | "
            f"{bk.get('percent_total', 0):.2f}% | {bk.get('calls', 0)} | `{name}` |"
        )

    lines += [
        f"",
        f"## Kernel Optimization Results",
        f"",
    ]

    for kr in kernel_results:
        status = "PASS" if kr.correct else ("COMPILE" if kr.compiled else "FAIL")
        lines += [
            f"### {kr.kernel_spec} ({kr.category})",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Status | **{status}** |",
            f"| Compiled | {'Yes' if kr.compiled else 'No'} |",
            f"| Correct | {'Yes' if kr.correct else 'No'} |",
            f"| Speedup | {kr.speedup:.3f}x |",
            f"| Score | {kr.score:.1f} |",
            f"| Baseline ms | {kr.baseline_ms:.3f} |",
            f"| Optimized ms | {kr.optimized_ms:.3f} |",
            f"| Iterations | {kr.iterations_used} |",
            f"| Re-injected | {'Yes' if kr.reinjected else 'No'} |",
        ]
        if kr.shape_mismatch:
            lines.append(f"| Shape mismatch | **YES** (test shapes differ from target model) |")
        if kr.error:
            lines.append(f"| Error | {kr.error[:100]} |")
        lines.append("")

    patches_applied = final_result.get("_kernel_patches_applied", False)
    patched_kernels = final_result.get("_patched_kernels", [])

    lines += [
        f"## Final E2E Performance",
        f"",
        f"| Metric | Baseline | Optimized | Change |",
        f"|--------|----------|-----------|--------|",
        f"| Output throughput (tok/s) | {baseline_tps:.2f} | {final_tps:.2f} | {(tps_ratio-1)*100:+.2f}% |",
        f"| Throughput ratio | 1.00x | {tps_ratio:.4f}x | |",
        f"| Kernel patches applied | {'Yes' if patches_applied else '**No**'} | {len(patched_kernels)} kernel(s) | |",
        f"",
    ]

    b_multi = baseline_result.get("_multi_run", {})
    f_multi = final_result.get("_multi_run", {})
    if b_multi or f_multi:
        lines += [
            f"## E2E Benchmark Statistics",
            f"",
            f"| Metric | Baseline | Optimized |",
            f"|--------|----------|-----------|",
        ]
        b_runs = b_multi.get("num_runs", 1) if b_multi else "N/A"
        f_runs = f_multi.get("num_runs", 1) if f_multi else "N/A"
        lines.append(f"| Runs | {b_runs} | {f_runs} |")
        lines.append(
            f"| Mean TPS | "
            f"{b_multi.get('mean_tps', baseline_tps):.1f} "
            f"+/- {b_multi.get('std_tps', 0):.1f} | "
            f"{f_multi.get('mean_tps', final_tps):.1f} "
            f"+/- {f_multi.get('std_tps', 0):.1f} |"
            if b_multi else
            f"| Mean TPS | {baseline_tps:.1f} | "
            f"{f_multi.get('mean_tps', final_tps):.1f} "
            f"+/- {f_multi.get('std_tps', 0):.1f} |"
        )
        lines.append(
            f"| CV | "
            f"{b_multi.get('cv_pct', 0):.2f}% | "
            f"{f_multi.get('cv_pct', 0):.2f}% |"
            if b_multi else
            f"| CV | N/A | {f_multi.get('cv_pct', 0):.2f}% |"
        )
        significant = _is_improvement_significant(baseline_result, final_result) \
            if b_multi and f_multi else True
        lines.append(
            f"| Significant? | -- | {'Yes' if significant else '**No** (improvement < 2*sigma)'} |"
        )
        note = b_multi.get("note") or f_multi.get("note")
        if note:
            lines.append(f"| Note | {note} | |")
        lines.append("")

    lines += [
        f"## Reward Scores",
        f"",
        f"| Component | Value |",
        f"|-----------|-------|",
        f"| Per-kernel scores | {reward.get('per_kernel_scores', [])} |",
        f"| Avg kernel score | {reward.get('avg_kernel_score', 0):.4f} |",
        f"| Normalized kernel score | {reward.get('normalized_kernel_score', 0):.4f} |",
        f"| **Model reward** | **{reward.get('model_reward', 0):.4f}** |",
        f"| Quality | {trajectory.trajectory_quality} |",
        f"",
        f"## Scoring Formulas",
        f"",
        f"**Kernel level:**",
        f"```",
        f"score = compiled x 20 + correct x 100 + (baseline_ms / optimized_ms) x 100",
        f"```",
        f"",
        f"**Model level:**",
        f"```",
        f"score = 0.5 x normalized_kernel_score + 0.5 x (optimized_tps / baseline_tps - 1)",
        f"```",
        f"",
        f"## Run Duration",
        f"",
    ]

    step_labels = {
        "benchmark": "Initial E2E benchmark",
        "identify": "Identify bottleneck kernels",
        "optimize": "Kernel optimization loop",
        "grade": "Re-grade solutions",
        "integrate": "Re-inject kernels",
        "benchmark_final": "Final E2E benchmark",
        "score": "Compute scores",
        "report": "Generate reports",
    }

    if step_timings:
        lines += [
            f"| Step | Duration |",
            f"|------|----------|",
        ]
        total_s = 0.0
        for step, label in step_labels.items():
            t = step_timings.get(step)
            if t is not None:
                total_s += t
                if t >= 60:
                    lines.append(f"| {label} | {t:.1f}s ({t/60:.1f} min) |")
                else:
                    lines.append(f"| {label} | {t:.1f}s |")
        lines.append(f"| **Total** | **{total_s:.1f}s ({total_s/60:.1f} min)** |")
    else:
        total_s = trajectory.total_duration_s
        lines.append(f"Total: {total_s:.1f}s ({total_s/60:.1f} min)")

    lines.append("")

    report_path = results_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


def _generate_replication_guide(
    config: WorkloadConfig,
    results_dir: Path,
) -> Path:
    """Generate a step-by-step CLI replication guide."""
    benchmark_path = config.skip_benchmark or "<path-to-benchmark_report.json>"
    magpie_cfg = Path(config.benchmark_config).absolute()

    lines = [
        f"# Replication Guide: Workload Optimization Trajectory",
        f"",
        f"Step-by-step instructions to reproduce this workload optimization run.",
        f"",
        f"## Prerequisites",
        f"",
        f"1. AMD GPU with ROCm installed (target: {config.gpu_arch})",
        f"2. Python 3.10+ with claude-code-sdk and anthropic packages:",
        f"   ```bash",
        f"   pip install claude-code-sdk anthropic pyyaml",
        f"   ```",
        f"3. Magpie installed at `{MAGPIE_ROOT}`:",
        f"   ```bash",
        f"   cd {MAGPIE_ROOT} && pip install -e .",
        f"   ```",
        f"4. Anthropic API key:",
        f"   ```bash",
        f"   export ANTHROPIC_API_KEY=<your-key>",
        f"   ```",
        f"",
        f"## Step 1: Run Initial E2E Benchmark (or skip if available)",
        f"",
        f"Run a fresh benchmark:",
        f"```bash",
        f"cd {MAGPIE_ROOT}",
        f"source .venv/bin/activate",
        f"python -m Magpie benchmark --benchmark-config {magpie_cfg}",
        f"```",
        f"",
        f"Or reuse an existing result:",
        f"```bash",
        f"# Available results:",
        f"ls {MAGPIE_ROOT}/results/benchmark_vllm_*/benchmark_report.json",
        f"```",
        f"",
        f"## Step 2: Run Full Optimization Trajectory",
        f"",
        f"### Option A: Skip initial benchmark (reuse existing)",
        f"```bash",
        f"cd {REPO_ROOT}",
        f"export ANTHROPIC_API_KEY=<your-key>",
        f"export MAGPIE_ROOT={MAGPIE_ROOT}",
        f"",
        f"/usr/bin/python3 workload_optimizer.py \\",
        f"  --benchmark-config {magpie_cfg} \\",
        f"  --skip-benchmark {benchmark_path} \\",
        f"  --kernel-types {','.join(config.kernel_types)} \\",
        f"  --top-k {config.top_k} \\",
        f"  --max-iterations {config.max_iterations} \\",
        f"  --max-turns {config.max_turns_per_iter} \\",
        f"  --agent-model {config.agent_model} \\",
        f"  --agent-version {config.agent_version} \\",
        f"  --output-dir {config.output_dir} \\",
        f"  --results-dir {results_dir} \\",
        f"  --leaderboard",
        f"```",
        f"",
        f"### Option B: Full run (includes benchmark)",
        f"```bash",
        f"cd {REPO_ROOT}",
        f"export ANTHROPIC_API_KEY=<your-key>",
        f"export MAGPIE_ROOT={MAGPIE_ROOT}",
        f"",
        f"/usr/bin/python3 workload_optimizer.py \\",
        f"  --benchmark-config {magpie_cfg} \\",
        f"  --kernel-types {','.join(config.kernel_types)} \\",
        f"  --top-k {config.top_k} \\",
        f"  --max-iterations {config.max_iterations} \\",
        f"  --leaderboard \\",
        f"  --results-dir {results_dir}",
        f"```",
        f"",
        f"## Step 3: Check Results",
        f"",
        f"```bash",
        f"# View the report",
        f"cat {results_dir}/report.md",
        f"",
        f"# View leaderboard",
        f"cat {results_dir}/leaderboard.json | python3 -m json.tool",
        f"",
        f"# View trajectory",
        f"cat {results_dir}/trajectory.json | python3 -m json.tool",
        f"",
        f"# View kernel optimization outputs",
        f"ls {config.output_dir}/workload__*/",
        f"```",
        f"",
        f"## Step 4: Re-run to Add Another Leaderboard Entry",
        f"",
        f"Each run appends a new entry to `leaderboard.json`. Just re-run Step 2.",
        f"The leaderboard accumulates runs and tracks the best scores.",
        f"",
        f"## Understanding the Scores",
        f"",
        f"- **Kernel score**: `compiled(20) + correct(100) + speedup_ratio(x100)`",
        f"  - Max theoretical per kernel: 320+ (with 2x speedup)",
        f"- **Model score**: `0.5 * norm_kernel + 0.5 * (optimized_tps/baseline_tps - 1)`",
        f"- **Arena score**: Combined total * 100 (for leaderboard ranking)",
        f"",
        f"## Troubleshooting",
        f"",
        f"- If agent fails: check `{config.output_dir}/workload__*/` for partial outputs",
        f"- If Magpie fails: ensure `MAGPIE_ROOT` is set and Magpie venv is active",
        f"- If scores are 0: check that config.yaml has correct baseline paths",
        f"- For API errors: verify ANTHROPIC_API_KEY is valid",
        f"",
    ]

    guide_path = results_dir / "replication_guide.md"
    guide_path.write_text("\n".join(lines))
    return guide_path


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_workload_optimization(config: WorkloadConfig) -> WorkloadTrajectoryRecord:
    t_start = time.monotonic()
    results_dir = config.effective_results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / "run.log"
    log_file = open(log_path, "w")

    class TeeWriter:
        def __init__(self, *writers):
            self.writers = writers
        def write(self, s):
            for w in self.writers:
                w.write(s)
                w.flush()
        def flush(self):
            for w in self.writers:
                w.flush()

    original_stdout = sys.stdout
    sys.stdout = TeeWriter(original_stdout, log_file)

    try:
        return _run_workload_optimization_inner(config, results_dir, t_start)
    finally:
        sys.stdout = original_stdout
        log_file.close()


def _run_workload_optimization_inner(
    config: WorkloadConfig,
    results_dir: Path,
    t_start: float,
) -> WorkloadTrajectoryRecord:

    with open(config.benchmark_config) as f:
        benchmark_cfg = yaml.safe_load(f)

    bench_section = benchmark_cfg.get("benchmark", {})
    model_id = bench_section.get("model", "unknown")
    framework = bench_section.get("framework", config.framework or "vllm")
    if not config.framework:
        config.framework = framework

    workload_id = f"workload__{framework}__{model_id.replace('/', '_')}"
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config.agent_version = f"v1.0-{run_ts}"

    trajectory = WorkloadTrajectoryRecord(
        workload_id=workload_id,
        agent_model=config.agent_model,
        agent_version=config.agent_version,
        benchmark_config_path=config.benchmark_config,
        benchmark_config=benchmark_cfg,
        framework=framework,
        model_id=model_id,
        gpu_arch=config.gpu_arch,
        skip_benchmark_used=config.skip_benchmark is not None,
    )

    print(f"\n{'='*65}")
    print(f"  WORKLOAD OPTIMIZATION TRAJECTORY")
    print(f"{'='*65}")
    print(f"  Workload:    {workload_id}")
    print(f"  Model:       {model_id}")
    print(f"  Framework:   {framework}")
    print(f"  GPU:         {config.gpu_arch}")
    print(f"  Agent:       {config.agent_model} ({config.agent_version})")
    print(f"  Config:      {config.benchmark_config}")
    print(f"  Results dir: {results_dir}")
    print(f"  Output dir:  {config.output_dir}")
    print(f"{'='*65}")

    # Guarantee fresh baseline and register session cleanup
    _ensure_clean_baseline()
    _register_session_handlers()

    state = PipelineState(results_dir)
    state.update({
        "benchmark_config_path": config.benchmark_config,
        "benchmark_config": benchmark_cfg,
        "model_id": model_id,
        "framework": framework,
        "gpu_arch": config.gpu_arch,
        "output_dir": str(config.output_dir),
        "agent_model": config.agent_model,
        "agent_version": config.agent_version,
    })

    env_snap = _snapshot_environment()
    trajectory.metadata = getattr(trajectory, "metadata", {})
    trajectory.metadata["environment_snapshot"] = env_snap
    print(f"  Environment: {len(env_snap.get('env_vars', {}))} AITER/TRITON vars, "
          f"{', '.join(f'{k}={v}' for k, v in env_snap.get('package_versions', {}).items())}")

    step_timings: dict[str, float] = {}

    # Step 1: Initial E2E Benchmark
    print(f"\n{'─'*65}")
    print(f"  Step 1: Initial E2E Benchmark")
    print(f"{'─'*65}")

    t_step = time.monotonic()
    baseline_result = _run_initial_benchmark(config)
    trajectory.baseline_benchmark = baseline_result
    baseline_tps = extract_tps(baseline_result)
    trajectory.baseline_tps = baseline_tps
    step_timings["benchmark"] = time.monotonic() - t_step

    state.update({
        "baseline_result": baseline_result,
        "baseline_tps": baseline_tps,
    })
    state.mark_step("benchmark")

    if baseline_result.get("error"):
        err = baseline_result["error"]
        print(f"  [error] Benchmark failed: {err[:200]}")
        trajectory.errors.append(f"Baseline benchmark failed: {err[:200]}")
    elif baseline_tps > 0:
        print(f"  Baseline TPS: {baseline_tps:.1f} tok/s")
    else:
        print(f"  [warn] Could not extract TPS from benchmark result")

    if config.skip_benchmark:
        print(f"  (Using cached benchmark)")
    print(f"  Duration: {step_timings['benchmark']:.1f}s")

    # Step 2-4: Bottleneck -> classify -> filter -> select
    print(f"\n{'─'*65}")
    print(f"  Step 2-4: Identify & Select Bottleneck Kernels")
    print(f"{'─'*65}")

    t_step = time.monotonic()
    selected = _select_kernels(baseline_result, config, state=state)

    # Gap analysis fallback chain: try multiple sources if profiler traces failed
    if not selected:
        # 1. Per-results-dir cache
        gap_cache = config.effective_results_dir / "gap_analysis_cache.json"
        if gap_cache.exists():
            print(f"  No kernels from benchmark — loading cached gap analysis...")
            try:
                cached = json.loads(gap_cache.read_text())
                selected = _select_kernels(cached, config, state=state)
            except Exception as e:
                print(f"  [warn] Could not load gap analysis cache: {e}")

        # 2. Per-run benchmark files (individual runs may have valid traces)
        if not selected:
            runs_dir = config.output_dir / "benchmark_runs"
            if runs_dir.exists():
                for run_file in sorted(runs_dir.glob("baseline_*.json")):
                    try:
                        run_data = json.loads(run_file.read_text())
                        run_ga = (run_data.get("gap_analysis") or {}).get("top_kernels", [])
                        if run_ga:
                            print(f"  Found gap analysis in {run_file.name}")
                            selected = _select_kernels(run_data, config, state=state)
                            if selected:
                                break
                    except Exception as e:
                        print(f"  [warn] Could not parse {run_file.name}: {e}")
                        continue

        # 3. Global cache (shared across results dirs for the same model)
        if not selected and config.benchmark_config:
            cache_key = _gap_cache_key(config.benchmark_config)
            global_cache = _GLOBAL_GAP_CACHE_DIR / f"{cache_key}.json"
            if global_cache.exists():
                print(f"  Loading gap analysis from global cache ({cache_key})...")
                try:
                    cached = json.loads(global_cache.read_text())
                    selected = _select_kernels(cached, config, state=state)
                except Exception as e:
                    print(f"  [warn] Global cache load failed: {e}")

        # 4. kernel_summary fallback (aggregated from traces, less detailed)
        if not selected:
            print(f"  Retrying with kernel_summary fallback...")
            kernel_summary = baseline_result.get("kernel_summary", {})
            if kernel_summary:
                selected = _select_kernels({"gap_analysis": {"top_kernels": [
                    {"name": k, "calls": v.get("calls", 1),
                     "self_cuda_total_us": v.get("total_us", 0),
                     "avg_time_us": v.get("avg_us", 0),
                     "pct_total": v.get("pct", 0)}
                    for k, v in kernel_summary.items()
                ]}, **baseline_result}, config, state=state)

    # Cache gap analysis for future runs (per-results-dir + global)
    if selected:
        gap_data = {"gap_analysis": baseline_result.get("gap_analysis", {})}
        # Per-results-dir cache
        gap_cache = config.effective_results_dir / "gap_analysis_cache.json"
        try:
            gap_cache.parent.mkdir(parents=True, exist_ok=True)
            gap_cache.write_text(json.dumps(gap_data, indent=2, default=str))
        except Exception as e:
            print(f"  [warn] Could not write gap analysis cache: {e}")
        # Global cache
        if config.benchmark_config:
            cache_key = _gap_cache_key(config.benchmark_config)
            global_cache = _GLOBAL_GAP_CACHE_DIR / f"{cache_key}.json"
            try:
                global_cache.parent.mkdir(parents=True, exist_ok=True)
                global_cache.write_text(json.dumps(gap_data, indent=2, default=str))
            except Exception as e:
                print(f"  [warn] Could not write global gap cache: {e}")

    trajectory.bottleneck_kernels = [k.to_dict() for k in selected]
    trajectory.kernel_type_filter = list(config.kernel_types)
    trajectory.selected_kernels = [k.matched_kernel_spec or k.name for k in selected]
    step_timings["identify"] = time.monotonic() - t_step

    if not selected:
        print(f"\n  No kernels selected for optimization. Exiting.")
        trajectory.errors.append("No kernels selected after filtering")
        trajectory.total_duration_s = time.monotonic() - t_start
        return trajectory
    print(f"  Duration: {step_timings['identify']:.1f}s")

    # Step 5: Per-kernel optimization loop
    print(f"\n{'─'*65}")
    print(f"  Step 5: Kernel Optimization Loop ({len(selected)} kernels)")
    print(f"{'─'*65}")

    t_step = time.monotonic()
    kernel_opt_results: list[KernelOptResult] = []

    for i, kernel in enumerate(selected, 1):
        print(f"\n  [{i}/{len(selected)}] Optimizing: "
              f"{kernel.matched_kernel_spec or kernel.name[:50]}")
        opt_result = _optimize_kernel(kernel, config, benchmark_cfg)
        kernel_opt_results.append(opt_result)
        trajectory.kernel_optimizations.append(opt_result.to_dict())
        print(f"    Result: compiled={opt_result.compiled} correct={opt_result.correct} "
              f"speedup={opt_result.speedup:.2f}x score={opt_result.score:.0f}")
    step_timings["optimize"] = time.monotonic() - t_step
    print(f"\n  Optimization duration: {step_timings['optimize']:.1f}s")

    # Step 6: Re-inject
    print(f"\n{'─'*65}")
    print(f"  Step 6: Re-inject Optimized Kernels")
    print(f"{'─'*65}")

    t_step = time.monotonic()
    model_cfg = _load_model_config(benchmark_cfg)
    reinjected = []
    for opt_result in kernel_opt_results:
        bk = BottleneckKernel(name=opt_result.kernel_name, matched_kernel_spec=opt_result.kernel_spec)
        task_id = _make_kernel_task_id(bk, config)
        task_dir = config.output_dir / task_id
        origin_lib = getattr(opt_result, "origin_library", "aiter") or "aiter"
        if _reinject_kernel(opt_result, task_dir, config, origin_library=origin_lib):
            reinjected.append(opt_result.kernel_spec)
            # Dispatch path validation (Fix 7)
            solution = find_solution(task_dir)
            baseline_sources = _find_baseline_sources(opt_result.kernel_spec, library=origin_lib)
            baseline_path = Path(baseline_sources[0]) if baseline_sources else None
            if solution:
                warning = _validate_optimization_relevance(
                    solution, baseline_path, benchmark_cfg, model_cfg,
                    opt_result.kernel_spec,
                )
                if warning:
                    print(f"    {warning}")

    trajectory.reinjected_kernels = reinjected
    step_timings["integrate"] = time.monotonic() - t_step
    print(f"  Re-injected {len(reinjected)} kernel(s): {reinjected}")
    print(f"  Duration: {step_timings['integrate']:.1f}s")

    # Step 7: Final E2E benchmark
    print(f"\n{'─'*65}")
    print(f"  Step 7: Final E2E Benchmark")
    print(f"{'─'*65}")

    t_step = time.monotonic()
    any_reinjected = len(reinjected) > 0
    if not any_reinjected and not config.dry_run:
        print(f"  Skipping final benchmark (no kernels met reinjection criteria)")
        final_result = baseline_result
        final_result["_kernel_patches_applied"] = False
    else:
        final_result = _run_final_benchmark(config, baseline_tps=baseline_tps)
    trajectory.final_benchmark = final_result
    final_tps = extract_tps(final_result)
    trajectory.final_tps = final_tps
    step_timings["benchmark_final"] = time.monotonic() - t_step

    if final_tps > 0:
        print(f"  Final TPS: {final_tps:.1f} tok/s")
    if baseline_tps > 0 and final_tps > 0:
        ratio = final_tps / baseline_tps
        print(f"  Throughput improvement: {ratio:.4f}x ({(ratio-1)*100:.2f}%)")
        if not _is_improvement_significant(baseline_result, final_result):
            trajectory.errors.append("E2E throughput improvement not statistically significant")
    if not final_result.get("_kernel_patches_applied", True):
        print(f"  WARNING: No kernel patches were applied for this benchmark run")
    print(f"  Duration: {step_timings['benchmark_final']:.1f}s")

    # Step 8: Compute trajectory reward
    print(f"\n{'─'*65}")
    print(f"  Step 8: Compute Trajectory Reward")
    print(f"{'─'*65}")
    t_step = time.monotonic()

    kr_dicts = [
        {"compiled": o.compiled, "correct": o.correct,
         "baseline_ms": o.baseline_ms, "optimized_ms": o.optimized_ms}
        for o in kernel_opt_results
    ]

    reward = trajectory_reward(
        kernel_results=kr_dicts,
        baseline_tps=baseline_tps,
        optimized_tps=final_tps,
    )
    trajectory.apply_reward(reward)

    print(f"  Kernel scores: {reward['per_kernel_scores']}")
    print(f"  Avg kernel score: {reward['avg_kernel_score']:.2f}")
    print(f"  Normalized kernel: {reward['normalized_kernel_score']:.4f}")
    print(f"  Model reward: {reward['model_reward']:.4f}")
    print(f"  Quality: {trajectory.trajectory_quality}")
    step_timings["score"] = time.monotonic() - t_step
    print(f"  Duration: {step_timings['score']:.1f}s")

    # Save trajectory
    trajectory.total_duration_s = time.monotonic() - t_start

    traj_dict = trajectory.to_dict() if hasattr(trajectory, "to_dict") else asdict(trajectory)
    traj_path = results_dir / "trajectory.json"
    with open(traj_path, "w") as f:
        json.dump(traj_dict, f, indent=2, default=str)
    print(f"\n  Trajectory saved: {traj_path}")

    store = get_store(config.trajectory_store)
    tid = store.save(trajectory)
    print(f"  Trajectory ID: {tid}")

    # Save results summary
    summary = {
        "trajectory_id": tid,
        "workload_id": workload_id,
        "model_id": model_id,
        "framework": framework,
        "gpu_arch": config.gpu_arch,
        "agent_model": config.agent_model,
        "agent_backend": config.agent_backend,
        "agent_version": config.agent_version,
        "baseline_tps": baseline_tps,
        "final_tps": final_tps,
        "throughput_ratio": final_tps / baseline_tps if baseline_tps > 0 else 0.0,
        "kernel_results": [o.to_dict() for o in kernel_opt_results],
        "reward": reward,
        "quality": trajectory.trajectory_quality,
        "duration_s": trajectory.total_duration_s,
        "step_timings": step_timings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = results_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Leaderboard
    if config.push_leaderboard:
        print(f"\n{'─'*65}")
        print(f"  Leaderboard")
        print(f"{'─'*65}")

        entry = LeaderboardEntry(
            agent_model=config.agent_model,
            agent_version=config.agent_version,
            task_id=workload_id,
            kernel_type="workload",
            model_id=model_id,
            gpu_arch=config.gpu_arch,
            kernel_score=reward["avg_kernel_score"],
            model_score=reward.get("model_reward", 0.0),
            arena_score=reward.get("model_reward", 0.0) * 100,
            baseline_tps=baseline_tps,
            optimized_tps=final_tps,
            throughput_ratio=final_tps / baseline_tps if baseline_tps > 0 else 0.0,
            speedup=sum(o.speedup for o in kernel_opt_results) / max(len(kernel_opt_results), 1),
            iterations_used=sum(o.iterations_used for o in kernel_opt_results),
            total_agent_turns=sum(o.agent_turns for o in kernel_opt_results),
            trajectory_id=tid,
        )
        _save_leaderboard(entry, results_dir)

    # Generate reports
    print(f"\n{'─'*65}")
    print(f"  Generating Reports")
    print(f"{'─'*65}")

    report_path = _generate_report(
        trajectory, kernel_opt_results, config,
        baseline_result, final_result, reward, results_dir,
        step_timings=step_timings,
    )
    print(f"  Report: {report_path}")

    guide_path = _generate_replication_guide(config, results_dir)
    print(f"  Replication guide: {guide_path}")

    # Summary
    print(f"\n{'='*65}")
    print(f"  WORKLOAD OPTIMIZATION COMPLETE")
    print(f"{'='*65}")
    print(f"  Workload:       {workload_id}")
    print(f"  Kernels:        {len(kernel_opt_results)} optimized, {len(reinjected)} re-injected")
    print(f"  Baseline TPS:   {baseline_tps:.1f}")
    print(f"  Final TPS:      {final_tps:.1f}")
    if baseline_tps > 0 and final_tps > 0:
        print(f"  Improvement:    {final_tps/baseline_tps:.4f}x")
    print(f"  Model reward:   {trajectory.model_reward:.4f} ({trajectory.trajectory_quality})")
    print(f"  Duration:       {trajectory.total_duration_s:.1f}s")
    print(f"  Trajectory ID:  {tid}")
    print(f"  Results dir:    {results_dir}")
    print(f"{'='*65}")

    return trajectory


# ---------------------------------------------------------------------------
# Subcommand handlers — each step can be run independently
# ---------------------------------------------------------------------------

def _load_benchmark_config(args) -> dict:
    """Load benchmark YAML config and set defaults in state."""
    with open(args.benchmark_config) as f:
        return yaml.safe_load(f)


def _init_config_from_args(args) -> WorkloadConfig:
    """Build WorkloadConfig from parsed CLI args."""
    from agents.backends import resolve_default_model

    results_dir = Path(args.results_dir)
    output_dir = Path(getattr(args, "output_dir", None) or str(results_dir / "output"))
    agent_backend = getattr(args, "agent_backend", "claude")
    agent_model = getattr(args, "agent_model", None) or resolve_default_model(agent_backend)

    return WorkloadConfig(
        benchmark_config=getattr(args, "benchmark_config", ""),
        skip_benchmark=getattr(args, "skip_benchmark", None),
        kernel_types=[t.strip() for t in getattr(args, "kernel_types", "all").split(",")],
        kernels=[k.strip() for k in getattr(args, "kernels", "all").split(",")],
        top_k=getattr(args, "top_k", 10),
        top_k_mode=getattr(args, "top_k_mode", "post-filter"),
        max_iterations=getattr(args, "max_iterations", 5),
        max_turns_per_iter=getattr(args, "max_turns", 25),
        score_threshold=getattr(args, "score_threshold", 300.0),
        agent_model=agent_model,
        agent_version=getattr(args, "agent_version", "v1.0"),
        agent_backend=agent_backend,
        framework=getattr(args, "framework", ""),
        gpu_arch=getattr(args, "gpu", "gfx950"),
        docker_image=getattr(args, "docker_image", ""),
        kernel_python=getattr(args, "kernel_python", ""),
        output_dir=output_dir,
        results_dir=results_dir,
        trajectory_store=getattr(args, "trajectory_store", "file"),
        push_leaderboard=getattr(args, "leaderboard", False),
        dry_run=getattr(args, "dry_run", False),
        num_benchmark_runs=getattr(args, "num_benchmark_runs", 5),
        benchmark_timeout=getattr(args, "benchmark_timeout", 5400),
    )


def cmd_benchmark(args):
    """Step 1: Run or load E2E benchmark."""
    t0 = time.monotonic()
    config = _init_config_from_args(args)
    state = PipelineState(config.effective_results_dir)

    benchmark_cfg = _load_benchmark_config(args)
    bench_section = benchmark_cfg.get("benchmark", {})
    model_id = bench_section.get("model", "unknown")
    framework = bench_section.get("framework", config.framework or "vllm")
    if not config.framework:
        config.framework = framework

    state.update({
        "benchmark_config_path": str(args.benchmark_config),
        "benchmark_config": benchmark_cfg,
        "model_id": model_id,
        "framework": framework,
        "gpu_arch": config.gpu_arch,
        "output_dir": str(config.output_dir),
    })

    print(f"\n  Step 1: Initial E2E Benchmark")
    print(f"  {'─'*55}")

    baseline_result = _run_initial_benchmark(config)
    baseline_tps = extract_tps(baseline_result)

    elapsed = time.monotonic() - t0
    state.update({
        "baseline_result_path": config.skip_benchmark or "",
        "baseline_result": baseline_result,
        "baseline_tps": baseline_tps,
    })
    state.mark_step("benchmark")
    state.record_step_time("benchmark", elapsed)

    if baseline_result.get("error"):
        print(f"  [error] Benchmark failed: {baseline_result['error'][:200]}")
    else:
        print(f"  Baseline TPS: {baseline_tps:.1f} tok/s")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  State saved to {state.path}")


def cmd_identify(args):
    """Step 2-4: Identify, classify, filter bottleneck kernels."""
    t0 = time.monotonic()
    config = _init_config_from_args(args)
    state = PipelineState(config.effective_results_dir)
    baseline_result = state.require("baseline_result", "benchmark")

    print(f"\n  Step 2-4: Identify & Select Bottleneck Kernels")
    print(f"  {'─'*55}")

    selected = _select_kernels(baseline_result, config, state=state)

    # Early patchability check
    patchable = [k for k in selected
                 if k.category == "triton"
                 or (k.category == "hip" and _is_hip_patchable(
                     k.matched_kernel_spec or "", k.origin_library))]
    non_patchable = [k for k in selected if k not in patchable]
    if non_patchable:
        print(f"\n  WARNING: {len(non_patchable)} selected kernel(s) cannot be reinjected:")
        for k in non_patchable:
            print(f"    - {k.matched_kernel_spec} ({k.category}, {k.origin_library}): "
                  f"monolithic .so or unsupported category")
    if not patchable and selected:
        print(f"\n  ERROR: No selected kernels are patchable. Optimization will not affect E2E performance.")
        print(f"  Consider running with --kernel-types triton to target patchable kernels only.")

    elapsed = time.monotonic() - t0
    state.update({
        "identified_kernels": [k.to_dict() for k in selected],
        "kernel_type_filter": list(config.kernel_types),
    })
    state.mark_step("identify")
    state.record_step_time("identify", elapsed)

    if not selected:
        print(f"\n  No kernels matched filters.")
    else:
        print(f"\n  {len(selected)} kernels identified ({len(patchable)} patchable). Use 'list-kernels' to view.")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  State saved to {state.path}")


def cmd_list_kernels(args):
    """Show identified kernels for interactive selection."""
    results_dir = Path(args.results_dir)
    state = PipelineState(results_dir)
    kernels_data = state.require("identified_kernels", "identify")

    kernels = [BottleneckKernel(**k) for k in kernels_data]

    print(f"\n  Available kernels for optimization ({len(kernels)}):")
    print(f"  {'─'*55}")
    print(format_bottleneck_table(kernels))
    print(f"\n  Kernel specs (use with --kernels flag):")
    for k in kernels:
        spec = k.matched_kernel_spec or k.name[:40]
        has_solution = ""
        output_dir = Path(state.get("output_dir", results_dir / "output"))
        framework = state.get("framework", "vllm")
        task_id = f"workload__{framework}__{spec}"
        task_dir = output_dir / task_id
        if (task_dir / "solution.py").exists() or (task_dir / "solution.hip").exists():
            has_solution = " [has solution]"
        print(f"    - {spec}{has_solution}")

    print(f"\n  Example: workload_optimizer.py optimize -r {results_dir} --kernels {kernels[0].matched_kernel_spec or 'name'}")


def cmd_optimize(args):
    """Step 5: Optimize selected kernels."""
    t0 = time.monotonic()
    config = _init_config_from_args(args)
    state = PipelineState(config.effective_results_dir)
    kernels_data = state.require("identified_kernels", "identify")
    benchmark_cfg = state.require("benchmark_config", "benchmark")

    all_kernels = [BottleneckKernel(**k) for k in kernels_data]

    if config.kernels and "all" not in config.kernels:
        selected = [k for k in all_kernels
                     if (k.matched_kernel_spec or "") in config.kernels
                     or k.name in config.kernels]
        if not selected:
            print(f"  [error] No kernels matched: {config.kernels}")
            print(f"  Available: {[k.matched_kernel_spec for k in all_kernels]}")
            return
    else:
        selected = all_kernels

    if not config.framework:
        config.framework = state.get("framework", "vllm")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config.agent_version = f"v1.0-{run_ts}"

    print(f"\n  Step 5: Kernel Optimization Loop ({len(selected)} kernels)")
    print(f"  {'─'*55}")

    existing_results = state.get("optimization_results", {})

    for i, kernel in enumerate(selected, 1):
        spec = kernel.matched_kernel_spec or kernel.name[:50]
        print(f"\n  [{i}/{len(selected)}] Optimizing: {spec}")
        opt_result = _optimize_kernel(kernel, config, benchmark_cfg)
        existing_results[spec] = opt_result.to_dict()
        print(f"    Result: compiled={opt_result.compiled} correct={opt_result.correct} "
              f"speedup={opt_result.speedup:.2f}x score={opt_result.score:.0f}")

    elapsed = time.monotonic() - t0
    state.update({
        "optimization_results": existing_results,
        "agent_model": config.agent_model,
        "agent_version": config.agent_version,
    })
    state.mark_step("optimize")
    state.record_step_time("optimize", elapsed)
    print(f"\n  Duration: {elapsed:.1f}s")
    print(f"  State saved to {state.path}")


def cmd_grade(args):
    """Re-grade existing solutions without re-running agent."""
    t0 = time.monotonic()
    config = _init_config_from_args(args)
    state = PipelineState(config.effective_results_dir)
    kernels_data = state.require("identified_kernels", "identify")

    if not config.framework:
        config.framework = state.get("framework", "vllm")

    all_kernels = [BottleneckKernel(**k) for k in kernels_data]
    if config.kernels and "all" not in config.kernels:
        selected = [k for k in all_kernels
                     if (k.matched_kernel_spec or "") in config.kernels]
    else:
        selected = all_kernels

    print(f"\n  Grading {len(selected)} kernel solutions")
    print(f"  {'─'*55}")

    existing_results = state.get("optimization_results", {})

    for kernel in selected:
        spec = kernel.matched_kernel_spec or "unknown"
        task_id = _make_kernel_task_id(kernel, config)
        task_dir = config.output_dir / task_id

        solution = find_solution(task_dir)
        if not solution:
            print(f"  {spec}: no solution found, skipping")
            continue

        origin_lib = getattr(kernel, "origin_library", "aiter") or "aiter"
        baseline_sources = _find_baseline_sources(spec, library=origin_lib)
        _create_task_config(task_dir, kernel, config, baseline_sources)

        print(f"  Grading {spec}...")
        kr = grade_task(task_dir)

        baseline_ms, optimized_ms = _extract_timing_from_raw(kr.raw)
        print(f"    compiled={kr.compiled} correct={kr.correct} "
              f"speedup={kr.speedup:.2f}x score={kr.score:.0f}"
              f" baseline={baseline_ms:.3f}ms optimized={optimized_ms:.3f}ms")

        prev = existing_results.get(spec, {})
        existing_results[spec] = {
            "kernel_name": kernel.name, "kernel_spec": spec,
            "category": kernel.category,
            "compiled": kr.compiled, "correct": kr.correct,
            "speedup": kr.speedup, "score": kr.score,
            "baseline_ms": baseline_ms,
            "optimized_ms": optimized_ms,
            "iterations_used": prev.get("iterations_used", 0),
            "reinjected": prev.get("reinjected", False),
            "agent_turns": prev.get("agent_turns", 0),
            "error": kr.error,
        }

    elapsed = time.monotonic() - t0
    state.set("optimization_results", existing_results)
    state.mark_step("grade")
    state.record_step_time("grade", elapsed)
    print(f"\n  Duration: {elapsed:.1f}s")
    print(f"  State saved to {state.path}")


def cmd_integrate(args):
    """Step 6: Re-inject optimized kernels."""
    t0 = time.monotonic()
    config = _init_config_from_args(args)
    state = PipelineState(config.effective_results_dir)
    opt_results = state.require("optimization_results", "optimize")

    if not config.framework:
        config.framework = state.get("framework", "vllm")

    if config.kernels and "all" not in config.kernels:
        specs_to_inject = config.kernels
    else:
        specs_to_inject = list(opt_results.keys())

    benchmark_cfg = state.get("benchmark_config", {})
    model_cfg = _load_model_config(benchmark_cfg)

    print(f"\n  Step 6: Re-inject Optimized Kernels")
    print(f"  {'─'*55}")

    reinjected = []
    for spec in specs_to_inject:
        if spec not in opt_results:
            print(f"  {spec}: not found in optimization results, skipping")
            continue
        result = opt_results[spec]
        opt = KernelOptResult(**{k: v for k, v in result.items() if k in KernelOptResult.__dataclass_fields__})

        bk = BottleneckKernel(name=opt.kernel_name, matched_kernel_spec=spec)
        task_id = _make_kernel_task_id(bk, config)
        task_dir = config.output_dir / task_id

        origin_lib = getattr(opt, "origin_library", "aiter") or "aiter"
        if _reinject_kernel(opt, task_dir, config, origin_library=origin_lib):
            reinjected.append(spec)
            opt_results[spec]["reinjected"] = True

            # Dispatch path validation (Fix 7)
            solution = find_solution(task_dir)
            baseline_sources = _find_baseline_sources(spec, library=origin_lib)
            baseline_path = Path(baseline_sources[0]) if baseline_sources else None
            if solution:
                warning = _validate_optimization_relevance(
                    solution, baseline_path, benchmark_cfg, model_cfg, spec,
                )
                if warning:
                    print(f"    {warning}")

    elapsed = time.monotonic() - t0
    state.update({
        "optimization_results": opt_results,
        "reinjected_kernels": reinjected,
    })
    state.mark_step("integrate")
    state.record_step_time("integrate", elapsed)

    print(f"  Re-injected {len(reinjected)} kernel(s): {reinjected}")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  State saved to {state.path}")


def cmd_benchmark_final(args):
    """Step 7: Run final E2E benchmark."""
    t0 = time.monotonic()
    config = _init_config_from_args(args)
    state = PipelineState(config.effective_results_dir)
    state.require("optimization_results", "optimize")

    if not config.benchmark_config:
        config.benchmark_config = state.require("benchmark_config_path", "benchmark")
    if not config.framework:
        config.framework = state.get("framework", "vllm")

    _ensure_clean_baseline()
    _register_session_handlers()

    print(f"\n  Step 7: Final E2E Benchmark")
    print(f"  {'─'*55}")

    baseline_tps = state.get("baseline_tps", 0)
    final_result = _run_final_benchmark(config, baseline_tps=baseline_tps)
    final_tps = extract_tps(final_result)
    baseline_result = state.get("baseline_result", {})

    elapsed = time.monotonic() - t0
    state.update({
        "final_result": final_result,
        "final_tps": final_tps,
    })
    state.mark_step("benchmark_final")
    state.record_step_time("benchmark_final", elapsed)

    print(f"  Final TPS: {final_tps:.1f} tok/s")
    if baseline_tps > 0 and final_tps > 0:
        ratio = final_tps / baseline_tps
        print(f"  Throughput improvement: {ratio:.4f}x ({(ratio-1)*100:.2f}%)")
        _is_improvement_significant(baseline_result, final_result)
    if not final_result.get("_kernel_patches_applied", True):
        print(f"  WARNING: No kernel patches were applied for this benchmark run")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  State saved to {state.path}")


def cmd_score(args):
    """Step 8: Compute trajectory reward and update leaderboard."""
    t0 = time.monotonic()
    config = _init_config_from_args(args)
    state = PipelineState(config.effective_results_dir)
    results_dir = config.effective_results_dir
    opt_results = state.require("optimization_results", "optimize")
    baseline_tps = state.require("baseline_tps", "benchmark")
    final_tps = state.get("final_tps", baseline_tps)

    if not config.framework:
        config.framework = state.get("framework", "vllm")

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config.agent_version = state.get("agent_version", f"v1.0-{run_ts}")

    print(f"\n  Step 8: Compute Trajectory Reward")
    print(f"  {'─'*55}")

    kr_dicts = []
    kernel_opt_results = []
    for spec, data in opt_results.items():
        kr_dicts.append({
            "compiled": data.get("compiled", False),
            "correct": data.get("correct", False),
            "baseline_ms": float(data.get("baseline_ms", 0)),
            "optimized_ms": float(data.get("optimized_ms", 0)),
        })
        kernel_opt_results.append(KernelOptResult(
            **{k: v for k, v in data.items() if k in KernelOptResult.__dataclass_fields__}
        ))

    reward = trajectory_reward(
        kernel_results=kr_dicts,
        baseline_tps=baseline_tps,
        optimized_tps=final_tps,
    )

    model_id = state.get("model_id", "unknown")
    workload_id = f"workload__{config.framework}__{model_id.replace('/', '_')}"
    tid = str(uuid.uuid4())

    print(f"  Kernel scores: {reward['per_kernel_scores']}")
    print(f"  Avg kernel score: {reward['avg_kernel_score']:.2f}")
    print(f"  Normalized kernel: {reward['normalized_kernel_score']:.4f}")
    print(f"  Model reward: {reward['model_reward']:.4f}")

    state.update({
        "reward": reward,
        "trajectory_id": tid,
        "workload_id": workload_id,
    })

    summary = {
        "trajectory_id": tid,
        "workload_id": workload_id,
        "model_id": model_id,
        "framework": config.framework,
        "gpu_arch": config.gpu_arch,
        "agent_model": config.agent_model,
        "agent_backend": config.agent_backend,
        "agent_version": config.agent_version,
        "baseline_tps": baseline_tps,
        "final_tps": final_tps,
        "throughput_ratio": final_tps / baseline_tps if baseline_tps > 0 else 0.0,
        "kernel_results": [o.to_dict() for o in kernel_opt_results],
        "reward": reward,
        "step_timings": state.step_timings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(results_dir / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    if config.push_leaderboard:
        entry = LeaderboardEntry(
            agent_model=config.agent_model,
            agent_version=config.agent_version,
            task_id=workload_id,
            kernel_type="workload",
            model_id=model_id,
            gpu_arch=config.gpu_arch,
            kernel_score=reward["avg_kernel_score"],
            model_score=reward.get("model_reward", 0.0),
            arena_score=reward.get("model_reward", 0.0) * 100,
            baseline_tps=baseline_tps,
            optimized_tps=final_tps,
            throughput_ratio=final_tps / baseline_tps if baseline_tps > 0 else 0.0,
            speedup=sum(o.speedup for o in kernel_opt_results) / max(len(kernel_opt_results), 1),
            iterations_used=sum(o.iterations_used for o in kernel_opt_results),
            total_agent_turns=sum(o.agent_turns for o in kernel_opt_results),
            trajectory_id=tid,
        )
        _save_leaderboard(entry, results_dir)

    elapsed = time.monotonic() - t0
    state.mark_step("score")
    state.record_step_time("score", elapsed)
    print(f"\n  Trajectory ID: {tid}")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  State saved to {state.path}")


def cmd_report(args):
    """Generate markdown report and replication guide."""
    config = _init_config_from_args(args)
    state = PipelineState(config.effective_results_dir)
    results_dir = config.effective_results_dir

    opt_results_raw = state.require("optimization_results", "optimize")
    baseline_tps = state.require("baseline_tps", "benchmark")
    final_tps = state.get("final_tps", baseline_tps)
    reward = state.get("reward", {})
    model_id = state.get("model_id", "unknown")
    if not config.framework:
        config.framework = state.get("framework", "vllm")
    if not config.benchmark_config:
        config.benchmark_config = state.get("benchmark_config_path", "")

    workload_id = state.get("workload_id",
                             f"workload__{config.framework}__{model_id.replace('/', '_')}")
    tid = state.get("trajectory_id", "unknown")
    config.agent_version = state.get("agent_version", config.agent_version)

    kernel_opt_results = []
    for spec, data in opt_results_raw.items():
        kernel_opt_results.append(KernelOptResult(
            **{k: v for k, v in data.items() if k in KernelOptResult.__dataclass_fields__}
        ))

    trajectory = WorkloadTrajectoryRecord(
        trajectory_id=tid,
        workload_id=workload_id,
        agent_model=config.agent_model,
        agent_version=config.agent_version,
        benchmark_config_path=config.benchmark_config,
        framework=config.framework,
        model_id=model_id,
        gpu_arch=config.gpu_arch,
        baseline_tps=baseline_tps,
        final_tps=final_tps,
        bottleneck_kernels=state.get("identified_kernels", []),
        total_duration_s=0,
    )
    trajectory.apply_reward(reward)

    baseline_result = state.get("baseline_result", {})

    report_path = _generate_report(
        trajectory, kernel_opt_results, config,
        baseline_result, state.get("final_result", {}), reward, results_dir,
        step_timings=state.step_timings,
    )
    print(f"  Report: {report_path}")

    guide_path = _generate_replication_guide(config, results_dir)
    print(f"  Replication guide: {guide_path}")

    state.mark_step("report")


def cmd_run(args):
    """Full pipeline (all steps sequentially) — backward compatible."""
    config = _init_config_from_args(args)
    run_workload_optimization(config)


def cmd_export_rl(args):
    """Export scored trajectories to RL training dataset format."""
    from export_rl_dataset import export
    import glob as _glob

    traj_dir = Path(args.trajectories_dir) if args.trajectories_dir else REPO_ROOT / "trajectories"
    export_output = Path(args.export_output_dir)

    results_dir = Path(args.results_dir)
    results_dirs = []
    if results_dir.is_dir():
        results_dirs.append(results_dir)
    for pattern in _glob.glob(str(results_dir.parent / "results_total_*")):
        p = Path(pattern)
        if p.is_dir() and p not in results_dirs:
            results_dirs.append(p)
    if not results_dirs:
        results_dirs = [REPO_ROOT / "output"]

    summary = export(
        trajectories_dir=traj_dir,
        results_dirs=results_dirs,
        output_dir=export_output,
        include_sft=args.sft,
        quality_filter=args.quality,
        min_score=args.min_score,
        gpu_arch=args.gpu_arch,
    )
    print(f"\nExport complete: {summary}")


# ---------------------------------------------------------------------------
# Standalone kernel optimization / grading (no E2E pipeline required)
# ---------------------------------------------------------------------------

@dataclass
class KernelStandaloneDefinition:
    """Everything needed to optimize or grade a single kernel standalone."""
    task_id: str = ""
    kernel_path: str = ""
    kernel_type: str = "triton"
    kernel_name: str = ""
    description: str = ""
    gpu_arch: str = "gfx950"
    framework: str = ""
    solution_path: str = ""
    ground_truth: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id:
            stem = Path(self.kernel_path).stem if self.kernel_path else "kernel"
            self.task_id = f"standalone__{stem}"
        if not self.kernel_name and self.kernel_path:
            self.kernel_name = Path(self.kernel_path).stem

    @property
    def gt_mode(self) -> str:
        return self.ground_truth.get("mode", "pytorch")


_STANDALONE_TESTCASE_TEMPLATE = '''\
#!/usr/bin/env python3
"""Correctness testcase: compare solution vs real library PyTorch reference."""
import importlib.util, os, sys, torch

_DIR = os.path.dirname(os.path.abspath(__file__))
ATOL, RTOL = 1e-3, 1e-3

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _find_fn(mod):
    for n in ("forward", "baseline_fn", "kernel_fn"):
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    for n in dir(mod):
        if not n.startswith("_"):
            obj = getattr(mod, n)
            if callable(obj) and not isinstance(obj, type):
                return obj
    return None

ref_mod = _load(os.path.join(_DIR, "reference.py"), "ref")
sol_mod = _load(os.path.join(_DIR, "{sol_filename}"), "sol")

ref_fn = _find_fn(ref_mod)
sol_fn = _find_fn(sol_mod)

try:
    shapes_mod = _load(os.path.join(_DIR, "test_shapes.py"), "shapes")
    test_inputs = shapes_mod.get_test_inputs()
except Exception:
    test_inputs = None

if test_inputs is None:
    import inspect
    sig = inspect.signature(ref_fn)
    nparams = len([p for p in sig.parameters.values()
                   if p.default is inspect.Parameter.empty])
    if nparams <= 1:
        test_inputs = [(torch.randn(256, 4096, dtype=torch.float16, device="cuda"),)]
    elif nparams == 2:
        test_inputs = [(torch.randn(256, 4096, dtype=torch.float16, device="cuda"),
                        torch.randn(4096, dtype=torch.float16, device="cuda"))]
    else:
        test_inputs = [(torch.randn(256, 4096, dtype=torch.float16, device="cuda"),) * nparams]

all_ok = True
for i, inputs in enumerate(test_inputs):
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
    with torch.no_grad():
        ref_out = ref_fn(*inputs)
        sol_out = sol_fn(*inputs)
    if isinstance(ref_out, (tuple, list)):
        ref_out = ref_out[0]
    if isinstance(sol_out, (tuple, list)):
        sol_out = sol_out[0]
    if torch.allclose(ref_out.float(), sol_out.float(), atol=ATOL, rtol=RTOL):
        md = (ref_out.float() - sol_out.float()).abs().max().item()
        print(f"PASS [shape {{i}}]: max_diff={{md:.6e}}")
    else:
        md = (ref_out.float() - sol_out.float()).abs().max().item()
        print(f"FAIL [shape {{i}}]: max_diff={{md:.6e}}")
        all_ok = False

if all_ok:
    print("\\nAll correctness checks PASSED")
    sys.exit(0)
else:
    print("\\nSome correctness checks FAILED")
    sys.exit(1)
'''


def _parse_kernel_spec(args) -> KernelStandaloneDefinition:
    """Build KernelStandaloneDefinition from --kernel-spec YAML or CLI args."""
    spec_path = getattr(args, "kernel_spec", None)

    if spec_path:
        with open(spec_path) as f:
            raw = yaml.safe_load(f)
        return KernelStandaloneDefinition(
            task_id=raw.get("task_id", ""),
            kernel_path=raw.get("kernel_path", ""),
            kernel_type=raw.get("kernel_type", "triton"),
            kernel_name=raw.get("kernel_name", ""),
            description=raw.get("description", ""),
            gpu_arch=raw.get("gpu_arch", getattr(args, "gpu", "gfx950")),
            framework=raw.get("framework", ""),
            solution_path=raw.get("solution_path", "") or getattr(args, "solution", "") or "",
            ground_truth=raw.get("ground_truth", {}),
        )

    kernel_path = getattr(args, "kernel", "")
    if not kernel_path:
        raise ValueError("Either --kernel-spec or --kernel must be provided")

    corr_mode = getattr(args, "correctness_mode", "pytorch")
    gt: dict = {"mode": corr_mode}

    if corr_mode == "pytorch":
        ref_path = getattr(args, "reference", "")
        if ref_path and Path(ref_path).exists():
            gt["pytorch_reference_code"] = Path(ref_path).read_text()
        shapes_path = getattr(args, "test_shapes", "")
        if shapes_path and Path(shapes_path).exists():
            gt["test_shapes_code"] = Path(shapes_path).read_text()
        # Auto-enrich from MANUAL_REGISTRY if no explicit reference was given
        if not gt.get("pytorch_reference_code"):
            kname = getattr(args, "kernel_name", "")
            if kname and get_ground_truth_spec is not None:
                gt_spec = get_ground_truth_spec(kname)
                if gt_spec and gt_spec.mode == "pytorch" and gt_spec.pytorch_reference_code:
                    gt["pytorch_reference_code"] = gt_spec.pytorch_reference_code
                    if gt_spec.test_shapes_code:
                        gt["test_shapes_code"] = gt_spec.test_shapes_code
                    print(f"    Auto-discovered PyTorch reference for '{kname}' from {gt_spec.source_library}")
    elif corr_mode == "library_test":
        gt["unit_test_command"] = getattr(args, "test_cmd", "")
        gt["repo_url"] = getattr(args, "repo_url", "")
        gt["working_directory"] = getattr(args, "working_directory", "")
    elif corr_mode == "accordo":
        acc_path = getattr(args, "accordo_config", "")
        if acc_path and Path(acc_path).exists():
            with open(acc_path) as f:
                gt["accordo_config"] = yaml.safe_load(f)

    return KernelStandaloneDefinition(
        kernel_path=kernel_path,
        kernel_type=getattr(args, "kernel_type", "triton"),
        kernel_name=getattr(args, "kernel_name", ""),
        description=getattr(args, "description", ""),
        gpu_arch=getattr(args, "gpu", "gfx950"),
        framework=getattr(args, "framework", ""),
        solution_path=getattr(args, "solution", "") or "",
        ground_truth=gt,
    )


def _create_standalone_task_config(
    task_dir: Path,
    kdef: KernelStandaloneDefinition,
) -> Path:
    """Create config.yaml for a standalone kernel task using ground truth mode."""
    from ground_truth import GroundTruthSpec

    kernel_python = _detect_kernel_python()
    ext = ".py" if kdef.kernel_type in ("triton", "pytorch") else ".hip"
    gt = kdef.ground_truth

    baseline_path = f"./baseline{ext}"

    # Build a temporary GroundTruthSpec so we can use the shared helper
    tmp_spec = GroundTruthSpec(
        kernel_type=kdef.kernel_name or kdef.kernel_type,
        mode=gt.get("mode", "pytorch"),
        pytorch_reference_code=gt.get("pytorch_reference_code", ""),
        test_shapes_code=gt.get("test_shapes_code", ""),
        unit_test_command=gt.get("unit_test_command", ""),
        repo_url=gt.get("repo_url", ""),
        accordo_config=gt.get("accordo_config", {}),
        source_library=gt.get("source_library", ""),
        source_file=gt.get("source_file", ""),
    )
    # For standalone, working_directory may come from CLI args directly
    correctness_cfg, gt_mode = build_correctness_config(tmp_spec)
    if gt_mode == "library_test" and gt.get("working_directory"):
        correctness_cfg["working_directory"] = gt["working_directory"]

    if gt_mode == "pytorch":
        correctness_cmd = f"{kernel_python} solution{ext}"
        ref_code = gt.get("pytorch_reference_code", "")
        shapes_code = gt.get("test_shapes_code", "")
        if ref_code:
            ref_path = task_dir / "reference.py"
            ref_preamble = "import torch\nimport math\n\n" if "torch" in ref_code and "import torch" not in ref_code else ""
            ref_path.write_text(ref_preamble + ref_code)
            if shapes_code:
                shapes_path = task_dir / "test_shapes.py"
                shapes_path.write_text(shapes_code)
            tc_path = task_dir / "testcase.py"
            sol_filename = f"solution{ext}"
            tc_path.write_text(
                _STANDALONE_TESTCASE_TEMPLATE.format(sol_filename=sol_filename)
            )
            correctness_cmd = f"{kernel_python} testcase.py"
            print(f"    Generated testcase from real library reference: {tc_path.name}")
        else:
            # No real pytorch reference available -- do NOT generate synthetic code.
            # Fall back to baseline-vs-solution comparison via Magpie.
            print(f"    Correctness mode: pytorch (baseline-vs-solution, no library reference found)")

        correctness_cfg["command"] = correctness_cmd

    # Hash pipeline-generated correctness files so the grader can detect
    # if the agent tampered with them (testcase.py, reference.py, etc.).
    protected_hashes: dict[str, str] = {}
    for fname in ("testcase.py", "reference.py", "test_shapes.py"):
        fpath = task_dir / fname
        if fpath.exists():
            protected_hashes[fname] = hashlib.sha256(fpath.read_bytes()).hexdigest()

    cfg = {
        "gpu": {"device": 0, "arch": kdef.gpu_arch},
        "baseline": {"path": baseline_path},
        "optimized": {"path": f"./solution{ext}"},
        "correctness": correctness_cfg,
        "performance": {
            "command": kernel_python,
            "warmup_iterations": 10,
            "iterations": 100,
        },
        "_pipeline_metadata": {
            "kernel_type": kdef.kernel_name or kdef.kernel_type,
            "framework": kdef.framework or "standalone",
            "generated_by": "workload_optimizer.py:standalone",
            "tamper_protected": True,
            "correctness_mode": gt_mode,
            "protected_file_hashes": protected_hashes,
        },
    }

    config_path = task_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return config_path


def _build_standalone_kernel_prompt(
    kdef: KernelStandaloneDefinition,
    task_dir: Path,
) -> str:
    """Build an optimization prompt from a standalone kernel definition.

    Works without profiler data, BottleneckKernel, or benchmark_config.
    """
    kernel_source = ""
    baseline = task_dir / "baseline.py"
    if not baseline.exists():
        baseline = task_dir / "baseline.hip"
    if baseline.exists():
        kernel_source = _read_source_code(str(baseline))

    spec = kdef.kernel_name or "kernel"
    gpu_arch = kdef.gpu_arch or DEFAULT_TARGET

    # Try to use the rich prompt system if the kernel matches a known spec
    rich_prompt = ""
    if spec in KERNEL_MAP:
        rich_prompt = _try_build_rich_prompt(
            spec, "standalone", kdef.framework or "vllm", gpu_arch, "aiter",
        )

    multi_strategy = _build_multi_strategy_block(spec) if spec in KERNEL_MAP else ""

    arch_desc = ARCH_MAP.get(gpu_arch, "")
    arch_section = f"## Target Architecture: {gpu_arch}\n{arch_desc}\n" if arch_desc else ""

    gt = kdef.ground_truth
    gt_mode = gt.get("mode", "pytorch")
    correctness_section = f"**Correctness mode:** {gt_mode}\n"
    if gt_mode == "pytorch":
        if gt.get("pytorch_reference_code"):
            correctness_section += f"""
The PyTorch reference implementation is in `{task_dir}/reference.py`.
Your solution must produce outputs matching this reference within tolerance (atol=1e-3, rtol=1e-3).
Run `mcp__magpie__compare` with the config.yaml to validate correctness and measure speedup.
"""
        else:
            correctness_section += f"""
Correctness is checked by comparing your solution output against the baseline using torch.allclose.
Use `mcp__source_finder__find_kernel_source` to locate the PyTorch reference implementation if needed.
Run `mcp__magpie__compare` with the config.yaml to validate correctness and measure speedup.
"""
    elif gt_mode == "library_test":
        correctness_section += f"""
Correctness is verified by the library test suite:
```
{gt.get('unit_test_command', '')}
```
Your solution must pass **all** tests in this suite. The test command is run with your
solution directory on PYTHONPATH, so your `solution.py` is importable during tests.
After correctness passes, `mcp__magpie__compare` measures speedup.
"""
    elif gt_mode == "accordo":
        correctness_section += """
Correctness is verified at the HSA runtime level using Accordo.
Your optimized binary must produce GPU buffer outputs matching the reference binary
within the configured tolerance. This is for HIP/C++ kernels where source-level
comparison is not feasible.
"""

    # Build skill recommendations based on kernel type
    skill_recommendations = []
    if kdef.kernel_type == "triton":
        skill_recommendations = [
            "tools/skills/triton-kernel-optimization/SKILL.md",
            "tools/skills/triton-kernel-reflection-prompts/SKILL.md",
        ]
    elif kdef.kernel_type == "hip":
        skill_recommendations = [
            "tools/skills/hip-kernel-optimization/SKILL.md",
            "tools/skills/mi300-hip-programming-insights/SKILL.md",
        ]
    elif kdef.kernel_type == "pytorch":
        skill_recommendations = [
            "tools/skills/pytorch-kernel-optimization/SKILL.md",
        ]
    skill_recommendations += [
        "tools/skills/gpu-architecture-fundamentals/SKILL.md",
        "tools/skills/mi300-cdna3-architecture/SKILL.md",
    ]
    skills_block = "\n".join(f"   - `{s}`" for s in skill_recommendations)

    sol_ext = "py" if kdef.kernel_type != "hip" else "hip"

    prompt = textwrap.dedent(f"""\
# Kernel Optimization Task: {kdef.kernel_name or kdef.kernel_type}

{kdef.description or f"Optimize the {kdef.kernel_type} kernel for AMD {gpu_arch}."}

{arch_section}

## Baseline Kernel

**Type:** {kdef.kernel_type}
**Framework:** {kdef.framework or 'standalone'}
**Task directory:** `{task_dir}`

```{'python' if kdef.kernel_type != 'hip' else 'cpp'}
{kernel_source or '(no source available — read baseline from task directory)'}
```

{correctness_section}

{multi_strategy}

## Step-by-Step Optimization Process

### Step 1: Understand the kernel
Read the baseline source above. Identify the computational pattern (GEMM, attention,
elementwise, reduction, MoE dispatch, etc.).

### Step 2: Read relevant skills
Before optimizing, read the skill files for domain knowledge:
{skills_block}

### Step 3: Use MCP tools for research

**Find reference implementations (any library):**
- Call `mcp__source_finder__find_kernel_source` with the kernel name to find implementations
  across all ROCm libraries (aiter, CK, vLLM, MIOpen, hipBLASLt, sglang, etc.)
- Call `mcp__source_finder__identify_kernel_origin` to determine which library this kernel
  comes from and find alternative implementations
- Call `mcp__source_finder__find_ck_template` if a Composable Kernel template exists

**Get optimization patterns:**
- Call `mcp__rag_server__get_optimization_playbook` for a step-by-step optimization plan
  specific to this kernel type
- Call `mcp__rag_server__search_kernel_optimization` for tiling, vectorization, fusion,
  and other optimization patterns
- Call `mcp__rag_server__get_optimization_snippet` for ready-to-use code snippets

**Get architecture-specific hints:**
- Call `mcp__gpu_info__get_arch_optimization_hints` for {gpu_arch}-specific tuning guidance
  (MFMA sizes, LDS capacity, wavefront occupancy, etc.)
- Call `mcp__gpu_info__get_gpu_specs` for memory bandwidth, compute throughput, cache sizes

**Check fusion opportunities:**
- Call `mcp__fusion_advisor__detect_fusion_opportunities` to find fusible operations
- Call `mcp__fusion_advisor__generate_fused_kernel` to generate a fused implementation

### Step 4: Profile and analyze
- Call `mcp__kernel_perf__roofline_analysis` to determine if the kernel is memory-bound
  or compute-bound — this determines the optimization strategy
- Call `mcp__kernel_perf__profile_kernel` for execution time, occupancy, memory throughput

### Step 5: Implement optimizations
Based on the research above, implement MULTIPLE optimization strategies. Common approaches:
- **Memory-bound kernels:** Improve coalescing, reduce global memory traffic, use LDS tiling
- **Compute-bound kernels:** Increase MFMA utilization, reduce register pressure, improve occupancy
- **Both:** Fuse adjacent operations, use vectorized loads/stores, optimize block/grid sizes

### Step 6: Write solution and validate
- Write the optimized kernel to: `{task_dir}/solution.{sol_ext}`
- Call `mcp__magpie__compare` with `{task_dir}/config.yaml` to validate correctness and measure speedup
- The config.yaml already has baseline path and correctness settings configured
- If correctness fails, iterate: read the error, adjust the solution, re-validate

### Step 7: Iterate for best speedup
- Try at least 2-3 different optimization strategies
- Keep the version with the highest speedup that passes correctness
- Save the best solution to `{task_dir}/solution.{sol_ext}`

## IMPORTANT Constraints
- Your solution must be functionally equivalent to the baseline (same inputs -> same outputs).
- Do NOT modify files outside `{task_dir}/`.
- Do NOT add `if __name__ == "__main__":` blocks with fake benchmarks or `sys.exit(0)`.
- Focus on real performance improvements, not just code style changes.
- Include the kernel function with the same signature as the baseline.
""")

    if rich_prompt:
        return rich_prompt + "\n\n" + prompt
    return prompt


def cmd_optimize_kernel(args):
    """Optimize a standalone kernel using an AI agent."""
    t0 = time.monotonic()
    kdef = _parse_kernel_spec(args)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(getattr(args, "output_dir", None) or str(results_dir / "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    task_dir = output_dir / kdef.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Copy baseline kernel to task directory
    ext = ".py" if kdef.kernel_type in ("triton", "pytorch") else ".hip"
    baseline_dst = task_dir / f"baseline{ext}"
    if kdef.kernel_path and Path(kdef.kernel_path).exists():
        shutil.copy2(kdef.kernel_path, baseline_dst)
        print(f"  Copied baseline: {kdef.kernel_path} -> {baseline_dst.name}")
    elif not baseline_dst.exists():
        print(f"  [error] Kernel path does not exist: {kdef.kernel_path}")
        return

    # Build WorkloadConfig for the agent loop
    config = WorkloadConfig(
        max_iterations=getattr(args, "max_iterations", 5),
        max_turns_per_iter=getattr(args, "max_turns", 25),
        score_threshold=getattr(args, "score_threshold", 300.0),
        agent_model=getattr(args, "agent_model", "") or "",
        agent_version=getattr(args, "agent_version", "v1.0"),
        agent_backend=getattr(args, "agent_backend", "claude"),
        framework=kdef.framework,
        gpu_arch=kdef.gpu_arch,
        kernel_python=getattr(args, "kernel_python", ""),
        output_dir=output_dir,
        results_dir=results_dir,
        dry_run=getattr(args, "dry_run", False),
    )

    print(f"\n{'='*60}")
    print(f"  Standalone Kernel Optimization")
    print(f"  Task:       {kdef.task_id}")
    print(f"  Kernel:     {kdef.kernel_name or kdef.kernel_path}")
    print(f"  Type:       {kdef.kernel_type}")
    print(f"  Correctness: {kdef.gt_mode}")
    print(f"  Agent:      {config.agent_backend} ({config.agent_model})")
    print(f"  Task dir:   {task_dir}")
    print(f"{'='*60}\n")

    _create_standalone_task_config(task_dir, kdef)
    prompt = _build_standalone_kernel_prompt(kdef, task_dir)

    best_kr: Optional[KernelResult] = None
    reflection_prompt = ""

    for iteration in range(1, config.max_iterations + 1):
        print(f"\n  --- Iteration {iteration}/{config.max_iterations} ---")

        print("  Running agent...")
        t_iter = time.monotonic()
        messages, solution_written = _run_agent_iteration(
            task_dir, prompt, config, iteration, reflection_prompt,
        )
        print(f"  Agent completed in {time.monotonic() - t_iter:.1f}s")

        if not solution_written and find_solution(task_dir):
            solution_written = True

        if not solution_written:
            print("  Agent did not write a solution.")
            if iteration < config.max_iterations:
                time.sleep(min(5 * iteration, 15))
            continue

        print("  Grading...")
        kr = grade_task(task_dir, isolate_caches=True, gpu_device=0)
        print(f"    compiled={kr.compiled} correct={kr.correct} "
              f"speedup={kr.speedup:.2f}x score={kr.score:.0f}")
        if kr.error:
            print(f"    error: {kr.error[:200]}")

        if best_kr is None or kr.score > best_kr.score:
            best_kr = kr
            best_sol = find_solution(task_dir)
            if best_sol:
                shutil.copy2(best_sol, task_dir / "solution_best.py")

        if kr.score >= config.score_threshold:
            print(f"  Target reached: score={kr.score:.0f}")
            break

        reflection_prompt = reflect(
            kr, task_dir, iteration,
            kernel_type=kdef.kernel_name or "kernel",
            target_speedup=config.score_threshold / 100.0,
            min_speedup=1.05,
        )

        if not should_continue(kr, iteration, config.max_iterations, config.score_threshold):
            break

    # Restore best
    best_snapshot = task_dir / "solution_best.py"
    if best_snapshot.exists():
        shutil.copy2(best_snapshot, task_dir / "solution.py")

    elapsed = time.monotonic() - t0
    print(f"\n{'='*60}")
    print(f"  Standalone Optimization Complete ({elapsed:.0f}s)")
    if best_kr:
        print(f"  compiled={best_kr.compiled} correct={best_kr.correct} "
              f"speedup={best_kr.speedup:.2f}x score={best_kr.score:.0f}")
    else:
        print(f"  No successful grading result.")
    print(f"  Task dir: {task_dir}")
    print(f"{'='*60}")

    # Save result summary
    summary = {
        "task_id": kdef.task_id,
        "kernel_path": kdef.kernel_path,
        "kernel_type": kdef.kernel_type,
        "kernel_name": kdef.kernel_name,
        "correctness_mode": kdef.gt_mode,
        "agent_backend": config.agent_backend,
        "agent_model": config.agent_model,
        "compiled": best_kr.compiled if best_kr else False,
        "correct": best_kr.correct if best_kr else False,
        "speedup": best_kr.speedup if best_kr else 0.0,
        "score": best_kr.score if best_kr else 0.0,
        "elapsed_seconds": round(elapsed, 1),
    }
    summary_path = results_dir / "standalone_result.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Result saved: {summary_path}")


def cmd_grade_kernel(args):
    """Grade an existing baseline + solution pair (no agent)."""
    t0 = time.monotonic()
    kdef = _parse_kernel_spec(args)

    results_dir = Path(getattr(args, "results_dir", None) or ".")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(getattr(args, "output_dir", None) or str(results_dir / "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    task_dir = output_dir / kdef.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    ext = ".py" if kdef.kernel_type in ("triton", "pytorch") else ".hip"

    # Copy baseline
    baseline_dst = task_dir / f"baseline{ext}"
    if kdef.kernel_path and Path(kdef.kernel_path).exists():
        shutil.copy2(kdef.kernel_path, baseline_dst)
    elif not baseline_dst.exists():
        print(f"  [error] Baseline kernel not found: {kdef.kernel_path}")
        return

    # Copy solution
    sol_src = kdef.solution_path or getattr(args, "solution", "")
    if not sol_src:
        print("  [error] --solution is required for grade-kernel")
        return
    sol_dst = task_dir / f"solution{ext}"
    if Path(sol_src).exists():
        shutil.copy2(sol_src, sol_dst)
    else:
        print(f"  [error] Solution file not found: {sol_src}")
        return

    print(f"\n{'='*60}")
    print(f"  Standalone Kernel Grading")
    print(f"  Task:       {kdef.task_id}")
    print(f"  Baseline:   {kdef.kernel_path}")
    print(f"  Solution:   {sol_src}")
    print(f"  Type:       {kdef.kernel_type}")
    print(f"  Correctness: {kdef.gt_mode}")
    print(f"{'='*60}\n")

    _create_standalone_task_config(task_dir, kdef)

    print("  Grading...")
    kr = grade_task(task_dir, isolate_caches=True, gpu_device=0)

    elapsed = time.monotonic() - t0
    print(f"\n{'='*60}")
    print(f"  Grading Complete ({elapsed:.1f}s)")
    print(f"  compiled={kr.compiled}  correct={kr.correct}  "
          f"speedup={kr.speedup:.2f}x  score={kr.score:.0f}")
    if kr.error:
        print(f"  error: {kr.error[:300]}")
    print(f"{'='*60}")

    summary = {
        "task_id": kdef.task_id,
        "baseline": kdef.kernel_path,
        "solution": sol_src,
        "kernel_type": kdef.kernel_type,
        "correctness_mode": kdef.gt_mode,
        "compiled": kr.compiled,
        "correct": kr.correct,
        "speedup": kr.speedup,
        "score": kr.score,
        "error": kr.error or "",
        "elapsed_seconds": round(elapsed, 1),
    }
    summary_path = results_dir / "grade_result.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Result saved: {summary_path}")

    if getattr(args, "json_output", False):
        print(json.dumps(summary, indent=2))


def _add_kernel_spec_args(parser):
    """Add --kernel-spec and individual CLI arg alternatives for kernel definitions."""
    g = parser.add_argument_group("kernel specification")
    g.add_argument("--kernel-spec", default=None,
                   help="Path to kernel spec YAML file (full definition)")
    g.add_argument("--kernel", default="",
                   help="Path to baseline kernel file (alternative to --kernel-spec)")
    g.add_argument("--kernel-type", default="triton",
                   choices=["triton", "hip", "pytorch"],
                   help="Kernel type: triton, hip, or pytorch (default: triton)")
    g.add_argument("--kernel-name", default="",
                   help="Human-readable kernel name (for prompts)")
    g.add_argument("--description", default="",
                   help="Description of the kernel for the agent prompt")
    g.add_argument("--framework", default="",
                   help="Framework context (vllm, sglang, etc.)")

    c = parser.add_argument_group("correctness definition")
    c.add_argument("--correctness-mode", default="pytorch",
                   choices=["pytorch", "library_test", "accordo"],
                   help="Correctness checking mode (default: pytorch)")
    c.add_argument("--reference", default="",
                   help="Path to PyTorch reference implementation file (pytorch mode)")
    c.add_argument("--test-shapes", default="",
                   help="Path to test shapes generator file (pytorch mode)")
    c.add_argument("--test-cmd", default="",
                   help="Unit test command (library_test mode)")
    c.add_argument("--repo-url", default="",
                   help="Repository URL (library_test mode)")
    c.add_argument("--working-directory", default="",
                   help="Working directory for library tests (library_test mode)")
    c.add_argument("--accordo-config", default="",
                   help="Path to Accordo config YAML (accordo mode)")


# ---------------------------------------------------------------------------
# CLI with subcommands
# ---------------------------------------------------------------------------

def _add_common_args(parser):
    """Add arguments shared across all subcommands."""
    parser.add_argument("-r", "--results-dir", required=True,
                        help="Directory for state, reports, leaderboard, trajectory")
    parser.add_argument("--output-dir",
                        help="Directory for kernel task outputs (default: results-dir/output)")
    parser.add_argument("--gpu", default="gfx950",
                        help="Target GPU architecture (default: gfx950)")
    parser.add_argument("--kernel-python", default="",
                        help="Python with torch+triton for kernel execution (auto-detected)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without GPU or API calls")


def _add_benchmark_args(parser):
    parser.add_argument("-b", "--benchmark-config", required=True,
                        help="Path to Magpie benchmark YAML config")
    parser.add_argument("--skip-benchmark",
                        help="Path to existing benchmark_report.json (skip benchmark)")
    parser.add_argument("--framework", default="",
                        help="Inference framework (default: auto-detect from config)")
    parser.add_argument("--num-benchmark-runs", type=int, default=5,
                        help="Number of benchmark runs for statistical averaging (default: 5)")
    parser.add_argument("--benchmark-timeout", type=int, default=5400)


def _add_kernel_filter_args(parser):
    parser.add_argument("--kernel-types", default="all",
                        help="Comma-separated: triton,hip,ck,asm,all (default: all)")
    parser.add_argument("--kernels", default="all",
                        help="Comma-separated kernel spec names, or 'all'")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top bottleneck kernels to consider (default: 10)")
    parser.add_argument("--top-k-mode", choices=["pre-filter", "post-filter"],
                        default="post-filter",
                        help="Filter order: post-filter (type filter before top-k) or "
                             "pre-filter (top-k before type filter). Default: post-filter")


def _add_agent_args(parser):
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="Max optimization iterations per kernel (default: 5)")
    parser.add_argument("--max-turns", type=int, default=25,
                        help="Max agent turns per iteration (default: 25)")
    parser.add_argument("--score-threshold", type=float, default=300.0,
                        help="Stop early if score exceeds this (default: 300)")
    parser.add_argument("--agent-model", default=None,
                        help="Override the backend-specific default agent model")
    parser.add_argument("--agent-version", default="v1.0")
    parser.add_argument("--agent-backend", default="claude", choices=["claude", "codex"])


def main():
    parser = argparse.ArgumentParser(
        description="Modular workload optimization trajectory pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline step to run")

    # -- benchmark --
    p = subparsers.add_parser("benchmark", help="Step 1: Run or load E2E benchmark")
    _add_common_args(p)
    _add_benchmark_args(p)

    # -- identify --
    p = subparsers.add_parser("identify",
                              help="Step 2-4: Identify & filter bottleneck kernels")
    _add_common_args(p)
    _add_kernel_filter_args(p)

    # -- list-kernels --
    p = subparsers.add_parser("list-kernels",
                              help="Show identified kernels for selection")
    p.add_argument("-r", "--results-dir", required=True)

    # -- optimize --
    p = subparsers.add_parser("optimize",
                              help="Step 5: Optimize selected kernels (agent + grading)")
    _add_common_args(p)
    _add_kernel_filter_args(p)
    _add_agent_args(p)

    # -- grade --
    p = subparsers.add_parser("grade",
                              help="Re-grade existing solutions (no agent)")
    _add_common_args(p)
    _add_kernel_filter_args(p)

    # -- integrate --
    p = subparsers.add_parser("integrate",
                              help="Step 6: Re-inject optimized kernels")
    _add_common_args(p)
    p.add_argument("--kernels", default="all",
                    help="Comma-separated kernel specs to integrate, or 'all'")

    # -- benchmark-final --
    p = subparsers.add_parser("benchmark-final",
                              help="Step 7: Run final E2E benchmark")
    _add_common_args(p)
    _add_benchmark_args(p)

    # -- score --
    p = subparsers.add_parser("score",
                              help="Step 8: Compute reward and update leaderboard")
    _add_common_args(p)
    p.add_argument("--leaderboard", action="store_true",
                    help="Push result to leaderboard")
    p.add_argument("--agent-model", default=None)
    p.add_argument("--agent-version", default="v1.0")
    p.add_argument("--agent-backend", default="claude", choices=["claude", "codex"])
    p.add_argument("--trajectory-store", default="file")

    # -- report --
    p = subparsers.add_parser("report",
                              help="Generate markdown report and replication guide")
    _add_common_args(p)
    p.add_argument("-b", "--benchmark-config", default="")
    p.add_argument("--agent-model", default=None)
    p.add_argument("--agent-version", default="v1.0")
    p.add_argument("--agent-backend", default="claude", choices=["claude", "codex"])

    # -- run --
    p = subparsers.add_parser("run",
                              help="Full pipeline (all steps sequentially)")
    _add_common_args(p)
    _add_benchmark_args(p)
    _add_kernel_filter_args(p)
    _add_agent_args(p)
    p.add_argument("--docker-image", default="")
    p.add_argument("--trajectory-store", default="file")
    p.add_argument("--leaderboard", action="store_true")

    # ── export-rl: export trajectories to RL training dataset format ──────
    p = subparsers.add_parser(
        "export-rl",
        help="Export scored trajectories to RL training dataset format (tasks.json + optional SFT JSONL)",
    )
    _add_common_args(p)
    p.add_argument("--trajectories-dir", type=str, default=None,
                   help="Directory with trajectory JSON files (default: <repo>/trajectories)")
    p.add_argument("--export-output-dir", type=str, required=True,
                   help="Output directory for tasks.json and sft_warmstart.jsonl")
    p.add_argument("--sft", action="store_true",
                   help="Also emit SFT warm-start JSONL from good trajectories")
    p.add_argument("--quality", type=str, default=None,
                   help="Filter SFT trajectories by quality (good|mediocre|bad)")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="Minimum kernel score to include as a task")
    p.add_argument("--gpu-arch", type=str, default="gfx950",
                   help="Target GPU architecture (default: gfx950)")

    # ── optimize-kernel: standalone kernel optimization with agent ─────
    p = subparsers.add_parser(
        "optimize-kernel",
        help="Optimize a standalone kernel using an AI agent (no full pipeline needed)",
    )
    p.add_argument("-r", "--results-dir", required=True,
                   help="Directory for results and artifacts")
    p.add_argument("--output-dir",
                   help="Directory for task outputs (default: results-dir/output)")
    p.add_argument("--gpu", default="gfx950",
                   help="Target GPU architecture (default: gfx950)")
    p.add_argument("--kernel-python", default="",
                   help="Python with torch+triton for kernel execution (auto-detected)")
    p.add_argument("--dry-run", action="store_true",
                   help="Simulate without GPU or API calls")
    _add_kernel_spec_args(p)
    _add_agent_args(p)

    # ── grade-kernel: grade an existing baseline + solution pair ─────
    p = subparsers.add_parser(
        "grade-kernel",
        help="Grade an existing baseline + solution pair (no agent, no pipeline)",
    )
    p.add_argument("-r", "--results-dir", default=".",
                   help="Directory for results (default: current directory)")
    p.add_argument("--output-dir",
                   help="Directory for task outputs (default: results-dir/output)")
    p.add_argument("--gpu", default="gfx950",
                   help="Target GPU architecture (default: gfx950)")
    p.add_argument("--kernel-python", default="",
                   help="Python with torch+triton for kernel execution (auto-detected)")
    p.add_argument("--dry-run", action="store_true")
    _add_kernel_spec_args(p)
    # grade-kernel only: --solution is intentionally NOT in _add_kernel_spec_args
    # since optimize-kernel does not accept a pre-existing solution.
    p.add_argument("--solution", default="",
                   help="Path to the solution file to grade (required unless provided in --kernel-spec)")
    p.add_argument("--json", dest="json_output", action="store_true",
                   help="Print result as JSON to stdout")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    handlers = {
        "benchmark": cmd_benchmark,
        "identify": cmd_identify,
        "list-kernels": cmd_list_kernels,
        "optimize": cmd_optimize,
        "grade": cmd_grade,
        "integrate": cmd_integrate,
        "benchmark-final": cmd_benchmark_final,
        "score": cmd_score,
        "report": cmd_report,
        "run": cmd_run,
        "export-rl": cmd_export_rl,
        "optimize-kernel": cmd_optimize_kernel,
        "grade-kernel": cmd_grade_kernel,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    # Ensure all print output is immediately flushed (important for piped/background runs)
    import functools
    print = functools.partial(print, flush=True)  # noqa: A001
    main()
