"""
cache_manager.py — GPU compilation cache isolation for reliable kernel benchmarking.

Handles isolation and cleanup of every cache layer that can cause stale compiled
kernels to be served instead of freshly-compiled ones:

  1. Triton compilation cache  (TRITON_CACHE_DIR / ~/.triton/cache)
  2. Python bytecode cache     (__pycache__ directories)
  3. torch.compile / Inductor  (~/.cache/torch/inductor, torch._dynamo)
  4. ROCm comgr cache          (~/.cache/comgr)
  5. Python module cache        (sys.modules entries for patched modules)
  6. GPU warmup                 (HIP device sync / memory flush between runs)
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


# ---------------------------------------------------------------------------
# 1. Triton cache isolation
# ---------------------------------------------------------------------------

@contextmanager
def isolated_triton_cache(prefix: str = "triton_cache_") -> Iterator[str]:
    """Run a block with a fresh, empty TRITON_CACHE_DIR.

    On exit the temporary directory is removed and the original env var
    is restored (or unset if it wasn't set before).
    """
    old_val = os.environ.get("TRITON_CACHE_DIR")
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    os.environ["TRITON_CACHE_DIR"] = tmpdir
    try:
        yield tmpdir
    finally:
        if old_val is not None:
            os.environ["TRITON_CACHE_DIR"] = old_val
        else:
            os.environ.pop("TRITON_CACHE_DIR", None)
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2. Python bytecode cache (__pycache__)
# ---------------------------------------------------------------------------

def clear_pycache(module_path: Path) -> None:
    """Remove __pycache__ for the directory containing a module file."""
    cache_dir = module_path.parent / "__pycache__"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def clear_pycache_tree(root: Path) -> int:
    """Recursively remove all __pycache__ dirs under *root*. Returns count."""
    count = 0
    for cache_dir in root.rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)
        count += 1
    return count


# ---------------------------------------------------------------------------
# 3. torch.compile / Inductor cache
# ---------------------------------------------------------------------------

_INDUCTOR_CACHE_DIRS = [
    Path.home() / ".cache" / "torch" / "inductor",
    Path.home() / ".cache" / "torch" / "_inductor",
]


def clear_torch_caches() -> None:
    """Clear torch.compile / Inductor caches (both on-disk and in-process).

    - Deletes ~/.cache/torch/inductor/ contents
    - Resets torch._dynamo if loaded
    - Clears torch._inductor code cache if available
    """
    for cache_dir in _INDUCTOR_CACHE_DIRS:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch._dynamo  # type: ignore[import-untyped]
        torch._dynamo.reset()
    except (ImportError, AttributeError):
        pass

    try:
        import torch._inductor.codecache  # type: ignore[import-untyped]
        if hasattr(torch._inductor.codecache, "PyCodeCache"):
            cc = torch._inductor.codecache.PyCodeCache
            if hasattr(cc, "cache"):
                cc.cache.clear()
    except (ImportError, AttributeError):
        pass


@contextmanager
def isolated_torch_cache() -> Iterator[None]:
    """Run a block with a fresh torch Inductor cache directory.

    Sets TORCHINDUCTOR_CACHE_DIR to a temp location and restores on exit.
    """
    old_val = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    tmpdir = tempfile.mkdtemp(prefix="torch_inductor_cache_")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tmpdir
    try:
        clear_torch_caches()
        yield
    finally:
        if old_val is not None:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = old_val
        else:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 4. ROCm comgr / hipcc cache
# ---------------------------------------------------------------------------

_COMGR_CACHE_DIRS = [
    Path.home() / ".cache" / "comgr",
    Path.home() / ".AMD",
]


def clear_comgr_cache() -> None:
    """Clear the ROCm compiler (comgr) on-disk cache."""
    for d in _COMGR_CACHE_DIRS:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)


@contextmanager
def isolated_comgr_cache() -> Iterator[str]:
    """Run a block with AMD_COMGR_CACHE_DIR pointed to a fresh temp directory."""
    old_val = os.environ.get("AMD_COMGR_CACHE_DIR")
    tmpdir = tempfile.mkdtemp(prefix="comgr_cache_")
    os.environ["AMD_COMGR_CACHE_DIR"] = tmpdir
    try:
        yield tmpdir
    finally:
        if old_val is not None:
            os.environ["AMD_COMGR_CACHE_DIR"] = old_val
        else:
            os.environ.pop("AMD_COMGR_CACHE_DIR", None)
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 5. Python sys.modules purge
# ---------------------------------------------------------------------------

def purge_modules(prefixes: list[str]) -> list[str]:
    """Remove all sys.modules entries whose name starts with any of *prefixes*.

    Returns list of purged module names.
    """
    to_remove = [
        name for name in sys.modules
        if any(name.startswith(p) or name == p for p in prefixes)
    ]
    for name in to_remove:
        del sys.modules[name]
    return to_remove


# Well-known module prefixes for GPU kernel libraries
KERNEL_MODULE_PREFIXES = [
    "aiter",
    "triton",
    "vllm._custom_ops",
    "vllm.model_executor",
]


def purge_kernel_modules() -> list[str]:
    """Purge cached imports for all known GPU kernel library modules."""
    return purge_modules(KERNEL_MODULE_PREFIXES)


# ---------------------------------------------------------------------------
# 6. GPU warmup / sync
# ---------------------------------------------------------------------------

def gpu_sync_and_flush(device: int = 0) -> None:
    """Synchronize and flush the GPU to get a clean baseline.

    - torch.cuda.synchronize (blocks until all GPU work finishes)
    - torch.cuda.empty_cache (releases cached allocator memory)
    - Optionally resets peak memory stats
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
    except (ImportError, RuntimeError):
        pass


def gpu_warmup(device: int = 0, iterations: int = 3) -> None:
    """Run a small dummy kernel to warm the GPU and stabilize clocks.

    This helps avoid cold-start variance between benchmark runs.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return
        with torch.cuda.device(device):
            for _ in range(iterations):
                a = torch.randn(1024, 1024, device=f"cuda:{device}")
                b = torch.randn(1024, 1024, device=f"cuda:{device}")
                _ = torch.mm(a, b)
            torch.cuda.synchronize(device)
            del a, b
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


# ---------------------------------------------------------------------------
# Composite context managers
# ---------------------------------------------------------------------------

@contextmanager
def isolated_grading_env(
    clear_torch: bool = True,
    clear_comgr: bool = True,
    warmup_gpu: bool = True,
    gpu_device: int = 0,
) -> Iterator[dict[str, str]]:
    """Full cache isolation for grading a single kernel.

    Combines all isolation layers into one context manager:
    - Fresh TRITON_CACHE_DIR
    - Fresh TORCHINDUCTOR_CACHE_DIR (if clear_torch)
    - Fresh AMD_COMGR_CACHE_DIR (if clear_comgr)
    - GPU sync + warmup (if warmup_gpu)

    Yields a dict with temp dir paths for inspection.
    """
    dirs: dict[str, str] = {}

    with isolated_triton_cache() as triton_dir:
        dirs["triton_cache"] = triton_dir

        comgr_ctx = isolated_comgr_cache() if clear_comgr else _noop_ctx()
        torch_ctx = isolated_torch_cache() if clear_torch else _noop_ctx()

        with comgr_ctx as comgr_result:
            if comgr_result:
                dirs["comgr_cache"] = comgr_result

            with torch_ctx:
                if warmup_gpu:
                    gpu_sync_and_flush(gpu_device)
                    gpu_warmup(gpu_device)

                yield dirs


@contextmanager
def isolated_benchmark_env(
    gpu_device: int = 0,
) -> Iterator[dict[str, str]]:
    """Full cache isolation for running an E2E benchmark.

    Same as isolated_grading_env but always enables all layers.
    """
    with isolated_grading_env(
        clear_torch=True,
        clear_comgr=True,
        warmup_gpu=True,
        gpu_device=gpu_device,
    ) as dirs:
        yield dirs


@contextmanager
def _noop_ctx():
    """No-op context manager that yields None."""
    yield None
