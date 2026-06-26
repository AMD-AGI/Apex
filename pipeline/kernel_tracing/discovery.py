"""Discover statically traceable kernels from supported ROCm checkouts."""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

from .registry import TraceKernelEntry


REPO_PACKAGE_ROOTS = {
    "aiter": Path("tools/rocm/aiter/aiter"),
    "vllm": Path("tools/rocm/vllm/vllm"),
    "sglang": Path("tools/rocm/sglang/python/sglang"),
}
EXCLUDED_PATH_PARTS = {
    ".git",
    "__pycache__",
    "benchmarks",
    "benchmark",
    "docs",
    "examples",
    "op_tests",
    "tests",
    "test",
}
GENERIC_TRITON_NAMES = {"kernel", "call_kernel"}
VLLM_TORCH_OP_NAMESPACES = {
    "_C",
    "_C_cache_ops",
    "_moe_C",
    "_rocm_C",
    "aiter",
    "vllm",
    "vllm_aiter",
}
SGLANG_CUSTOM_DECORATORS = {"register_custom_op", "debug_kernel_api"}

KNOWN_ID_OVERRIDES = {
    (
        "aiter",
        "triton",
        "kernel_unified_attention_2d",
        "tools/rocm/aiter/aiter/ops/triton/attention/unified_attention.py",
    ): "aiter.triton.unified_attention_2d",
    (
        "aiter",
        "triton",
        "_paged_attn_decode_v1_wo_dot_kernel",
        "tools/rocm/aiter/aiter/ops/triton/attention/pa_decode.py",
    ): "aiter.triton.paged_attn_decode_v1_wo_dot",
    (
        "aiter",
        "hip",
        "moe_sorting_fwd",
        "tools/rocm/aiter/aiter/ops/moe_sorting.py",
    ): "aiter.hip.moe_sorting_fwd",
    (
        "vllm",
        "triton",
        "_gumbel_sample_kernel",
        "tools/rocm/vllm/vllm/v1/worker/gpu/sample/gumbel.py",
    ): "vllm.triton.gumbel_sample",
    (
        "vllm",
        "triton",
        "reshape_and_cache_kernel_flash",
        "tools/rocm/vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py",
    ): "vllm.triton.reshape_and_cache_flash",
    (
        "vllm",
        "hip",
        "reshape_and_cache_flash",
        "tools/rocm/vllm/vllm/_custom_ops.py",
    ): "vllm.hip.reshape_and_cache_flash",
}


def _skip_file(path: Path) -> bool:
    if path.name.startswith("test_") or path.name.endswith("_test.py"):
        return True
    return bool(set(path.parts) & EXCLUDED_PATH_PARTS)


def _dotted_name(expr: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.append(expr.id)
    return ".".join(reversed(parts))


def _decorator_call_target(decorator: ast.AST) -> ast.AST:
    return decorator.func if isinstance(decorator, ast.Call) else decorator


def _decorator_name(decorator: ast.AST) -> str:
    return _dotted_name(_decorator_call_target(decorator))


def _is_triton_decorator(decorator: ast.AST) -> bool:
    name = _decorator_name(decorator)
    return name in {"triton.jit", "triton.heuristics"} or name.endswith(
        (".triton.jit", ".triton.heuristics")
    )


def _subscript_target_name(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


def _triton_import_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if "triton" not in node.module and "_triton_kernels" not in node.module:
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            names.add(alias.asname or alias.name)
    return names


def _compile_ops_load_name(decorator: ast.AST, function_name: str) -> str | None:
    if not isinstance(decorator, ast.Call):
        return None
    if not _decorator_name(decorator).endswith("compile_ops"):
        return None
    for keyword in decorator.keywords:
        if (
            keyword.arg == "fc_name"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ):
            return keyword.value.value
    return function_name


def _torch_ops_namespace(call: ast.Call) -> str | None:
    cur = call.func
    names: list[str] = []
    while isinstance(cur, ast.Attribute):
        names.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name) and cur.id == "torch" and len(names) >= 2:
        if names[-1] == "ops":
            return names[-2]
    return None


def _function_calls_vllm_custom_op(function: ast.AST) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            namespace = _torch_ops_namespace(node)
            if namespace in VLLM_TORCH_OP_NAMESPACES:
                return True
    return False


def _function_has_sglang_custom_decorator(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        _decorator_name(decorator).split(".")[-1] in SGLANG_CUSTOM_DECORATORS
        for decorator in function.decorator_list
    )


def _sanitize_id_part(value: str) -> str:
    value = value.strip("_").lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "kernel"


def _path_slug(kernel_file: str, repo: str) -> str:
    parts = Path(kernel_file).with_suffix("").parts
    repo_root = REPO_PACKAGE_ROOTS[repo]
    root_parts = repo_root.parts
    rel_parts = parts
    for idx in range(0, len(parts) - len(root_parts) + 1):
        if parts[idx : idx + len(root_parts)] == root_parts:
            rel_parts = parts[idx + len(root_parts) :]
            break
    return "_".join(_sanitize_id_part(part) for part in rel_parts[-3:])


def _entry_without_id(
    *,
    repo: str,
    kernel_type: str,
    kernel_name: str,
    kernel_file: str,
    trace_mode: str,
) -> dict[str, str]:
    return {
        "repo": repo,
        "kernel_type": kernel_type,
        "kernel_name": kernel_name,
        "kernel_file": kernel_file,
        "trace_mode": trace_mode,
        "patch_strategy": "static",
    }


def _parse_python(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


def _discover_repo_entries(repo_root: Path, repo: str) -> list[dict[str, str]]:
    package_root = repo_root / REPO_PACKAGE_ROOTS[repo]
    entries: list[dict[str, str]] = []
    for path in sorted(package_root.rglob("*.py")):
        rel_file = path.relative_to(repo_root).as_posix()
        rel_to_package = path.relative_to(package_root)
        if _skip_file(rel_to_package):
            continue
        tree = _parse_python(path)
        if tree is None:
            continue

        triton_defs = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(_is_triton_decorator(decorator) for decorator in node.decorator_list)
        }
        triton_targets = triton_defs | _triton_import_names(tree)
        triton_seen: set[str] = set()
        if triton_targets:
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Subscript):
                    continue
                target = _subscript_target_name(node.func.value)
                if (
                    target
                    and target in triton_targets
                    and target not in GENERIC_TRITON_NAMES
                    and target not in triton_seen
                ):
                    triton_seen.add(target)
                    entries.append(
                        _entry_without_id(
                            repo=repo,
                            kernel_type="triton",
                            kernel_name=target,
                            kernel_file=rel_file,
                            trace_mode="triton-launch",
                        )
                    )

        if repo == "aiter":
            compile_seen: set[str] = set()
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for decorator in node.decorator_list:
                    load_name = _compile_ops_load_name(decorator, node.name)
                    if load_name and load_name not in compile_seen:
                        compile_seen.add(load_name)
                        entries.append(
                            _entry_without_id(
                                repo=repo,
                                kernel_type="hip",
                                kernel_name=load_name,
                                kernel_file=rel_file,
                                trace_mode="aiter-compile-ops",
                            )
                        )
        elif repo == "vllm":
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if _function_calls_vllm_custom_op(node):
                        entries.append(
                            _entry_without_id(
                                repo=repo,
                                kernel_type="hip",
                                kernel_name=node.name,
                                kernel_file=rel_file,
                                trace_mode="vllm-custom-op",
                            )
                        )
        elif repo == "sglang":
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if _function_has_sglang_custom_decorator(node):
                        entries.append(
                            _entry_without_id(
                                repo=repo,
                                kernel_type="hip",
                                kernel_name=node.name,
                                kernel_file=rel_file,
                                trace_mode="sglang-custom-op",
                            )
                        )
    return entries


def discover_trace_kernel_entries(repo_root: Path) -> list[TraceKernelEntry]:
    """Return statically traceable kernel entries discovered from source."""
    raw_entries: list[dict[str, str]] = []
    for repo in sorted(REPO_PACKAGE_ROOTS):
        raw_entries.extend(_discover_repo_entries(repo_root, repo))

    base_ids: list[str] = []
    for entry in raw_entries:
        override = KNOWN_ID_OVERRIDES.get(
            (
                entry["repo"],
                entry["kernel_type"],
                entry["kernel_name"],
                entry["kernel_file"],
            )
        )
        if override:
            base_ids.append(override)
        else:
            base_ids.append(
                f"{entry['repo']}.{entry['kernel_type']}.{_sanitize_id_part(entry['kernel_name'])}"
            )

    counts = Counter(base_ids)
    used: set[str] = set()
    discovered: list[TraceKernelEntry] = []
    for entry, base_id in zip(raw_entries, base_ids, strict=True):
        kernel_id = base_id
        if counts[base_id] > 1:
            kernel_id = (
                f"{entry['repo']}.{entry['kernel_type']}."
                f"{_path_slug(entry['kernel_file'], entry['repo'])}."
                f"{_sanitize_id_part(entry['kernel_name'])}"
            )
        if kernel_id in used:
            suffix = 2
            candidate = f"{kernel_id}_{suffix}"
            while candidate in used:
                suffix += 1
                candidate = f"{kernel_id}_{suffix}"
            kernel_id = candidate
        used.add(kernel_id)
        discovered.append(TraceKernelEntry(id=kernel_id, **entry))

    return sorted(discovered, key=lambda e: (e.repo, e.kernel_type, e.kernel_file, e.kernel_name))
