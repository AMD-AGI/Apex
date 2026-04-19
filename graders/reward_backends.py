#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
reward_backends.py — Backend-specific helpers for the reward pipeline.

This module keeps backend dispatch, answer normalization, and static preflight
checks separate from the shared reward math in ``reward_fn.py``.

Ported from keystone-rl-training/reward_backends.py and extended with
Apex-specific kernel-spec-to-backend mapping.
"""

from __future__ import annotations

import ast
import re


KERNEL_BACKEND_TRITON = "triton"
KERNEL_BACKEND_HIP = "hip"
KERNEL_BACKEND_GLUON = "gluon"

SUPPORTED_KERNEL_BACKENDS: frozenset[str] = frozenset({
    KERNEL_BACKEND_TRITON,
    KERNEL_BACKEND_HIP,
    KERNEL_BACKEND_GLUON,
})

# Apex kernel spec → backend mapping
TRITON_SPECS: frozenset[str] = frozenset({
    "flash_attn_prefill", "paged_attn_decode", "mla_attn", "fused_moe",
    "rms_norm", "rope_embedding", "kv_cache_ops", "act_quant_fp8", "silu_mul",
    "gelu_tanh",
})

HIP_SPECS: frozenset[str] = frozenset({
    "gemm_w8a8", "gemm_bf16", "all_reduce",
})

# Apex kernel specs that correspond to a Gluon (@gluon.jit) implementation.
# These are passed through to the gluon backend for static screening, so the
# checker accepts `from triton.experimental.gluon import language as gl` etc.
GLUON_SPECS: frozenset[str] = frozenset({
    "paged_attn_decode_gluon",
})


def resolve_kernel_backend(kernel_type_str: str) -> str:
    """
    Normalize and validate the kernel backend string.

    Accepts either a raw backend name (``"triton"``, ``"hip"``) or an
    Apex kernel spec name (``"fused_moe"`` → ``"triton"``).

    Parameters
    ----------
    kernel_type_str : str
        Backend identifier or Apex kernel spec name.

    Returns
    -------
    str
        Canonical backend constant.

    Raises
    ------
    ValueError
        If the string cannot be resolved to a supported backend.
    """
    backend = kernel_type_str.strip().lower()
    if backend in SUPPORTED_KERNEL_BACKENDS:
        return backend
    if backend in GLUON_SPECS or backend.endswith("_gluon"):
        return KERNEL_BACKEND_GLUON
    if backend in TRITON_SPECS:
        return KERNEL_BACKEND_TRITON
    if backend in HIP_SPECS:
        return KERNEL_BACKEND_HIP
    raise ValueError(
        f"Unsupported kernel type {kernel_type_str!r}; "
        f"expected one of {sorted(SUPPORTED_KERNEL_BACKENDS)} "
        f"or a known spec name"
    )


def strip_markdown_code_fence(text: str) -> str:
    """Strip one outer fenced code block if the answer is wrapped in Markdown."""
    stripped = text.strip()
    match = re.fullmatch(r"```[^\n`]*\n(?P<body>.*)\n```", stripped, flags=re.DOTALL)
    if match:
        return match.group("body").strip()
    return stripped


def normalize_answer_for_backend(answer: str, backend: str) -> str:
    """
    Apply lightweight backend-aware normalization before static checking.

    Parameters
    ----------
    answer : str
        Raw answer content (from ``<answer>`` tags or bare solution file).
    backend : str
        Canonical backend constant.

    Returns
    -------
    str
        Normalized code string ready for static analysis.
    """
    normalized = strip_markdown_code_fence(answer).strip()
    if backend == KERNEL_BACKEND_HIP:
        return normalized
    if backend in (KERNEL_BACKEND_TRITON, KERNEL_BACKEND_GLUON):
        return normalized
    raise ValueError(f"Unsupported backend: {backend}")


# ── Triton static checks ─────────────────────────────────────────────────────

ALLOWED_IMPORTS: frozenset[str] = frozenset({
    "triton",
    "triton.language",
    "math",
    "torch",
})

# Gluon kernels live under triton.experimental.gluon and use the same
# whitelist plus the gluon namespace.
GLUON_ALLOWED_IMPORTS: frozenset[str] = frozenset({
    "triton",
    "triton.language",
    "triton.experimental",
    "triton.experimental.gluon",
    "triton.experimental.gluon.language",
    "triton.experimental.gluon.language.amd",
    "triton.experimental.gluon.language.amd.cdna3",
    "triton.experimental.gluon.language.amd.cdna4",
    "triton.experimental.gluon.language.nvidia",
    "math",
    "torch",
})

BLOCKED_TORCH_ATTRS: frozenset[str] = frozenset({
    "matmul", "mm", "bmm", "einsum",
    "conv1d", "conv2d", "conv3d", "linear",
    "softmax", "sigmoid", "relu", "layer_norm", "batch_norm", "cross_entropy",
    "dot", "mv", "addmm", "baddbmm", "tensordot", "inner", "outer",
    "linalg",
    "nn",
    "autograd",
    "functional",
    "compile",
    "jit",
    "full",
})

BLOCKED_MODULES: frozenset[str] = frozenset({
    "os", "sys", "subprocess", "importlib", "builtins",
    "ctypes", "cffi", "socket", "urllib", "http",
    "requests", "multiprocessing", "concurrent", "threading",
})

BLOCKED_BUILTINS: frozenset[str] = frozenset({
    "exec", "eval", "compile", "__import__", "open", "input",
})


def run_triton_static_check(
    code: str,
    allowed_imports: list[str] | None = None,
) -> tuple[bool, str]:
    """
    Parse ``code`` into an AST and enforce the Triton kernel whitelist.

    Checks applied in order:
    1. Syntax — ``ast.parse()``; ``SyntaxError`` → ``(False, "SyntaxError: ...")``
    2. Import whitelist enforcement
    3. Blocked builtin calls (``exec``, ``eval``, etc.)
    4. Blocked module access and torch computation attrs
    5. ``@triton.jit`` decorator requirement
    6. Hardcoded output detection (``return torch.tensor([literals])``)

    Parameters
    ----------
    code : str
        Triton kernel source code to analyze.
    allowed_imports : list[str] | None
        Override for the default import whitelist. If ``None``, uses
        ``ALLOWED_IMPORTS``.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` on pass; ``(False, reason)`` on any violation.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    allowed_set: frozenset[str] = (
        frozenset(allowed_imports) if allowed_imports is not None else ALLOWED_IMPORTS
    )

    alias_map: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                alias_map[local] = alias.name

    def _canonical(name: str) -> str:
        return alias_map.get(name, name)

    def _is_literal_tree(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, (ast.List, ast.Tuple)):
            return all(_is_literal_tree(child) for child in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return _is_literal_tree(node.operand)
        return False

    # Import whitelist
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if alias.name not in allowed_set and top not in allowed_set:
                    return False, f"blocked_import: '{alias.name}' is not whitelisted"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0]
            if module not in allowed_set and top not in allowed_set:
                return False, f"blocked_import: 'from {module} import ...' is not whitelisted"
            if top == "torch" or _canonical(top) == "torch":
                for alias in node.names:
                    if alias.name in BLOCKED_TORCH_ATTRS:
                        return False, (
                            f"blocked_import: 'from torch import {alias.name}' "
                            "imports a blocked torch computation"
                        )

    # Blocked builtins
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BLOCKED_BUILTINS:
                return False, f"blocked_builtin: call to '{func.id}()' is forbidden"

    # Blocked module access and torch attrs
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value  # type: ignore[assignment]
            if not isinstance(root, ast.Name):
                continue
            canonical = _canonical(root.id)
            top_canonical = canonical.split(".")[0]

            if top_canonical in BLOCKED_MODULES:
                return False, (
                    f"blocked_module: access to '{root.id}.{node.attr}' is forbidden"
                )
            if canonical == "torch" or top_canonical == "torch":
                if node.attr in BLOCKED_TORCH_ATTRS:
                    return False, (
                        f"blocked_torch_attr: '{root.id}.{node.attr}' "
                        "is a blocked torch computation"
                    )

    # @triton.jit decorator requirement
    has_triton_jit = False
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Attribute) and dec.attr == "jit":
                root = dec.value
                if isinstance(root, ast.Name) and _canonical(root.id) == "triton":
                    has_triton_jit = True
    if not has_triton_jit:
        return False, "no_triton_jit: no @triton.jit decorated function found"

    # Hardcoded output detection
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        val = node.value
        if (
            isinstance(val, ast.Call)
            and isinstance(val.func, ast.Attribute)
            and isinstance(val.func.value, ast.Name)
            and _canonical(val.func.value.id) == "torch"
            and val.func.attr == "tensor"
            and val.args
            and _is_literal_tree(val.args[0])
        ):
            return False, "hardcoded_output: 'return torch.tensor(...)' with literal values"

    return True, ""


# ── Gluon static checks ──────────────────────────────────────────────────────


def run_gluon_static_check(
    code: str,
    allowed_imports: list[str] | None = None,
) -> tuple[bool, str]:
    """
    Parse a Gluon kernel source and enforce a whitelist similar to Triton's.

    The differences vs. ``run_triton_static_check`` are:
    1. The import whitelist allows the ``triton.experimental.gluon`` namespace.
    2. The decorator requirement is ``@gluon.jit`` (or ``@triton.experimental.gluon.jit``)
       *or* ``@triton.jit`` — both are valid in Gluon-friendly solutions.
    3. All other safety checks (blocked builtins, blocked modules, blocked
       torch attrs, hardcoded literal returns) are identical.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    allowed_set: frozenset[str] = (
        frozenset(allowed_imports) if allowed_imports is not None
        else GLUON_ALLOWED_IMPORTS
    )

    alias_map: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                alias_map[local] = alias.name
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                local = alias.asname or alias.name
                full = f"{node.module}.{alias.name}" if node.module else alias.name
                alias_map[local] = full

    def _canonical(name: str) -> str:
        return alias_map.get(name, name)

    def _is_literal_tree(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, (ast.List, ast.Tuple)):
            return all(_is_literal_tree(child) for child in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return _is_literal_tree(node.operand)
        return False

    # Import whitelist
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if alias.name not in allowed_set and top not in allowed_set:
                    return False, f"blocked_import: '{alias.name}' is not whitelisted"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            top = module.split(".")[0]
            if module not in allowed_set and top not in allowed_set:
                return False, f"blocked_import: 'from {module} import ...' is not whitelisted"
            if top == "torch" or _canonical(top) == "torch":
                for alias in node.names:
                    if alias.name in BLOCKED_TORCH_ATTRS:
                        return False, (
                            f"blocked_import: 'from torch import {alias.name}' "
                            "imports a blocked torch computation"
                        )

    # Blocked builtins
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BLOCKED_BUILTINS:
                return False, f"blocked_builtin: call to '{func.id}()' is forbidden"

    # Blocked module access and torch attrs
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value  # type: ignore[assignment]
            if not isinstance(root, ast.Name):
                continue
            canonical = _canonical(root.id)
            top_canonical = canonical.split(".")[0]

            if top_canonical in BLOCKED_MODULES:
                return False, (
                    f"blocked_module: access to '{root.id}.{node.attr}' is forbidden"
                )
            if canonical == "torch" or top_canonical == "torch":
                if node.attr in BLOCKED_TORCH_ATTRS:
                    return False, (
                        f"blocked_torch_attr: '{root.id}.{node.attr}' "
                        "is a blocked torch computation"
                    )

    # @gluon.jit OR @triton.jit decorator requirement (Gluon kernels are
    # legal Triton too — accept either so wrappers / helper kernels pass).
    has_jit = False
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Attribute) and dec.attr == "jit":
                root = dec.value
                if isinstance(root, ast.Name):
                    canon = _canonical(root.id)
                    canon_top = canon.split(".")[0]
                    if canon_top in ("triton", "gluon"):
                        has_jit = True
                    if "gluon" in canon:
                        has_jit = True
                elif isinstance(root, ast.Attribute) and root.attr == "gluon":
                    has_jit = True
    if not has_jit:
        return False, "no_jit: no @gluon.jit or @triton.jit decorated function found"

    # Hardcoded output detection
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        val = node.value
        if (
            isinstance(val, ast.Call)
            and isinstance(val.func, ast.Attribute)
            and isinstance(val.func.value, ast.Name)
            and _canonical(val.func.value.id) == "torch"
            and val.func.attr == "tensor"
            and val.args
            and _is_literal_tree(val.args[0])
        ):
            return False, "hardcoded_output: 'return torch.tensor(...)' with literal values"

    return True, ""


# ── HIP static checks ────────────────────────────────────────────────────────

_HIP_REQUIRED_MARKERS: tuple[str, ...] = (
    "PYBIND11_MODULE",
    "m.def(",
)

_HIP_HINT_MARKERS: tuple[str, ...] = (
    "#include <hip/hip_runtime.h>",
    "__global__",
    "hipLaunchKernelGGL",
    "ATen/hip",
    "c10/hip",
)

_HIP_BLOCKED_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\b(system|popen|fork|execv|execve|dlopen|dlsym)\s*\(", "blocked_hip_api"),
    (r"#include\s*<\s*(unistd\.h|sys/socket\.h|curl/curl\.h)\s*>", "blocked_hip_include"),
    (r"\b(socket|connect|bind|listen|accept|send|recv)\s*\(", "blocked_hip_api"),
)


def run_hip_static_check(code: str) -> tuple[bool, str]:
    """
    Perform lightweight static screening for HIP/C++ extension answers.

    Parameters
    ----------
    code : str
        HIP/C++ kernel source code to analyze.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` on pass; ``(False, reason)`` on any violation.
    """
    normalized = code.strip()
    if not normalized:
        return False, "no_hip_code: empty answer"

    for pattern, reason_tag in _HIP_BLOCKED_PATTERNS:
        match = re.search(pattern, normalized)
        if match:
            return False, f"{reason_tag}: disallowed token '{match.group(0)}'"

    for marker in _HIP_REQUIRED_MARKERS:
        if marker not in normalized:
            return False, f"no_hip_binding: missing required marker '{marker}'"

    if not any(marker in normalized for marker in _HIP_HINT_MARKERS):
        return False, "no_hip_kernel: answer does not look like HIP/C++ extension code"

    if not re.search(r'm\.def\(\s*"(\w+)"', normalized):
        return False, "no_hip_binding: no pybind entry function declared via m.def(...)"

    return True, ""


def run_backend_static_check(
    code: str,
    backend: str,
    allowed_imports: list[str] | None = None,
) -> tuple[bool, str]:
    """
    Dispatch backend-specific static checks.

    Parameters
    ----------
    code : str
        Kernel source code.
    backend : str
        Canonical backend constant (``"triton"`` or ``"hip"``).
    allowed_imports : list[str] | None
        Import whitelist override (Triton only).

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` on pass; ``(False, reason)`` on violation.
    """
    if backend == KERNEL_BACKEND_TRITON:
        return run_triton_static_check(code, allowed_imports=allowed_imports)
    if backend == KERNEL_BACKEND_GLUON:
        return run_gluon_static_check(code, allowed_imports=allowed_imports)
    if backend == KERNEL_BACKEND_HIP:
        return run_hip_static_check(code)
    raise ValueError(f"Unsupported backend: {backend}")
