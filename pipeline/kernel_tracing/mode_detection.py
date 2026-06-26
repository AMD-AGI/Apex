"""Trace mode detection for kernel tracing."""

from __future__ import annotations

from pathlib import Path

from .patch_triton import has_triton_launch


TRACE_MODES = {
    "auto",
    "triton-launch",
    "aiter-compile-ops",
    "vllm-custom-op",
    "sglang-custom-op",
    "agent",
}


def normalize_trace_mode(mode: str = "auto", kernel_type: str = "") -> str:
    mode = (mode or "auto").strip().lower()
    kernel_type = (kernel_type or "").strip().lower()
    if mode == "auto" and kernel_type:
        if kernel_type == "triton":
            return "triton-launch"
        if kernel_type == "hip":
            return "auto"
    if mode not in TRACE_MODES:
        raise ValueError(f"Unsupported trace mode: {mode}")
    return mode


def detect_trace_mode(source_path: Path, kernel_name: str, requested: str = "auto") -> str:
    requested = normalize_trace_mode(requested)
    if requested != "auto":
        return requested

    source = source_path.read_text(encoding="utf-8")
    path_s = str(source_path)

    # Generic aliases hide the real target identity; let Agent fallback choose
    # the semantically useful patch point.
    if kernel_name in {"kernel", "call_kernel"}:
        return "agent"

    if has_triton_launch(source, kernel_name):
        return "triton-launch"
    if "@triton.jit" in source or "triton.jit" in source:
        return "agent"
    if "@compile_ops" in source or "compile_ops(" in source or "aiter/jit/core.py" in path_s:
        return "aiter-compile-ops"
    if (
        "direct_register_custom_op" in source
        or "torch.ops.vllm" in source
        or "torch.ops._C" in source
        or "CustomOp" in source
        or "vllm/" in path_s
    ):
        return "vllm-custom-op"
    if (
        "register_custom_op" in source
        or "debug_kernel_api" in source
        or "debug_torch_op" in source
        or "torch.ops.sgl_kernel" in source
        or "sglang/" in path_s
    ):
        return "sglang-custom-op"
    return "agent"
