#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
testcase_generator.py — Generate per-kernel correctness testcase scripts.

Produces a standalone testcase.py that:
  1. Dynamically loads baseline and solution modules
  2. Generates inputs for multiple shapes (small, production, edge-case)
  3. Runs both kernels and compares outputs with torch.allclose
  4. Exits 0 only if all checks pass (compatible with Magpie TESTCASE mode)
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional


# Shape derivation per kernel spec family
_ATTENTION_SPECS = {"flash_attn_prefill", "paged_attn_decode", "paged_attn_decode_gluon", "mla_attn"}
_GEMM_SPECS = {"gemm_bf16", "gemm_w8a8"}
_NORM_SPECS = {"rms_norm"}
_MOE_SPECS = {"fused_moe"}
_ACTIVATION_SPECS = {"silu_mul", "act_quant_fp8"}
_ROPE_SPECS = {"rope_embedding"}
_CACHE_SPECS = {"kv_cache_ops"}


def _derive_test_shapes(
    kernel_spec: str,
    model_config: dict,
    gap_analysis: Optional[dict] = None,
) -> list[dict]:
    """Derive test shapes from model config and optional profiler gap_analysis."""
    hidden = model_config.get("hidden_size", 4096)
    intermediate = model_config.get("intermediate_size", hidden * 4)
    num_heads = model_config.get("num_attention_heads", 32)
    num_kv_heads = model_config.get("num_key_value_heads", num_heads)
    head_dim = hidden // num_heads if num_heads > 0 else 128
    num_experts = model_config.get("num_local_experts", 8)
    vocab_size = model_config.get("vocab_size", 32000)

    if kernel_spec in _ATTENTION_SPECS:
        return [
            {"batch": 1, "seq_len": 128, "num_heads": num_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim, "label": "small"},
            {"batch": 4, "seq_len": 1024, "num_heads": num_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim, "label": "production"},
            {"batch": 3, "seq_len": 517, "num_heads": num_heads, "num_kv_heads": num_kv_heads, "head_dim": head_dim, "label": "edge_case"},
        ]
    elif kernel_spec in _GEMM_SPECS:
        return [
            {"M": 16, "N": hidden, "K": hidden, "label": "small"},
            {"M": 512, "N": hidden, "K": intermediate, "label": "production"},
            {"M": 137, "N": hidden, "K": intermediate, "label": "edge_case"},
        ]
    elif kernel_spec in _MOE_SPECS:
        return [
            {"tokens": 32, "hidden": hidden, "intermediate": intermediate, "num_experts": num_experts, "top_k": min(4, num_experts), "label": "small"},
            {"tokens": 256, "hidden": hidden, "intermediate": intermediate, "num_experts": num_experts, "top_k": min(4, num_experts), "label": "production"},
            {"tokens": 97, "hidden": hidden, "intermediate": intermediate, "num_experts": num_experts, "top_k": min(4, num_experts), "label": "edge_case"},
        ]
    elif kernel_spec in _NORM_SPECS:
        return [
            {"batch": 4, "seq_len": 128, "hidden": hidden, "label": "small"},
            {"batch": 16, "seq_len": 1024, "hidden": hidden, "label": "production"},
            {"batch": 7, "seq_len": 513, "hidden": hidden, "label": "edge_case"},
        ]
    elif kernel_spec in _ACTIVATION_SPECS:
        return [
            {"batch": 4, "seq_len": 128, "hidden": intermediate, "label": "small"},
            {"batch": 16, "seq_len": 1024, "hidden": intermediate, "label": "production"},
            {"batch": 7, "seq_len": 333, "hidden": intermediate, "label": "edge_case"},
        ]
    elif kernel_spec in _ROPE_SPECS:
        return [
            {"batch": 2, "seq_len": 128, "num_heads": num_heads, "head_dim": head_dim, "label": "small"},
            {"batch": 8, "seq_len": 1024, "num_heads": num_heads, "head_dim": head_dim, "label": "production"},
            {"batch": 3, "seq_len": 511, "num_heads": num_heads, "head_dim": head_dim, "label": "edge_case"},
        ]
    else:
        return [
            {"M": 64, "N": hidden, "K": hidden, "label": "small"},
            {"M": 512, "N": hidden, "K": intermediate, "label": "production"},
            {"M": 131, "N": hidden, "K": intermediate, "label": "edge_case"},
        ]


TESTCASE_TEMPLATE = textwrap.dedent('''\
#!/usr/bin/env python3
"""Auto-generated correctness testcase for kernel: {kernel_spec}

Loads baseline and solution modules, runs both with multiple shapes,
and compares outputs with torch.allclose. Exits 0 only if all pass.
"""
import importlib.util
import sys
import torch

ATOL = {atol}
RTOL = {rtol}
SHAPES = {shapes}


def _load_module(path, name):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {{path}}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_callable(mod):
    """Find the main callable in a module (kernel function or forward)."""
    for name in ("forward", "kernel_fn", "main", "{kernel_spec}", "__call__"):
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    # Fallback: find first public callable that isn't a class
    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if callable(obj) and not isinstance(obj, type):
            return obj
    return None


def _generate_inputs(shape_dict):
    """Generate random tensor inputs based on shape dict."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    if "M" in shape_dict:
        M, N, K = shape_dict["M"], shape_dict["N"], shape_dict["K"]
        A = torch.randn(M, K, device=device, dtype=dtype)
        B = torch.randn(K, N, device=device, dtype=dtype)
        return (A, B)
    elif "num_heads" in shape_dict and "seq_len" in shape_dict:
        batch = shape_dict.get("batch", 1)
        seq_len = shape_dict["seq_len"]
        num_heads = shape_dict["num_heads"]
        head_dim = shape_dict["head_dim"]
        Q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        K = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        V = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        return (Q, K, V)
    elif "tokens" in shape_dict:
        tokens = shape_dict["tokens"]
        hidden = shape_dict["hidden"]
        X = torch.randn(tokens, hidden, device=device, dtype=dtype)
        return (X,)
    elif "hidden" in shape_dict:
        batch = shape_dict.get("batch", 1)
        seq_len = shape_dict.get("seq_len", 128)
        hidden = shape_dict["hidden"]
        X = torch.randn(batch * seq_len, hidden, device=device, dtype=dtype)
        return (X,)
    else:
        return (torch.randn(64, 64, device="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype),)


def run_tests():
    baseline_path = "{baseline_path}"
    solution_path = "{solution_path}"

    try:
        baseline_mod = _load_module(baseline_path, "baseline_mod")
    except Exception as e:
        print(f"FAIL: Cannot load baseline: {{e}}")
        return False

    try:
        solution_mod = _load_module(solution_path, "solution_mod")
    except Exception as e:
        print(f"FAIL: Cannot load solution: {{e}}")
        return False

    baseline_fn = _find_callable(baseline_mod)
    solution_fn = _find_callable(solution_mod)

    if baseline_fn is None:
        print("FAIL: No callable found in baseline module")
        return False
    if solution_fn is None:
        print("FAIL: No callable found in solution module")
        return False

    all_passed = True
    for shape in SHAPES:
        label = shape.get("label", "unknown")
        try:
            inputs = _generate_inputs(shape)
            with torch.no_grad():
                baseline_out = baseline_fn(*inputs)
                solution_out = solution_fn(*inputs)

            if baseline_out is None and solution_out is None:
                print(f"PASS [{{label}}]: both returned None")
                continue

            if isinstance(baseline_out, (tuple, list)):
                baseline_out = baseline_out[0]
            if isinstance(solution_out, (tuple, list)):
                solution_out = solution_out[0]

            if not isinstance(baseline_out, torch.Tensor):
                print(f"SKIP [{{label}}]: baseline output is not a tensor ({{type(baseline_out)}})")
                continue
            if not isinstance(solution_out, torch.Tensor):
                print(f"FAIL [{{label}}]: solution output is not a tensor ({{type(solution_out)}})")
                all_passed = False
                continue

            if baseline_out.shape != solution_out.shape:
                print(f"FAIL [{{label}}]: shape mismatch: baseline={{baseline_out.shape}} vs solution={{solution_out.shape}}")
                all_passed = False
                continue

            b_f = baseline_out.float()
            s_f = solution_out.float()

            if torch.isnan(s_f).any():
                print(f"FAIL [{{label}}]: solution output contains NaN")
                all_passed = False
                continue

            if torch.isinf(s_f).any():
                print(f"FAIL [{{label}}]: solution output contains Inf")
                all_passed = False
                continue

            if torch.allclose(b_f, s_f, atol=ATOL, rtol=RTOL):
                max_diff = (b_f - s_f).abs().max().item()
                print(f"PASS [{{label}}]: allclose OK (max_diff={{max_diff:.6e}})")
            else:
                max_diff = (b_f - s_f).abs().max().item()
                mean_diff = (b_f - s_f).abs().mean().item()
                print(f"FAIL [{{label}}]: allclose FAILED (max_diff={{max_diff:.6e}}, mean_diff={{mean_diff:.6e}})")
                all_passed = False

        except Exception as e:
            print(f"FAIL [{{label}}]: exception: {{e}}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    passed = run_tests()
    if passed:
        print("\\nAll correctness checks PASSED")
        sys.exit(0)
    else:
        print("\\nSome correctness checks FAILED")
        sys.exit(1)
''')


def generate_testcase(
    task_dir: Path,
    baseline_path: Path,
    kernel_spec: str,
    model_config: dict,
    gap_analysis: Optional[dict] = None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> Path:
    """Generate testcase.py that does baseline-vs-solution allclose comparison.

    Returns the path to the generated testcase script.
    """
    shapes = _derive_test_shapes(kernel_spec, model_config, gap_analysis)

    script = TESTCASE_TEMPLATE.format(
        kernel_spec=kernel_spec,
        baseline_path=str(baseline_path.resolve()),
        solution_path=str((task_dir / "solution.py").resolve()),
        shapes=repr(shapes),
        atol=atol,
        rtol=rtol,
    )

    testcase_path = task_dir / "testcase.py"
    testcase_path.write_text(script)
    return testcase_path
