# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
ground_truth_templates.py — Per-kernel-type CPU baseline and test-shape generators.

Each kernel_spec in Apex's KernelSpec registry gets a template that produces:
  - cpu_baseline_code: source of ``baseline_fn(*inputs) -> flat 1-D tensor``
  - test_shapes_code:  source of ``get_test_inputs(device) -> list[tuple]``

These are used by the RL evaluator to check correctness of
model-generated Triton kernels during GRPO training.

Usage:
    from ground_truth_templates import get_ground_truth, get_instruction, OP_TYPE_MAP
    gt = get_ground_truth("rms_norm", baseline_code)
    # gt == {"cpu_baseline_code": "def baseline_fn(...)...", "test_shapes_code": "..."}
"""

from __future__ import annotations

from typing import Any

# kernel_spec -> "memory_bound" | "compute_bound"
OP_TYPE_MAP: dict[str, str] = {
    "flash_attn_prefill": "compute_bound",
    "paged_attn_decode": "memory_bound",
    "mla_attn": "compute_bound",
    "fused_moe": "compute_bound",
    "gemm_w8a8": "compute_bound",
    "gemm_bf16": "compute_bound",
    "rms_norm": "memory_bound",
    "rope_embedding": "memory_bound",
    "kv_cache_ops": "memory_bound",
    "all_reduce": "memory_bound",
    "act_quant_fp8": "memory_bound",
    "silu_mul": "memory_bound",
}

# ── Instruction templates ────────────────────────────────────────────────────

_INSTRUCTION_TEMPLATE = (
    "You are an expert AMD GPU AI Compiler Engineer specializing in Triton "
    "kernels for AMD Instinct {gpu_name} ({cdna_gen}, {gpu_arch}). Your task is "
    "to optimize the provided baseline {description} kernel for maximum GPU "
    "throughput while maintaining numerical correctness. Focus on: memory "
    "coalescing, tiling, vectorized loads/stores, LDS usage, MFMA utilization, "
    "register pressure, and occupancy."
)

_GPU_INFO = {
    "gfx950": {"gpu_name": "MI355X", "cdna_gen": "CDNA4"},
    "gfx942": {"gpu_name": "MI300X", "cdna_gen": "CDNA3"},
    "gfx940": {"gpu_name": "MI300A", "cdna_gen": "CDNA3"},
    "gfx90a": {"gpu_name": "MI250X", "cdna_gen": "CDNA2"},
}

_KERNEL_DESCRIPTIONS: dict[str, str] = {
    "flash_attn_prefill": "Flash Attention prefill (prompt-phase multi-head attention)",
    "paged_attn_decode": "Paged Attention decode (single-token autoregressive decoding)",
    "mla_attn": "Multi-Head Latent Attention (MLA, compressed KV)",
    "fused_moe": "Fused Mixture-of-Experts (gate + topk routing + expert GEMM)",
    "gemm_w8a8": "FP8 Weight-Activation GEMM (W8A8)",
    "gemm_bf16": "BF16 General Matrix Multiply (GEMM)",
    "rms_norm": "RMS Normalization",
    "rope_embedding": "Rotary Position Embedding (RoPE)",
    "kv_cache_ops": "KV Cache reshape/copy/quantization",
    "all_reduce": "Tensor-parallel All-Reduce",
    "act_quant_fp8": "Dynamic per-token FP8 activation quantization",
    "silu_mul": "Fused SiLU-gate (SwiGLU) activation",
}


def get_instruction(kernel_spec: str, gpu_arch: str = "gfx950") -> str:
    """Return the optimization instruction/persona for a kernel type."""
    info = _GPU_INFO.get(gpu_arch, _GPU_INFO["gfx950"])
    desc = _KERNEL_DESCRIPTIONS.get(kernel_spec, kernel_spec)
    return _INSTRUCTION_TEMPLATE.format(
        gpu_name=info["gpu_name"],
        cdna_gen=info["cdna_gen"],
        gpu_arch=gpu_arch,
        description=desc,
    )


# ── Ground truth templates ───────────────────────────────────────────────────
# Each value is a dict with "cpu_baseline_code" and "test_shapes_code" strings.
# These are designed to be safe for exec() in the RL evaluator sandbox.

_TEMPLATES: dict[str, dict[str, str]] = {}

# ── rms_norm ─────────────────────────────────────────────────────────────────

_TEMPLATES["rms_norm"] = {
    "cpu_baseline_code": """\
def baseline_fn(x, weight, eps=1e-6):
    import torch
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    result = (x_normed * weight.float()).to(x.dtype)
    return result.reshape(-1)
""",
    "test_shapes_code": """\
def get_test_inputs(device='cuda'):
    import torch
    shapes = [(32, 1024), (64, 4096), (128, 8192), (256, 4096), (512, 1024)]
    inputs = []
    for i, (rows, hidden) in enumerate(shapes):
        torch.manual_seed(42 + i)
        x = torch.randn(rows, hidden, device=device, dtype=torch.float16)
        weight = torch.randn(hidden, device=device, dtype=torch.float16)
        inputs.append((x, weight))
    return inputs
""",
}

# ── silu_mul ─────────────────────────────────────────────────────────────────

_TEMPLATES["silu_mul"] = {
    "cpu_baseline_code": """\
def baseline_fn(x):
    import torch
    import torch.nn.functional as F
    d = x.shape[-1] // 2
    gate = x[:, :d].float()
    up = x[:, d:].float()
    gate_silu = F.silu(gate).clamp(max=7.0)
    up_clamped = up.clamp(-7.0, 7.0)
    result = (gate_silu * up_clamped).to(x.dtype)
    return result.reshape(-1)
""",
    "test_shapes_code": """\
def get_test_inputs(device='cuda'):
    import torch
    shapes = [(32, 512), (64, 1024), (128, 2048), (256, 4096), (512, 8192)]
    inputs = []
    for i, (batch, two_d) in enumerate(shapes):
        torch.manual_seed(42 + i)
        x = torch.randn(batch, two_d, device=device, dtype=torch.float16)
        inputs.append((x,))
    return inputs
""",
}

# ── act_quant_fp8 ────────────────────────────────────────────────────────────

_TEMPLATES["act_quant_fp8"] = {
    "cpu_baseline_code": """\
def baseline_fn(x, group_size):
    import torch
    x_flat = x.reshape(-1, x.shape[-1])
    M, N = x_flat.shape
    num_groups = N // group_size
    x_grouped = x_flat.reshape(M, num_groups, group_size).float()
    absmax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    fp8_max = 240.0
    scales = absmax / fp8_max
    x_scaled = (x_grouped / scales).clamp(-fp8_max, fp8_max)
    x_q_sim = x_scaled.round().clamp(-fp8_max, fp8_max)
    x_deq = (x_q_sim * scales).reshape(M, N)
    return x_deq.reshape(-1)
""",
    "test_shapes_code": """\
def get_test_inputs(device='cuda'):
    import torch
    shapes = [(32, 256, 128), (64, 512, 128), (128, 1024, 128),
              (256, 2048, 128), (64, 1024, 64)]
    inputs = []
    for i, (M, N, gs) in enumerate(shapes):
        torch.manual_seed(42 + i)
        x = torch.randn(M, N, device=device, dtype=torch.float16)
        inputs.append((x, gs))
    return inputs
""",
}

# ── rope_embedding ───────────────────────────────────────────────────────────

_TEMPLATES["rope_embedding"] = {
    "cpu_baseline_code": """\
def baseline_fn(q, k, cos, sin):
    import torch
    num_tokens, qk_dim = q.shape[0], q.shape[1]
    half_rd = cos.shape[-1]
    q_out = q.clone().float()
    k_out = k.clone().float()
    cos_f = cos.float()
    sin_f = sin.float()
    n_qh = qk_dim // (half_rd * 2)
    head_size = half_rd * 2
    for h in range(n_qh):
        off = h * head_size
        x1 = q_out[:, off:off + half_rd]
        x2 = q_out[:, off + half_rd:off + head_size]
        q_out[:, off:off + half_rd] = x1 * cos_f - x2 * sin_f
        q_out[:, off + half_rd:off + head_size] = x2 * cos_f + x1 * sin_f
    n_kh = k.shape[1] // head_size
    for h in range(n_kh):
        off = h * head_size
        x1 = k_out[:, off:off + half_rd]
        x2 = k_out[:, off + half_rd:off + head_size]
        k_out[:, off:off + half_rd] = x1 * cos_f - x2 * sin_f
        k_out[:, off + half_rd:off + head_size] = x2 * cos_f + x1 * sin_f
    return torch.cat([q_out.to(q.dtype).reshape(-1), k_out.to(k.dtype).reshape(-1)])
""",
    "test_shapes_code": """\
def get_test_inputs(device='cuda'):
    import torch
    configs = [
        (32, 8, 8, 128, 128),
        (64, 16, 4, 128, 128),
        (128, 32, 8, 128, 64),
        (256, 64, 8, 128, 128),
        (16, 8, 2, 64, 64),
    ]
    inputs = []
    for i, (num_tokens, n_qh, n_kh, head_size, rotary_dim) in enumerate(configs):
        torch.manual_seed(42 + i)
        half_rd = rotary_dim // 2
        q = torch.randn(num_tokens, n_qh * head_size, device=device, dtype=torch.float16)
        k = torch.randn(num_tokens, n_kh * head_size, device=device, dtype=torch.float16)
        cos = torch.randn(num_tokens, half_rd, device=device, dtype=torch.float16)
        sin = torch.randn(num_tokens, half_rd, device=device, dtype=torch.float16)
        inputs.append((q, k, cos, sin))
    return inputs
""",
}

# ── gemm_bf16 ────────────────────────────────────────────────────────────────

_TEMPLATES["gemm_bf16"] = {
    "cpu_baseline_code": """\
def baseline_fn(A, B):
    import torch
    result = torch.mm(A.float(), B.float()).to(A.dtype)
    return result.reshape(-1)
""",
    "test_shapes_code": """\
def get_test_inputs(device='cuda'):
    import torch
    shapes = [(32, 8192, 8192), (128, 4096, 4096), (256, 2048, 8192),
              (512, 1024, 4096), (64, 8192, 2048)]
    inputs = []
    for i, (M, N, K) in enumerate(shapes):
        torch.manual_seed(42 + i)
        A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
        B = torch.randn(K, N, device=device, dtype=torch.bfloat16)
        inputs.append((A, B))
    return inputs
""",
}

# ── gemm_w8a8 ────────────────────────────────────────────────────────────────

_TEMPLATES["gemm_w8a8"] = {
    "cpu_baseline_code": """\
def baseline_fn(A, B, scale_a, scale_b):
    import torch
    A_f = A.float() * scale_a
    B_f = B.float() * scale_b
    result = torch.mm(A_f, B_f).to(torch.bfloat16)
    return result.reshape(-1)
""",
    "test_shapes_code": """\
def get_test_inputs(device='cuda'):
    import torch
    shapes = [(32, 8192, 8192), (128, 4096, 4096), (256, 2048, 8192),
              (512, 1024, 4096), (64, 8192, 2048)]
    inputs = []
    fp8_dtype = torch.float8_e4m3fnuz if hasattr(torch, 'float8_e4m3fnuz') else torch.float8_e4m3fn
    for i, (M, N, K) in enumerate(shapes):
        torch.manual_seed(42 + i)
        A = torch.randn(M, K, device=device, dtype=torch.bfloat16).to(fp8_dtype)
        B = torch.randn(K, N, device=device, dtype=torch.bfloat16).to(fp8_dtype)
        scale_a = torch.tensor(0.01, device=device, dtype=torch.float32)
        scale_b = torch.tensor(0.01, device=device, dtype=torch.float32)
        inputs.append((A, B, scale_a, scale_b))
    return inputs
""",
}

# ── flash_attn_prefill ───────────────────────────────────────────────────────

_TEMPLATES["flash_attn_prefill"] = {
    "cpu_baseline_code": """\
def baseline_fn(Q, K, V):
    import torch
    import torch.nn.functional as F
    scale = Q.shape[-1] ** -0.5
    out = F.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(), scale=scale
    ).to(Q.dtype)
    return out.reshape(-1)
""",
    "test_shapes_code": """\
def get_test_inputs(device='cuda'):
    import torch
    configs = [
        (1, 8, 128, 128),
        (1, 16, 256, 128),
        (1, 32, 512, 128),
        (2, 8, 128, 64),
        (1, 64, 64, 128),
    ]
    inputs = []
    for i, (batch, heads, seq_len, head_dim) in enumerate(configs):
        torch.manual_seed(42 + i)
        Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        inputs.append((Q, K, V))
    return inputs
""",
}

# ── paged_attn_decode ────────────────────────────────────────────────────────

_TEMPLATES["paged_attn_decode"] = {
    "cpu_baseline_code": """\
def baseline_fn(Q, K_cache, V_cache, scale):
    import torch
    import torch.nn.functional as F
    out = F.scaled_dot_product_attention(
        Q.float(), K_cache.float(), V_cache.float(), scale=scale
    ).to(Q.dtype)
    return out.reshape(-1)
""",
    "test_shapes_code": """\
def get_test_inputs(device='cuda'):
    import torch
    configs = [
        (1, 8, 1, 512, 128),
        (4, 8, 1, 1024, 128),
        (8, 16, 1, 256, 128),
        (16, 8, 1, 2048, 128),
        (1, 64, 1, 128, 128),
    ]
    inputs = []
    for i, (batch, heads, q_len, kv_len, head_dim) in enumerate(configs):
        torch.manual_seed(42 + i)
        scale = head_dim ** -0.5
        Q = torch.randn(batch, heads, q_len, head_dim, device=device, dtype=torch.float16)
        K = torch.randn(batch, heads, kv_len, head_dim, device=device, dtype=torch.float16)
        V = torch.randn(batch, heads, kv_len, head_dim, device=device, dtype=torch.float16)
        inputs.append((Q, K, V, scale))
    return inputs
""",
}

# ── mla_attn ─────────────────────────────────────────────────────────────────

_TEMPLATES["mla_attn"] = {
    "cpu_baseline_code": """\
def baseline_fn(Q, K_compressed, V_compressed, W_UK, W_UV):
    import torch
    K = torch.matmul(K_compressed.float(), W_UK.float())
    V = torch.matmul(V_compressed.float(), W_UV.float())
    scale = Q.shape[-1] ** -0.5
    attn = torch.matmul(Q.float(), K.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    out = torch.matmul(attn, V).to(Q.dtype)
    return out.reshape(-1)
""",
    "test_shapes_code": """\
def get_test_inputs(device='cuda'):
    import torch
    configs = [
        (1, 8, 128, 128, 512),
        (1, 16, 64, 128, 512),
        (2, 8, 256, 128, 512),
        (1, 32, 128, 64, 256),
        (1, 8, 512, 128, 512),
    ]
    inputs = []
    for i, (batch, heads, seq_len, head_dim, latent_dim) in enumerate(configs):
        torch.manual_seed(42 + i)
        Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
        K_c = torch.randn(batch, heads, seq_len, latent_dim, device=device, dtype=torch.float16)
        V_c = torch.randn(batch, heads, seq_len, latent_dim, device=device, dtype=torch.float16)
        W_UK = torch.randn(latent_dim, head_dim, device=device, dtype=torch.float16)
        W_UV = torch.randn(latent_dim, head_dim, device=device, dtype=torch.float16)
        inputs.append((Q, K_c, V_c, W_UK, W_UV))
    return inputs
""",
}

# ── fused_moe ────────────────────────────────────────────────────────────────

_TEMPLATES["fused_moe"] = {
    "cpu_baseline_code": """\
def baseline_fn(x, gate_logits, expert_weights, topk):
    import torch
    scores = torch.softmax(gate_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(scores, topk, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    M, D = x.shape
    E, D_out, _ = expert_weights.shape
    output = torch.zeros(M, D_out, device=x.device, dtype=torch.float32)
    for i in range(topk):
        eid = topk_ids[:, i]
        w = topk_weights[:, i:i+1]
        for e in range(E):
            mask = eid == e
            if mask.any():
                x_e = x[mask].float()
                out_e = torch.mm(x_e, expert_weights[e].float().T)
                output[mask] += out_e * w[mask]
    return output.to(x.dtype).reshape(-1)
""",
    "test_shapes_code": """\
def get_test_inputs(device='cuda'):
    import torch
    configs = [
        (32, 1024, 16, 4, 1024),
        (64, 2048, 8, 2, 2048),
        (128, 4096, 16, 4, 4096),
        (256, 1024, 8, 2, 1024),
        (16, 2048, 16, 4, 2048),
    ]
    inputs = []
    for i, (M, D, E, topk, D_out) in enumerate(configs):
        torch.manual_seed(42 + i)
        x = torch.randn(M, D, device=device, dtype=torch.float16)
        gate = torch.randn(M, E, device=device, dtype=torch.float16)
        W = torch.randn(E, D_out, D, device=device, dtype=torch.float16)
        inputs.append((x, gate, W, topk))
    return inputs
""",
}

# ── kv_cache_ops ─────────────────────────────────────────────────────────────

_TEMPLATES["kv_cache_ops"] = {
    "cpu_baseline_code": """\
def baseline_fn(key, value, cache_k, cache_v, slot_mapping):
    import torch
    out_ck = cache_k.clone()
    out_cv = cache_v.clone()
    for i, slot in enumerate(slot_mapping):
        block_idx = slot // cache_k.shape[1]
        block_off = slot % cache_k.shape[1]
        if block_idx < out_ck.shape[0]:
            out_ck[block_idx, block_off] = key[i]
            out_cv[block_idx, block_off] = value[i]
    return torch.cat([out_ck.reshape(-1), out_cv.reshape(-1)])
""",
    "test_shapes_code": """\
def get_test_inputs(device='cuda'):
    import torch
    configs = [
        (16, 8, 128, 16, 64),
        (64, 8, 128, 16, 128),
        (128, 16, 128, 16, 256),
        (32, 8, 64, 16, 32),
        (256, 8, 128, 16, 512),
    ]
    inputs = []
    for i, (num_tokens, heads, head_dim, block_size, num_blocks) in enumerate(configs):
        torch.manual_seed(42 + i)
        key = torch.randn(num_tokens, heads, head_dim, device=device, dtype=torch.float16)
        value = torch.randn(num_tokens, heads, head_dim, device=device, dtype=torch.float16)
        cache_k = torch.zeros(num_blocks, block_size, heads, head_dim, device=device, dtype=torch.float16)
        cache_v = torch.zeros(num_blocks, block_size, heads, head_dim, device=device, dtype=torch.float16)
        total_slots = num_blocks * block_size
        slot_mapping = torch.randperm(total_slots, device=device)[:num_tokens]
        inputs.append((key, value, cache_k, cache_v, slot_mapping))
    return inputs
""",
}

# ── all_reduce ───────────────────────────────────────────────────────────────

_TEMPLATES["all_reduce"] = {
    "cpu_baseline_code": """\
def baseline_fn(tensor):
    import torch
    return tensor.reshape(-1).clone()
""",
    "test_shapes_code": """\
def get_test_inputs(device='cuda'):
    import torch
    sizes = [1024, 16384, 65536, 262144, 1048576]
    inputs = []
    for i, size in enumerate(sizes):
        torch.manual_seed(42 + i)
        t = torch.randn(size, device=device, dtype=torch.bfloat16)
        inputs.append((t,))
    return inputs
""",
}


# ── Public API ───────────────────────────────────────────────────────────────

def get_ground_truth(
    kernel_spec: str,
    baseline_code: str | None = None,
    model_config: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Return {"cpu_baseline_code": str, "test_shapes_code": str} for a kernel type.

    Falls back to a generic wrapper around the baseline Triton kernel if no
    specific template exists for the given kernel_spec.
    """
    if kernel_spec in _TEMPLATES:
        return dict(_TEMPLATES[kernel_spec])

    if baseline_code:
        return _generic_baseline_wrapper(kernel_spec, baseline_code)

    raise ValueError(
        f"No ground truth template for kernel_spec={kernel_spec!r} "
        f"and no baseline_code provided for generic fallback."
    )


def _generic_baseline_wrapper(kernel_spec: str, baseline_code: str) -> dict[str, str]:
    """Wrap the baseline kernel itself as its own correctness oracle.

    The generated cpu_baseline_code imports and runs the baseline on GPU,
    then returns the flattened output. This works when the baseline kernel
    is already correct and we just need a reference to compare against.
    """
    cpu_code = (
        "def baseline_fn(*args):\n"
        "    import torch, importlib, types, textwrap\n"
        "    _src = " + repr(baseline_code) + "\n"
        "    _mod = types.ModuleType('_baseline')\n"
        "    exec(compile(_src, '<baseline>', 'exec'), _mod.__dict__)\n"
        "    fns = [v for v in vars(_mod).values()\n"
        "           if callable(v) and not v.__name__.startswith('_')]\n"
        "    assert fns, 'no callable found in baseline code'\n"
        "    main_fn = fns[-1]\n"
        "    result = main_fn(*args)\n"
        "    if isinstance(result, tuple):\n"
        "        return torch.cat([r.reshape(-1) for r in result])\n"
        "    return result.reshape(-1)\n"
    )
    test_code = (
        "def get_test_inputs(device='cuda'):\n"
        "    import torch\n"
        "    inputs = []\n"
        "    for i in range(3):\n"
        "        torch.manual_seed(42 + i)\n"
        "        x = torch.randn(64, 1024, device=device, dtype=torch.float16)\n"
        "        inputs.append((x,))\n"
        "    return inputs\n"
    )
    return {"cpu_baseline_code": cpu_code, "test_shapes_code": test_code}


def list_supported_kernels() -> list[str]:
    """Return kernel_spec names that have dedicated templates."""
    return sorted(_TEMPLATES.keys())
