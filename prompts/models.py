"""
models.py — Registry of open-source LLM models available in SGLang and vLLM
with AMD/ROCm-optimized kernels.

Model selection criteria:
  - Actively supported in SGLang ≥ 0.4 and/or vLLM ≥ 0.6 on ROCm
  - Represents diversity in: architecture, attention type, size, and MoE vs dense
  - AMD-specific kernel paths confirmed or likely (flash attn, MLA, FusedMoE, GEMM)
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    hf_id:          str          # HuggingFace model ID
    family:         str          # Model family (llama, qwen, deepseek, …)
    params_b:       float        # Parameter count in billions
    attention:      str          # mha | gqa | mqa | mla | sliding_window | moe_gqa
    num_heads:      int          # Query heads
    num_kv_heads:   int          # KV heads (= num_heads for MHA)
    head_dim:       int
    hidden_dim:     int
    num_layers:     int
    mlp_type:       str          # dense | moe | moe_shared
    num_experts:    int          # 1 for dense
    active_experts: int          # experts activated per token
    context_len:    int          # max context length (tokens)
    frameworks:     list[str]    = field(default_factory=list)  # sglang | vllm | both
    notes:          str          = ""


# ── Model registry ─────────────────────────────────────────────────────────────
# 19 models; ~15 distinct kernel shapes for good RL diversity.

MODELS: list[ModelConfig] = [

    # ── Llama 3 family ───────────────────────────────────────────────────────
    ModelConfig(
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        family="llama3", params_b=1.2,
        attention="gqa",    num_heads=32, num_kv_heads=8,  head_dim=64,
        hidden_dim=2048,    num_layers=16,
        mlp_type="dense",   num_experts=1, active_experts=1,
        context_len=128_000,
        frameworks=["sglang", "vllm"],
        notes="Smallest Llama 3; decode-bound, good for occupancy tuning",
    ),
    ModelConfig(
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        family="llama3", params_b=8,
        attention="gqa",    num_heads=32, num_kv_heads=8,  head_dim=128,
        hidden_dim=4096,    num_layers=32,
        mlp_type="dense",   num_experts=1, active_experts=1,
        context_len=128_000,
        frameworks=["sglang", "vllm"],
        notes="Reference dense model; MLPerf LLM-v0.5",
    ),
    ModelConfig(
        hf_id="meta-llama/Llama-3.1-70B-Instruct",
        family="llama3", params_b=70,
        attention="gqa",    num_heads=64, num_kv_heads=8,  head_dim=128,
        hidden_dim=8192,    num_layers=80,
        mlp_type="dense",   num_experts=1, active_experts=1,
        context_len=128_000,
        frameworks=["sglang", "vllm"],
        notes="Large dense; GEMM-bound at batch>1; MLPerf Llama-2-70B successor",
    ),
    ModelConfig(
        hf_id="meta-llama/Llama-3.3-70B-Instruct",
        family="llama3", params_b=70,
        attention="gqa",    num_heads=64, num_kv_heads=8,  head_dim=128,
        hidden_dim=8192,    num_layers=80,
        mlp_type="dense",   num_experts=1, active_experts=1,
        context_len=128_000,
        frameworks=["sglang", "vllm"],
        notes="Latest Llama 3 release; instruction-tuned",
    ),

    # ── Mistral / Mixtral ────────────────────────────────────────────────────
    ModelConfig(
        hf_id="mistralai/Mistral-7B-Instruct-v0.3",
        family="mistral", params_b=7,
        attention="gqa",    num_heads=32, num_kv_heads=8,  head_dim=128,
        hidden_dim=4096,    num_layers=32,
        mlp_type="dense",   num_experts=1, active_experts=1,
        context_len=32_768,
        frameworks=["sglang", "vllm"],
        notes="Sliding-window attention (window=4096); tests chunked prefill kernels",
    ),
    ModelConfig(
        hf_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        family="mixtral", params_b=46.7,
        attention="gqa",    num_heads=32, num_kv_heads=8,  head_dim=128,
        hidden_dim=4096,    num_layers=32,
        mlp_type="moe",     num_experts=8, active_experts=2,
        context_len=32_768,
        frameworks=["sglang", "vllm"],
        notes="FusedMoE critical path; top-2 routing; MLPerf v4",
    ),
    ModelConfig(
        hf_id="mistralai/Mixtral-8x22B-Instruct-v0.1",
        family="mixtral", params_b=141,
        attention="gqa",    num_heads=48, num_kv_heads=8,  head_dim=128,
        hidden_dim=6144,    num_layers=56,
        mlp_type="moe",     num_experts=8, active_experts=2,
        context_len=65_536,
        frameworks=["sglang", "vllm"],
        notes="Largest Mixtral; tests FusedMoE + large GEMM on MI355",
    ),

    # ── Qwen 2.5 family ──────────────────────────────────────────────────────
    ModelConfig(
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        family="qwen", params_b=7.6,
        attention="gqa",    num_heads=28, num_kv_heads=4,  head_dim=128,
        hidden_dim=3584,    num_layers=28,
        mlp_type="dense",   num_experts=1, active_experts=1,
        context_len=131_072,
        frameworks=["sglang", "vllm"],
        notes="Non-power-of-2 head counts; tests generalised attention tiling",
    ),
    ModelConfig(
        hf_id="Qwen/Qwen2.5-32B-Instruct",
        family="qwen", params_b=32.5,
        attention="gqa",    num_heads=40, num_kv_heads=8,  head_dim=128,
        hidden_dim=5120,    num_layers=64,
        mlp_type="dense",   num_experts=1, active_experts=1,
        context_len=131_072,
        frameworks=["sglang", "vllm"],
        notes="Mid-size; memory-bandwidth bound at small batch",
    ),
    ModelConfig(
        hf_id="Qwen/Qwen2.5-72B-Instruct",
        family="qwen", params_b=72.7,
        attention="gqa",    num_heads=64, num_kv_heads=8,  head_dim=128,
        hidden_dim=8192,    num_layers=80,
        mlp_type="dense",   num_experts=1, active_experts=1,
        context_len=131_072,
        frameworks=["sglang", "vllm"],
        notes="Comparable size to Llama-70B; different head config",
    ),
    ModelConfig(
        hf_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        family="qwen", params_b=32.5,
        attention="gqa",    num_heads=40, num_kv_heads=8,  head_dim=128,
        hidden_dim=5120,    num_layers=64,
        mlp_type="dense",   num_experts=1, active_experts=1,
        context_len=131_072,
        frameworks=["sglang", "vllm"],
        notes="Code-generation workload; long-context prefill stress",
    ),

    # ── Google Gemma 2 ───────────────────────────────────────────────────────
    ModelConfig(
        hf_id="google/gemma-2-9b-it",
        family="gemma2", params_b=9.2,
        attention="mha",    num_heads=16, num_kv_heads=8,  head_dim=256,
        hidden_dim=3584,    num_layers=42,
        mlp_type="dense",   num_experts=1, active_experts=1,
        context_len=8_192,
        frameworks=["sglang", "vllm"],
        notes="Alternating sliding-window + global attention; head_dim=256 unusual",
    ),
    ModelConfig(
        hf_id="google/gemma-2-27b-it",
        family="gemma2", params_b=27.2,
        attention="mha",    num_heads=32, num_kv_heads=16, head_dim=128,
        hidden_dim=4608,    num_layers=46,
        mlp_type="dense",   num_experts=1, active_experts=1,
        context_len=8_192,
        frameworks=["sglang", "vllm"],
        notes="Larger Gemma; tests intermediate head_dim and deeper nets",
    ),

    # ── DeepSeek (MLA + MoE) ─────────────────────────────────────────────────
    ModelConfig(
        hf_id="deepseek-ai/DeepSeek-R1",
        family="deepseek", params_b=671,
        attention="mla",    num_heads=128, num_kv_heads=128, head_dim=128,
        hidden_dim=7168,    num_layers=61,
        mlp_type="moe",     num_experts=256, active_experts=8,
        context_len=163_840,
        frameworks=["sglang", "vllm"],
        notes="Multi-Head Latent Attention (MLA); SGLang AMD blog post; FP8 recommended",
    ),
    ModelConfig(
        hf_id="deepseek-ai/DeepSeek-V3",
        family="deepseek", params_b=671,
        attention="mla",    num_heads=128, num_kv_heads=128, head_dim=128,
        hidden_dim=7168,    num_layers=61,
        mlp_type="moe_shared", num_experts=256, active_experts=8,
        context_len=163_840,
        frameworks=["sglang", "vllm"],
        notes="MoE with 1 shared expert + 8 routed; FP8 only on MI300+; MLA",
    ),
    ModelConfig(
        hf_id="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        family="deepseek_distill", params_b=70,
        attention="gqa",    num_heads=64, num_kv_heads=8,  head_dim=128,
        hidden_dim=8192,    num_layers=80,
        mlp_type="dense",   num_experts=1, active_experts=1,
        context_len=131_072,
        frameworks=["sglang", "vllm"],
        notes="DeepSeek reasoning distilled into Llama arch; dense, no MLA",
    ),

    # ── Microsoft Phi ────────────────────────────────────────────────────────
    ModelConfig(
        hf_id="microsoft/Phi-3.5-mini-instruct",
        family="phi", params_b=3.8,
        attention="gqa",    num_heads=32, num_kv_heads=8,  head_dim=96,
        hidden_dim=3072,    num_layers=32,
        mlp_type="dense",   num_experts=1, active_experts=1,
        context_len=131_072,
        frameworks=["sglang", "vllm"],
        notes="Small, decode-efficient; head_dim=96 (non-power-of-2 padding needed)",
    ),
    ModelConfig(
        hf_id="microsoft/phi-4",
        family="phi", params_b=14.7,
        attention="gqa",    num_heads=40, num_kv_heads=10, head_dim=128,
        hidden_dim=5120,    num_layers=40,
        mlp_type="dense",   num_experts=1, active_experts=1,
        context_len=16_384,
        frameworks=["sglang", "vllm"],
        notes="Mid-size Phi; GQA with unusual 10 KV heads",
    ),

    # ── Falcon ───────────────────────────────────────────────────────────────
    ModelConfig(
        hf_id="tiiuae/falcon-7b-instruct",
        family="falcon", params_b=7,
        attention="mqa",    num_heads=71, num_kv_heads=1,  head_dim=64,
        hidden_dim=4544,    num_layers=32,
        mlp_type="dense",   num_experts=1, active_experts=1,
        context_len=8_192,
        frameworks=["vllm"],
        notes="MQA (1 KV head); extreme KV compression; non-power-of-2 heads",
    ),
]

# ── Convenience groupings ──────────────────────────────────────────────────────

def by_attention(attn_type: str) -> list[ModelConfig]:
    return [m for m in MODELS if m.attention == attn_type]

def by_family(family: str) -> list[ModelConfig]:
    return [m for m in MODELS if m.family == family]

def moe_models() -> list[ModelConfig]:
    return [m for m in MODELS if m.mlp_type in ("moe", "moe_shared")]

def dense_models() -> list[ModelConfig]:
    return [m for m in MODELS if m.mlp_type == "dense"]

def in_framework(framework: str) -> list[ModelConfig]:
    return [m for m in MODELS if framework in m.frameworks]


if __name__ == "__main__":
    print(f"Total models: {len(MODELS)}")
    print(f"  Dense:    {len(dense_models())}")
    print(f"  MoE:      {len(moe_models())}")
    print(f"  GQA:      {len(by_attention('gqa'))}")
    print(f"  MLA:      {len(by_attention('mla'))}")
    print(f"  MQA:      {len(by_attention('mqa'))}")
    print(f"  SGLang:   {len(in_framework('sglang'))}")
    print(f"  vLLM:     {len(in_framework('vllm'))}")
    for m in MODELS:
        print(f"  {m.hf_id:55s}  {m.params_b:6.1f}B  {m.attention:15s}  {m.mlp_type}")
