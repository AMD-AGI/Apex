"""Required repository cases for tracing patch validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceTestCase:
    repo: str
    kind: str
    target: str
    file: str
    patch_path: str
    static_expected: bool


TRACE_TEST_CASES: list[TraceTestCase] = [
    TraceTestCase("aiter", "triton", "kernel_unified_attention_2d", "tools/rocm/aiter/aiter/ops/triton/attention/unified_attention.py", "triton-launch", True),
    TraceTestCase("aiter", "triton", "_paged_attn_decode_v1_wo_dot_kernel", "tools/rocm/aiter/aiter/ops/triton/attention/pa_decode.py", "triton-launch", True),
    TraceTestCase("aiter", "triton", "_attn_fwd", "tools/rocm/aiter/aiter/ops/triton/attention/mha.py", "triton-launch", True),
    TraceTestCase("aiter", "triton", "_fwd_kernel_stage1", "tools/rocm/aiter/aiter/ops/triton/attention/mla_decode.py", "triton-launch", True),
    TraceTestCase("aiter", "triton", "kernel", "tools/rocm/aiter/aiter/ops/triton/attention/pa_mqa_logits.py", "agent", False),
    TraceTestCase("aiter", "hip", "reshape_and_cache", "tools/rocm/aiter/aiter/ops/cache.py", "aiter-compile-ops", True),
    TraceTestCase("aiter", "hip", "rms_norm", "tools/rocm/aiter/aiter/ops/rmsnorm.py", "aiter-compile-ops", True),
    TraceTestCase("aiter", "hip", "biased_grouped_topk", "tools/rocm/aiter/aiter/ops/topk.py", "aiter-compile-ops", True),
    TraceTestCase("aiter", "hip", "gemm_a8w8", "tools/rocm/aiter/aiter/ops/gemm_op_a8w8.py", "aiter-compile-ops", True),
    TraceTestCase("aiter", "hip", "fmoe", "tools/rocm/aiter/aiter/ops/moe_op.py", "aiter-compile-ops", True),
    TraceTestCase("vllm", "triton", "fused_moe_kernel", "tools/rocm/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py", "triton-launch", True),
    TraceTestCase("vllm", "triton", "_fwd_kernel_stage1", "tools/rocm/vllm/vllm/v1/attention/ops/triton_decode_attention.py", "triton-launch", True),
    TraceTestCase("vllm", "triton", "_fwd_grouped_kernel_stage1", "tools/rocm/vllm/vllm/v1/attention/ops/triton_decode_attention.py", "triton-launch", True),
    TraceTestCase("vllm", "triton", "reshape_and_cache_kernel_flash", "tools/rocm/vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py", "triton-launch", True),
    TraceTestCase("vllm", "triton", "call_kernel", "tools/rocm/vllm/vllm/v1/attention/ops/common.py", "agent", False),
    TraceTestCase("vllm", "hip", "paged_attention_v1", "tools/rocm/vllm/vllm/_custom_ops.py", "vllm-custom-op", True),
    TraceTestCase("vllm", "hip", "paged_attention_v2", "tools/rocm/vllm/vllm/_custom_ops.py", "vllm-custom-op", True),
    TraceTestCase("vllm", "hip", "rms_norm", "tools/rocm/vllm/vllm/_custom_ops.py", "vllm-custom-op", True),
    TraceTestCase("vllm", "hip", "w8a8_gemm", "tools/rocm/vllm/vllm/_aiter_ops.py", "vllm-custom-op", True),
    TraceTestCase("vllm", "hip", "fused_moe", "tools/rocm/vllm/vllm/_aiter_ops.py", "vllm-custom-op", True),
    TraceTestCase("sglang", "triton", "fused_rmsnorm_kernel", "tools/rocm/sglang/python/sglang/srt/layers/elementwise.py", "triton-launch", True),
    TraceTestCase("sglang", "triton", "_fwd_kernel_stage1", "tools/rocm/sglang/python/sglang/srt/layers/attention/triton_ops/decode_attention.py", "triton-launch", True),
    TraceTestCase("sglang", "triton", "_fwd_kernel", "tools/rocm/sglang/python/sglang/srt/layers/attention/triton_ops/extend_attention.py", "triton-launch", True),
    TraceTestCase("sglang", "triton", "kernel", "tools/rocm/sglang/python/sglang/srt/layers/quantization/fp8_kernel.py", "agent", False),
    TraceTestCase("sglang", "triton", "fused_moe_kernel", "tools/rocm/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py", "triton-launch", True),
    TraceTestCase("sglang", "hip", "store_cache", "tools/rocm/sglang/python/sglang/jit_kernel/kvcache.py", "sglang-custom-op", True),
    TraceTestCase("sglang", "hip", "apply_rope_inplace", "tools/rocm/sglang/python/sglang/jit_kernel/rope.py", "sglang-custom-op", True),
    TraceTestCase("sglang", "hip", "_per_token_group_quant_8bit_custom_op", "tools/rocm/sglang/python/sglang/jit_kernel/per_token_group_quant_8bit.py", "sglang-custom-op", True),
    TraceTestCase("sglang", "hip", "_scaled_fp4_quant_custom_op", "tools/rocm/sglang/python/sglang/jit_kernel/nvfp4.py", "sglang-custom-op", True),
    TraceTestCase("sglang", "hip", "rocm_aiter_asm_moe_tkw1", "tools/rocm/sglang/python/sglang/srt/layers/moe/rocm_moe_utils.py", "sglang-custom-op", True),
]
