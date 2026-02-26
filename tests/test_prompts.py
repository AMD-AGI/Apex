"""
test_prompts.py — Unit tests for prompts/models.py, configs.py,
                  kernel_prompt.py, and model_prompt.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "prompts"))
from models        import MODELS, ModelConfig, moe_models, dense_models
from configs       import CONFIGS, InferenceConfig
from kernel_prompt import (
    all_prompts as kernel_all_prompts,
    applicable_kernels,
    build_kernel_prompt,
    KERNEL_SPECS,
    detect_gpu,
    DEFAULT_TARGET,
    make_task_id as kernel_make_task_id,
)
from model_prompt import (
    all_prompts as model_all_prompts,
    build_model_prompt,
    make_task_id as model_make_task_id,
)


# ── models.py ─────────────────────────────────────────────────────────────────

class TestModelRegistry:
    def test_non_empty(self):
        assert len(MODELS) >= 10

    def test_hf_ids_unique(self):
        ids = [m.hf_id for m in MODELS]
        assert len(ids) == len(set(ids)), "Duplicate hf_id found in MODELS"

    def test_kv_heads_lte_q_heads(self):
        for m in MODELS:
            assert m.num_kv_heads <= m.num_heads, \
                f"{m.hf_id}: num_kv_heads ({m.num_kv_heads}) > num_heads ({m.num_heads})"

    def test_params_positive(self):
        for m in MODELS:
            assert m.params_b > 0, f"{m.hf_id}: params_b must be > 0"

    def test_context_len_positive(self):
        for m in MODELS:
            assert m.context_len > 0

    def test_active_experts_lte_num_experts(self):
        for m in MODELS:
            assert m.active_experts <= m.num_experts, \
                f"{m.hf_id}: active_experts > num_experts"

    def test_moe_models_have_multiple_experts(self):
        for m in moe_models():
            assert m.num_experts > 1, f"{m.hf_id}: MoE model should have >1 experts"

    def test_dense_models_have_one_expert(self):
        for m in dense_models():
            assert m.num_experts == 1 and m.active_experts == 1, \
                f"{m.hf_id}: dense model should have num_experts=1"

    def test_frameworks_non_empty(self):
        for m in MODELS:
            assert len(m.frameworks) >= 1, f"{m.hf_id}: must list at least one framework"

    def test_attention_types_known(self):
        known = {"gqa", "mha", "mqa", "mla", "sliding_window", "moe_gqa"}
        for m in MODELS:
            assert m.attention in known, \
                f"{m.hf_id}: unknown attention type '{m.attention}'"

    def test_at_least_one_moe_model(self):
        assert len(moe_models()) >= 1

    def test_at_least_one_mla_model(self):
        mla = [m for m in MODELS if m.attention == "mla"]
        assert len(mla) >= 1

    def test_at_least_one_large_model(self):
        large = [m for m in MODELS if m.params_b >= 70]
        assert len(large) >= 2


# ── configs.py ────────────────────────────────────────────────────────────────

class TestConfigRegistry:
    def test_non_empty(self):
        assert len(CONFIGS) >= 10

    def test_config_ids_unique(self):
        ids = [c.config_id for c in CONFIGS]
        assert len(ids) == len(set(ids)), "Duplicate config_id found in CONFIGS"

    def test_token_lengths_positive(self):
        for c in CONFIGS:
            assert c.input_len  > 0
            assert c.output_len > 0

    def test_concurrency_positive(self):
        for c in CONFIGS:
            assert c.concurrency >= 1

    def test_precision_known(self):
        known = {"bf16", "fp16", "fp8", "fp4", "int8"}
        for c in CONFIGS:
            assert c.precision in known, \
                f"{c.config_id}: unknown precision '{c.precision}'"

    def test_source_known(self):
        known = {"mlperf", "inferencemax", "custom"}
        for c in CONFIGS:
            assert c.source in known, f"{c.config_id}: unknown source '{c.source}'"

    def test_has_mlperf_configs(self):
        assert any(c.source == "mlperf" for c in CONFIGS)

    def test_has_fp8_configs(self):
        assert any(c.precision == "fp8" for c in CONFIGS)

    def test_has_long_context_config(self):
        assert any(c.input_len >= 4096 for c in CONFIGS)

    def test_has_high_concurrency_config(self):
        assert any(c.concurrency >= 128 for c in CONFIGS)


# ── kernel_prompt.py ──────────────────────────────────────────────────────────

class TestKernelPromptGeneration:
    @pytest.fixture(scope="class")
    def prompts(self):
        return list(kernel_all_prompts(framework="sglang", gpu_arch=DEFAULT_TARGET))

    def test_generates_many_prompts(self, prompts):
        assert len(prompts) > 50

    def test_task_ids_unique(self, prompts):
        ids = [p["task_id"] for p in prompts]
        assert len(ids) == len(set(ids)), "Duplicate task_id in kernel prompts"

    def test_prompt_contains_model_id(self, prompts):
        for p in prompts:
            assert p["model_id"] in p["prompt"], \
                f"{p['task_id']}: model_id missing from prompt"

    def test_prompt_contains_gpu_arch(self, prompts):
        for p in prompts:
            assert p["gpu_arch"] in p["prompt"], \
                f"{p['task_id']}: gpu_arch missing from prompt"

    def test_prompt_contains_output_instruction(self, prompts):
        for p in prompts:
            assert "output/" in p["prompt"], \
                f"{p['task_id']}: output/ instruction missing from prompt"

    def test_prompt_contains_task_id_in_output_path(self, prompts):
        for p in prompts:
            assert f"output/{p['task_id']}" in p["prompt"], \
                f"{p['task_id']}: task_id missing from output path in prompt"

    def test_all_fields_present(self, prompts):
        required = {"task_id", "model_id", "kernel_type", "framework", "gpu_arch", "prompt"}
        for p in prompts:
            assert required <= set(p), f"Missing fields in {p.get('task_id')}"

    def test_applicable_kernels_non_empty(self):
        for m in MODELS:
            kernels = applicable_kernels(m)
            assert len(kernels) > 0, f"{m.hf_id}: no applicable kernels"

    def test_moe_models_get_fused_moe_kernel(self):
        for m in moe_models():
            types = {k.kernel_type for k in applicable_kernels(m)}
            assert "fused_moe" in types, \
                f"{m.hf_id}: MoE model missing fused_moe kernel"

    def test_mla_models_get_mla_kernel(self):
        for m in MODELS:
            if m.attention == "mla":
                types = {k.kernel_type for k in applicable_kernels(m)}
                assert "mla_attn" in types, \
                    f"{m.hf_id}: MLA model missing mla_attn kernel"

    def test_detect_gpu_returns_valid_arch(self):
        arch = detect_gpu()
        assert arch.startswith("gfx"), f"detect_gpu returned unexpected: {arch}"

    def test_detect_gpu_fallback_is_default(self, monkeypatch):
        import subprocess
        monkeypatch.setattr(subprocess, "check_output",
                            lambda *a, **kw: (_ for _ in ()).throw(FileNotFoundError()))
        arch = detect_gpu()
        assert arch == DEFAULT_TARGET


# ── model_prompt.py ───────────────────────────────────────────────────────────

class TestModelPromptGeneration:
    @pytest.fixture(scope="class")
    def prompts(self):
        return list(model_all_prompts(framework="sglang", gpu_arch=DEFAULT_TARGET))

    def test_generates_many_prompts(self, prompts):
        assert len(prompts) > 100

    def test_task_ids_unique(self, prompts):
        ids = [p["task_id"] for p in prompts]
        assert len(ids) == len(set(ids)), "Duplicate task_id in model prompts"

    def test_prompt_contains_model_id(self, prompts):
        for p in prompts[:20]:   # spot-check first 20
            assert p["model_id"] in p["prompt"]

    def test_prompt_contains_output_instruction(self, prompts):
        for p in prompts[:20]:
            assert "output/" in p["prompt"]

    def test_prompt_contains_benchmark_yaml(self, prompts):
        for p in prompts[:20]:
            assert "benchmark.yaml" in p["prompt"]

    def test_all_fields_present(self, prompts):
        required = {"task_id", "model_id", "config_id", "framework",
                    "gpu_arch", "precision", "input_len", "output_len", "prompt"}
        for p in prompts[:20]:
            assert required <= set(p), f"Missing fields in {p.get('task_id')}"

    def test_fp8_prompts_mention_fp8(self, prompts):
        fp8 = [p for p in prompts if p["precision"] == "fp8"]
        assert len(fp8) > 0
        for p in fp8[:5]:
            assert "fp8" in p["prompt"].lower() or "FP8" in p["prompt"]

    def test_long_context_prompts_mention_chunked_prefill(self, prompts):
        long = [p for p in prompts if p["input_len"] >= 4096]
        assert len(long) > 0
        for p in long[:3]:
            assert "chunked" in p["prompt"].lower() or "long" in p["prompt"].lower()
