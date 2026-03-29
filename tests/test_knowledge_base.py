# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Tests for the persistent knowledge base."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from knowledge_base import (
    KnowledgeBase,
    OptimizationOutcome,
    _infer_strategy,
)


@pytest.fixture
def kb_path(tmp_path):
    return tmp_path / "kb.json"


class TestKnowledgeBase:
    def test_record_and_query(self, kb_path):
        kb = KnowledgeBase(path=kb_path)
        outcome = OptimizationOutcome(
            kernel_spec="rms_norm", kernel_type="triton",
            gpu_arch="gfx950", speedup=2.5, correct=True, score=75.0,
            strategy_used="autotuned_triton",
        )
        kb.record(outcome)
        results = kb.query(kernel_spec="rms_norm")
        assert len(results) == 1
        assert results[0].speedup == 2.5

    def test_query_filters(self, kb_path):
        kb = KnowledgeBase(path=kb_path)
        for spec, ktype, spd in [
            ("rms_norm", "triton", 2.0),
            ("silu_mul", "triton", 1.5),
            ("gemm", "hip", 3.0),
        ]:
            kb.record(OptimizationOutcome(
                kernel_spec=spec, kernel_type=ktype,
                speedup=spd, correct=True, score=50.0,
            ))

        assert len(kb.query(kernel_type="triton")) == 2
        assert len(kb.query(kernel_type="hip")) == 1
        assert len(kb.query(min_speedup=2.0)) == 2
        assert len(kb.query(top_k=1)) == 1

    def test_corrupt_json_recovery(self, kb_path):
        kb_path.write_text("NOT VALID JSON{{{")
        kb = KnowledgeBase(path=kb_path)
        results = kb._load()
        assert results == []

    def test_summarize_for_prompt(self, kb_path):
        kb = KnowledgeBase(path=kb_path)
        kb.record(OptimizationOutcome(
            kernel_spec="rms_norm", kernel_type="triton",
            speedup=2.0, correct=True, strategy_used="autotuned",
            key_insight="block size 256 works best",
        ))
        summary = kb.summarize_for_prompt("rms_norm", "triton")
        assert "rms_norm" in summary
        assert "2.00x" in summary
        assert "autotuned" in summary

    def test_record_from_opt_result(self, kb_path):
        kb = KnowledgeBase(path=kb_path)
        kb.record_from_opt_result(
            {"kernel_spec": "fused_moe", "correct": True, "speedup": 1.8, "score": 60},
            kernel_type="triton", gpu_arch="gfx950", agent_model="test",
            solution_code="@triton.autotune(configs=[], key=['M'])\ndef kernel(): pass",
        )
        results = kb.query(kernel_spec="fused_moe")
        assert len(results) == 1
        assert "autotuned_triton" in results[0].strategy_used


class TestInferStrategy:
    def test_autotune_pattern(self):
        name, desc, tags = _infer_strategy("@triton.autotune\ndef kernel(): pass")
        assert "autotuned_triton" in name
        assert "triton" in tags

    def test_mfma_pattern(self):
        name, desc, tags = _infer_strategy("result = tl.dot(a, b)")
        assert "mfma_matmul" in name

    def test_library_dispatch(self):
        name, desc, tags = _infer_strategy("out = torch.mm(a, b)")
        assert "library_dispatch" in name

    def test_fusion_pattern(self):
        name, desc, tags = _infer_strategy("def fused_kernel(): pass")
        assert "kernel_fusion" in name

    def test_no_pattern_fallback(self):
        name, desc, tags = _infer_strategy("x = 1 + 2")
        assert name == "custom_optimization"
