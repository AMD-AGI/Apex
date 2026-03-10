"""
configs.py — Inference serving configurations for prompt diversity.

Drawn from:
  - MLPerf Inference v4 / v5 datacenter scenarios
  - AMD InferenceMAX benchmark suite
  - Real-world deployment patterns (chatbot, RAG, code, long-context)

Each InferenceConfig represents one (workload, serving) combination that
affects which kernels are bottlenecked and by how much.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    config_id:    str    # short unique ID
    scenario:     str    # offline | server | interactive
    input_len:    int    # prompt tokens
    output_len:   int    # generated tokens
    concurrency:  int    # simultaneous requests (server) or batch size (offline)
    precision:    str    # bf16 | fp8 | fp4 | int8
    framework:    str    # sglang | vllm | both
    source:       str    # mlperf | inferencemax | custom
    notes:        str    = ""


CONFIGS: list[InferenceConfig] = [

    # ── MLPerf Inference v4/v5 scenarios ─────────────────────────────────────
    # Latency constraints: TTFT ≤ 2000ms, TPOT ≤ 200ms (server), offline = max throughput

    InferenceConfig(
        config_id="mlperf-server-short",
        scenario="server", input_len=512, output_len=128, concurrency=32,
        precision="fp8", framework="both",
        source="mlperf",
        notes="MLPerf server scenario; FP8 required on MI300+; TPOT ≤ 200ms",
    ),
    InferenceConfig(
        config_id="mlperf-offline-chat",
        scenario="offline", input_len=512, output_len=512, concurrency=128,
        precision="fp8", framework="both",
        source="mlperf",
        notes="MLPerf offline; Mixtral 8x7B reference; max throughput",
    ),
    InferenceConfig(
        config_id="mlperf-llama31-summ",
        scenario="server", input_len=2048, output_len=128, concurrency=64,
        precision="fp8", framework="both",
        source="mlperf",
        notes="MLPerf Llama-3.1-8B summarization; long-input, short-output",
    ),
    InferenceConfig(
        config_id="mlperf-deepseek-reason",
        scenario="server", input_len=2048, output_len=512, concurrency=32,
        precision="fp8", framework="both",
        source="mlperf",
        notes="MLPerf DeepSeek-R1 reasoning; TPOT ≤ 80ms; MLA + MoE stress",
    ),
    InferenceConfig(
        config_id="mlperf-llama2-70b",
        scenario="offline", input_len=1024, output_len=1024, concurrency=64,
        precision="fp8", framework="both",
        source="mlperf",
        notes="Classic MLPerf Llama 2 70B; balanced prefill/decode",
    ),

    # ── AMD InferenceMAX configurations ──────────────────────────────────────
    # InferenceMAX sweeps: precision × token_shape × concurrency

    InferenceConfig(
        config_id="imax-fp16-short-low",
        scenario="server", input_len=128, output_len=128, concurrency=8,
        precision="bf16", framework="both",
        source="inferencemax",
        notes="Baseline BF16; decode-dominant; low concurrency — latency-sensitive",
    ),
    InferenceConfig(
        config_id="imax-fp16-short-high",
        scenario="offline", input_len=128, output_len=128, concurrency=256,
        precision="bf16", framework="both",
        source="inferencemax",
        notes="BF16; high batch; tests max decode GEMM throughput",
    ),
    InferenceConfig(
        config_id="imax-fp8-short-med",
        scenario="server", input_len=128, output_len=512, concurrency=64,
        precision="fp8", framework="both",
        source="inferencemax",
        notes="FP8 chatbot; decode-heavy; quantized GEMM + attention",
    ),
    InferenceConfig(
        config_id="imax-fp8-long-in",
        scenario="server", input_len=8192, output_len=256, concurrency=16,
        precision="fp8", framework="both",
        source="inferencemax",
        notes="Long-context prefill (RAG); prefill-dominated; FlashAttn critical",
    ),
    InferenceConfig(
        config_id="imax-fp8-long-out",
        scenario="offline", input_len=256, output_len=8192, concurrency=8,
        precision="fp8", framework="both",
        source="inferencemax",
        notes="Long generation (code, story); decode-bottleneck; KV-cache pressure",
    ),
    InferenceConfig(
        config_id="imax-fp4-med",
        scenario="offline", input_len=512, output_len=512, concurrency=128,
        precision="fp4", framework="both",
        source="inferencemax",
        notes="FP4 (MXFP4/NF4) quantization; tests weight-loading & dequant kernels",
    ),
    InferenceConfig(
        config_id="imax-fp8-very-long",
        scenario="server", input_len=32768, output_len=512, concurrency=4,
        precision="fp8", framework="sglang",
        source="inferencemax",
        notes="Very long context (128K model window); chunked prefill; RadixAttn",
    ),

    # ── Realistic deployment patterns ─────────────────────────────────────────

    InferenceConfig(
        config_id="chatbot-interactive",
        scenario="interactive", input_len=256, output_len=256, concurrency=1,
        precision="bf16", framework="both",
        source="custom",
        notes="Single-user interactive chat; TTFT critical; decode-latency bound",
    ),
    InferenceConfig(
        config_id="rag-short-answer",
        scenario="server", input_len=3072, output_len=128, concurrency=32,
        precision="fp8", framework="both",
        source="custom",
        notes="Retrieval-augmented generation; large context, short answer",
    ),
    InferenceConfig(
        config_id="code-completion",
        scenario="server", input_len=4096, output_len=256, concurrency=16,
        precision="fp8", framework="both",
        source="custom",
        notes="Code completion (Copilot-style); medium context, speculative decode friendly",
    ),
    InferenceConfig(
        config_id="batch-summarization",
        scenario="offline", input_len=2048, output_len=256, concurrency=256,
        precision="fp8", framework="both",
        source="custom",
        notes="Document summarization pipeline; max throughput; prefill-heavy at scale",
    ),
    InferenceConfig(
        config_id="moe-high-concurrency",
        scenario="offline", input_len=512, output_len=512, concurrency=512,
        precision="fp8", framework="both",
        source="custom",
        notes="High-concurrency MoE (DeepSeek/Mixtral); expert routing stress test",
    ),
]


# ── Subsets ───────────────────────────────────────────────────────────────────

def by_precision(p: str)  -> list[InferenceConfig]: return [c for c in CONFIGS if c.precision == p]
def by_scenario(s: str)   -> list[InferenceConfig]: return [c for c in CONFIGS if c.scenario == s]
def by_source(src: str)   -> list[InferenceConfig]: return [c for c in CONFIGS if c.source == src]
def long_context()        -> list[InferenceConfig]: return [c for c in CONFIGS if c.input_len >= 2048]
def high_concurrency()    -> list[InferenceConfig]: return [c for c in CONFIGS if c.concurrency >= 64]


if __name__ == "__main__":
    print(f"Total configs: {len(CONFIGS)}")
    print(f"  MLPerf:       {len(by_source('mlperf'))}")
    print(f"  InferenceMAX: {len(by_source('inferencemax'))}")
    print(f"  Custom:       {len(by_source('custom'))}")
    print(f"  FP8:          {len(by_precision('fp8'))}")
    print(f"  BF16:         {len(by_precision('bf16'))}")
    print(f"  FP4:          {len(by_precision('fp4'))}")
    print(f"  Long context: {len(long_context())}")
    for c in CONFIGS:
        print(f"  {c.config_id:30s}  {c.input_len:6d}→{c.output_len:<6d}  "
              f"conc={c.concurrency:4d}  {c.precision:4s}  {c.scenario}")
