#!/usr/bin/env python3
"""
test_and_bench.py — Correctness test + benchmark for fused MoE kernel.

Usage:
    python test_and_bench.py                    # test baseline only
    python test_and_bench.py --solution PATH    # test solution, compare to baseline
    python test_and_bench.py --bench-only       # skip correctness, benchmark only

Output: JSON to stdout with {"compiled", "correct", "baseline_ms", "optimized_ms", "speedup"}
"""

import argparse
import importlib.util
import json
import sys
import time
import traceback

import torch

# ── Load module from path ────────────────────────────────────────────────────

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Create test data (Mixtral-8x7B-like dimensions) ──────────────────────────

def make_test_data(num_tokens=256, hidden_dim=4096, ffn_dim=14336,
                   num_experts=8, top_k=2, dtype=torch.float16, device="cuda"):
    torch.manual_seed(42)
    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
    expert_weights = torch.randn(num_experts, ffn_dim, hidden_dim, dtype=dtype, device=device) * 0.02

    scores = torch.randn(num_tokens, num_experts, device=device)
    topk_weights, topk_ids = torch.topk(torch.softmax(scores, dim=-1), top_k, dim=-1)
    topk_ids = topk_ids.to(torch.int32)

    return hidden_states, expert_weights, topk_ids, topk_weights


# ── Correctness test ─────────────────────────────────────────────────────────

def test_correctness(mod, label="kernel"):
    """Test a module's fused_moe_forward against the PyTorch reference."""
    from baseline_fused_moe import moe_reference

    for num_tokens in [16, 64, 256]:
        hidden_states, expert_weights, topk_ids, topk_weights = make_test_data(
            num_tokens=num_tokens, hidden_dim=512, ffn_dim=1024,
            num_experts=8, top_k=2,
        )
        ref = moe_reference(hidden_states, expert_weights, topk_ids, topk_weights)
        got = mod.fused_moe_forward(hidden_states, expert_weights, topk_ids, topk_weights)

        if ref.shape != got.shape:
            return False, f"shape mismatch: ref={ref.shape} vs got={got.shape} (tokens={num_tokens})"

        diff = (ref.float() - got.float()).abs().max().item()
        rel_err = diff / (ref.float().abs().max().item() + 1e-8)
        if rel_err > 0.02:  # 2% relative tolerance for fp16
            return False, f"mismatch: max_abs_diff={diff:.6f}, rel_err={rel_err:.4f} (tokens={num_tokens})"

    return True, "all tests passed"


# ── Benchmark ────────────────────────────────────────────────────────────────

def benchmark(mod, num_tokens=512, hidden_dim=4096, ffn_dim=14336,
              num_experts=8, top_k=2, warmup=10, iters=50):
    """Benchmark fused_moe_forward, return median time in ms."""
    hidden_states, expert_weights, topk_ids, topk_weights = make_test_data(
        num_tokens=num_tokens, hidden_dim=hidden_dim, ffn_dim=ffn_dim,
        num_experts=num_experts, top_k=top_k,
    )

    # Warmup
    for _ in range(warmup):
        mod.fused_moe_forward(hidden_states, expert_weights, topk_ids, topk_weights)
    torch.cuda.synchronize()

    # Timed iterations
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mod.fused_moe_forward(hidden_states, expert_weights, topk_ids, topk_weights)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]  # median


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default=None, help="Path to solution module")
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--num-tokens", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--ffn-dim", type=int, default=14336)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    result = {
        "compiled": False,
        "correct": False,
        "baseline_ms": 0.0,
        "optimized_ms": 0.0,
        "speedup": 0.0,
        "error": None,
    }

    # Load baseline
    baseline = load_module("baseline_fused_moe",
                           os.path.join(os.path.dirname(__file__), "baseline_fused_moe.py"))

    # Load solution (or use baseline as both)
    if args.solution:
        try:
            solution = load_module("solution", args.solution)
            result["compiled"] = True
        except Exception as e:
            result["error"] = f"import failed: {e}"
            print(json.dumps(result))
            return
    else:
        solution = baseline
        result["compiled"] = True

    # Correctness
    if not args.bench_only:
        try:
            ok, msg = test_correctness(solution)
            result["correct"] = ok
            if not ok:
                result["error"] = msg
        except Exception as e:
            result["correct"] = False
            result["error"] = f"correctness exception: {traceback.format_exc()}"

    # Benchmark
    try:
        bench_kwargs = dict(
            num_tokens=args.num_tokens, hidden_dim=args.hidden_dim,
            ffn_dim=args.ffn_dim, num_experts=args.num_experts, top_k=args.top_k,
        )
        result["baseline_ms"] = round(benchmark(baseline, **bench_kwargs), 4)
        result["optimized_ms"] = round(benchmark(solution, **bench_kwargs), 4)
        if result["optimized_ms"] > 0:
            result["speedup"] = round(result["baseline_ms"] / result["optimized_ms"], 4)
    except Exception as e:
        result["error"] = (result.get("error") or "") + f"; bench failed: {e}"

    print(json.dumps(result))


if __name__ == "__main__":
    main()
