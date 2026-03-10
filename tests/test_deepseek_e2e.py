#!/usr/bin/env python3
"""
test_deepseek_e2e.py — End-to-end test of model grader with DeepSeek R1 FP8.

This script:
  1. Checks/installs minimal dependencies (torch-rocm, sglang, magpie)
  2. Downloads the model via huggingface-cli (only the files sglang needs)
  3. Creates a minimal task in output/ with a stock (unoptimized) kernel
  4. Runs the model grader and reports results

Usage:
    # Quick test with 8B distill (single GPU, ~16 GB):
    python3 tests/test_deepseek_e2e.py

    # Full DeepSeek R1 671B FP8 (needs 8 GPUs, TP=8):
    python3 tests/test_deepseek_e2e.py --model deepseek-ai/DeepSeek-R1 --tp 8

    # Install dependencies only:
    python3 tests/test_deepseek_e2e.py --setup-only

    # Dry run (mock Magpie, test scoring pipeline):
    python3 tests/test_deepseek_e2e.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"

sys.path.insert(0, str(REPO_ROOT / "graders"))
sys.path.insert(0, str(REPO_ROOT / "prompts"))

# ── Default configuration ────────────────────────────────────────────────────

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_TP    = 1
DEFAULT_PREC  = "fp8"

FULL_R1_MODEL = "deepseek-ai/DeepSeek-R1"
FULL_R1_TP    = 8


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DEPENDENCY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def run(cmd: str, check: bool = True, timeout: int = 600, **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, check=check, timeout=timeout, **kwargs)


def check_rocm() -> bool:
    try:
        r = run("rocm-smi --showid", check=False, capture_output=True, text=True)
        if r.returncode == 0 and "MI355X" in r.stdout:
            gpu_count = r.stdout.count("Device Name")
            print(f"  ROCm OK: {gpu_count}x MI355X detected")
            return True
        print(f"  ROCm: GPUs detected but not MI355X")
        return r.returncode == 0
    except Exception as e:
        print(f"  ROCm check failed: {e}")
        return False


def check_torch_rocm() -> bool:
    try:
        r = run(
            'python3 -c "import torch; assert torch.cuda.is_available(); print(f\'torch {torch.__version__} hip={torch.version.hip} gpus={torch.cuda.device_count()}\')"',
            check=False, capture_output=True, text=True,
        )
        if r.returncode == 0:
            print(f"  {r.stdout.strip()}")
            return True
        print(f"  torch not available: {r.stderr.strip()[:120]}")
        return False
    except Exception:
        return False


def install_torch_rocm() -> bool:
    print("  Installing PyTorch for ROCm 7.0 ...")
    r = run(
        "pip3 install --pre torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/rocm7.0",
        check=False, timeout=600,
    )
    return r.returncode == 0


def check_sglang() -> bool:
    r = run('python3 -c "import sglang; print(f\'sglang {sglang.__version__}\')"',
            check=False, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  {r.stdout.strip()}")
        return True
    return False


def install_sglang() -> bool:
    print("  Installing SGLang ...")
    r = run('pip3 install "sglang[all]"', check=False, timeout=600)
    return r.returncode == 0


def check_magpie() -> bool:
    """Check if Magpie CLI or local copy is available."""
    if shutil.which("magpie"):
        print("  magpie CLI found in PATH")
        return True
    local = REPO_ROOT / "tools" / "magpie" / "main.py"
    if local.exists():
        print(f"  magpie found at {local}")
        return True
    return False


def install_magpie() -> bool:
    magpie_dir = REPO_ROOT / "tools" / "magpie"
    if not (magpie_dir / ".git").exists():
        print("  Cloning AMD-AGI/Magpie ...")
        r = run(f"git clone --depth=1 https://github.com/AMD-AGI/Magpie.git {magpie_dir}",
                check=False, timeout=300)
        if r.returncode != 0:
            return False
    print("  Installing Magpie ...")
    r = run(f"pip3 install -e {magpie_dir}", check=False, timeout=120)
    return r.returncode == 0


def ensure_deps(skip_install: bool = False) -> dict[str, bool]:
    """Check and optionally install all required dependencies."""
    status = {}

    print("\n--- Checking dependencies ---")

    # ROCm
    status["rocm"] = check_rocm()

    # PyTorch ROCm
    status["torch"] = check_torch_rocm()
    if not status["torch"] and not skip_install:
        status["torch"] = install_torch_rocm() and check_torch_rocm()

    # SGLang
    status["sglang"] = check_sglang()
    if not status["sglang"] and not skip_install:
        status["sglang"] = install_sglang() and check_sglang()

    # Magpie
    status["magpie"] = check_magpie()
    if not status["magpie"] and not skip_install:
        status["magpie"] = install_magpie() and check_magpie()

    print("\n  Dependency status:")
    for name, ok in status.items():
        print(f"    {'OK' if ok else 'MISSING':>7s}  {name}")

    return status


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MODEL DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════

def download_model(model_id: str) -> str:
    """Download model via huggingface-cli. Returns the cache path."""
    print(f"\n--- Downloading model: {model_id} ---")
    print("  (This checks the HF cache first and only downloads missing files.)")

    # Use huggingface-cli download which handles caching
    r = run(
        f'python3 -c "'
        f"from huggingface_hub import snapshot_download; "
        f"p = snapshot_download('{model_id}'); "
        f'print(p)"',
        check=False, capture_output=True, text=True, timeout=7200,
    )
    if r.returncode == 0:
        cache_path = r.stdout.strip().split("\n")[-1]
        print(f"  Model cached at: {cache_path}")
        return cache_path
    else:
        print(f"  Download failed: {r.stderr.strip()[:200]}")
        # Fall back to huggingface-cli
        r2 = run(f"huggingface-cli download {model_id}", check=False, timeout=7200)
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TASK SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def make_task_id(model_id: str, precision: str) -> str:
    slug = model_id.split("/")[-1].replace(".", "-").lower()
    return f"{slug}__{precision}__e2e-test"


SOLUTION_PY = textwrap.dedent("""\
    # solution.py — Stock (unoptimized) RMSNorm kernel.
    # This is a baseline test: no actual optimization applied.
    # The model grader should report ~1.0x speedup.
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _rms_norm_fwd_kernel(
        X, W, Y,
        stride_x, N,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        rms = tl.sqrt(tl.sum(x * x) / N + eps)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x / rms * w
        tl.store(Y + row * stride_x + cols, y.to(tl.float16), mask=mask)

    def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        M, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _rms_norm_fwd_kernel[(M,)](x, weight, y, x.stride(0), N, eps, BLOCK_N)
        return y
""")


def setup_task(
    model_id: str,
    precision: str,
    tp: int,
    framework: str = "sglang",
) -> Path:
    """Create the output/<task_id>/ directory with solution + configs."""
    task_id  = make_task_id(model_id, precision)
    task_dir = OUTPUT_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Write the solution kernel (stock RMSNorm — intentionally unoptimized)
    (task_dir / "solution.py").write_text(SOLUTION_PY)

    # Write config.yaml for kernel-level grading (Magpie compare)
    config_yaml = textwrap.dedent(f"""\
        gpu:
          device: 0
          arch: gfx950
        baseline:
          path: ./solution.py
        optimized:
          path: ./solution.py
        correctness:
          command: "python3 -c \\"import solution; import torch; x=torch.randn(4,128,device='cuda',dtype=torch.float16); w=torch.ones(128,device='cuda',dtype=torch.float16); y=solution.rms_norm(x,w); assert y.shape==x.shape; print('PASS')\\""
        performance:
          command: "python3 -c \\"import solution,torch,time; x=torch.randn(32,4096,device='cuda',dtype=torch.float16); w=torch.ones(4096,device='cuda',dtype=torch.float16); [solution.rms_norm(x,w) for _ in range(10)]; torch.cuda.synchronize(); t0=time.time(); [solution.rms_norm(x,w) for _ in range(100)]; torch.cuda.synchronize(); print(f'{{\\\\\\"baseline_ms\\\\\\":1.0,\\\\\\"optimized_ms\\\\\\":{{}}}}'.format((time.time()-t0)*10))\\""
          iterations: 100
    """)
    (task_dir / "config.yaml").write_text(config_yaml)

    # Write benchmark.yaml for model-level grading (Magpie benchmark)
    num_prompts = 50 if tp <= 2 else 200
    benchmark_yaml = textwrap.dedent(f"""\
        framework: {framework}
        model: {model_id}
        gpu:
          device: 0
          arch: gfx950
        baseline:
          framework_config: {{}}
        optimized:
          patch: ./solution.py
        workload:
          input_len: 512
          output_len: 128
          num_prompts: {num_prompts}
          concurrency: {min(tp * 4, 32)}
        precision: {precision}
        tensor_parallel: {tp}
    """)
    (task_dir / "benchmark.yaml").write_text(benchmark_yaml)

    print(f"\n--- Task created: {task_dir} ---")
    print(f"  solution.py:    stock RMSNorm Triton kernel")
    print(f"  config.yaml:    kernel-level Magpie compare config")
    print(f"  benchmark.yaml: model-level Magpie benchmark config")
    print(f"  model:          {model_id}")
    print(f"  precision:      {precision}")
    print(f"  TP:             {tp}")

    return task_dir


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GRADING
# ═══════════════════════════════════════════════════════════════════════════════

def grade_with_model_grader(task_dir: Path, model_filter: str | None = None) -> dict:
    """Run the real model_grader.grade_all() and return the summary."""
    from model_grader import grade_all, summarise
    results = grade_all(task_dir.parent, model_filter=task_dir.name)
    return summarise(results), results


def grade_direct_sglang(
    model_id: str,
    precision: str,
    tp: int,
    input_len: int = 512,
    output_len: int = 128,
    num_prompts: int = 50,
) -> dict:
    """
    Bypass Magpie: directly benchmark with sglang and compute a score.
    This is the fallback when Magpie is not available.
    """
    from score import ModelResult

    print("\n--- Direct SGLang benchmark (Magpie bypass) ---")

    bench_script = textwrap.dedent(f"""\
        import subprocess, json, sys, time, signal, os

        MODEL    = "{model_id}"
        TP       = {tp}
        PREC     = "{precision}"
        ISL      = {input_len}
        OSL      = {output_len}
        NUM_REQ  = {num_prompts}

        # Start sglang server
        print("Starting sglang server ...", flush=True)
        quant_flag = "--quantization fp8" if PREC == "fp8" else ""
        server_cmd = (
            f"python3 -m sglang.launch_server "
            f"--model-path {{MODEL}} --tp {{TP}} {{quant_flag}} "
            f"--host 127.0.0.1 --port 30000 "
            f"--disable-custom-all-reduce"
        )
        server = subprocess.Popen(server_cmd, shell=True, preexec_fn=os.setsid)

        # Wait for server to be ready
        import urllib.request
        for i in range(120):
            try:
                urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=2)
                print(f"  Server ready after {{i+1}}s", flush=True)
                break
            except Exception:
                time.sleep(1)
        else:
            print("ERROR: server did not start in 120s")
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            sys.exit(1)

        # Run benchmark
        print(f"Running benchmark: {{NUM_REQ}} requests, ISL={{ISL}}, OSL={{OSL}} ...", flush=True)
        bench_cmd = (
            f"python3 -m sglang.bench_serving "
            f"--backend sglang --host 127.0.0.1 --port 30000 "
            f"--dataset-name random --random-input-len {{ISL}} --random-output-len {{OSL}} "
            f"--num-prompts {{NUM_REQ}} --output-file /tmp/sglang_bench_result.json"
        )
        r = subprocess.run(bench_cmd, shell=True, capture_output=True, text=True, timeout=600)
        print(r.stdout[-500:] if r.stdout else "", flush=True)
        if r.stderr:
            print(r.stderr[-200:], file=sys.stderr, flush=True)

        # Cleanup
        os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        server.wait(timeout=30)

        # Parse result
        try:
            with open("/tmp/sglang_bench_result.json") as f:
                data = json.load(f)
            tps = data.get("output_throughput", data.get("output_token_throughput", 0))
            print(json.dumps({{"throughput_tps": tps, "model": MODEL, "precision": PREC}}))
        except Exception as e:
            print(json.dumps({{"error": str(e)}}))
    """)

    bench_file = REPO_ROOT / "output" / "_bench_direct.py"
    bench_file.write_text(bench_script)

    try:
        r = subprocess.run(
            [sys.executable, str(bench_file)],
            capture_output=True, text=True, timeout=900,
        )
        print(r.stdout[-1000:] if r.stdout else "")
        if r.stderr:
            print(r.stderr[-500:], file=sys.stderr)

        # Parse the last JSON line
        for line in reversed(r.stdout.strip().split("\n")):
            try:
                data = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        else:
            data = {"error": "no JSON output from benchmark"}

        if "error" in data:
            return ModelResult(
                model_id=model_id,
                error=data["error"],
            )

        tps = data.get("throughput_tps", 0)
        # For a baseline-vs-baseline test, ratio = 1.0 (no optimization)
        return ModelResult(
            model_id=model_id,
            kernel_score=120.0,  # compiled + correct, no speedup
            e2e_throughput_ratio=1.0,
            raw=data,
        )
    except subprocess.TimeoutExpired:
        return ModelResult(model_id=model_id, error="benchmark timed out")
    except Exception as e:
        return ModelResult(model_id=model_id, error=str(e))
    finally:
        bench_file.unlink(missing_ok=True)


def grade_dry_run(model_id: str, precision: str) -> dict:
    """Mock grading — no GPU needed. Tests the scoring pipeline only."""
    from score import ModelResult

    print("\n--- Dry run: mock Magpie results ---")

    # Simulate: compiled=True, correct=True, speedup=1.0 (baseline=baseline)
    result = ModelResult(
        model_id=model_id,
        kernel_score=120.0,   # PTS_COMPILED(20) + PTS_CORRECT(100)
        e2e_throughput_ratio=1.0,
        raw={
            "note": "dry run — mock results",
            "model": model_id,
            "precision": precision,
            "simulated_tps": 1000.0,
        },
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def print_result(result, model_id: str, elapsed: float):
    from score import PTS_COMPILED, PTS_CORRECT

    d = result.to_dict() if hasattr(result, "to_dict") else result

    print(f"\n{'=' * 60}")
    print(f"  MODEL GRADER RESULT")
    print(f"{'=' * 60}")
    print(f"  Model:            {model_id}")
    print(f"  Kernel score:     {d.get('kernel_score', 0):.1f}")
    print(f"  E2E throughput:   {d.get('e2e_throughput_ratio', 0):.3f}x")
    print(f"  TOTAL SCORE:      {d.get('score', 0):.1f}")
    if d.get("error"):
        print(f"  Error:            {d['error']}")
    print(f"  Elapsed:          {elapsed:.1f}s")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end model grader test with DeepSeek R1 FP8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Quick test (8B distill, single GPU):
              python3 tests/test_deepseek_e2e.py

              # Full DeepSeek R1 671B FP8:
              python3 tests/test_deepseek_e2e.py --model deepseek-ai/DeepSeek-R1 --tp 8

              # Just install deps:
              python3 tests/test_deepseek_e2e.py --setup-only

              # Dry run (no GPU, tests scoring only):
              python3 tests/test_deepseek_e2e.py --dry-run
        """),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--tp", type=int, default=DEFAULT_TP,
                        help=f"Tensor parallelism degree (default: {DEFAULT_TP})")
    parser.add_argument("--precision", default=DEFAULT_PREC, choices=["fp8", "bf16", "fp4"],
                        help="Model precision (default: fp8)")
    parser.add_argument("--framework", default="sglang", choices=["sglang", "vllm"],
                        help="Inference framework (default: sglang)")
    parser.add_argument("--setup-only", action="store_true",
                        help="Only install dependencies, don't run the test")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mock Magpie results, test scoring pipeline only (no GPU)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip model download (assume already cached)")
    parser.add_argument("--direct", action="store_true",
                        help="Bypass Magpie, benchmark directly with sglang")
    parser.add_argument("--skip-install", action="store_true",
                        help="Skip dependency installation")
    args = parser.parse_args()

    print("")
    print("=" * 60)
    print("  DeepSeek R1 FP8 — Model Grader E2E Test")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Precision:  {args.precision}")
    print(f"  TP:         {args.tp}")
    print(f"  Framework:  {args.framework}")
    print(f"  Mode:       {'dry-run' if args.dry_run else 'direct' if args.direct else 'magpie'}")
    print("")

    t0 = time.time()

    # ── Step 1: Dependencies ──────────────────────────────────────────────────
    if args.dry_run:
        print("--- Skipping dependency checks (dry-run) ---")
        dep_status = {"rocm": True, "torch": True, "sglang": True, "magpie": True}
    else:
        dep_status = ensure_deps(skip_install=args.skip_install)

    if args.setup_only:
        print(f"\n  Setup complete in {time.time()-t0:.1f}s")
        missing = [k for k, v in dep_status.items() if not v]
        if missing:
            print(f"  Still missing: {', '.join(missing)}")
            sys.exit(1)
        sys.exit(0)

    # ── Step 2: Download model ────────────────────────────────────────────────
    if not args.dry_run and not args.skip_download:
        download_model(args.model)

    # ── Step 3: Create task ───────────────────────────────────────────────────
    task_dir = setup_task(
        model_id=args.model,
        precision=args.precision,
        tp=args.tp,
        framework=args.framework,
    )

    # ── Step 4: Grade ─────────────────────────────────────────────────────────
    if args.dry_run:
        result = grade_dry_run(args.model, args.precision)
    elif args.direct or not dep_status.get("magpie"):
        if not dep_status.get("magpie"):
            print("\n  Magpie not available — falling back to direct sglang benchmark")
        result = grade_direct_sglang(
            model_id=args.model,
            precision=args.precision,
            tp=args.tp,
        )
    else:
        # Use the real model_grader with Magpie
        print("\n--- Running model_grader.grade_all() ---")
        try:
            summary, results = grade_with_model_grader(task_dir)
            if results:
                result = results[0]
            else:
                from score import ModelResult
                result = ModelResult(model_id=args.model, error="no results from model_grader")
        except Exception as e:
            from score import ModelResult
            result = ModelResult(model_id=args.model, error=str(e))

    # ── Step 5: Report ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print_result(result, args.model, elapsed)

    # Write JSON result
    result_file = task_dir / "grader_result.json"
    result_file.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"  Result saved: {result_file}")

    sys.exit(0 if not result.error else 1)


if __name__ == "__main__":
    main()
