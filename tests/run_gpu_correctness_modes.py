#!/usr/bin/env python3
"""
run_gpu_correctness_modes.py — GPU validation for all three correctness modes.

Exercises the full grading pipeline on real MI355X GPUs:
  Mode 1 (pytorch):      Magpie compare for correctness + perf
  Mode 2 (library_test): pytest for correctness, then Magpie for perf (new)
  Mode 3 (accordo):      accordo validate for correctness, then Magpie for perf (new)

Usage:
    source $HOME/Kernel/.venv/bin/activate
    export MAGPIE_ROOT=$HOME/code_combine/Magpie
    python3 tests/run_gpu_correctness_modes.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "graders"))

from kernel_grader import grade_task, summarise, _try_magpie_perf_measurement, _detect_kernel_type
from score import KernelResult, PTS_COMPILED, PTS_CORRECT, parse_compare_result, run_magpie_compare

BASELINE_KERNEL = textwrap.dedent("""\
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
        return out

    if __name__ == "__main__":
        import sys
        import time
        x = torch.randn(1024 * 1024, device="cuda")
        y = torch.randn(1024 * 1024, device="cuda")
        for _ in range(10):
            vector_add(x, y)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            vector_add(x, y)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000 / 100
        print(f"BENCHMARK_MS: {elapsed_ms:.4f}")
""")

OPTIMIZED_KERNEL = textwrap.dedent("""\
    import torch
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 256}),
            triton.Config({'BLOCK_SIZE': 512}),
            triton.Config({'BLOCK_SIZE': 1024}),
            triton.Config({'BLOCK_SIZE': 2048}),
        ],
        key=['n_elements'],
    )
    @triton.jit
    def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
        vector_add_kernel[grid](x, y, out, n)
        return out

    if __name__ == "__main__":
        import sys
        import time
        x = torch.randn(1024 * 1024, device="cuda")
        y = torch.randn(1024 * 1024, device="cuda")
        for _ in range(10):
            vector_add(x, y)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            vector_add(x, y)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000 / 100
        print(f"BENCHMARK_MS: {elapsed_ms:.4f}")
""")

PYTEST_TEST = textwrap.dedent("""\
    import torch
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from solution import vector_add

    def test_correctness_small():
        x = torch.randn(128, device="cuda")
        y = torch.randn(128, device="cuda")
        out = vector_add(x, y)
        ref = x + y
        assert torch.allclose(out, ref, atol=1e-5)

    def test_correctness_large():
        x = torch.randn(1024 * 1024, device="cuda")
        y = torch.randn(1024 * 1024, device="cuda")
        out = vector_add(x, y)
        ref = x + y
        assert torch.allclose(out, ref, atol=1e-5)

    def test_dtypes():
        for dtype in [torch.float32, torch.float16]:
            x = torch.randn(512, device="cuda", dtype=dtype)
            y = torch.randn(512, device="cuda", dtype=dtype)
            out = vector_add(x, y)
            ref = x + y
            assert torch.allclose(out, ref, atol=1e-3)
""")


def banner(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def setup_task(tmp: Path, mode: str) -> Path:
    """Create a task directory with appropriate config for the given mode."""
    task_dir = tmp / f"task_{mode}"
    task_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "baseline.py").write_text(BASELINE_KERNEL)
    (task_dir / "solution.py").write_text(OPTIMIZED_KERNEL)

    if mode == "pytorch":
        (task_dir / "config.yaml").write_text(textwrap.dedent(f"""\
            gpu:
              device: 0
              arch: gfx950
            baseline:
              path: ./baseline.py
            optimized:
              path: ./solution.py
            correctness:
              mode: pytorch
              command: python3 solution.py
            performance:
              command: python3 solution.py --benchmark
              mode: magpie_builtin
            _pipeline_metadata:
              tamper_protected: true
        """))

    elif mode == "library_test":
        (task_dir / "test_solution.py").write_text(PYTEST_TEST)
        (task_dir / "config.yaml").write_text(textwrap.dedent(f"""\
            gpu:
              device: 0
              arch: gfx950
            baseline:
              path: ./baseline.py
            optimized:
              path: ./solution.py
            correctness:
              mode: library_test
              unit_test_command: python -m pytest test_solution.py -v
              working_directory: {task_dir}
            performance:
              command: python3 solution.py --benchmark
            _pipeline_metadata:
              tamper_protected: true
        """))

    return task_dir


def report_result(result: KernelResult, mode: str) -> bool:
    """Print result details and return True if all checks pass."""
    ok = True
    print(f"  Task ID:    {result.task_id}")
    print(f"  Compiled:   {result.compiled}")
    print(f"  Correct:    {result.correct}")
    print(f"  Speedup:    {result.speedup:.4f}x")
    print(f"  Score:      {result.score:.1f}")
    if result.error:
        print(f"  Error:      {result.error}")

    if result.raw:
        perf_source = result.raw.get("_magpie_perf_source", "script_benchmark")
        print(f"  Perf src:   {perf_source}")
        if "_magpie_speedup" in result.raw:
            print(f"  Magpie spd: {result.raw['_magpie_speedup']:.4f}x")
        if "_no_perf_data" in result.raw:
            print(f"  (no perf data — used 1.0x fallback)")

    if not result.compiled:
        print(f"  FAIL: kernel did not compile")
        ok = False
    if not result.correct:
        print(f"  FAIL: kernel not correct")
        ok = False
    if result.speedup <= 0:
        print(f"  FAIL: speedup is zero or negative")
        ok = False
    elif result.speedup < 1.0:
        print(f"  [OK] speedup {result.speedup:.4f}x (< 1.0 is expected for trivial kernels)")

    if mode == "library_test" and result.raw:
        if result.raw.get("_magpie_perf_source") == "magpie_compare":
            print(f"  [OK] Magpie perf measurement used for library_test mode")
        else:
            print(f"  [INFO] Magpie perf not used (fell back to script benchmark)")

    return ok


def test_try_magpie_perf_direct(tmp: Path):
    """Directly test _try_magpie_perf_measurement with real GPU files.

    Magpie may skip perf for simple kernels — the helper should either
    inject speedup or fail gracefully (no crash, no corrupt raw dict).
    Both outcomes are valid; the grader falls back to script benchmarking.
    """
    banner("Direct _try_magpie_perf_measurement test")

    baseline = tmp / "direct_test" / "baseline.py"
    solution = tmp / "direct_test" / "solution.py"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(BASELINE_KERNEL)
    solution.write_text(OPTIMIZED_KERNEL)

    raw: dict = {"compiled": True, "correct": True}
    _try_magpie_perf_measurement(
        raw=raw,
        baseline_path=str(baseline),
        optimized_path=str(solution),
        task_dir=baseline.parent,
        compare_timeout=300,
        solution=solution,
        testcase_cmd=f"python3 {solution}",
    )

    if "_magpie_speedup" in raw:
        print(f"  Magpie speedup: {raw['_magpie_speedup']:.4f}x")
        print(f"  Perf source:    {raw.get('_magpie_perf_source')}")
        print(f"  [OK] _try_magpie_perf_measurement injected speedup")
    else:
        print(f"  [OK] Magpie skipped perf (expected for simple kernels)")
        print(f"       Grader will fall back to script-level BENCHMARK_MS")

    # Success = no crash, raw dict not corrupted
    assert raw["compiled"] is True, "raw['compiled'] was corrupted"
    assert raw["correct"] is True, "raw['correct'] was corrupted"
    print(f"  [OK] raw dict integrity preserved")
    return True


def test_run_magpie_compare_direct(tmp: Path):
    """Directly test run_magpie_compare with real kernels."""
    banner("Direct run_magpie_compare test")

    baseline = tmp / "magpie_direct" / "baseline.py"
    solution = tmp / "magpie_direct" / "solution.py"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(BASELINE_KERNEL)
    solution.write_text(OPTIMIZED_KERNEL)

    result = run_magpie_compare(
        baseline_path=str(baseline),
        optimized_path=str(solution),
        kernel_type="triton",
        working_dir=str(baseline.parent),
        timeout=300,
    )

    if "error" in result:
        print(f"  Magpie compare error: {result['error']}")
        print(f"  stderr: {result.get('stderr', 'N/A')[:300]}")
        return False

    compiled, correct, speedup = parse_compare_result(result)
    print(f"  Compiled: {compiled}")
    print(f"  Correct:  {correct}")
    print(f"  Speedup:  {speedup:.4f}x")
    print(f"  [OK] run_magpie_compare returned valid result")
    return True


def main():
    print("\n" + "=" * 70)
    print("  GPU Correctness Modes — End-to-End Validation")
    print("  Tests: pytorch, library_test, _try_magpie_perf, magpie_compare")
    print("=" * 70)

    results = {}

    with tempfile.TemporaryDirectory(prefix="apex_gpu_test_") as tmp_str:
        tmp = Path(tmp_str)

        # ── Test 1: Direct Magpie compare ────────────────────────────────
        results["magpie_compare"] = test_run_magpie_compare_direct(tmp)

        # ── Test 2: Direct _try_magpie_perf_measurement ──────────────────
        results["try_magpie_perf"] = test_try_magpie_perf_direct(tmp)

        # ── Test 3: pytorch mode (default) ───────────────────────────────
        banner("Mode 1: pytorch (Magpie compare for correctness + perf)")
        task_dir = setup_task(tmp, "pytorch")
        r = grade_task(task_dir, isolate_caches=False, trust_agent_config=True)
        results["pytorch"] = report_result(r, "pytorch")

        # ── Test 4: library_test mode ────────────────────────────────────
        banner("Mode 2: library_test (pytest + Magpie perf)")
        task_dir = setup_task(tmp, "library_test")
        r = grade_task(task_dir, isolate_caches=False, trust_agent_config=True)
        results["library_test"] = report_result(r, "library_test")

    # ── Summary ──────────────────────────────────────────────────────────
    banner("SUMMARY")
    all_ok = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s} {status}")
        if not passed:
            all_ok = False

    if all_ok:
        print("\n  All GPU tests passed.")
        sys.exit(0)
    else:
        print("\n  Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
