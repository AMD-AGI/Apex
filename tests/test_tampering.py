# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Tests for benchmark tampering detection and penalty logic."""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "graders"))

from kernel_grader import _detect_benchmark_tampering, _apply_tampering_penalties


def _write_solution(code: str) -> Path:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
    )
    f.write(code)
    f.close()
    return Path(f.name)


class TestDetectBenchmarkTampering:
    def test_clean_solution(self):
        path = _write_solution("def kernel(x):\n    return x * 2\n")
        warnings = _detect_benchmark_tampering(path, str(path))
        assert warnings == []
        path.unlink()

    def test_detect_systemexit(self):
        path = _write_solution(
            'import sys\ndef kernel(x): return x\n'
            'if __name__ == "__main__":\n    sys.exit(0)\n'
        )
        warnings = _detect_benchmark_tampering(path, str(path))
        assert len(warnings) == 1
        assert "tampering" in warnings[0].lower() or "SystemExit" in warnings[0]
        path.unlink()

    def test_detect_hardcoded_pass(self):
        path = _write_solution(
            'def kernel(x): return x\n'
            'if __name__ == "__main__":\n    print("PASS")\n'
        )
        warnings = _detect_benchmark_tampering(path, str(path))
        assert len(warnings) == 1
        assert "PASS" in warnings[0]
        path.unlink()

    def test_detect_fake_benchmark(self):
        path = _write_solution(
            'def kernel(x): return x\n'
            'if __name__ == "__main__":\n    print("BENCHMARK_MS: 0.001")\n'
        )
        warnings = _detect_benchmark_tampering(path, str(path))
        assert len(warnings) == 1
        assert "BENCHMARK_MS" in warnings[0]
        path.unlink()


class TestApplyTamperingPenalties:
    def test_no_tampering(self):
        path = _write_solution("def kernel(x): return x * 2\n")
        raw = {}
        correct, speedup = _apply_tampering_penalties(
            raw, path, str(path), compiled=True, correct=True, speedup=2.0
        )
        assert correct is True
        assert speedup == 2.0
        assert "_tampering_warnings" not in raw
        path.unlink()

    def test_fake_pass_rejects_correctness(self):
        path = _write_solution(
            'def kernel(x): return x\n'
            'if __name__ == "__main__":\n    print("PASS")\n'
        )
        raw = {}
        correct, speedup = _apply_tampering_penalties(
            raw, path, str(path), compiled=True, correct=True, speedup=1.5
        )
        assert correct is False
        assert raw.get("_correctness_rejected_for_tampering") is True
        path.unlink()

    def test_speedup_cap(self):
        path = _write_solution(
            'def kernel(x): return x\n'
            'if __name__ == "__main__":\n    print("BENCHMARK_MS: 0.001")\n'
        )
        raw = {}
        correct, speedup = _apply_tampering_penalties(
            raw, path, str(path), compiled=True, correct=True, speedup=5.0,
            speedup_cap=2.0,
        )
        assert speedup == 2.0
        assert raw.get("_speedup_capped_for_tampering") is True
        path.unlink()


class TestBroadenedTampering:
    """Tests for whole-AST tampering detection outside __main__ guards."""

    def test_sys_exit_at_module_level(self):
        path = _write_solution("import sys\nsys.exit(0)\ndef kernel(x): return x\n")
        warnings = _detect_benchmark_tampering(path, str(path))
        assert any("sys.exit" in w and "outside __main__" in w for w in warnings)
        path.unlink()

    def test_quit_at_module_level(self):
        path = _write_solution("quit()\ndef kernel(x): return x\n")
        warnings = _detect_benchmark_tampering(path, str(path))
        assert any("quit" in w and "outside __main__" in w for w in warnings)
        path.unlink()

    def test_exec_with_benchmark_ms(self):
        path = _write_solution(
            "exec(\"print('BENCHMARK_MS: 0.1')\")\ndef kernel(x): return x\n"
        )
        warnings = _detect_benchmark_tampering(path, str(path))
        assert any("exec" in w and "suspicious" in w.lower() for w in warnings)
        path.unlink()

    def test_print_pass_outside_main(self):
        path = _write_solution("print(\"PASS\")\ndef kernel(x): return x\n")
        warnings = _detect_benchmark_tampering(path, str(path))
        assert any("PASS" in w and "outside" in w for w in warnings)
        path.unlink()

    def test_os_exit_outside_main(self):
        path = _write_solution(
            "import os\ndef helper():\n    os._exit(0)\ndef kernel(x): return x\n"
        )
        warnings = _detect_benchmark_tampering(path, str(path))
        assert any("os._exit" in w and "outside __main__" in w for w in warnings)
        path.unlink()

    def test_clean_solution_no_warnings(self):
        path = _write_solution(
            "import torch\ndef kernel(x):\n    return x * 2\n"
        )
        warnings = _detect_benchmark_tampering(path, str(path))
        assert warnings == []
        path.unlink()

    def test_penalties_sys_exit_marks_incorrect(self):
        path = _write_solution("import sys\nsys.exit(0)\ndef kernel(x): return x\n")
        raw = {}
        correct, speedup = _apply_tampering_penalties(
            raw, path, str(path), compiled=True, correct=True, speedup=2.0,
        )
        assert correct is False
        assert raw.get("_correctness_rejected_for_tampering") is True
        path.unlink()
