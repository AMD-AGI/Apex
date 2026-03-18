"""
config_generator.py — Pipeline-controlled config.yaml generation for kernel grading.

The agent must ONLY write solution.py / solution.hip. The pipeline generates
(or overwrites) the config.yaml with trusted correctness and performance
commands that the agent cannot tamper with.

This prevents the agent from:
  - Writing trivial tests that always pass
  - Fabricating benchmark numbers
  - Pointing baseline.path to a deliberately slow implementation
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(REPO_ROOT / "prompts"))
try:
    from kernel_prompt import KERNEL_MAP, KernelSpec
except ImportError:
    KERNEL_MAP = {}

    @dataclass
    class KernelSpec:  # type: ignore[no-redef]
        kernel_type: str = ""
        description: str = ""
        applies_to: str = "all"
        vllm_path: str = ""
        sglang_path: str = ""
        triton: bool = False


# ---------------------------------------------------------------------------
# Trusted baseline paths per kernel type × framework
# ---------------------------------------------------------------------------

_BASELINE_PATHS: dict[str, dict[str, str]] = {
    "flash_attn_prefill": {
        "vllm": "vllm/attention/backends/rocm_flash_attn.py",
        "sglang": "sglang/srt/layers/attention/triton_ops/prefill_attention.py",
    },
    "paged_attn_decode": {
        "vllm": "vllm/attention/backends/rocm_flash_attn.py",
        "sglang": "sglang/srt/layers/attention/triton_ops/decode_attention.py",
    },
    "mla_attn": {
        "vllm": "vllm/attention/backends/mla/rocm_mla_attn.py",
        "sglang": "sglang/srt/layers/attention/mla_attn_backend.py",
    },
    "fused_moe": {
        "vllm": "vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py",
        "sglang": "sglang/srt/layers/moe/fused_moe_triton.py",
    },
    "gemm_w8a8": {
        "vllm": "vllm/model_executor/layers/quantization/utils/w8a8_utils.py",
        "sglang": "sglang/srt/layers/quantization/fp8_kernel.py",
    },
    "gemm_bf16": {
        "vllm": "vllm/model_executor/layers/linear.py",
        "sglang": "sglang/srt/layers/linear.py",
    },
    "rms_norm": {
        "vllm": "vllm/model_executor/layers/layernorm.py",
        "sglang": "sglang/srt/layers/layernorm.py",
    },
    "rope_embedding": {
        "vllm": "vllm/model_executor/layers/rotary_embedding.py",
        "sglang": "sglang/srt/layers/rotary_embedding.py",
    },
    "kv_cache_ops": {
        "vllm": "vllm/attention/ops/paged_attn.py",
        "sglang": "sglang/srt/mem_cache/memory_pool.py",
    },
    "all_reduce": {
        "vllm": "vllm/distributed/communication_op.py",
        "sglang": "sglang/srt/distributed/all_reduce.py",
    },
    "act_quant_fp8": {
        "vllm": "vllm/model_executor/layers/quantization/fp8.py",
        "sglang": "sglang/srt/layers/quantization/fp8_kernel.py",
    },
    "silu_mul": {
        "vllm": "vllm/model_executor/layers/activation.py",
        "sglang": "sglang/srt/layers/activation.py",
    },
}


def _detect_kernel_type(task_dir: Path) -> Optional[str]:
    """Infer kernel_type from the task directory name."""
    name = task_dir.name.lower()
    for kt in KERNEL_MAP:
        if kt in name:
            return kt
    return None


def _detect_framework(task_dir: Path) -> str:
    """Infer framework from task directory name or existing config."""
    name = task_dir.name.lower()
    if "vllm" in name:
        return "vllm"
    if "sglang" in name:
        return "sglang"
    cfg_path = task_dir / "config.yaml"
    if cfg_path.exists() and yaml is not None:
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            fw = cfg.get("framework", "")
            if fw in ("vllm", "sglang"):
                return fw
        except Exception:
            pass
    return "vllm"


def _find_solution(task_dir: Path) -> Optional[Path]:
    for name in ("solution.py", "solution.hip", "solution.cu"):
        p = task_dir / name
        if p.exists():
            return p
    return None


def _solution_hash(solution_path: Path) -> str:
    """SHA-256 of the solution file, recorded in config for tamper detection."""
    return hashlib.sha256(solution_path.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def _resolve_baseline_path(
    task_dir: Path,
    kernel_type: str,
    framework: str,
) -> str:
    """Resolve a baseline path that actually exists on disk.

    Priority:
      1. baseline_ref.py / baseline_ref.hip in task_dir (pipeline-generated reference)
      2. baseline.py / baseline.hip in task_dir (copied from source tree by pipeline)
      3. Absolute path resolved from _BASELINE_PATHS via tools/rocm/ source tree
      4. Raw _BASELINE_PATHS value as fallback
    """
    for ext in (".py", ".hip"):
        ref = task_dir / f"baseline_ref{ext}"
        if ref.exists():
            return f"./baseline_ref{ext}"
        base = task_dir / f"baseline{ext}"
        if base.exists():
            return f"./baseline{ext}"

    raw_path = _BASELINE_PATHS.get(kernel_type, {}).get(framework, "")
    if not raw_path:
        return ""

    repo_root = Path(__file__).parent.parent
    for lib_prefix in ("", "vllm/", "aiter/"):
        candidate = repo_root / "tools" / "rocm" / lib_prefix / raw_path
        if candidate.exists():
            return str(candidate.resolve())

    return raw_path


def generate_config(
    task_dir: Path,
    kernel_type: Optional[str] = None,
    framework: Optional[str] = None,
    gpu_arch: str = "gfx950",
    gpu_device: int = 0,
    baseline_path: Optional[str] = None,
) -> dict:
    """Generate a trusted Magpie config.yaml for a task directory.

    The pipeline controls:
      - baseline.path   (from the canonical registry, not agent-chosen)
      - correctness     (Magpie's built-in comparison, not an agent script)
      - performance     (Magpie's built-in timing, not an agent script)

    The agent only controls:
      - solution.py contents (which is what we're evaluating)
    """
    kernel_type = kernel_type or _detect_kernel_type(task_dir)
    framework = framework or _detect_framework(task_dir)
    solution = _find_solution(task_dir)

    if not kernel_type:
        raise ValueError(f"Cannot determine kernel_type for {task_dir.name}")

    is_triton = KERNEL_MAP.get(kernel_type, KernelSpec(
        kernel_type=kernel_type, description="", applies_to="all"
    )).triton

    if baseline_path is None:
        baseline_path = _resolve_baseline_path(task_dir, kernel_type, framework)

    sol_relpath = f"./{solution.name}" if solution else "./solution.py"
    sol_hash = _solution_hash(solution) if solution else ""

    config = {
        "gpu": {
            "device": gpu_device,
            "arch": gpu_arch,
        },
        "baseline": {
            "path": baseline_path,
        },
        "optimized": {
            "path": sol_relpath,
        },
        "correctness": {
            "mode": "magpie_builtin",
            "tolerance": 1e-3,
            "num_tests": 10,
        },
        "performance": {
            "mode": "magpie_builtin",
            "warmup_iterations": 10,
            "iterations": 100,
        },
        "_pipeline_metadata": {
            "kernel_type": kernel_type,
            "framework": framework,
            "generated_by": "config_generator.py",
            "solution_hash": sol_hash,
            "tamper_protected": True,
        },
    }

    return config


def write_config(task_dir: Path, **kwargs) -> Path:
    """Generate and write config.yaml to the task directory."""
    config = generate_config(task_dir, **kwargs)
    config_path = task_dir / "config.yaml"
    if yaml is not None:
        config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
    else:
        config_path.write_text(json.dumps(config, indent=2))
    return config_path


# ---------------------------------------------------------------------------
# Config validation (detect agent tampering)
# ---------------------------------------------------------------------------

_FORBIDDEN_CORRECTNESS_PATTERNS = [
    "echo",
    "exit 0",
    "true",
    "/dev/null",
    "pass",
    "print('PASS')",
    'print("PASS")',
]

_FORBIDDEN_PERFORMANCE_PATTERNS = [
    "echo",
    "sleep",
    "print(",
    "/dev/null",
]


@dataclass
class ConfigValidation:
    valid: bool
    warnings: list[str]
    errors: list[str]


def validate_config(task_dir: Path) -> ConfigValidation:
    """Validate that a config.yaml hasn't been tampered with by the agent.

    Returns validation result with warnings and errors.
    """
    config_path = task_dir / "config.yaml"
    warnings: list[str] = []
    errors: list[str] = []

    if not config_path.exists():
        errors.append("config.yaml missing")
        return ConfigValidation(valid=False, warnings=warnings, errors=errors)

    try:
        raw_text = config_path.read_text()
        if yaml is not None:
            config = yaml.safe_load(raw_text)
        else:
            config = json.loads(raw_text)
    except Exception as e:
        errors.append(f"config.yaml parse error: {e}")
        return ConfigValidation(valid=False, warnings=warnings, errors=errors)

    meta = config.get("_pipeline_metadata", {})
    if not meta.get("tamper_protected"):
        warnings.append("config.yaml was not generated by the pipeline — will be regenerated")

    correctness = config.get("correctness", {})
    if "command" in correctness:
        cmd = correctness["command"].lower()
        for pattern in _FORBIDDEN_CORRECTNESS_PATTERNS:
            if pattern in cmd:
                errors.append(
                    f"Suspicious correctness command contains '{pattern}': {correctness['command']}"
                )

    performance = config.get("performance", {})
    if "command" in performance:
        cmd = performance["command"].lower()
        for pattern in _FORBIDDEN_PERFORMANCE_PATTERNS:
            if pattern in cmd:
                errors.append(
                    f"Suspicious performance command contains '{pattern}': {performance['command']}"
                )

    solution = _find_solution(task_dir)
    if solution and meta.get("solution_hash"):
        current_hash = _solution_hash(solution)
        if current_hash != meta["solution_hash"]:
            warnings.append(
                "Solution was modified after config was generated "
                "(hash mismatch) — config will be regenerated"
            )

    return ConfigValidation(
        valid=len(errors) == 0,
        warnings=warnings,
        errors=errors,
    )
