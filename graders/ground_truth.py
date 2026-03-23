"""
ground_truth.py — Auto-discovery of kernel ground truth across tools/rocm/.

Scans ALL libraries (aiter, vLLM, SGLang, CK, triton, TransformerEngine, etc.)
for PyTorch reference implementations, unit test commands, and input generators.

Three mutually exclusive ground truth modes:
  A) pytorch   — pytorch_reference_code + test_shapes_code
  B) library_test — repo_url + unit_test_command
  C) accordo   — Magpie config dict with correctness.backend: accordo
"""

from __future__ import annotations

import ast
import json
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
ROCM_DIR = REPO_ROOT / "tools" / "rocm"

# ── Repo URL mapping ────────────────────────────────────────────────────────

REPO_URLS: dict[str, str] = {
    "aiter": "https://github.com/ROCm/aiter",
    "vllm": "https://github.com/vllm-project/vllm",
    "sglang": "https://github.com/sgl-project/sglang",
    "composable_kernel": "https://github.com/ROCm/composable_kernel",
    "triton": "https://github.com/triton-lang/triton",
    "TransformerEngine": "https://github.com/NVIDIA/TransformerEngine",
    "MIOpen": "https://github.com/ROCm/MIOpen",
    "rocBLAS": "https://github.com/ROCm/rocBLAS",
    "hipBLASLt": "https://github.com/ROCm/hipBLASLt",
    "rccl": "https://github.com/ROCm/rccl",
    "AMDMIGraphX": "https://github.com/ROCm/AMDMIGraphX",
}

# Function name prefixes that indicate a PyTorch reference implementation
_REF_PREFIXES = (
    "torch_", "ref_", "baseline_", "naive_", "reference_", "gold_", "cpu_",
)

# Function name prefixes/patterns for input generators
_INPUT_PREFIXES = (
    "generate_", "get_inputs", "get_test_inputs", "get_init_inputs",
)

# Kernel category -> op_type classification
_MEMORY_BOUND_KEYWORDS = {
    "attention", "attn", "norm", "rmsnorm", "layernorm", "softmax",
    "activation", "silu", "gelu", "rope", "embedding", "kv_cache",
    "reduce", "scatter", "gather", "copy", "pad",
}
_COMPUTE_BOUND_KEYWORDS = {
    "gemm", "matmul", "moe", "linear", "conv", "fft", "quant",
    "grouped_gemm", "gmm", "tgmm",
}


@dataclass
class GroundTruthSpec:
    """Ground truth for a single kernel op."""

    kernel_type: str
    mode: str  # "pytorch" | "library_test" | "accordo"

    # Mode A (pytorch)
    pytorch_reference_code: str = ""
    test_shapes_code: str = ""

    # Mode B (library_test)
    repo_url: str = ""
    unit_test_command: str = ""

    # Mode C (accordo)
    accordo_config: dict = field(default_factory=dict)

    # Metadata
    source_file: str = ""
    source_library: str = ""
    difficulty_level: int = 2
    op_type: str = "memory_bound"

    def to_ground_truth_dict(self) -> dict:
        """Return the ground_truth sub-dict for the dataset schema."""
        return {
            "pytorch_reference_code": self.pytorch_reference_code,
            "test_shapes_code": self.test_shapes_code,
            "repo_url": self.repo_url,
            "unit_test_command": self.unit_test_command,
            "accordo_config": self.accordo_config,
        }

    def to_dict(self) -> dict:
        return asdict(self)


# ── AST-based source extraction ─────────────────────────────────────────────

def _extract_function_source(filepath: Path, func_name: str) -> Optional[str]:
    """Extract the source code of a function from a Python file using AST."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return None

    lines = source.splitlines(keepends=True)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 1
            return "".join(lines[start:end])
    return None


def _find_functions_in_file(filepath: Path) -> list[tuple[str, int, int]]:
    """Return list of (func_name, start_line, end_line) for top-level functions."""
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError):
        return []

    results = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else node.lineno
            results.append((node.name, node.lineno, end))
    return results


def _is_ref_function(name: str) -> bool:
    return any(name.startswith(p) for p in _REF_PREFIXES)


def _is_input_generator(name: str) -> bool:
    return any(name.startswith(p) for p in _INPUT_PREFIXES)


def _classify_op_type(kernel_type: str, filepath: str) -> str:
    combined = (kernel_type + " " + filepath).lower()
    if any(kw in combined for kw in _COMPUTE_BOUND_KEYWORDS):
        return "compute_bound"
    return "memory_bound"


def _estimate_difficulty(source_code: str) -> int:
    loc = len(source_code.strip().splitlines())
    if loc > 80:
        return 3
    if loc > 30:
        return 2
    return 1


def _infer_kernel_type(filepath: Path, func_name: str) -> str:
    """Infer kernel type from the file path and function name."""
    parts = str(filepath).lower()
    name = func_name.lower()
    combined = parts + " " + name

    type_hints = [
        ("flash_attn", ["flash_attn", "flash_attention"]),
        ("paged_attn_decode", ["paged_attn", "pa_decode", "paged_attention"]),
        ("mla_attn", ["mla"]),
        ("fused_moe", ["fused_moe", "moe"]),
        ("rms_norm", ["rmsnorm", "rms_norm"]),
        ("rope_embedding", ["rope"]),
        ("silu_mul", ["silu", "activation"]),
        ("gemm_w8a8", ["w8a8", "a8w8", "fp8"]),
        ("gemm_bf16", ["gemm", "matmul", "gmm"]),
        ("act_quant_fp8", ["quant", "quantiz"]),
        ("kv_cache_ops", ["kv_cache"]),
        ("all_reduce", ["all_reduce", "allreduce"]),
        ("layernorm", ["layernorm", "layer_norm"]),
        ("softmax", ["softmax"]),
        ("conv", ["conv"]),
    ]

    for kernel_type, keywords in type_hints:
        if any(kw in combined for kw in keywords):
            return kernel_type

    stem = filepath.stem.replace("test_", "").replace("bench_", "")
    return stem


def _detect_library(filepath: Path) -> str:
    """Detect which library a file belongs to under tools/rocm/."""
    try:
        rel = filepath.relative_to(ROCM_DIR)
        return rel.parts[0]
    except ValueError:
        return "unknown"


# ── Scanner ──────────────────────────────────────────────────────────────────

_MAX_FILE_BYTES = 150_000  # skip files > 150KB to keep AST parsing fast


def _iter_test_files(rocm_dir: Path, max_files: int) -> list[Path]:
    """Collect test files efficiently by scanning known test directories first."""
    files: list[Path] = []
    seen: set[Path] = set()

    # High-value directories first (most kernel tests), then broad search
    priority_dirs = [
        "aiter/op_tests",
        "aiter/csrc",
        "vllm/tests/kernels",
        "sglang/sgl-kernel/tests",
        "sglang/test/srt/cpu",
        "triton/python/test",
        "triton/third_party/amd/python/test",
        "TransformerEngine/tests",
        "Primus-Turbo",
        "composable_kernel",
        "rocBLAS",
        "hipBLASLt",
        "MIOpen",
        "rccl",
    ]

    for subdir in priority_dirs:
        d = rocm_dir / subdir
        if not d.exists():
            continue
        for py_file in d.rglob("*.py"):
            if len(files) >= max_files:
                return files
            name = py_file.name
            if not (name.startswith("test_") or name.endswith("_test.py")
                    or "ref" in name.lower() or "utils" in name.lower()
                    or "baseline" in name.lower()):
                continue
            try:
                if py_file.stat().st_size > _MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            if py_file not in seen:
                seen.add(py_file)
                files.append(py_file)

    return files


def scan_rocm_ground_truth(
    rocm_dir: Path | None = None,
    max_files: int = 2000,
) -> list[GroundTruthSpec]:
    """Scan tools/rocm/ for ground truth specs.

    Walks all test files, extracts PyTorch reference functions and input
    generators, and pairs them into GroundTruthSpec objects.
    """
    rocm_dir = rocm_dir or ROCM_DIR
    if not rocm_dir.exists():
        return []

    specs: list[GroundTruthSpec] = []
    test_files = _iter_test_files(rocm_dir, max_files)

    for py_file in test_files:
        functions = _find_functions_in_file(py_file)
        if not functions:
            continue

        library = _detect_library(py_file)
        repo_url = REPO_URLS.get(library, "")
        rel_path = str(py_file.relative_to(rocm_dir))

        ref_funcs = [(n, s, e) for n, s, e in functions if _is_ref_function(n)]
        input_funcs = [(n, s, e) for n, s, e in functions if _is_input_generator(n)]

        if not ref_funcs and not input_funcs:
            continue

        for func_name, start, end in ref_funcs:
            ref_source = _extract_function_source(py_file, func_name)
            if not ref_source or len(ref_source.strip()) < 10:
                continue

            kernel_type = _infer_kernel_type(py_file, func_name)

            input_source = ""
            for inp_name, _, _ in input_funcs:
                extracted = _extract_function_source(py_file, inp_name)
                if extracted:
                    input_source = extracted
                    break

            spec = GroundTruthSpec(
                kernel_type=kernel_type,
                mode="pytorch",
                pytorch_reference_code=ref_source.strip(),
                test_shapes_code=input_source.strip(),
                source_file=rel_path,
                source_library=library,
                difficulty_level=_estimate_difficulty(ref_source),
                op_type=_classify_op_type(kernel_type, rel_path),
            )
            specs.append(spec)

        # For test files with no extractable ref function but valid test commands
        if not ref_funcs and input_funcs:
            kernel_type = _infer_kernel_type(py_file, input_funcs[0][0])
            test_cmd = f"python -m pytest {rel_path}"
            spec = GroundTruthSpec(
                kernel_type=kernel_type,
                mode="library_test",
                repo_url=repo_url,
                unit_test_command=test_cmd,
                source_file=rel_path,
                source_library=library,
                difficulty_level=2,
                op_type=_classify_op_type(kernel_type, rel_path),
            )
            specs.append(spec)

    return specs


def scan_test_commands(
    rocm_dir: Path | None = None,
    max_files: int = 2000,
) -> list[GroundTruthSpec]:
    """Scan for test files that can serve as Mode B (library_test) ground truth.

    This finds test files even if they don't have extractable ref functions,
    producing unit_test_command entries.
    """
    rocm_dir = rocm_dir or ROCM_DIR
    if not rocm_dir.exists():
        return []

    specs: list[GroundTruthSpec] = []
    test_files = _iter_test_files(rocm_dir, max_files)

    for py_file in test_files:
        name = py_file.name
        if not (name.startswith("test_") or name.endswith("_test.py")):
            continue

        library = _detect_library(py_file)
        repo_url = REPO_URLS.get(library, "")
        rel_path = str(py_file.relative_to(rocm_dir))

        kernel_type = _infer_kernel_type(py_file, "")
        test_cmd = f"python -m pytest {rel_path}"

        spec = GroundTruthSpec(
            kernel_type=kernel_type,
            mode="library_test",
            repo_url=repo_url,
            unit_test_command=test_cmd,
            source_file=rel_path,
            source_library=library,
            difficulty_level=2,
            op_type=_classify_op_type(kernel_type, rel_path),
        )
        specs.append(spec)

    return specs


# ── Manual registry for core kernel types ────────────────────────────────────

MANUAL_REGISTRY: dict[str, GroundTruthSpec] = {}


def _build_manual_registry() -> None:
    """Populate MANUAL_REGISTRY with hand-curated specs for the 12 KERNEL_MAP
    types. These take priority over auto-discovered ones."""

    entries = [
        GroundTruthSpec(
            kernel_type="rms_norm",
            mode="pytorch",
            pytorch_reference_code=textwrap.dedent("""\
                def baseline_fn(x, weight, eps=1e-6):
                    import torch
                    x_f32 = x.float()
                    rms = torch.sqrt(torch.mean(x_f32 * x_f32, dim=-1, keepdim=True) + eps)
                    return (x_f32 / rms * weight.float()).to(x.dtype)
            """).strip(),
            test_shapes_code=textwrap.dedent("""\
                def get_test_inputs(device='cuda'):
                    import torch
                    configs = [(128, 4096), (256, 8192), (512, 4096)]
                    inputs = []
                    for M, N in configs:
                        x = torch.randn(M, N, dtype=torch.float16, device=device)
                        weight = torch.randn(N, dtype=torch.float16, device=device)
                        inputs.append((x, weight))
                    return inputs
            """).strip(),
            source_file="aiter/op_tests/triton_tests/normalization/test_rmsnorm.py",
            source_library="aiter",
            difficulty_level=1,
            op_type="memory_bound",
        ),
        GroundTruthSpec(
            kernel_type="silu_mul",
            mode="pytorch",
            pytorch_reference_code=textwrap.dedent("""\
                def baseline_fn(x):
                    import torch
                    import torch.nn.functional as F
                    d = x.shape[-1] // 2
                    return F.silu(x[..., :d]) * x[..., d:]
            """).strip(),
            test_shapes_code=textwrap.dedent("""\
                def get_test_inputs(device='cuda'):
                    import torch
                    configs = [(128, 8192), (256, 16384), (512, 8192)]
                    inputs = []
                    for M, N in configs:
                        x = torch.randn(M, N * 2, dtype=torch.float16, device=device)
                        inputs.append((x,))
                    return inputs
            """).strip(),
            source_file="aiter/op_tests/test_activation.py",
            source_library="aiter",
            difficulty_level=1,
            op_type="memory_bound",
        ),
        GroundTruthSpec(
            kernel_type="fused_moe",
            mode="pytorch",
            pytorch_reference_code=textwrap.dedent("""\
                def baseline_fn(hidden_states, w1, w2, topk_weights, topk_ids):
                    import torch
                    import torch.nn.functional as F
                    B, D = hidden_states.shape
                    E, _, N = w1.shape
                    K = topk_ids.shape[1]
                    out = torch.zeros(B, D, dtype=hidden_states.dtype, device=hidden_states.device)
                    for i in range(B):
                        for j in range(K):
                            eid = topk_ids[i, j].item()
                            gate = F.silu(hidden_states[i] @ w1[eid].T)
                            expert_out = gate @ w2[eid].T
                            out[i] += topk_weights[i, j] * expert_out
                    return out
            """).strip(),
            test_shapes_code=textwrap.dedent("""\
                def get_test_inputs(device='cuda'):
                    import torch
                    configs = [
                        (32, 4096, 14336, 8, 2),
                        (64, 8192, 14336, 16, 4),
                    ]
                    inputs = []
                    for B, D, N, E, K in configs:
                        hidden = torch.randn(B, D, dtype=torch.float16, device=device)
                        w1 = torch.randn(E, N, D, dtype=torch.float16, device=device)
                        w2 = torch.randn(E, D, N, dtype=torch.float16, device=device)
                        topk_w = torch.softmax(torch.randn(B, K, device=device), dim=-1).half()
                        topk_ids = torch.randint(0, E, (B, K), device=device)
                        inputs.append((hidden, w1, w2, topk_w, topk_ids))
                    return inputs
            """).strip(),
            source_file="aiter/op_tests/triton_tests/moe/test_moe.py",
            source_library="aiter",
            difficulty_level=3,
            op_type="compute_bound",
        ),
        GroundTruthSpec(
            kernel_type="flash_attn_prefill",
            mode="pytorch",
            pytorch_reference_code=textwrap.dedent("""\
                def baseline_fn(q, k, v, causal=True):
                    import torch
                    import torch.nn.functional as F
                    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            """).strip(),
            test_shapes_code=textwrap.dedent("""\
                def get_test_inputs(device='cuda'):
                    import torch
                    configs = [
                        (2, 16, 128, 128),
                        (4, 32, 256, 128),
                    ]
                    inputs = []
                    for B, H, S, D in configs:
                        q = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
                        k = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
                        v = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
                        inputs.append((q, k, v))
                    return inputs
            """).strip(),
            source_file="aiter/op_tests/triton_tests/attention/test_prefill_attention.py",
            source_library="aiter",
            difficulty_level=3,
            op_type="memory_bound",
        ),
        GroundTruthSpec(
            kernel_type="paged_attn_decode",
            mode="pytorch",
            pytorch_reference_code=textwrap.dedent("""\
                def baseline_fn(query, key_cache, value_cache, head_mapping, scale):
                    import torch
                    num_heads = query.shape[1]
                    head_dim = query.shape[2]
                    outputs = []
                    for h in range(num_heads):
                        q = query[:, h, :].unsqueeze(1)
                        k = key_cache[:, h, :, :]
                        v = value_cache[:, h, :, :]
                        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
                        out = torch.matmul(attn, v)
                        outputs.append(out.squeeze(1))
                    return torch.stack(outputs, dim=1)
            """).strip(),
            test_shapes_code=textwrap.dedent("""\
                def get_test_inputs(device='cuda'):
                    import torch
                    configs = [(1, 32, 128, 256), (4, 64, 128, 512)]
                    inputs = []
                    for B, H, D, S in configs:
                        q = torch.randn(B, H, D, dtype=torch.float16, device=device)
                        k = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
                        v = torch.randn(B, H, S, D, dtype=torch.float16, device=device)
                        mapping = torch.arange(H, device=device)
                        inputs.append((q, k, v, mapping, 1.0 / (D ** 0.5)))
                    return inputs
            """).strip(),
            source_file="aiter/op_tests/triton_tests/attention/test_pa_decode.py",
            source_library="aiter",
            difficulty_level=3,
            op_type="memory_bound",
        ),
        GroundTruthSpec(
            kernel_type="gemm_bf16",
            mode="pytorch",
            pytorch_reference_code=textwrap.dedent("""\
                def baseline_fn(a, b):
                    import torch
                    return torch.matmul(a, b)
            """).strip(),
            test_shapes_code=textwrap.dedent("""\
                def get_test_inputs(device='cuda'):
                    import torch
                    configs = [(1024, 4096, 4096), (2048, 8192, 4096), (4096, 4096, 8192)]
                    inputs = []
                    for M, K, N in configs:
                        a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
                        b = torch.randn(K, N, dtype=torch.bfloat16, device=device)
                        inputs.append((a, b))
                    return inputs
            """).strip(),
            source_file="aiter/op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py",
            source_library="aiter",
            difficulty_level=2,
            op_type="compute_bound",
        ),
        GroundTruthSpec(
            kernel_type="gemm_w8a8",
            mode="pytorch",
            pytorch_reference_code=textwrap.dedent("""\
                def baseline_fn(a, b, a_scale, b_scale):
                    import torch
                    return torch.matmul(a.float() * a_scale, b.float() * b_scale)
            """).strip(),
            test_shapes_code=textwrap.dedent("""\
                def get_test_inputs(device='cuda'):
                    import torch
                    configs = [(1024, 4096, 4096), (2048, 8192, 4096)]
                    inputs = []
                    for M, K, N in configs:
                        a = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=device)
                        b = torch.randint(-127, 127, (K, N), dtype=torch.int8, device=device)
                        a_s = torch.tensor(0.01, device=device)
                        b_s = torch.tensor(0.01, device=device)
                        inputs.append((a, b, a_s, b_s))
                    return inputs
            """).strip(),
            source_file="aiter/op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py",
            source_library="aiter",
            difficulty_level=2,
            op_type="compute_bound",
        ),
        GroundTruthSpec(
            kernel_type="rope_embedding",
            mode="pytorch",
            pytorch_reference_code=textwrap.dedent("""\
                def baseline_fn(x, cos, sin):
                    import torch
                    x1 = x[..., : x.shape[-1] // 2]
                    x2 = x[..., x.shape[-1] // 2 :]
                    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
            """).strip(),
            test_shapes_code=textwrap.dedent("""\
                def get_test_inputs(device='cuda'):
                    import torch
                    configs = [(2, 32, 128, 128), (4, 64, 256, 128)]
                    inputs = []
                    for B, H, S, D in configs:
                        x = torch.randn(B, S, H, D, dtype=torch.float16, device=device)
                        cos = torch.randn(S, D // 2, dtype=torch.float16, device=device)
                        sin = torch.randn(S, D // 2, dtype=torch.float16, device=device)
                        inputs.append((x, cos, sin))
                    return inputs
            """).strip(),
            source_file="aiter/op_tests/test_rope.py",
            source_library="aiter",
            difficulty_level=2,
            op_type="memory_bound",
        ),
        GroundTruthSpec(
            kernel_type="act_quant_fp8",
            mode="pytorch",
            pytorch_reference_code=textwrap.dedent("""\
                def baseline_fn(x):
                    import torch
                    scale = x.abs().max() / 448.0
                    x_q = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fnuz)
                    return x_q, scale
            """).strip(),
            test_shapes_code=textwrap.dedent("""\
                def get_test_inputs(device='cuda'):
                    import torch
                    configs = [(128, 4096), (256, 8192), (512, 4096)]
                    inputs = []
                    for M, N in configs:
                        x = torch.randn(M, N, dtype=torch.float16, device=device)
                        inputs.append((x,))
                    return inputs
            """).strip(),
            source_file="aiter/op_tests/triton_tests/quant/test_quant.py",
            source_library="aiter",
            difficulty_level=2,
            op_type="compute_bound",
        ),
        GroundTruthSpec(
            kernel_type="kv_cache_ops",
            mode="library_test",
            repo_url="https://github.com/ROCm/aiter",
            unit_test_command="python -m pytest aiter/op_tests/triton_tests/fusions/test_fused_kv_cache.py",
            source_file="aiter/op_tests/triton_tests/fusions/test_fused_kv_cache.py",
            source_library="aiter",
            difficulty_level=2,
            op_type="memory_bound",
        ),
        GroundTruthSpec(
            kernel_type="all_reduce",
            mode="library_test",
            repo_url="https://github.com/ROCm/aiter",
            unit_test_command="python -m pytest aiter/op_tests/multigpu_tests/test_quick_all_reduce.py",
            source_file="aiter/op_tests/multigpu_tests/test_quick_all_reduce.py",
            source_library="aiter",
            difficulty_level=2,
            op_type="memory_bound",
        ),
        GroundTruthSpec(
            kernel_type="mla_attn",
            mode="library_test",
            repo_url="https://github.com/ROCm/aiter",
            unit_test_command="python -m pytest aiter/op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py",
            source_file="aiter/op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py",
            source_library="aiter",
            difficulty_level=3,
            op_type="memory_bound",
        ),
    ]

    for spec in entries:
        MANUAL_REGISTRY[spec.kernel_type] = spec


_build_manual_registry()


# ── Public API ───────────────────────────────────────────────────────────────

def scan_hip_kernels_for_accordo(
    rocm_dir: Path | None = None,
) -> list[GroundTruthSpec]:
    """Scan CK, rocBLAS, hipBLASLt, MIOpen for HIP/C++ kernels -> Accordo specs.

    These are Mode C (Accordo) since they have no Python test infrastructure.
    """
    rocm_dir = rocm_dir or ROCM_DIR
    specs: list[GroundTruthSpec] = []

    # CK examples: each numbered directory is a distinct kernel operation
    ck_examples = rocm_dir / "composable_kernel" / "example"
    if ck_examples.exists():
        for example_dir in sorted(ck_examples.iterdir()):
            if not example_dir.is_dir():
                continue
            name = example_dir.name
            # Parse "01_gemm", "15_grouped_gemm", etc.
            parts = name.split("_", 1)
            if len(parts) < 2:
                continue
            kernel_name = parts[1]  # e.g. "gemm", "grouped_gemm"

            # Find any binary targets (from CMakeLists.txt)
            cmake = example_dir / "CMakeLists.txt"
            binary_name = f"example_{name}"

            accordo_config = {
                "correctness": {
                    "backend": "accordo",
                    "accordo": {
                        "kernel_name": kernel_name,
                        "reference_binary": f"build/bin/{binary_name}",
                        "optimized_binary": f"build/bin/{binary_name}_opt",
                        "tolerance": 0.001,
                        "timeout_seconds": 60,
                        "working_directory": "${CK_HOME}",
                    },
                }
            }

            specs.append(GroundTruthSpec(
                kernel_type=f"ck_{kernel_name}",
                mode="accordo",
                accordo_config=accordo_config,
                source_file=f"composable_kernel/example/{name}",
                source_library="composable_kernel",
                difficulty_level=3,
                op_type=_classify_op_type(kernel_name, name),
            ))

    # rocBLAS routines
    rocblas_src = rocm_dir / "rocBLAS" / "library" / "src"
    if rocblas_src.exists():
        for blas_dir in sorted(rocblas_src.iterdir()):
            if not blas_dir.is_dir():
                continue
            name = blas_dir.name
            if name.startswith("blas"):
                level = name  # e.g. "blas1", "blas2", "blas3"
                for cpp_file in blas_dir.glob("*.cpp"):
                    routine = cpp_file.stem.replace("rocblas_", "")
                    specs.append(GroundTruthSpec(
                        kernel_type=f"rocblas_{routine}",
                        mode="accordo",
                        accordo_config={
                            "correctness": {
                                "backend": "accordo",
                                "accordo": {
                                    "kernel_name": routine,
                                    "reference_binary": f"build/clients/staging/rocblas-bench",
                                    "tolerance": 0.001,
                                    "timeout_seconds": 60,
                                    "working_directory": "${ROCBLAS_HOME}",
                                },
                            }
                        },
                        source_file=f"rocBLAS/library/src/{name}/{cpp_file.name}",
                        source_library="rocBLAS",
                        difficulty_level=3,
                        op_type="compute_bound",
                    ))

    # hipBLASLt routines
    hipblaslt_src = rocm_dir / "hipBLASLt" / "tensilelite" / "Tensile" / "Components"
    if not hipblaslt_src.exists():
        hipblaslt_src = rocm_dir / "hipBLASLt" / "library" / "src"
    if hipblaslt_src.exists():
        specs.append(GroundTruthSpec(
            kernel_type="hipblaslt_gemm",
            mode="accordo",
            accordo_config={
                "correctness": {
                    "backend": "accordo",
                    "accordo": {
                        "kernel_name": "hipblaslt_gemm",
                        "reference_binary": "build/clients/staging/hipblaslt-bench",
                        "tolerance": 0.001,
                        "timeout_seconds": 60,
                        "working_directory": "${HIPBLASLT_HOME}",
                    },
                }
            },
            source_file="hipBLASLt",
            source_library="hipBLASLt",
            difficulty_level=3,
            op_type="compute_bound",
        ))

    # MIOpen kernels
    miopen_src = rocm_dir / "MIOpen" / "src"
    if miopen_src.exists():
        miopen_ops = ["convolution", "batchnorm", "pooling", "softmax",
                      "lrn", "dropout", "rnn", "ctc", "layernorm", "groupnorm"]
        for op in miopen_ops:
            op_dir = miopen_src / op
            if op_dir.exists() or (miopen_src / f"{op}.cpp").exists():
                specs.append(GroundTruthSpec(
                    kernel_type=f"miopen_{op}",
                    mode="accordo",
                    accordo_config={
                        "correctness": {
                            "backend": "accordo",
                            "accordo": {
                                "kernel_name": op,
                                "reference_binary": f"build/bin/MIOpenDriver",
                                "tolerance": 0.001,
                                "timeout_seconds": 60,
                                "working_directory": "${MIOPEN_HOME}",
                            },
                        }
                    },
                    source_file=f"MIOpen/src/{op}",
                    source_library="MIOpen",
                    difficulty_level=3,
                    op_type=_classify_op_type(op, op),
                ))

    return specs


def generate_accordo_config(
    kernel_name: str,
    reference_binary: str,
    optimized_binary: str = "",
    tolerance: float = 0.001,
    timeout_seconds: int = 60,
    working_directory: str = "",
) -> dict:
    """Generate a Magpie-compatible Accordo correctness config dict.

    At evaluation time, the RL evaluator writes this to a temp YAML and runs:
        python -m Magpie compare --kernel-config <temp.yaml>
    """
    config: dict = {
        "correctness": {
            "backend": "accordo",
            "accordo": {
                "kernel_name": kernel_name,
                "reference_binary": reference_binary,
                "tolerance": tolerance,
                "timeout_seconds": timeout_seconds,
            },
        }
    }
    if optimized_binary:
        config["correctness"]["accordo"]["optimized_binary"] = optimized_binary
    if working_directory:
        config["correctness"]["accordo"]["working_directory"] = working_directory
    return config


def discover_all(
    rocm_dir: Path | None = None,
    include_manual: bool = True,
    max_files: int = 5000,
) -> list[GroundTruthSpec]:
    """Return all discovered ground truth specs.

    Manual registry entries take priority (appear first) and are followed
    by auto-discovered specs from scanning tools/rocm/.
    """
    results: list[GroundTruthSpec] = []

    if include_manual:
        results.extend(MANUAL_REGISTRY.values())

    auto_specs = scan_rocm_ground_truth(rocm_dir=rocm_dir, max_files=max_files)
    seen_types = {s.kernel_type for s in results}
    for spec in auto_specs:
        if spec.kernel_type not in seen_types:
            seen_types.add(spec.kernel_type)
            results.append(spec)

    # Accordo specs for HIP/C++ kernels
    accordo_specs = scan_hip_kernels_for_accordo(rocm_dir=rocm_dir)
    for spec in accordo_specs:
        if spec.kernel_type not in seen_types:
            seen_types.add(spec.kernel_type)
            results.append(spec)

    return results


def get_spec(kernel_type: str, rocm_dir: Path | None = None) -> Optional[GroundTruthSpec]:
    """Look up ground truth for a specific kernel type.

    Checks manual registry first, then scans if not found.
    """
    if kernel_type in MANUAL_REGISTRY:
        return MANUAL_REGISTRY[kernel_type]

    for spec in scan_rocm_ground_truth(rocm_dir=rocm_dir, max_files=2000):
        if spec.kernel_type == kernel_type:
            return spec
    return None


def discover_by_library(
    rocm_dir: Path | None = None,
    max_files: int = 5000,
) -> dict[str, list[GroundTruthSpec]]:
    """Group discovered specs by source library."""
    specs = discover_all(rocm_dir=rocm_dir, max_files=max_files)
    by_lib: dict[str, list[GroundTruthSpec]] = {}
    for s in specs:
        by_lib.setdefault(s.source_library, []).append(s)
    return by_lib
