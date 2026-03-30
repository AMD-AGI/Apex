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
import logging
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NamedTuple, Optional

logger = logging.getLogger(__name__)

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
    ref_func_name: str = ""
    sol_entry_func: str = ""
    input_func_name: str = ""

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
#
# Every entry here points to a REAL library test file that contains an actual
# PyTorch reference function (torch_*, ref_*, reference_*).  Apex never
# generates synthetic reference code -- it only uses references that already
# exist in ROCm libraries.
#
# For "pytorch" mode entries the ref_function / input_generator names tell the
# auto-extractor which function to pull from the test file.  The actual source
# is read lazily via _extract_function_source() at discovery time so it always
# reflects the latest library code.

MANUAL_REGISTRY: dict[str, GroundTruthSpec] = {}


class _RegistryEntry(NamedTuple):
    kernel_type: str
    mode: str
    source_file: str
    library: str
    ref_func: str
    input_func: str
    sol_entry_func: str
    test_cmd: str
    difficulty: int
    op_type: str


_REGISTRY_ENTRIES: list[_RegistryEntry] = [
    #                                                                                                                                              ref_func               input_func                 sol_entry_func         test_cmd                                                                        diff  op_type
    _RegistryEntry("rms_norm",         "pytorch",      "aiter/op_tests/triton_tests/normalization/test_rmsnorm.py",                      "aiter", "torch_rmsnorm",       "generate_rmsnorm_inputs", "rms_norm",            "",                                                                                 1, "memory_bound"),
    _RegistryEntry("silu_mul",         "library_test", "aiter/op_tests/triton_tests/test_activation.py",                                 "aiter", "",                    "",                        "",                    "python -m pytest aiter/op_tests/triton_tests/test_activation.py",               1, "memory_bound"),
    _RegistryEntry("fused_moe",        "pytorch",      "aiter/op_tests/triton_tests/moe/test_moe.py",                                   "aiter", "torch_moe_ref",       "",                        "",                    "",                                                                              3, "compute_bound"),
    _RegistryEntry("flash_attn_prefill","pytorch",     "aiter/op_tests/triton_tests/attention/test_la.py",                               "aiter", "reference_attention", "",                        "",                    "",                                                                              3, "memory_bound"),
    _RegistryEntry("paged_attn_decode","pytorch",      "aiter/op_tests/triton_tests/attention/test_unified_attention.py",                "aiter", "ref_paged_attn",      "",                        "",                    "",                                                                              3, "memory_bound"),
    _RegistryEntry("gemm_bf16",        "library_test", "aiter/op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py",                    "aiter", "",                    "",                        "",                    "python -m pytest aiter/op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py",   2, "compute_bound"),
    _RegistryEntry("gemm_w8a8",        "library_test", "aiter/op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py",                      "aiter", "",                    "",                        "",                    "python -m pytest aiter/op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py",     2, "compute_bound"),
    _RegistryEntry("rope_embedding",   "pytorch",      "aiter/op_tests/test_rope.py",                                                   "aiter", "ref_rope_sbhd_fwd",   "",                        "",                    "",                                                                              2, "memory_bound"),
    _RegistryEntry("act_quant_fp8",    "library_test", "aiter/op_tests/triton_tests/quant/test_quant.py",                               "aiter", "",                    "",                        "",                    "python -m pytest aiter/op_tests/triton_tests/quant/test_quant.py",              2, "compute_bound"),
    _RegistryEntry("kv_cache_ops",     "library_test", "aiter/op_tests/triton_tests/fusions/test_fused_kv_cache.py",                    "aiter", "",                    "",                        "",                    "python -m pytest aiter/op_tests/triton_tests/fusions/test_fused_kv_cache.py",   2, "memory_bound"),
    _RegistryEntry("all_reduce",       "library_test", "aiter/op_tests/multigpu_tests/test_quick_all_reduce.py",                        "aiter", "",                    "",                        "",                    "python -m pytest aiter/op_tests/multigpu_tests/test_quick_all_reduce.py",       2, "memory_bound"),
    _RegistryEntry("mla_attn",         "pytorch",      "aiter/op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py",    "aiter", "reference_torch",     "",                        "",                    "",                                                                              3, "memory_bound"),
    _RegistryEntry("layernorm",        "library_test", "aiter/op_tests/triton_tests/normalization/test_layernorm.py",                   "aiter", "",                    "",                        "",                    "python -m pytest aiter/op_tests/triton_tests/normalization/test_layernorm.py",  1, "memory_bound"),
    _RegistryEntry("softmax",          "library_test", "aiter/op_tests/triton_tests/test_softmax.py",                                   "aiter", "",                    "",                        "",                    "python -m pytest aiter/op_tests/triton_tests/test_softmax.py",                  1, "memory_bound"),
    _RegistryEntry("mla_decode_rope",  "pytorch",      "aiter/op_tests/triton_tests/attention/test_mla_decode_rope.py",                 "aiter", "ref_compute",         "",                        "",                    "",                                                                              3, "memory_bound"),
]


def _build_manual_registry() -> None:
    """Populate MANUAL_REGISTRY from _REGISTRY_ENTRIES.

    For 'pytorch' mode entries, we extract the actual reference function
    source from the library test file (no synthetic code).  If the file
    or function cannot be found, we downgrade to 'library_test' mode.
    """
    for entry in _REGISTRY_ENTRIES:
        repo_url = REPO_URLS.get(entry.library, "")
        mode = entry.mode
        test_cmd = entry.test_cmd

        if mode == "pytorch" and entry.ref_func:
            filepath = ROCM_DIR / entry.source_file
            ref_code = _extract_function_source(filepath, entry.ref_func) if filepath.exists() else None
            inp_code = _extract_function_source(filepath, entry.input_func) if (filepath.exists() and entry.input_func) else None

            if ref_code and len(ref_code.strip()) >= 10:
                MANUAL_REGISTRY[entry.kernel_type] = GroundTruthSpec(
                    kernel_type=entry.kernel_type,
                    mode="pytorch",
                    pytorch_reference_code=ref_code.strip(),
                    test_shapes_code=(inp_code or "").strip(),
                    source_file=entry.source_file,
                    source_library=entry.library,
                    difficulty_level=entry.difficulty,
                    op_type=entry.op_type,
                    ref_func_name=entry.ref_func,
                    sol_entry_func=entry.sol_entry_func,
                    input_func_name=entry.input_func,
                )
                continue

            logger.info(
                "kernel_type=%s: could not extract ref function '%s' from %s; "
                "downgrading to library_test mode",
                entry.kernel_type, entry.ref_func, entry.source_file,
            )
            mode = "library_test"
            if not test_cmd:
                test_cmd = f"python -m pytest {entry.source_file}"

        MANUAL_REGISTRY[entry.kernel_type] = GroundTruthSpec(
            kernel_type=entry.kernel_type,
            mode="library_test",
            repo_url=repo_url,
            unit_test_command=test_cmd or f"python -m pytest {entry.source_file}",
            source_file=entry.source_file,
            source_library=entry.library,
            difficulty_level=entry.difficulty,
            op_type=entry.op_type,
        )


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
    ck_build_bin = rocm_dir / "composable_kernel" / "build" / "bin"
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

            binary_name = f"example_{name}"

            # Prefer absolute path to built binary if it exists
            built_binary = ck_build_bin / binary_name
            is_built = built_binary.exists() and built_binary.stat().st_mode & 0o111
            ref_binary_path = str(built_binary) if is_built else f"build/bin/{binary_name}"

            accordo_config = {
                "correctness": {
                    "backend": "accordo",
                    "accordo": {
                        "kernel_name": kernel_name,
                        "reference_binary": ref_binary_path,
                        "optimized_binary": f"build/bin/{binary_name}_opt",
                        "tolerance": 0.001,
                        "timeout_seconds": 60,
                        "working_directory": str(rocm_dir / "composable_kernel") if is_built else "${CK_HOME}",
                        "built": is_built,
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


def build_correctness_config(
    gt_spec: Optional[GroundTruthSpec],
    rocm_root: Optional[Path] = None,
) -> tuple[dict, str]:
    """Build a correctness config dict from a GroundTruthSpec.

    Returns (correctness_cfg, resolved_mode) where resolved_mode is the
    effective mode after any fallbacks (e.g. pytorch -> library_test when
    no reference code is available).

    This is the single source of truth for correctness config construction,
    used by config_generator, _create_task_config, and _create_standalone_task_config.
    """
    rocm_root = rocm_root or ROCM_DIR

    if gt_spec is None:
        return {"mode": "pytorch", "tolerance": 1e-3, "num_tests": 10}, "pytorch"

    mode = gt_spec.mode

    if mode == "accordo" and gt_spec.accordo_config:
        accordo_inner = gt_spec.accordo_config.get("correctness", {})
        return {
            "mode": "accordo",
            "backend": "accordo",
            "accordo": accordo_inner.get("accordo", {}),
        }, "accordo"

    if mode == "accordo" and not gt_spec.accordo_config:
        logger.warning(
            "kernel_type=%s: mode is 'accordo' but accordo_config is missing; "
            "falling through to library_test/pytorch",
            gt_spec.kernel_type,
        )

    if mode == "library_test" and gt_spec.unit_test_command:
        lib_dir = str(rocm_root / gt_spec.source_library) if gt_spec.source_library else ""
        return {
            "mode": "library_test",
            "unit_test_command": gt_spec.unit_test_command,
            "repo_url": gt_spec.repo_url,
            "working_directory": lib_dir,
        }, "library_test"

    if mode == "library_test" and not gt_spec.unit_test_command:
        logger.warning(
            "kernel_type=%s: mode is 'library_test' but unit_test_command is empty; "
            "falling through to pytorch",
            gt_spec.kernel_type,
        )

    if mode == "pytorch" and gt_spec.pytorch_reference_code:
        return {"mode": "pytorch", "tolerance": 1e-3, "num_tests": 10}, "pytorch"

    if mode == "pytorch" and not gt_spec.pytorch_reference_code:
        logger.warning(
            "kernel_type=%s: mode is 'pytorch' but no reference code available; "
            "attempting downgrade to library_test",
            gt_spec.kernel_type,
        )

    if gt_spec.unit_test_command or gt_spec.source_file:
        lib_dir = str(rocm_root / gt_spec.source_library) if gt_spec.source_library else ""
        test_cmd = gt_spec.unit_test_command or f"python -m pytest {gt_spec.source_file}"
        return {
            "mode": "library_test",
            "unit_test_command": test_cmd,
            "repo_url": gt_spec.repo_url,
            "working_directory": lib_dir,
        }, "library_test"

    return {"mode": "pytorch", "tolerance": 1e-3, "num_tests": 10}, "pytorch"


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
