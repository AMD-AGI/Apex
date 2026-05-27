import py_compile
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from kernel_tracing.mode_detection import detect_trace_mode
from kernel_tracing.overlay import infer_module_mapping, overlay_path_for
from kernel_tracing.patch_triton import patch_triton_launch_file
from kernel_tracing.patch_wrapper import patch_aiter_compile_ops_file, patch_wrapper_entry_file
from kernel_tracing.test_cases import TRACE_TEST_CASES


def test_required_case_matrix_has_30_cases():
    assert len(TRACE_TEST_CASES) == 30
    for repo in ("aiter", "vllm", "sglang"):
        assert sum(1 for c in TRACE_TEST_CASES if c.repo == repo and c.kind == "triton") == 5
        assert sum(1 for c in TRACE_TEST_CASES if c.repo == repo and c.kind == "hip") == 5


@pytest.mark.parametrize(
    "case",
    TRACE_TEST_CASES,
    ids=lambda c: f"{c.repo}-{c.kind}-{c.target}",
)
def test_required_repo_case_patchability(case, tmp_path):
    source = REPO_ROOT / case.file
    assert source.exists(), case.file
    mode = detect_trace_mode(source, case.target, "auto")
    if not case.static_expected:
        assert mode == "agent"
        return

    if case.patch_path == "aiter-compile-ops":
        module_name = "aiter.jit.core"
        package_rel_path = "aiter/jit/core.py"
        source = REPO_ROOT / "tools/rocm/aiter/aiter/jit/core.py"
    else:
        module_name, package_rel_path = infer_module_mapping(source, REPO_ROOT)
    output = overlay_path_for(tmp_path / "patched_files", package_rel_path)
    if case.patch_path == "triton-launch":
        result = patch_triton_launch_file(
            source_path=source,
            output_path=output,
            kernel_name=case.target,
            module_name=module_name,
            package_rel_path=package_rel_path,
        )
    elif case.patch_path == "aiter-compile-ops":
        result = patch_aiter_compile_ops_file(
            source_path=source,
            output_path=output,
            trace_kind="hip_python_op",
            module_name=module_name,
            package_rel_path=package_rel_path,
        )
    else:
        kind = {
            "vllm-custom-op": "vllm_python_op",
            "sglang-custom-op": "sglang_python_op",
        }[case.patch_path]
        result = patch_wrapper_entry_file(
            source_path=source,
            output_path=output,
            kernel_name=case.target,
            trace_kind=kind,
            module_name=module_name,
            package_rel_path=package_rel_path,
        )
    assert result.events
    assert "apex_trace_event" in output.read_text(encoding="utf-8")
    py_compile.compile(str(output), doraise=True)
