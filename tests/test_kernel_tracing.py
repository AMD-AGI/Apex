import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from kernel_tracing.agent_harness import AgentPatchRequest, run_agent_patch_fallback
from kernel_tracing.overlay import ModuleMapping, write_overlay_support
from kernel_tracing.patch_triton import patch_triton_launch_file
from kernel_tracing.postprocess import postprocess_trace
from kernel_tracing.runner import TraceKernelConfig, _trace_event_flags, run_trace_kernel
from kernel_tracing.runtime import write_runtime_file
from kernel_tracing.serializer import serialize_value
from kernel_tracing.patch_wrapper import patch_aiter_compile_ops_file, patch_wrapper_entry_file


def _synthetic_source() -> str:
    return """
class DummyKernel:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return "ok"
        return launch

some_kernel = DummyKernel()

def wrapper(q, k, block, config):
    return some_kernel[(q.shape[0], block)](
        q,
        key=k,
        BLOCK_SIZE=block,
        **config,
    )
"""


def test_serialize_tensor_metadata_cpu():
    torch = pytest.importorskip("torch")
    x = torch.empty((2, 3, 4), dtype=torch.bfloat16).transpose(0, 1)
    out = serialize_value(x)
    assert out["type"] == "tensor"
    assert out["shape"] == [3, 2, 4]
    assert out["dtype"] == "torch.bfloat16"
    assert out["stride"] == list(x.stride())
    assert "data_ptr_hash" in out
    assert "values" not in out


def test_patch_synthetic_triton_launch_compiles(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text(_synthetic_source(), encoding="utf-8")
    out = tmp_path / "patched" / "mod.py"
    result = patch_triton_launch_file(
        source_path=src,
        output_path=out,
        kernel_name="some_kernel",
        module_name="mod",
        package_rel_path="mod.py",
    )
    text = out.read_text(encoding="utf-8")
    assert "apex_trace_event" in text
    assert "some_kernel" in text
    assert result.events[0]["kind"] == "triton_launch"
    compile(text, str(out), "exec")


def test_patch_nested_branch_inserts_inside_branch(tmp_path):
    src = tmp_path / "branch_mod.py"
    src.write_text("""
class DummyKernel:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return "ok"
        return launch

some_kernel = DummyKernel()

def wrapper(q, flag):
    if flag:
        config = {"BLOCK": 1}
        some_kernel[(q.shape[0],)](q, **config)
""", encoding="utf-8")
    out = tmp_path / "patched" / "branch_mod.py"
    patch_triton_launch_file(
        source_path=src,
        output_path=out,
        kernel_name="some_kernel",
        module_name="branch_mod",
        package_rel_path="branch_mod.py",
    )
    text = out.read_text(encoding="utf-8")
    event_idx = text.index("apex_trace_event", text.index("config ="))
    assert text.index("config =") < event_idx
    assert event_idx < text.index("some_kernel[")
    compile(text, str(out), "exec")


def test_patch_aiter_compile_ops_central_hook(tmp_path):
    src = tmp_path / "core.py"
    src.write_text("""
def torch_compile_guard(**_kwargs):
    def deco(fn):
        return fn
    return deco

def compile_ops(_md_name, fc_name=None, ffi_type="pybind", develop=False):
    def decorator(func):
        loadName = fc_name if fc_name is not None else func.__name__
        if ffi_type == "ctypes":
            def ctypes_wrapper(*args, **kwargs):
                return ctypes_caller(*args, **kwargs)
            return ctypes_wrapper
        elif ffi_type == "pybind":
            def wrapper(*args, custom_build_args={}, **kwargs):
                return op(*args, **kwargs)
            return wrapper
    return decorator
""", encoding="utf-8")
    out = tmp_path / "patched" / "aiter" / "jit" / "core.py"
    result = patch_aiter_compile_ops_file(
        source_path=src,
        output_path=out,
        trace_kind="hip_python_op",
        module_name="aiter.jit.core",
        package_rel_path="aiter/jit/core.py",
    )
    text = out.read_text(encoding="utf-8")
    assert "compile_ops.ctypes_wrapper" in text
    assert "compile_ops.pybind_wrapper" in text
    assert "kernel_name=_apex_trace_load_name" in text
    assert "'develop': _apex_trace_develop" in text
    assert len(result.events) == 2
    compile(text, str(out), "exec")


def test_patch_aiter_compile_ops_handles_wrapper_local_load_name(tmp_path, monkeypatch):
    src = tmp_path / "core.py"
    src.write_text("""
import functools

def op(*args, **kwargs):
    return "ok"

def compile_ops(_md_name, fc_name=None, gen_func=None, gen_fake=None):
    def decorator(func):
        func.arg_checked = False

        @functools.wraps(func)
        def wrapper(*args, custom_build_args={}, **kwargs):

            loadName = fc_name
            md_name = _md_name
            if fc_name is None:
                loadName = func.__name__
            return op(*args, **kwargs)

        return wrapper
    return decorator
""", encoding="utf-8")
    out = tmp_path / "patched" / "aiter" / "jit" / "core.py"
    result = patch_aiter_compile_ops_file(
        source_path=src,
        output_path=out,
        trace_kind="hip_python_op",
        module_name="aiter.jit.core",
        package_rel_path="aiter/jit/core.py",
    )
    text = out.read_text(encoding="utf-8")
    assert "kernel_name=loadName" not in text
    assert "kernel_name=_apex_trace_load_name" in text
    assert len(result.events) == 1
    compile(text, str(out), "exec")

    events = []
    runtime = type(sys)("apex_kernel_tracing_runtime")
    runtime.apex_trace_event = lambda **event: events.append(event)
    monkeypatch.setitem(sys.modules, "apex_kernel_tracing_runtime", runtime)

    namespace = {"__file__": str(out)}
    exec(compile(text, str(out), "exec"), namespace)

    def moe_sorting_fwd():
        return None

    wrapped = namespace["compile_ops"]("module_moe_sorting")(moe_sorting_fwd)
    assert wrapped("token_ids", dispatch_policy=0) == "ok"
    assert events[0]["kernel_name"] == "moe_sorting_fwd"
    assert events[0]["extra"]["load_name"] == "moe_sorting_fwd"
    assert events[0]["extra"]["develop"] is False


def test_patch_aiter_compile_ops_skips_uncheckable_annotations(tmp_path, monkeypatch):
    src = tmp_path / "core.py"
    src.write_text("""
import inspect
import types
import typing

def op(*args, **kwargs):
    return "ok"

def compile_ops(_md_name, fc_name=None, gen_func=None, gen_fake=None):
    def decorator(func):
        func.arg_checked = False

        def wrapper(*args, custom_build_args={}, **kwargs):
            loadName = fc_name
            if fc_name is None:
                loadName = func.__name__

            def check_args():
                sig = inspect.signature(func)
                func.__signature__ = sig
                ann = {k: v.annotation for k, v in sig.parameters.items()}
                ann["return"] = sig.return_annotation
                callargs = inspect.getcallargs(func, *args, **kwargs)
                enum_types = []
                for el, arg in callargs.items():
                    expected_type = ann[el]
                    got_type = type(arg)
                    origin = typing.get_origin(expected_type)
                    sub_t = typing.get_args(expected_type)

                    if origin is None:
                        if not isinstance(arg, expected_type) and not (
                            any(el in str(expected_type) for el in enum_types)
                            and isinstance(arg, int)
                        ):
                            raise TypeError(
                                f"{loadName}: {el} needs to be {expected_type} but got {got_type}"
                            )
                    elif origin is list:
                        if not isinstance(arg, list):
                            raise TypeError(
                                f"{loadName}: {el} needs to be List[{sub_t}] but got {arg}"
                            )
                    elif origin is typing.Union or origin is types.UnionType:
                        if arg is not None and not isinstance(arg, sub_t):
                            raise TypeError(
                                f"{loadName}: {el} needs to be Optional[{sub_t}] but got {arg}"
                            )
                    else:
                        raise TypeError(f"Unsupported type: {expected_type}")
                return True

            func.arg_checked = check_args()
            return op(*args, **kwargs)

        return wrapper
    return decorator
""", encoding="utf-8")
    out = tmp_path / "patched" / "aiter" / "jit" / "core.py"
    patch_aiter_compile_ops_file(
        source_path=src,
        output_path=out,
        trace_kind="hip_python_op",
        module_name="aiter.jit.core",
        package_rel_path="aiter/jit/core.py",
    )
    text = out.read_text(encoding="utf-8")
    assert "expected_type is inspect._empty" in text
    assert "not isinstance(expected_type, type)" in text
    compile(text, str(out), "exec")

    events = []
    runtime = type(sys)("apex_kernel_tracing_runtime")
    runtime.apex_trace_event = lambda **event: events.append(event)
    monkeypatch.setitem(sys.modules, "apex_kernel_tracing_runtime", runtime)

    namespace = {"__file__": str(out)}
    exec(compile(text, str(out), "exec"), namespace)

    def rmsnorm2d_fwd(x):
        return x

    wrapped = namespace["compile_ops"]("module_rmsnorm")(rmsnorm2d_fwd)
    assert wrapped("tensor") == "ok"
    assert events[0]["kernel_name"] == "rmsnorm2d_fwd"


def test_patch_wrapper_trace_all_instruments_top_level_functions(tmp_path):
    src = tmp_path / "custom_ops.py"
    src.write_text("""
def first(x):
    return x

def second(x):
    return x

class Helper:
    def method(self, x):
        return x

def outer(x):
    def nested(y):
        return y
    return nested(x)
""", encoding="utf-8")
    out = tmp_path / "patched" / "custom_ops.py"
    result = patch_wrapper_entry_file(
        source_path=src,
        output_path=out,
        kernel_name="first",
        trace_kind="vllm_python_op",
        module_name="custom_ops",
        package_rel_path="custom_ops.py",
        trace_all=True,
    )
    names = {event["kernel_name"] for event in result.events}
    assert names == {"first", "second", "outer"}
    text = out.read_text(encoding="utf-8")
    assert "kernel_name='second'" in text
    assert "kernel_name='method'" not in text
    assert "kernel_name='nested'" not in text
    compile(text, str(out), "exec")


def test_trace_event_flags_separate_any_and_target(tmp_path):
    raw = tmp_path / "trace_raw"
    raw.mkdir()
    events = [
        {"kind": "module_import", "kernel_name": "target"},
        {"kind": "vllm_python_op", "kernel_name": "other"},
    ]
    (raw / "trace_pid1_rank0.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    assert _trace_event_flags(tmp_path, "target") == {
        "any_event_found": True,
        "target_event_found": False,
    }


def test_local_overlay_import_smoke(tmp_path):
    results = tmp_path / "results"
    patched_dir = results / "patched_files"
    write_runtime_file(patched_dir)
    patched = patched_dir / "overlay" / "mod.py"
    patched.parent.mkdir(parents=True)
    patched.write_text(
        "from apex_kernel_tracing_runtime import apex_trace_event\n"
        "apex_trace_event(kind='module_import', kernel_name='k', source_file=__file__, line=1)\n",
        encoding="utf-8",
    )
    write_overlay_support(
        results_dir=results,
        mappings=[ModuleMapping("mod", "mod.py", tmp_path / "mod.py", patched)],
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{patched_dir}:{env.get('PYTHONPATH', '')}"
    env["APEX_TRACE_ENABLED"] = "1"
    env["APEX_TRACE_OUTPUT_DIR"] = str(results / "trace_raw")
    env["APEX_TRACE_KERNEL_NAME"] = "k"
    subprocess.run([sys.executable, "-c", "import mod"], env=env, check=True)
    raw = "\n".join(p.read_text() for p in (results / "trace_raw").glob("*.jsonl"))
    assert '"kind": "module_import"' in raw


def test_overlay_import_preserves_package_sibling_utils(tmp_path):
    site_root = tmp_path / "site"
    real_pkg = site_root / "pkg" / "jit"
    real_utils = real_pkg / "utils"
    real_utils.mkdir(parents=True)
    (site_root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (real_pkg / "__init__.py").write_text("", encoding="utf-8")
    (real_utils / "chip_info.py").write_text("VALUE = 'real-utils'\n", encoding="utf-8")
    real_core = real_pkg / "core.py"
    real_core.write_text("from chip_info import VALUE\n", encoding="utf-8")

    results = tmp_path / "results"
    patched_dir = results / "patched_files"
    write_runtime_file(patched_dir)
    patched = patched_dir / "overlay" / "pkg" / "jit" / "core.py"
    patched.parent.mkdir(parents=True)
    patched.write_text("from chip_info import VALUE\n", encoding="utf-8")
    write_overlay_support(
        results_dir=results,
        mappings=[ModuleMapping("pkg.jit.core", "pkg/jit/core.py", real_core, patched)],
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{patched_dir}:{site_root}:{env.get('PYTHONPATH', '')}"
    env["APEX_TRACE_ENABLED"] = "1"
    env["APEX_TRACE_OUTPUT_DIR"] = str(results / "trace_raw")
    env["APEX_TRACE_KERNEL_NAME"] = "k"
    env["APEX_TRACE_ALLOW_PACKAGE_PATH_INSERT"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import pathlib; import pkg.jit.core as c; "
                "assert c.VALUE == 'real-utils'; "
                f"assert pathlib.Path(c.__file__) == pathlib.Path({str(real_core)!r})"
            ),
        ],
        env=env,
        check=True,
    )
    raw = "\n".join(p.read_text() for p in (results / "trace_raw").glob("*.jsonl"))
    assert '"module_name": "pkg.jit.core"' in raw


def test_module_import_bypasses_sampling_and_max_records(tmp_path):
    results = tmp_path / "results"
    patched_dir = results / "patched_files"
    write_runtime_file(patched_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{patched_dir}:{env.get('PYTHONPATH', '')}"
    env["APEX_TRACE_ENABLED"] = "1"
    env["APEX_TRACE_OUTPUT_DIR"] = str(results / "trace_raw")
    env["APEX_TRACE_KERNEL_NAME"] = "target_kernel"
    env["APEX_TRACE_MAX_RECORDS"] = "0"
    env["APEX_TRACE_SAMPLE_RATE"] = "0"
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from apex_kernel_tracing_runtime import apex_trace_event\n"
                "apex_trace_event(kind='module_import', kernel_name='other', "
                "source_file='x.py', line=1)\n"
                "apex_trace_event(kind='triton_launch', kernel_name='target_kernel', "
                "source_file='x.py', line=2)\n"
            ),
        ],
        env=env,
        check=True,
    )
    raw = "\n".join(p.read_text() for p in (results / "trace_raw").glob("*.jsonl"))
    assert '"kind": "module_import"' in raw
    assert '"kind": "triton_launch"' not in raw


def test_run_trace_kernel_run_cmd_smoke(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text(_synthetic_source(), encoding="utf-8")
    script = tmp_path / "smoke.py"
    script.write_text("""
class T:
    shape = (2, 3)
    dtype = "fake"
    device = "cpu"
    layout = "strided"
    requires_grad = False
    def stride(self): return (3, 1)
    def is_contiguous(self): return True
    def numel(self): return 6
    def element_size(self): return 4
    def data_ptr(self): return 123

import mod
mod.wrapper(T(), T(), 64, {"EXTRA": True})
""", encoding="utf-8")
    cmd = f"{sys.executable} {script}"
    result = run_trace_kernel(TraceKernelConfig(
        results_dir=tmp_path / "results",
        kernel_name="some_kernel",
        kernel_file=src,
        run_cmd=cmd,
        max_records=10,
        repo_root=tmp_path,
    ))
    assert result["success"] is True
    raw = (tmp_path / "results" / "trace_raw.jsonl").read_text(encoding="utf-8")
    assert "some_kernel" in raw
    ranges = json.loads((tmp_path / "results" / "workload_ranges.json").read_text())
    assert ranges["total_calls"] == 1


def test_run_trace_kernel_aiter_mode_patches_compile_ops_core(tmp_path):
    core = tmp_path / "tools" / "rocm" / "aiter" / "aiter" / "jit" / "core.py"
    core.parent.mkdir(parents=True)
    core.write_text("""
def compile_ops(_md_name, fc_name=None, ffi_type="pybind", develop=False):
    def decorator(func):
        loadName = fc_name if fc_name is not None else func.__name__
        if ffi_type == "ctypes":
            def ctypes_wrapper(*args, **kwargs):
                return ctypes_caller(*args, **kwargs)
            return ctypes_wrapper
        elif ffi_type == "pybind":
            def wrapper(*args, custom_build_args={}, **kwargs):
                return op(*args, **kwargs)
            return wrapper
    return decorator
""", encoding="utf-8")
    wrapper = tmp_path / "tools" / "rocm" / "aiter" / "aiter" / "ops" / "moe_op.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("def fmoe(): pass\n", encoding="utf-8")
    result = run_trace_kernel(TraceKernelConfig(
        results_dir=tmp_path / "results",
        kernel_name="fmoe",
        kernel_file=wrapper,
        trace_mode="aiter-compile-ops",
        dry_run=True,
        repo_root=tmp_path,
    ))
    assert result["mode"] == "aiter-compile-ops"
    assert result["patched_file"].endswith("patched_files/overlay/aiter/jit/core.py")
    manifest = json.loads(
        (tmp_path / "results" / "patched_files" / "patch_manifest.json").read_text()
    )
    assert "aiter.jit.core" in manifest["overlay_modules"]


def test_postprocess_shape_ranges(tmp_path):
    raw = tmp_path / "trace_raw"
    raw.mkdir()
    events = [
        {"kind": "triton_launch", "kernel_name": "k", "kwargs": {"q": {"type": "tensor", "shape": [1, 64], "dtype": "torch.bfloat16", "layout": "torch.strided"}}, "source_file": "x.py", "line": 1},
        {"kind": "triton_launch", "kernel_name": "k", "kwargs": {"q": {"type": "tensor", "shape": [8, 64], "dtype": "torch.bfloat16", "layout": "torch.strided"}}, "source_file": "x.py", "line": 1},
    ]
    (raw / "trace_pid1_rank0.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n",
        encoding="utf-8",
    )
    result = postprocess_trace(tmp_path)
    assert result["total_calls"] == 2
    shape_ranges = result["groups"][0]["shape_ranges"]
    assert shape_ranges["q.shape.0"] == {"min": 1, "max": 8}
    assert shape_ranges["q.shape.1"] == {"min": 64, "max": 64}
    summary = (tmp_path / "workload_summary.md").read_text(encoding="utf-8")
    assert "| Field | Value |" in summary
    assert "| Tensor | Dim | Min | Max |" in summary
    assert "| `q` | 0 | 1 | 8 |" in summary


def test_agent_fallback_uses_existing_backend(monkeypatch, tmp_path):
    def fake_run_agent_task(**kwargs):
        manifest = kwargs["solution_path"]
        manifest.parent.mkdir(parents=True, exist_ok=True)
        patched = tmp_path / "patched_files" / "overlay" / "x.py"
        patched.parent.mkdir(parents=True)
        patched.write_text("apex_trace_event(kind='module_import', kernel_name='x', source_file=__file__, line=1)\n")
        manifest.write_text(json.dumps({
            "strategy": "agent",
            "patched_files": [{"patched_file": str(patched)}],
            "expected_events": [],
        }))
        return [], True

    monkeypatch.setattr("agents.backends.run_agent_task", fake_run_agent_task)
    monkeypatch.setattr("agents.backends.resolve_default_model", lambda agent: "mock-model")
    src = tmp_path / "source.py"
    src.write_text("def f(): pass\n", encoding="utf-8")
    manifest = run_agent_patch_fallback(AgentPatchRequest(
        results_dir=tmp_path,
        apex_root=REPO_ROOT,
        kernel_name="x",
        kernel_file=src,
        trace_mode="agent",
        agent_backend="codex",
    ))
    assert manifest["strategy"] == "agent"
