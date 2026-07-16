import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

import kernel_tracing.patch_triton as patch_triton
from kernel_tracing.agent_harness import AgentPatchRequest, run_agent_patch_fallback
from kernel_tracing.mode_detection import detect_trace_mode, normalize_trace_mode
from kernel_tracing.overlay import ModuleMapping, write_docker_wrapper, write_overlay_support
from kernel_tracing.patch_triton import patch_triton_launch_file
from kernel_tracing.postprocess import postprocess_trace
from kernel_tracing.registry import VLLM_TRACE_IMAGE
from kernel_tracing.runner import (
    TraceKernelConfig,
    TraceKernelTarget,
    _base_trace_env,
    _merge_benchmark_envs,
    _prepare_no_cudagraph_benchmark_script,
    _rewrite_benchmark_script_disable_cuda_graph,
    _source_for_patch,
    _temporary_env,
    _trace_event_flags,
    run_trace_kernel,
)
from kernel_tracing.runtime import RUNTIME_SOURCE, write_runtime_file
from kernel_tracing.serializer import runtime_serializer_source, serialize_value
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


def _synthetic_two_kernel_source() -> str:
    return """
class DummyKernel:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return "ok"
        return launch

first_kernel = DummyKernel()
second_kernel = DummyKernel()

def wrapper(q, k, block):
    first_kernel[(q.shape[0], block)](q, key=k, BLOCK_SIZE=block)
    second_kernel[(k.shape[0], block)](k, query=q, BLOCK_SIZE=block)
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


def test_serialize_fake_tensor_metadata_without_torch():
    class MockTensor:
        shape = (2, 3)
        dtype = "float16"
        device = "cuda:0"
        layout = "strided"
        requires_grad = True

        def stride(self):
            return (3, 1)

        def is_contiguous(self):
            return False

        def numel(self):
            return 6

        def element_size(self):
            return 2

        def data_ptr(self):
            return 123456

    out = serialize_value(MockTensor())
    assert out["type"] == "tensor"
    assert out["shape"] == [2, 3]
    assert out["dtype"] == "float16"
    assert out["device"] == "cuda:0"
    assert out["stride"] == [3, 1]
    assert out["is_contiguous"] is False
    assert out["numel"] == 6
    assert out["element_size"] == 2
    assert out["data_ptr_hash"] != "123456"


def test_serialize_tensor_metadata_tolerates_access_errors():
    class FlakyTensor:
        shape = (4,)
        dtype = "float32"

        def stride(self):
            raise RuntimeError("stride unavailable")

        def is_contiguous(self):
            raise RuntimeError("contiguity unavailable")

        def numel(self):
            raise RuntimeError("numel unavailable")

        def element_size(self):
            raise RuntimeError("element size unavailable")

        def data_ptr(self):
            raise RuntimeError("data pointer unavailable")

    out = serialize_value(FlakyTensor())
    assert out["type"] == "tensor"
    assert out["stride"] == []
    assert out["is_contiguous"] is False
    assert "numel" not in out
    assert "element_size" not in out
    assert "data_ptr_hash" not in out


def test_serialize_proxy_values_are_skipped_before_tensor_detection():
    class ProxyTensor:
        shape = (8,)
        dtype = "float16"

        def stride(self):
            return (1,)

    ProxyTensor.__module__ = "torch.fx.proxy"
    out = serialize_value(ProxyTensor())
    assert out == {
        "type": "ProxyTensor",
        "module": "torch.fx.proxy",
        "skipped": "torch_tracing_proxy",
    }


def test_serialize_collections_truncates_and_preserves_shapes():
    wide = {f"k{i}": i for i in range(35)}
    out = serialize_value({
        "wide": wide,
        "tuple": (1, "x"),
        "list": [1, 2],
        "callable": lambda: None,
        "bytes": b"abc",
        "deep": [[[[["too deep"]]]]],
    })
    assert out["wide"]["_truncated"] == 3
    assert "k31" in out["wide"]
    assert "k32" not in out["wide"]
    assert out["tuple"] == {"type": "tuple", "items": [1, "x"]}
    assert out["list"] == [1, 2]
    assert out["callable"]["type"] == "callable"
    assert out["bytes"]["type"] == "bytes"
    assert out["deep"][0][0][0] == {"type": "list", "truncated": True}


def test_runtime_source_embeds_shared_serializer():
    assert runtime_serializer_source().strip() in RUNTIME_SOURCE
    namespace = {"__file__": "apex_kernel_tracing_runtime.py"}
    exec(compile(RUNTIME_SOURCE.lstrip(), "apex_kernel_tracing_runtime.py", "exec"), namespace)
    sample = {"items": (1, [2, "x"])}
    assert namespace["serialize_value"](sample) == serialize_value(sample)
    assert "_serialize" not in namespace


def test_triton_patch_module_has_single_entrypoint():
    source = Path(patch_triton.__file__).read_text(encoding="utf-8")
    parsed = ast.parse(source)
    defs = [node.name for node in parsed.body if isinstance(node, ast.FunctionDef)]
    assert defs.count("patch_triton_launch_file") == 1


def test_normalize_trace_mode_handles_kernel_type_and_invalid_values():
    assert normalize_trace_mode("AUTO", kernel_type="triton") == "triton-launch"
    assert normalize_trace_mode("auto", kernel_type="hip") == "auto"
    assert normalize_trace_mode(" vllm-custom-op ") == "vllm-custom-op"
    with pytest.raises(ValueError, match="Unsupported trace mode"):
        normalize_trace_mode("unknown")


@pytest.mark.parametrize(
    ("path_parts", "source", "kernel_name", "expected"),
    [
        (("mod.py",), "some_kernel[(1,)]()", "some_kernel", "triton-launch"),
        (("mod.py",), "@triton.jit\ndef some_kernel():\n    pass\n", "some_kernel", "agent"),
        (("aiter", "jit", "core.py"), "def compile_ops():\n    pass\n", "moe", "aiter-compile-ops"),
        (("vllm", "_custom_ops.py"), "torch.ops.vllm.reshape_and_cache_flash()\n", "op", "vllm-custom-op"),
        (("sglang", "ops.py"), "register_custom_op('x')\n", "op", "sglang-custom-op"),
        (("plain.py",), "def wrapper():\n    return 1\n", "wrapper", "agent"),
    ],
)
def test_detect_trace_mode_auto_patterns(tmp_path, path_parts, source, kernel_name, expected):
    path = tmp_path.joinpath(*path_parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    assert detect_trace_mode(path, kernel_name) == expected


def test_detect_trace_mode_keeps_explicit_request_and_generic_alias(tmp_path):
    path = tmp_path / "mod.py"
    path.write_text("kernel[(1,)]()\n", encoding="utf-8")
    assert detect_trace_mode(path, "whatever", requested="sglang-custom-op") == "sglang-custom-op"
    assert detect_trace_mode(path, "kernel") == "agent"


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
        "any_target_event_found": False,
        "partial_coverage": False,
        "target_event_found": False,
    }


def test_trace_event_flags_tracks_multiple_targets_and_load_names(tmp_path):
    raw = tmp_path / "trace_raw"
    raw.mkdir()
    (raw / "trace_pid1_rank0.jsonl").write_text(
        "\n".join([
            "{not json",
            json.dumps({"kind": "module_import", "kernel_name": "ignored"}),
            json.dumps({"kind": "triton_launch", "kernel_name": "first"}),
            json.dumps({
                "kind": "hip_python_op",
                "kernel_name": "wrapper",
                "extra": {"load_name": "second"},
            }),
        ]) + "\n",
        encoding="utf-8",
    )

    assert _trace_event_flags(tmp_path, ["first", "second"]) == {
        "any_event_found": True,
        "any_target_event_found": True,
        "partial_coverage": False,
        "target_event_found": True,
    }
    details = _trace_event_flags(
        tmp_path,
        ["first", "second", "missing"],
        include_details=True,
    )
    assert details["any_target_event_found"] is True
    assert details["partial_coverage"] is True
    assert details["target_event_found"] is False
    assert details["target_events_found"] == {
        "first": True,
        "second": True,
        "missing": False,
    }
    assert details["missing_kernel_names"] == ["missing"]


def test_base_trace_env_uses_deduped_targets_and_trace_all(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "existing")
    kernel = tmp_path / "kernel.py"
    config = TraceKernelConfig(
        results_dir=tmp_path / "results",
        kernel_name="first",
        kernel_file=kernel,
        max_records=7,
        sample_rate=0.25,
        small_tensor_stats=True,
        targets=[
            TraceKernelTarget(kernel_name="first", kernel_file=kernel),
            TraceKernelTarget(kernel_name="second", kernel_file=kernel),
            TraceKernelTarget(kernel_name="first", kernel_file=kernel),
        ],
    )

    env = _base_trace_env(config, docker=False)
    assert env["APEX_TRACE_KERNEL_NAME"] == "first,second"
    assert env["APEX_TRACE_KERNEL_NAMES"] == "first,second"
    assert env["APEX_TRACE_MAX_RECORDS"] == "7"
    assert env["APEX_TRACE_SAMPLE_RATE"] == "0.25"
    assert env["APEX_TRACE_SMALL_TENSOR_STATS"] == "1"
    assert env["APEX_TRACE_OUTPUT_DIR"] == str(tmp_path / "results" / "trace_raw")
    assert env["PYTHONPATH"] == f"{tmp_path / 'results' / 'patched_files'}:existing"

    docker_env = _base_trace_env(config, docker=True)
    assert docker_env["APEX_TRACE_PATCH_MANIFEST"] == "/apex_trace/patched_files/patch_manifest.json"
    assert docker_env["APEX_TRACE_OUTPUT_DIR"] == "/apex_trace/trace_raw"

    config.trace_all = True
    trace_all_env = _base_trace_env(config, docker=False)
    assert trace_all_env["APEX_TRACE_KERNEL_NAME"] == ""
    assert trace_all_env["APEX_TRACE_KERNEL_NAMES"] == ""


def test_merge_benchmark_envs_pins_docker_image_and_preserves_existing_envs(tmp_path):
    bench = tmp_path / "bench.yaml"
    bench.write_text(
        "benchmark:\n"
        "  envs:\n"
        "    PYTHONPATH: /existing/path\n"
        "    KEEP_ME: keep\n"
        "  docker_image: vllm/vllm-openai-rocm:nightly\n",
        encoding="utf-8",
    )
    config = TraceKernelConfig(
        results_dir=tmp_path / "results",
        kernel_name="target",
        kernel_file=tmp_path / "kernel.py",
        max_records=3,
        docker_image=VLLM_TRACE_IMAGE,
    )

    out = _merge_benchmark_envs(str(bench), config, docker=True)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["benchmark"]["docker_image"] == VLLM_TRACE_IMAGE
    envs = data["benchmark"]["envs"]
    assert envs["KEEP_ME"] == "keep"
    assert envs["PYTHONPATH"] == "/apex_trace/patched_files:/existing/path"
    assert envs["APEX_TRACE_ENABLED"] == "1"
    assert envs["APEX_TRACE_KERNEL_NAMES"] == "target"
    assert envs["APEX_TRACE_MAX_RECORDS"] == "3"

    original = yaml.safe_load(bench.read_text(encoding="utf-8"))
    assert original["benchmark"]["docker_image"] == "vllm/vllm-openai-rocm:nightly"

    local_out = _merge_benchmark_envs(str(bench), config, docker=False)
    local_data = yaml.safe_load(local_out.read_text(encoding="utf-8"))
    assert local_data["benchmark"]["docker_image"] == "vllm/vllm-openai-rocm:nightly"


def test_temporary_env_restores_original_environment(monkeypatch):
    monkeypatch.setenv("APEX_TRACE_TEST_KEEP", "old")
    monkeypatch.delenv("APEX_TRACE_TEST_NEW", raising=False)

    with _temporary_env({
        "APEX_TRACE_TEST_KEEP": "new",
        "APEX_TRACE_TEST_NEW": "value",
    }):
        assert os.environ["APEX_TRACE_TEST_KEEP"] == "new"
        assert os.environ["APEX_TRACE_TEST_NEW"] == "value"

    assert os.environ["APEX_TRACE_TEST_KEEP"] == "old"
    assert "APEX_TRACE_TEST_NEW" not in os.environ


def test_source_for_patch_extracts_container_source(tmp_path):
    class Result:
        returncode = 0
        stderr = ""
        stdout = "/site/pkg/mod.py\n__APEX_SOURCE_BEGIN__\nVALUE = 1\n"

    fallback = tmp_path / "host_mod.py"
    config = TraceKernelConfig(
        results_dir=tmp_path / "results",
        kernel_name="target",
        kernel_file=fallback,
        benchmark_config=str(tmp_path / "bench.yaml"),
    )

    with patch("kernel_tracing.runner._detect_magpie_run_mode", return_value="docker"), \
            patch("kernel_tracing.runner._resolve_benchmark_docker_image", return_value="image:tag"), \
            patch("kernel_tracing.runner.shutil.which", return_value="/usr/bin/docker"), \
            patch("kernel_tracing.runner.subprocess.run", return_value=Result()) as run:
        source = _source_for_patch(config, "pkg.mod", "pkg/mod.py", fallback)

    assert source == tmp_path / "results" / "container_sources" / "pkg" / "mod.py"
    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert source.with_suffix(".py.container_path").read_text(encoding="utf-8") == "/site/pkg/mod.py"
    args = run.call_args.args[0]
    assert args[:5] == ["docker", "run", "--rm", "--entrypoint", "python3"]
    assert args[5] == "image:tag"


def test_source_for_patch_uses_fallback_outside_docker(tmp_path):
    fallback = tmp_path / "host_mod.py"
    config = TraceKernelConfig(
        results_dir=tmp_path / "results",
        kernel_name="target",
        kernel_file=tmp_path / "kernel.py",
    )
    assert _source_for_patch(config, "pkg.mod", "pkg/mod.py", fallback) == fallback


def test_source_for_patch_rejects_prepatched_container_source(tmp_path):
    class Result:
        returncode = 0
        stderr = ""
        stdout = "/site/pkg/mod.py\n__APEX_SOURCE_BEGIN__\napex_trace_event()\n"

    config = TraceKernelConfig(
        results_dir=tmp_path / "results",
        kernel_name="target",
        kernel_file=tmp_path / "kernel.py",
        benchmark_config=str(tmp_path / "bench.yaml"),
    )

    with patch("kernel_tracing.runner._detect_magpie_run_mode", return_value="docker"), \
            patch("kernel_tracing.runner._resolve_benchmark_docker_image", return_value="image:tag"), \
            patch("kernel_tracing.runner.shutil.which", return_value="/usr/bin/docker"), \
            patch("kernel_tracing.runner.subprocess.run", return_value=Result()), \
            pytest.raises(RuntimeError, match="already contains apex_trace_event"):
        _source_for_patch(config, "pkg.mod", "pkg/mod.py")


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


def test_overlay_manifest_uses_absolute_host_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = Path("relative_results")
    patched = results / "patched_files" / "overlay" / "mod.py"
    patched.parent.mkdir(parents=True)
    patched.write_text("VALUE = 1\n", encoding="utf-8")
    source = tmp_path / "source_mod.py"
    source.write_text("VALUE = 0\n", encoding="utf-8")

    manifest_path = write_overlay_support(
        results_dir=results,
        mappings=[ModuleMapping("mod", "mod.py", source, patched)],
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = manifest["patched_files"][0]
    assert Path(manifest["mounts"]["host_results_dir"]).is_absolute()
    assert Path(item["source_file"]).is_absolute()
    assert Path(item["patched_file"]).is_absolute()


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
    target_shapes = json.loads(
        (tmp_path / "results" / "target_kernel_tensor_shapes.json").read_text()
    )
    assert target_shapes["trace_result"]["success"] is True
    assert target_shapes["targets"]["some_kernel"]["events"] == 1


def test_run_trace_kernel_multiple_targets_same_file(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text(_synthetic_two_kernel_source(), encoding="utf-8")
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
mod.wrapper(T(), T(), 64)
""", encoding="utf-8")
    cmd = f"{sys.executable} {script}"
    result = run_trace_kernel(TraceKernelConfig(
        results_dir=tmp_path / "results",
        kernel_name="first_kernel",
        kernel_file=src,
        run_cmd=cmd,
        max_records=10,
        repo_root=tmp_path,
        targets=[
            TraceKernelTarget(
                kernel_name="first_kernel",
                kernel_file=src,
                kernel_id="synthetic.first",
                trace_mode="triton-launch",
                kernel_type="triton",
                patch_strategy="static",
            ),
            TraceKernelTarget(
                kernel_name="second_kernel",
                kernel_file=src,
                kernel_id="synthetic.second",
                trace_mode="triton-launch",
                kernel_type="triton",
                patch_strategy="static",
            ),
        ],
    ))
    assert result["success"] is True
    assert result["target_events_found"] == {
        "first_kernel": True,
        "second_kernel": True,
    }
    raw = (tmp_path / "results" / "trace_raw.jsonl").read_text(encoding="utf-8")
    assert "first_kernel" in raw
    assert "second_kernel" in raw
    ranges = json.loads((tmp_path / "results" / "workload_ranges.json").read_text())
    assert ranges["total_calls"] == 2
    target_shapes = json.loads(
        (tmp_path / "results" / "target_kernel_tensor_shapes.json").read_text()
    )
    assert set(target_shapes["targets"]) == {"first_kernel", "second_kernel"}
    assert target_shapes["targets"]["first_kernel"]["events"] == 1
    assert target_shapes["targets"]["second_kernel"]["events"] == 1


def test_run_trace_kernel_multiple_targets_partial_coverage(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text("""
class DummyKernel:
    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return "ok"
        return launch

first_kernel = DummyKernel()
second_kernel = DummyKernel()

def wrapper(q, k, block, launch_second=False):
    first_kernel[(q.shape[0], block)](q, key=k, BLOCK_SIZE=block)
    if launch_second:
        second_kernel[(k.shape[0], block)](k, query=q, BLOCK_SIZE=block)
""", encoding="utf-8")
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
mod.wrapper(T(), T(), 64)
""", encoding="utf-8")
    cmd = f"{sys.executable} {script}"

    result = run_trace_kernel(TraceKernelConfig(
        results_dir=tmp_path / "results",
        kernel_name="first_kernel",
        kernel_file=src,
        run_cmd=cmd,
        max_records=10,
        repo_root=tmp_path,
        targets=[
            TraceKernelTarget(
                kernel_name="first_kernel",
                kernel_file=src,
                kernel_id="synthetic.first",
                trace_mode="triton-launch",
                kernel_type="triton",
                patch_strategy="static",
            ),
            TraceKernelTarget(
                kernel_name="second_kernel",
                kernel_file=src,
                kernel_id="synthetic.second",
                trace_mode="triton-launch",
                kernel_type="triton",
                patch_strategy="static",
            ),
        ],
    ))

    assert result["success"] is True
    assert result["event_found"] is True
    assert result["any_target_event_found"] is True
    assert result["target_event_found"] is False
    assert result["partial_coverage"] is True
    assert result["missing_kernel_names"] == ["second_kernel"]
    assert result["target_events_found"] == {
        "first_kernel": True,
        "second_kernel": False,
    }
    target_shapes = json.loads(
        (tmp_path / "results" / "target_kernel_tensor_shapes.json").read_text()
    )
    assert target_shapes["trace_result"]["partial_coverage"] is True
    assert target_shapes["targets"]["first_kernel"]["events"] == 1
    assert target_shapes["targets"]["second_kernel"]["events"] == 0


def test_run_trace_kernel_aiter_mode_patches_compile_ops_core(tmp_path, monkeypatch):
    monkeypatch.setenv("MAGPIE_RUN_MODE", "local")
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
    bench = tmp_path / "benchmark.yaml"
    bench.write_text("benchmark: {}\n", encoding="utf-8")
    result = run_trace_kernel(TraceKernelConfig(
        results_dir=tmp_path / "results",
        kernel_name="fmoe",
        kernel_file=wrapper,
        trace_mode="aiter-compile-ops",
        benchmark_config=str(bench),
        dry_run=True,
        repo_root=tmp_path,
    ))
    assert result["mode"] == "aiter-compile-ops"
    assert ((tmp_path / "results" / "trace_raw").stat().st_mode & 0o777) == 0o777
    assert result["patched_file"].endswith("patched_files/overlay/aiter/jit/core.py")
    trace_config = json.loads((tmp_path / "results" / "trace_config.json").read_text())
    assert Path(trace_config["benchmark_config"]).is_absolute()
    manifest = json.loads(
        (tmp_path / "results" / "patched_files" / "patch_manifest.json").read_text()
    )
    assert "aiter.jit.core" in manifest["overlay_modules"]


def test_rewrite_sglang_deepseek_cuda_graph_script():
    source = """#!/usr/bin/env bash
python3 -m sglang.launch_server \\
--model-path $MODEL \\
--cuda-graph-max-bs=128 \\
--max-running-requests 128 > $SERVER_LOG 2>&1 &
"""
    rewritten = _rewrite_benchmark_script_disable_cuda_graph(
        source,
        source_path=Path("dsr1_fp4_mi355x.sh"),
    )
    assert "--cuda-graph-max-bs" not in rewritten
    assert "--disable-cuda-graph" in rewritten
    assert "--disable-piecewise-cuda-graph" in rewritten
    assert "--max-running-requests 128" in rewritten


def test_rewrite_sglang_qwen_cuda_graph_variable_arg():
    source = """#!/usr/bin/env bash
python3 -m sglang.launch_server \\
    --model-path $MODEL \\
    --cuda-graph-max-bs $CONC \\
    --disable-radix-cache \\
    --page-size 16 > $SERVER_LOG 2>&1 &
"""
    rewritten = _rewrite_benchmark_script_disable_cuda_graph(
        source,
        source_path=Path("qwen3.5_fp8_mi355x.sh"),
    )
    assert "--cuda-graph-max-bs" not in rewritten
    assert "    --disable-cuda-graph \\" in rewritten
    assert "    --disable-piecewise-cuda-graph \\" in rewritten
    assert "    --disable-radix-cache \\" in rewritten


def test_rewrite_vllm_kimi_adds_enforce_eager():
    source = """#!/usr/bin/env bash
vllm serve $MODEL --port $PORT \\
--tensor-parallel-size=$TP \\
--trust-remote-code > $SERVER_LOG 2>&1 &
"""
    rewritten = _rewrite_benchmark_script_disable_cuda_graph(
        source,
        source_path=Path("kimik2.5_fp4_mi355x.sh"),
    )
    assert "--enforce-eager \\" in rewritten
    assert "--trust-remote-code" in rewritten


def test_prepare_no_cudagraph_benchmark_script_resolves_config_script(tmp_path):
    repo_root = tmp_path
    script = (
        repo_root
        / "tools"
        / "magpie"
        / "InferenceX"
        / "benchmarks"
        / "single_node"
        / "fixed_seq_len"
        / "qwen3.5_fp8_mi355x.sh"
    )
    script.parent.mkdir(parents=True)
    script.write_text(
        """#!/usr/bin/env bash
python3 -m sglang.launch_server \\
    --model-path $MODEL \\
    --cuda-graph-max-bs $CONC \\
    --max-running-requests $CONC
""",
        encoding="utf-8",
    )
    bench = tmp_path / "bench.yaml"
    bench.write_text(
        """benchmark:
  framework: sglang
  benchmark_script: single_node/fixed_seq_len/qwen3.5_fp8_mi355x.sh
""",
        encoding="utf-8",
    )

    host_script, container_script = _prepare_no_cudagraph_benchmark_script(
        TraceKernelConfig(
            results_dir=tmp_path / "results",
            kernel_name="some_kernel",
            kernel_file=tmp_path / "kernel.py",
            benchmark_config=str(bench),
            repo_root=repo_root,
        )
    )

    text = host_script.read_text(encoding="utf-8")
    assert host_script.is_file()
    assert container_script == (
        "/opt/InferenceX/benchmarks/single_node/fixed_seq_len/qwen3.5_fp8_mi355x.sh"
    )
    assert "--cuda-graph-max-bs" not in text
    assert "--disable-cuda-graph" in text


def test_write_docker_wrapper_includes_extra_mounts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    host_script = tmp_path / "no_cudagraph.sh"
    host_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    wrapper_dir = write_docker_wrapper(
        Path("relative_results"),
        extra_mounts=[
            (
                host_script,
                "/opt/InferenceX/benchmarks/single_node/fixed_seq_len/no_cudagraph.sh",
            )
        ],
    )
    assert wrapper_dir.is_absolute()
    extra_mounts = (wrapper_dir / "extra_mounts.tsv").read_text(encoding="utf-8")
    wrapper = (wrapper_dir / "docker").read_text(encoding="utf-8")
    assert str(host_script.resolve()) in extra_mounts
    assert "/opt/InferenceX/benchmarks/single_node/fixed_seq_len/no_cudagraph.sh" in extra_mounts
    assert "extra_mounts.tsv" in wrapper


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
    target_shapes = json.loads((tmp_path / "target_kernel_tensor_shapes.json").read_text())
    assert target_shapes["targets"]["k"]["events"] == 2
    assert target_shapes["targets"]["k"]["group_count"] == 1
    assert target_shapes["workload_ranges"]["total_calls"] == 2


def test_postprocess_signature_includes_attention_meta_kwargs(tmp_path):
    raw = tmp_path / "trace_raw"
    raw.mkdir()
    events = [
        {
            "kind": "triton_launch",
            "kernel_name": "paged_attn",
            "kwargs": {"num_kv_heads": 8, "q": {"type": "tensor", "shape": [1, 64]}},
            "source_file": "x.py",
            "line": 1,
        },
        {
            "kind": "triton_launch",
            "kernel_name": "paged_attn",
            "kwargs": {"num_kv_heads": 16, "q": {"type": "tensor", "shape": [1, 64]}},
            "source_file": "x.py",
            "line": 1,
        },
    ]
    (raw / "trace_pid1_rank0.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n",
        encoding="utf-8",
    )

    result = postprocess_trace(tmp_path)

    assert result["total_calls"] == 2
    assert len(result["groups"]) == 2
    assert {
        group["signature"]["num_kv_heads"]
        for group in result["groups"]
    } == {8, 16}


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
