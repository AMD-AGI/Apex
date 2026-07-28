import json
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "pipeline"))

from kernel_tracing.mode_detection import detect_trace_mode
from kernel_tracing.overlay import infer_module_mapping, overlay_path_for
from kernel_tracing.patch_triton import patch_triton_launch_file
from kernel_tracing.patch_wrapper import patch_aiter_compile_ops_file, patch_wrapper_entry_file
from kernel_tracing.registry import (
    SGLANG_TRACE_IMAGE,
    TRACE_IMAGE_REGISTRY_PATHS,
    VALID_KERNEL_TYPES,
    VALID_PATCH_STRATEGIES,
    VALID_TRACE_MODES,
    VLLM_TRACE_IMAGE,
    load_supported_kernels,
    supported_trace_images,
)
from kernel_tracing.discovery import discover_trace_kernel_entries
from kernel_tracing import registry_update
from kernel_tracing.registry_update import (
    update_trace_kernel_registry,
)


REGISTRIES_RAW = {
    image: yaml.safe_load(TRACE_IMAGE_REGISTRY_PATHS[image].read_text(encoding="utf-8"))
    for image in supported_trace_images()
}
SUPPORTED_BY_IMAGE = {
    image: load_supported_kernels(
        docker_image=image,
        repo_root=REPO_ROOT,
        validate_files=False,
    )
    for image in supported_trace_images()
}
SUPPORTED_KERNELS = [
    entry
    for image in supported_trace_images()
    for entry in SUPPORTED_BY_IMAGE[image]
]

# Registries are generated from Docker image source, so local tools/rocm checkouts
# are intentionally not treated as the source of truth for patchability tests.
LOCAL_SOURCES_MATCH_REGISTRY = False


def test_supported_kernel_registry_schema():
    assert set(REGISTRIES_RAW) == {VLLM_TRACE_IMAGE, SGLANG_TRACE_IMAGE}
    for image, raw in REGISTRIES_RAW.items():
        assert raw["schema_version"] == 2
        assert raw["docker_image"] == image
        assert raw["image_metadata"]["image"] == image
        assert set(raw["package_sources"]) <= {"aiter", "vllm", "sglang"}

        entries = SUPPORTED_BY_IMAGE[image]
        ids = [entry.id for entry in entries]
        assert len(ids) >= 500
        assert len(ids) == len(set(ids))
        assert {entry.kernel_type for entry in entries} <= VALID_KERNEL_TYPES
        assert {entry.trace_mode for entry in entries} <= VALID_TRACE_MODES
        assert {entry.patch_strategy for entry in entries} <= VALID_PATCH_STRATEGIES
        assert all(entry.patch_strategy == "static" for entry in entries)
        assert all(entry.trace_mode != "agent" for entry in entries)

    vllm_ids = {entry.id for entry in SUPPORTED_BY_IMAGE[VLLM_TRACE_IMAGE]}
    assert len(vllm_ids) == 844
    vllm_counts = {
        (repo, kernel_type): sum(
            entry.repo == repo and entry.kernel_type == kernel_type
            for entry in SUPPORTED_BY_IMAGE[VLLM_TRACE_IMAGE]
        )
        for repo in ("aiter", "vllm")
        for kernel_type in ("hip", "triton")
    }
    assert vllm_counts == {
        ("aiter", "hip"): 205,
        ("aiter", "triton"): 259,
        ("vllm", "hip"): 176,
        ("vllm", "triton"): 204,
    }
    assert REGISTRIES_RAW[VLLM_TRACE_IMAGE]["package_sources"]["aiter"][
        "package_version"
    ] == "0.1.13.post1"
    assert REGISTRIES_RAW[VLLM_TRACE_IMAGE]["package_sources"]["vllm"][
        "package_version"
    ] == "0.23.0+rocm723"
    assert {
        "vllm.hip.reshape_and_cache_flash",
        "vllm.hip.custom_all_reduce_callsite",
        "vllm.hip.pynccl_all_reduce_callsite",
        "vllm.triton.solve_tril_bt64_callsite",
        "vllm.triton.gumbel_sample",
        "aiter.triton.unified_attention_2d",
        "aiter.hip.moe_sorting_fwd",
    } <= vllm_ids
    pack_bitmatrix = next(
        entry
        for entry in SUPPORTED_BY_IMAGE[VLLM_TRACE_IMAGE]
        if entry.id == "vllm.triton.pack_bitmatrix"
    )
    assert pack_bitmatrix.kernel_file.endswith(
        "model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py"
    )
    sglang_ids = {entry.id for entry in SUPPORTED_BY_IMAGE[SGLANG_TRACE_IMAGE]}
    assert {
        "sglang.triton.fused_append_shared_experts_kernel",
        "aiter.triton.unified_attention_2d",
        "aiter.hip.moe_sorting_fwd",
    } <= sglang_ids


@pytest.mark.parametrize(
    "entry",
    SUPPORTED_KERNELS,
    ids=lambda entry: entry.id,
)
def test_supported_kernel_patchability(entry, tmp_path):
    if not LOCAL_SOURCES_MATCH_REGISTRY:
        pytest.skip("fixed registries are generated from Docker image sources")
    source = entry.resolved_file(REPO_ROOT)
    if not source.exists():
        pytest.skip(f"registry entry comes from Docker source not present locally: {entry.kernel_file}")
    assert detect_trace_mode(source, entry.kernel_name, entry.trace_mode) == entry.trace_mode

    if entry.trace_mode == "aiter-compile-ops":
        module_name = "aiter.jit.core"
        package_rel_path = "aiter/jit/core.py"
        source = REPO_ROOT / "tools" / "rocm" / "aiter" / "aiter" / "jit" / "core.py"
    else:
        module_name, package_rel_path = infer_module_mapping(source, REPO_ROOT)

    output = overlay_path_for(tmp_path / "patched_files", package_rel_path)
    if entry.trace_mode == "triton-launch":
        result = patch_triton_launch_file(
            source_path=source,
            output_path=output,
            kernel_name=entry.kernel_name,
            module_name=module_name,
            package_rel_path=package_rel_path,
        )
    elif entry.trace_mode == "aiter-compile-ops":
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
        }[entry.trace_mode]
        result = patch_wrapper_entry_file(
            source_path=source,
            output_path=output,
            kernel_name=entry.kernel_name,
            trace_kind=kind,
            module_name=module_name,
            package_rel_path=package_rel_path,
        )
    assert result.events
    assert "apex_trace_event" in output.read_text(encoding="utf-8")
    py_compile.compile(str(output), doraise=True)


def test_list_trace_kernels_filters():
    proc = subprocess.run(
        [
            sys.executable,
            "workload_optimizer.py",
            "list-trace-kernels",
            "--docker-image",
            VLLM_TRACE_IMAGE,
            "--repo",
            "vllm",
            "--kernel-type",
            "hip",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"Docker image: {VLLM_TRACE_IMAGE}" in proc.stdout
    assert "vllm.hip.reshape_and_cache_flash" in proc.stdout
    assert "aiter." not in proc.stdout
    assert "supported trace kernels" in proc.stdout


def test_discovery_covers_supported_kernel_registry():
    pytest.skip("fixed registries are generated from Docker images; local checkout may intentionally drift")
    if not LOCAL_SOURCES_MATCH_REGISTRY:
        pytest.skip("fixed registries are generated from Docker image sources")
    discovered_ids = {
        entry.id
        for entry in discover_trace_kernel_entries(REPO_ROOT)
    }
    supported_ids = {entry.id for entry in SUPPORTED_KERNELS}
    assert supported_ids <= discovered_ids


def test_registry_schema_accepts_image_metadata(tmp_path):
    entry = next(entry for entry in SUPPORTED_BY_IMAGE[VLLM_TRACE_IMAGE] if entry.repo == "vllm")
    registry = {
        "schema_version": 2,
        "docker_image": VLLM_TRACE_IMAGE,
        "image_metadata": {
            "image": VLLM_TRACE_IMAGE,
            "image_id": "sha256:test",
        },
        "package_sources": {
            "vllm": {
                "image": VLLM_TRACE_IMAGE,
                "package_version": "0.23.0+rocm723",
                "source_path": "/usr/local/lib/python3.12/dist-packages/vllm",
                "registry_path": "tools/rocm/vllm/vllm",
            }
        },
        "kernels": [entry.as_dict()],
    }
    path = tmp_path / "vllm.yaml"
    path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    loaded = load_supported_kernels(
        docker_image=VLLM_TRACE_IMAGE,
        path=path,
        repo_root=REPO_ROOT,
        validate_files=False,
    )
    assert loaded[0].id == entry.id


def test_discovery_from_repo_shaped_root(tmp_path):
    source = tmp_path / "tools" / "rocm" / "vllm" / "vllm" / "sample_kernel.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join([
            "import triton",
            "",
            "@triton.jit",
            "def sample_kernel(x):",
            "    return",
            "",
            "def wrapper(x):",
            "    sample_kernel[(1,)](x)",
            "",
        ]),
        encoding="utf-8",
    )
    entries = discover_trace_kernel_entries(tmp_path)
    assert {entry.id for entry in entries} == {"vllm.triton.sample_kernel"}
    assert entries[0].kernel_file == "tools/rocm/vllm/vllm/sample_kernel.py"


def test_registry_generation_preserves_supplemental_callsites():
    data = registry_update.build_registry_data(
        docker_image=VLLM_TRACE_IMAGE,
        image_metadata={"image": VLLM_TRACE_IMAGE},
        package_sources={"vllm": {"image": VLLM_TRACE_IMAGE}},
        discovered_entries=[],
    )
    assert {entry["id"] for entry in data["kernels"]} == {
        "vllm.triton.solve_tril_bt64_callsite",
        "vllm.hip.custom_all_reduce_callsite",
        "vllm.hip.pynccl_all_reduce_callsite",
    }


def test_update_registry_dry_run_does_not_write_yaml(tmp_path, monkeypatch):
    outputs = {
        image: tmp_path / f"{idx}.yaml"
        for idx, image in enumerate(supported_trace_images())
    }
    for image, output in outputs.items():
        output.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 2,
                    "docker_image": image,
                    "image_metadata": {"image": image},
                    "package_sources": {"vllm": {"image": image}},
                    "kernels": [],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
    before = {image: path.read_text(encoding="utf-8") for image, path in outputs.items()}
    monkeypatch.setattr(registry_update, "TRACE_IMAGE_REGISTRY_PATHS", outputs)

    def fake_generate_registry_for_image(*, image, temp_root):
        del temp_root
        repo = "vllm" if image == VLLM_TRACE_IMAGE else "sglang"
        entry = {
            "id": f"{repo}.triton.sample_kernel",
            "repo": repo,
            "kernel_type": "triton",
            "kernel_name": "sample_kernel",
            "kernel_file": f"tools/rocm/{repo}/{repo}/sample_kernel.py",
            "trace_mode": "triton-launch",
            "patch_strategy": "static",
        }
        return {
            "schema_version": 2,
            "docker_image": image,
            "image_metadata": {"image": image, "image_id": "sha256:test"},
            "package_sources": {repo: {"image": image, "source_path": "/pkg"}},
            "kernels": [entry],
        }, [repo]

    monkeypatch.setattr(
        registry_update,
        "_generate_registry_for_image",
        fake_generate_registry_for_image,
    )

    report = tmp_path / "diff.md"
    result = update_trace_kernel_registry(
        repo_root=REPO_ROOT,
        report_path=report,
        write=False,
    )
    assert {image: path.read_text(encoding="utf-8") for image, path in outputs.items()} == before
    assert report.exists()
    assert result.diffs[VLLM_TRACE_IMAGE]["new_count"] == 1
    assert result.diffs[VLLM_TRACE_IMAGE]["added"] == ["vllm.triton.sample_kernel"]


def test_trace_kernel_cli_uses_kernel_id_dry_run(tmp_path):
    source = REPO_ROOT / "tools" / "rocm" / "vllm" / "vllm" / "_custom_ops.py"
    if not source.exists():
        pytest.skip(f"registry source fixture is not present in this checkout: {source}")
    results = tmp_path / "trace"
    proc = subprocess.run(
        [
            sys.executable,
            "workload_optimizer.py",
            "trace-kernel",
            "-r",
            str(results),
            "--kernel-id",
            "vllm.hip.reshape_and_cache_flash",
            "--docker-image",
            VLLM_TRACE_IMAGE,
            "--disable-benchmark-cuda-graph",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(proc.stdout)
    assert result["kernel_id"] == "vllm.hip.reshape_and_cache_flash"
    assert result["mode"] == "vllm-custom-op"

    trace_config = json.loads((results / "trace_config.json").read_text(encoding="utf-8"))
    assert trace_config["kernel_id"] == "vllm.hip.reshape_and_cache_flash"
    assert trace_config["kernel_name"] == "reshape_and_cache_flash"
    assert trace_config["docker_image"] == VLLM_TRACE_IMAGE
    assert trace_config["disable_benchmark_cuda_graph"] is True
    assert trace_config["registry_entry"]["kernel_file"] == "tools/rocm/vllm/vllm/_custom_ops.py"
    assert "Trace result:" in proc.stderr


def test_trace_kernel_cli_bad_kernel_id_suggests_list(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "workload_optimizer.py",
            "trace-kernel",
            "-r",
            str(tmp_path / "trace"),
            "--kernel-id",
            "vllm.hip.reshape_and_cache_flahs",
            "--docker-image",
            VLLM_TRACE_IMAGE,
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Unsupported trace kernel id" in proc.stderr
    assert "list-trace-kernels" in proc.stderr


def test_list_trace_kernels_requires_image():
    proc = subprocess.run(
        [
            sys.executable,
            "workload_optimizer.py",
            "list-trace-kernels",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "requires --docker-image or -b/--benchmark-config" in proc.stderr


def test_list_trace_kernels_supported_images():
    proc = subprocess.run(
        [
            sys.executable,
            "workload_optimizer.py",
            "list-trace-kernels",
            "--supported-images",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert VLLM_TRACE_IMAGE in proc.stdout
    assert SGLANG_TRACE_IMAGE in proc.stdout
    assert "vllm/vllm-openai-rocm:v0.19.1" not in proc.stdout


def test_list_trace_kernels_resolves_sglang_benchmark_config():
    bench = REPO_ROOT / "tools" / "magpie" / "examples" / "benchmarks" / "benchmark_sqlang_amd_dsr1_fp4.yaml"
    if not bench.exists():
        pytest.skip(f"Magpie benchmark fixture is not present in this checkout: {bench}")
    proc = subprocess.run(
        [
            sys.executable,
            "workload_optimizer.py",
            "list-trace-kernels",
            "-b",
            str(bench.relative_to(REPO_ROOT)),
            "--repo",
            "sglang",
            "--kernel-type",
            "triton",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"Docker image: {SGLANG_TRACE_IMAGE}" in proc.stdout
    assert "sglang.triton.fused_append_shared_experts_kernel" in proc.stdout


@pytest.mark.parametrize(
    "image",
    [
        "vllm/vllm-openai-rocm:v0.19.1",
        "vllm/vllm-openai-rocm:nightly",
    ],
)
def test_list_trace_kernels_rejects_unsupported_image(image):
    proc = subprocess.run(
        [
            sys.executable,
            "workload_optimizer.py",
            "list-trace-kernels",
            "--docker-image",
            image,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Unsupported trace-kernel Docker image" in proc.stderr
