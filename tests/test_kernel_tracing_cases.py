import json
import py_compile
import re
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
    SUPPORTED_KERNELS_PATH,
    VALID_KERNEL_TYPES,
    VALID_PATCH_STRATEGIES,
    VALID_TRACE_MODES,
    load_supported_kernels,
)
from kernel_tracing.discovery import discover_trace_kernel_entries
from kernel_tracing import registry_update
from kernel_tracing.registry_update import (
    CommandResult,
    resolve_vllm_tag_commit,
    update_trace_kernel_registry,
    vllm_tag_from_version,
)


REGISTRY_RAW = yaml.safe_load(SUPPORTED_KERNELS_PATH.read_text(encoding="utf-8"))
SUPPORTED_KERNELS = load_supported_kernels(repo_root=REPO_ROOT, validate_files=False)


def _local_sources_match_registry() -> bool:
    roots = {
        "aiter": REPO_ROOT / "tools" / "rocm" / "aiter",
        "vllm": REPO_ROOT / "tools" / "rocm" / "vllm",
        "sglang": REPO_ROOT / "tools" / "rocm" / "sglang",
    }
    commits = REGISTRY_RAW.get("source_commits") or {}
    for repo, root in roots.items():
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return False
        if proc.stdout.strip() != commits.get(repo):
            return False
    return True


LOCAL_SOURCES_MATCH_REGISTRY = _local_sources_match_registry()


def test_supported_kernel_registry_schema():
    raw = REGISTRY_RAW
    assert raw["schema_version"] == 1
    assert set(raw["source_commits"]) == {"aiter", "vllm", "sglang"}
    for commit in raw["source_commits"].values():
        assert re.fullmatch(r"[0-9a-f]{40}", commit)
    if "source_images" in raw:
        assert isinstance(raw["source_images"], dict)
        assert set(raw["source_images"]) <= {"aiter", "vllm", "sglang"}

    ids = [entry.id for entry in SUPPORTED_KERNELS]
    assert len(ids) >= 1000
    assert len(ids) == len(set(ids))
    assert {entry.repo for entry in SUPPORTED_KERNELS} == {"aiter", "vllm", "sglang"}
    assert {entry.kernel_type for entry in SUPPORTED_KERNELS} <= VALID_KERNEL_TYPES
    assert {entry.trace_mode for entry in SUPPORTED_KERNELS} <= VALID_TRACE_MODES
    assert {entry.patch_strategy for entry in SUPPORTED_KERNELS} <= VALID_PATCH_STRATEGIES
    assert all(entry.patch_strategy == "static" for entry in SUPPORTED_KERNELS)
    assert all(entry.trace_mode != "agent" for entry in SUPPORTED_KERNELS)
    assert {
        "vllm.hip.reshape_and_cache_flash",
        "vllm.triton.gumbel_sample",
        "aiter.triton.unified_attention_2d",
        "aiter.hip.moe_sorting_fwd",
    } <= set(ids)


@pytest.mark.parametrize(
    "entry",
    SUPPORTED_KERNELS,
    ids=lambda entry: entry.id,
)
def test_supported_kernel_patchability(entry, tmp_path):
    if not LOCAL_SOURCES_MATCH_REGISTRY:
        pytest.skip("local tools/rocm checkout does not match registry source_commits")
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
    assert "vllm.hip.reshape_and_cache_flash" in proc.stdout
    assert "aiter." not in proc.stdout
    assert "supported trace kernels" in proc.stdout


def test_discovery_covers_supported_kernel_registry():
    if REGISTRY_RAW.get("source_images"):
        pytest.skip("registry was generated from Docker images; local checkout may intentionally drift")
    if not LOCAL_SOURCES_MATCH_REGISTRY:
        pytest.skip("local tools/rocm checkout does not match registry source_commits")
    discovered_ids = {
        entry.id
        for entry in discover_trace_kernel_entries(REPO_ROOT)
    }
    supported_ids = {entry.id for entry in SUPPORTED_KERNELS}
    assert supported_ids <= discovered_ids


def test_registry_schema_accepts_source_images(tmp_path):
    registry = {
        "schema_version": 1,
        "source_commits": {
            "aiter": "a" * 40,
            "vllm": "b" * 40,
            "sglang": "c" * 40,
        },
        "source_images": {
            "vllm": {
                "image": "vllm/vllm-openai-rocm:v0.19.1",
                "package_version": "0.19.1+rocm721",
                "commit_resolution": "git ls-remote",
            }
        },
        "kernels": [SUPPORTED_KERNELS[0].as_dict()],
    }
    path = tmp_path / "supported_kernels.yaml"
    path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    loaded = load_supported_kernels(path=path, repo_root=REPO_ROOT, validate_files=False)
    assert loaded[0].id == SUPPORTED_KERNELS[0].id


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


def test_vllm_version_maps_to_remote_tag():
    assert vllm_tag_from_version("0.19.1+rocm721") == "v0.19.1"

    def fake_run(cmd):
        assert "refs/tags/v0.19.1" in cmd
        return CommandResult(
            stdout="b1388b1fbf5aaef47937fabe98931211684666a6\trefs/tags/v0.19.1\n",
            stderr="",
        )

    commit, tag, method = resolve_vllm_tag_commit("0.19.1+rocm721", run=fake_run)
    assert commit == "b1388b1fbf5aaef47937fabe98931211684666a6"
    assert tag == "v0.19.1"
    assert "git ls-remote" in method


def test_update_registry_dry_run_does_not_write_yaml(tmp_path, monkeypatch):
    output = tmp_path / "supported_kernels.yaml"
    output.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "source_commits": {
                    "aiter": "a" * 40,
                    "vllm": "b" * 40,
                    "sglang": "c" * 40,
                },
                "kernels": [],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    before = output.read_text(encoding="utf-8")

    monkeypatch.setattr(
        registry_update,
        "resolve_framework_images",
        lambda **_: {"vllm": "fixture/vllm:latest"},
    )

    def fake_copy_vllm_sources(*, image, temp_root, source_commits, source_images, vllm_commit=""):
        del image, vllm_commit
        source = temp_root / "tools" / "rocm" / "vllm" / "vllm" / "sample_kernel.py"
        source.parent.mkdir(parents=True)
        source.write_text(
            "\n".join([
                "import triton",
                "@triton.jit",
                "def sample_kernel(x):",
                "    return",
                "def wrapper(x):",
                "    sample_kernel[(1,)](x)",
            ]),
            encoding="utf-8",
        )
        source_commits["vllm"] = "d" * 40
        source_images["vllm"] = {
            "image": "fixture/vllm:latest",
            "package_version": "0.19.1+rocm721",
            "commit": "d" * 40,
            "commit_resolution": "fixture",
        }

    monkeypatch.setattr(registry_update, "_copy_vllm_sources", fake_copy_vllm_sources)

    report = tmp_path / "diff.md"
    result = update_trace_kernel_registry(
        repo_root=REPO_ROOT,
        frameworks="vllm",
        output_path=output,
        report_path=report,
        write=False,
    )
    assert output.read_text(encoding="utf-8") == before
    assert report.exists()
    assert result.diff["new_count"] == 1
    assert result.diff["added"] == ["vllm.triton.sample_kernel"]


def test_trace_kernel_cli_uses_kernel_id_dry_run(tmp_path):
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
    assert trace_config["registry_entry"]["kernel_file"] == "tools/rocm/vllm/vllm/_custom_ops.py"


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
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Unsupported trace kernel id" in proc.stderr
    assert "list-trace-kernels" in proc.stderr
