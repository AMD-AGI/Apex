# Dynamic Kernel Workload Tracing

Dynamic Kernel Workload Tracing is an Apex `trace-kernel` workflow for collecting real runtime workload information from model E2E benchmarks. It temporarily patches Python-visible kernel launch or wrapper sites, reruns a workload, writes JSONL trace events, and postprocesses them into workload ranges for later kernel optimization.

This feature is meant to answer one practical question:

> When this Triton kernel or Python-visible HIP/custom op runs inside the real model benchmark, what shapes, dtypes, strides, flags, and meta parameters does it actually see?

## Basic Command

```bash
python3 workload_optimizer.py trace-kernel \
  -r /path/to/results_trace \
  --kernel-name kernel_unified_attention_2d \
  --kernel-file /root/Apex/tools/rocm/aiter/aiter/ops/triton/attention/unified_attention.py \
  --trace-mode triton-launch \
  --patch-strategy static \
  --max-records 200 \
  --sample-rate 1.0 \
  -b /root/Magpie/examples/benchmarks/benchmark_vllm_gptoss_20b.yaml
```

`trace-kernel` has two phases:

1. Build a temporary patched overlay under the results directory.
2. Run either a Magpie benchmark or a user-provided command with that overlay enabled.

It does not edit the source checkout in place. For Docker benchmarks, Apex bind-mounts the results directory into the benchmark container and injects the patched overlay through `PYTHONPATH`.

## Outputs

All outputs are written under `-r/--results-dir`.

```text
<RESULTS_DIR>/
+-- trace_config.json
+-- trace_result.json
+-- trace_raw.jsonl
+-- trace_raw/
|   +-- trace_pid<PID>_rank<RANK>.jsonl
+-- workload_ranges.json
+-- workload_summary.md
+-- patched_files/
|   +-- apex_kernel_tracing_runtime.py
|   +-- apex_kernel_tracing_importer.py
|   +-- sitecustomize.py
|   +-- patch_manifest.json
|   +-- overlay/...
+-- container_sources/
|   +-- ... extracted source from the Docker image when available
+-- benchmark/
    +-- trace_benchmark_config.yaml
    +-- benchmark_result.json
```

Important files:

- `trace_raw/*.jsonl`: raw per-process trace events.
- `trace_raw.jsonl`: merged raw trace events.
- `workload_ranges.json`: grouped shape/dtype/scalar ranges.
- `workload_summary.md`: compact human-readable summary.
- `patched_files/patch_manifest.json`: module-to-overlay mapping.
- `trace_result.json`: final status, benchmark result, and whether a target event was found.

## CLI Arguments

| Argument | Required | Description |
|---|---:|---|
| `-r, --results-dir` | yes | Directory for trace outputs, patched overlay, benchmark result, and postprocessed summaries. |
| `--kernel-name` | yes | Target Triton kernel name or Python-visible op name. Used for patching and runtime filtering unless `--trace-all` is set. |
| `--kernel-file` | yes | Source file used to infer the package module and patch strategy. For Triton this is the launch file. For wrapper/custom-op modes this is the wrapper file. |
| `--trace-mode` | no | Trace strategy. Use `auto`, `triton-launch`, `aiter-compile-ops`, `vllm-custom-op`, `sglang-custom-op`, or `agent`. |
| `--kernel-type` | no | Compatibility alias. `triton` maps to `triton-launch`; `hip` leaves mode detection on `auto`. Prefer `--trace-mode` for new usage. |
| `--patch-strategy` | no | `static`, `agent`, or `auto`. Static patching handles known patterns; agent fallback is for irregular source patterns. |
| `-b, --benchmark-config` | one of `-b` or `--run-cmd` | Magpie benchmark YAML to run after patching. |
| `--run-cmd` | one of `-b` or `--run-cmd` | Local command to run after patching. Useful for op tests and small repros. |
| `--max-records` | no | Maximum non-diagnostic trace events per process. `module_import` diagnostics do not consume this budget. |
| `--sample-rate` | no | Sampling probability for non-diagnostic events. Use `1.0` for smoke tests, then lower it for high-frequency kernels. |
| `--small-tensor-stats` | no | Collect min/max/percentile-like small tensor content summaries where supported. Disabled by default because it can synchronize GPU work. |
| `--trace-all` | no | Disable runtime filtering by `--kernel-name`. Useful for central hooks, especially `aiter-compile-ops`, when the real low-level op name is unknown. |
| `--benchmark-timeout` | no | Timeout in seconds for the benchmark or run command. |
| `--docker-image` | no | Override benchmark Docker image. Otherwise Apex uses the benchmark config or default vLLM ROCm image. |
| `--framework` | no | Framework passed to Magpie benchmark, usually `vllm` or `sglang`. Defaults to `vllm`. |
| `--dry-run` | no | Generate and compile the patched overlay, then stop before running workload. |

## Trace Modes

### `triton-launch`

Patches Python Triton launch sites:

```python
some_kernel[grid](arg0, arg1, META=value, **config)
```

into:

```python
apex_trace_event(
    kind="triton_launch",
    kernel_name="some_kernel",
    grid=grid,
    args=[arg0, arg1],
    kwargs={"META": value, **config},
    extra={"wrapper": "..."},
)
some_kernel[grid](arg0, arg1, META=value, **config)
```

Use this for Triton kernels in aiter, vLLM, or SGLang when the launch expression is visible in Python.

### `aiter-compile-ops`

Patches the central aiter HIP wrapper factory:

```text
aiter.jit.core.compile_ops()
```

This records calls through both:

- `compile_ops.ctypes_wrapper`
- `compile_ops.pybind_wrapper`

The recorded `kernel_name` is the runtime `loadName`, not necessarily the high-level Python helper name. For MoE, for example, the high-level path may be `fused_moe`, while the compile-op calls may be `fmoe`, `fmoe_g1u1_tkw1`, or another low-level op. Use `--trace-all` for the first discovery run.

### `vllm-custom-op`

Patches Python functions in vLLM custom-op wrapper files such as:

```text
vllm/_custom_ops.py
vllm/_aiter_ops.py
```

This captures the tensor/scalar metadata before the wrapper calls `torch.ops._C...`, `torch.ops._C_cache_ops...`, or a vLLM aiter bridge.

### `sglang-custom-op`

Patches Python-visible SGLang custom-op wrappers. This is similar to `vllm-custom-op`, but aimed at SGLang `jit_kernel` and ROCm wrapper modules.

### `agent`

Uses the configured Apex agent backend to generate a patch when the static patcher cannot handle the source pattern. Use this for aliases such as `kernel`, indirect launch helpers, or deeply custom dispatch logic.

## Docker Overlay Behavior

For `-b/--benchmark-config` runs, Apex usually runs through Magpie. If Magpie selects Docker:

1. Apex creates `<RESULTS_DIR>/patched_files`.
2. Apex writes `sitecustomize.py`, `apex_kernel_tracing_importer.py`, and `apex_kernel_tracing_runtime.py`.
3. Apex writes patched modules under `patched_files/overlay/...`.
4. Apex wraps `docker run` so the results directory is mounted at `/apex_trace`.
5. Apex injects:

```bash
PYTHONPATH=/apex_trace/patched_files
APEX_TRACE_PATCH_MANIFEST=/apex_trace/patched_files/patch_manifest.json
APEX_TRACE_OUTPUT_DIR=/apex_trace/trace_raw
```

The importer records a `module_import` event whenever a patched overlay module is imported. This diagnostic event is never sampled and does not count against `--max-records`.

## Interpreting Results

### Case 1: `module_import` and target events exist

The overlay was imported and the target launch/wrapper executed. This is the ideal result. Inspect `workload_ranges.json` and `workload_summary.md`.

### Case 2: only `module_import` exists

The overlay was imported, but the target launch or wrapper did not execute. This usually means the workload did not take that runtime path.

Example:

```json
{"kind": "module_import", "extra": {"module_name": "vllm.v1.worker.gpu.sample.gumbel"}}
```

This means the patched module loaded, but if there is no `_gumbel_sample_kernel` event, the gumbel Triton launch did not run in that benchmark.

### Case 3: no events exist

The patched module was not imported, or tracing was not enabled in the process that executed the workload. Check:

- `PYTHONPATH` in `benchmark/trace_benchmark_config.yaml`
- `patched_files/patch_manifest.json`
- whether the benchmark uses a different framework path or package version
- whether the target path is inside a subprocess with a different environment

## Practical Workflow

Use this workflow for a new target:

1. Start with `--sample-rate 1.0 --max-records 200`.
2. For central hooks, add `--trace-all`.
3. Confirm `module_import` exists.
4. Confirm target runtime events exist.
5. If target events do not exist, choose a different wrapper or use `--trace-all` to discover the real low-level name.
6. Once the target is confirmed, lower `--sample-rate` and raise `--max-records` for distribution collection.

## Examples

Set up the environment first:

```bash
cd /root/Apex
source .venv/bin/activate
export MAGPIE_ROOT=$(cd ../Magpie && pwd)
export MAGPIE_RUN_MODE=docker
```

### 1. aiter Triton attention launch

This traces a known aiter Triton launch used by GPT-OSS 20B when vLLM routes attention through aiter unified attention.

```bash
python3 workload_optimizer.py trace-kernel \
  -r /root/Apex/results_trace_gptoss20b_aiter_unified_attention_2d \
  --kernel-name kernel_unified_attention_2d \
  --kernel-file /root/Apex/tools/rocm/aiter/aiter/ops/triton/attention/unified_attention.py \
  --trace-mode triton-launch \
  --patch-strategy static \
  --max-records 200 \
  --sample-rate 1.0 \
  --benchmark-timeout 2700 \
  -b /root/Magpie/examples/benchmarks/benchmark_vllm_gptoss_20b.yaml
```

Expected useful output:

- `triton_launch` events for `kernel_unified_attention_2d`
- tensor metadata for query/output/cache tensors
- constexpr/meta values such as `BLOCK_SIZE`, `HEAD_SIZE`, `SLIDING_WINDOW`

### 2. vLLM HIP custom op wrapper

This traces the Python-visible vLLM cache wrapper before it calls the compiled cache op.

```bash
python3 workload_optimizer.py trace-kernel \
  -r /root/Apex/results_trace_gptoss20b_vllm_reshape_and_cache_flash \
  --kernel-name reshape_and_cache_flash \
  --kernel-file /root/Apex/tools/rocm/vllm/vllm/_custom_ops.py \
  --trace-mode vllm-custom-op \
  --patch-strategy static \
  --max-records 200 \
  --sample-rate 1.0 \
  --benchmark-timeout 2700 \
  -b /root/Magpie/examples/benchmarks/benchmark_vllm_gptoss_20b.yaml
```

Expected useful output:

- `vllm_python_op` events for `reshape_and_cache_flash`
- key/value/cache tensor shapes, strides, dtype, and `kv_cache_dtype`

### 3. aiter HIP compile-ops discovery run

Use this when you know a high-level aiter path is involved, but you do not yet know the low-level `compile_ops` `loadName`.

```bash
python3 workload_optimizer.py trace-kernel \
  -r /root/Apex/results_trace_gptoss20b_aiter_compile_ops_discovery \
  --kernel-name fused_moe \
  --kernel-file /root/Apex/tools/rocm/aiter/aiter/fused_moe.py \
  --trace-mode aiter-compile-ops \
  --patch-strategy static \
  --trace-all \
  --max-records 200 \
  --sample-rate 1.0 \
  --benchmark-timeout 2700 \
  -b /root/Magpie/examples/benchmarks/benchmark_vllm_gptoss_20b.yaml
```

After the run, inspect:

```bash
python3 - <<'PY'
import json
from pathlib import Path
root = Path("/root/Apex/results_trace_gptoss20b_aiter_compile_ops_discovery")
names = {}
for path in (root / "trace_raw").glob("*.jsonl"):
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("kind") == "hip_python_op":
            names[event.get("kernel_name")] = names.get(event.get("kernel_name"), 0) + 1
print(names)
PY
```

Then rerun with the actual `--kernel-name`, for example:

```bash
python3 workload_optimizer.py trace-kernel \
  -r /root/Apex/results_trace_gptoss20b_aiter_fmoe \
  --kernel-name fmoe \
  --kernel-file /root/Apex/tools/rocm/aiter/aiter/fused_moe.py \
  --trace-mode aiter-compile-ops \
  --patch-strategy static \
  --max-records 10000 \
  --sample-rate 0.01 \
  --benchmark-timeout 2700 \
  -b /root/Magpie/examples/benchmarks/benchmark_vllm_gptoss_20b.yaml
```

### 4. Dry-run patch validation

Use `--dry-run` to validate that the patch can be generated and compiled without running a benchmark.

```bash
python3 workload_optimizer.py trace-kernel \
  -r /tmp/apex_trace_dry_unified_attention \
  --kernel-name kernel_unified_attention_2d \
  --kernel-file /root/Apex/tools/rocm/aiter/aiter/ops/triton/attention/unified_attention.py \
  --trace-mode triton-launch \
  --patch-strategy static \
  --dry-run
```

This should produce `patched_files/overlay/...` and `trace_config.json`, but no raw runtime events.

### 5. Local command instead of Magpie

Use `--run-cmd` when you want to run a small op test or repro command instead of an E2E Magpie benchmark.

```bash
python3 workload_optimizer.py trace-kernel \
  -r /root/Apex/results_trace_local_pa_decode \
  --kernel-name _paged_attn_decode_v1_wo_dot_kernel \
  --kernel-file /root/Apex/tools/rocm/aiter/aiter/ops/triton/attention/pa_decode.py \
  --trace-mode triton-launch \
  --patch-strategy static \
  --max-records 200 \
  --sample-rate 1.0 \
  --run-cmd 'python3 -m pytest /root/Apex/tools/rocm/aiter/op_tests/triton_tests/attention/test_pa_decode.py -x'
```

The command runs on the host with `PYTHONPATH` pointing at the patched overlay.

## Notes and Limitations

- Triton mode patches launch sites, not `@triton.jit` kernel bodies.
- HIP mode records Python-visible workload metadata. It does not identify the final HSA code object, CK template instance, or C++ internal branch unless that information is exposed through Python arguments.
- `module_import` is diagnostic only. It proves the overlay loaded, not that the target kernel executed.
- `--small-tensor-stats` can add overhead and should be disabled for normal distribution collection unless token-length or routing statistics are important.
- Magpie gap-analysis failures do not necessarily mean the tracing run failed. Check `trace_result.json`, `trace_raw/*.jsonl`, and `workload_ranges.json`.
