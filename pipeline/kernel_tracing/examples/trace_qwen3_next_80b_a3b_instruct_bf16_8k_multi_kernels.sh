#!/usr/bin/env bash
set -euo pipefail

# Trace Qwen3-Next BF16 kernel tensor shapes on an 8K-input workload.
#
# The vLLM Qwen3-Next implementation runs synthetic GDN kernel warmups while
# sizing the KV cache. Those events are present in the auditable full trace but
# are excluded from serving_only/. The accepted distribution is restricted to
# Magpie's benchmark client interval, and the run fails validation if a worker
# reaches APEX_TRACE_MAX_RECORDS (which could let warmup consume trace budget).
#
# Basic use:
#   HF_CACHE_PATH=/path/to/huggingface \
#     bash pipeline/kernel_tracing/examples/trace_qwen3_next_80b_a3b_instruct_bf16_8k_multi_kernels.sh
#
# Patch/source validation without loading the model:
#   DRY_RUN=1 \
#     bash pipeline/kernel_tracing/examples/trace_qwen3_next_80b_a3b_instruct_bf16_8k_multi_kernels.sh

umask 0022

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APEX_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$APEX_ROOT"

if [[ -f "$APEX_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$APEX_ROOT/.venv/bin/activate"
fi

if [[ -z "${MAGPIE_ROOT:-}" ]]; then
  if [[ -d "$APEX_ROOT/../Magpie" ]]; then
    export MAGPIE_ROOT="$(cd "$APEX_ROOT/../Magpie" && pwd)"
  elif [[ -d "$APEX_ROOT/tools/magpie" ]]; then
    export MAGPIE_ROOT="$APEX_ROOT/tools/magpie"
  else
    echo "MAGPIE_ROOT is not set and no Magpie checkout was found." >&2
    exit 1
  fi
fi

# The script resolves this environment variable into a run-local benchmark
# config before invoking Magpie. Magpie v0.23 does not expand hf_cache_path.
export HF_CACHE_PATH="${HF_CACHE_PATH:-/tmp/qwen3_next_hf_cache}"

DOCKER_IMAGE="vllm/vllm-openai-rocm:v0.23.0"
EXPECTED_IMAGE_ID="sha256:648be227ec3ee60b566f9def3485d29713f3d76464081e10a5d9ac56d25732cb"
export MAGPIE_RUN_MODE=docker

MAX_RECORDS="${MAX_RECORDS:-500000}"
SAMPLE_RATE="${SAMPLE_RATE:-0.01}"
BENCHMARK_TIMEOUT="${BENCHMARK_TIMEOUT:-14400}"
DRY_RUN="${DRY_RUN:-0}"
REQUIRE_ALL="${REQUIRE_ALL:-1}"
DISABLE_BENCHMARK_CUDA_GRAPH="${DISABLE_BENCHMARK_CUDA_GRAPH:-1}"

DEFAULT_TRACE_BENCH="$SCRIPT_DIR/benchmark_vllm_qwen3_next_80b_a3b_instruct_bf16.yaml"
TRACE_BENCH="${BENCH_CONFIG:-$DEFAULT_TRACE_BENCH}"
if [[ ! -f "$TRACE_BENCH" ]]; then
  echo "Benchmark config not found: $TRACE_BENCH" >&2
  exit 1
fi
TRACE_BENCH="$(readlink -f "$TRACE_BENCH")"

TS="$(date -u +%Y%m%d_%H%M%S)"
DEFAULT_RESULTS_DIR="$APEX_ROOT/tmp/results_trace_qwen3_next_80b_a3b_bf16_8k_${TS}"
RESULTS_DIR="${RESULTS_DIR:-$DEFAULT_RESULTS_DIR}"
if [[ "$RESULTS_DIR" != /* ]]; then
  RESULTS_DIR="$APEX_ROOT/$RESULTS_DIR"
fi
RESULTS_DIR="$(realpath -m "$RESULTS_DIR")"

if [[ -e "$RESULTS_DIR" && ! -d "$RESULTS_DIR" ]]; then
  echo "RESULTS_DIR exists but is not a directory: $RESULTS_DIR" >&2
  exit 1
fi
if [[ -d "$RESULTS_DIR" ]] \
  && [[ -n "$(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to mix a new trace with non-empty RESULTS_DIR: $RESULTS_DIR" >&2
  exit 1
fi
mkdir -p "$RESULTS_DIR"
chmod 0755 "$RESULTS_DIR"

# Qwen3-Next has 36 Gated DeltaNet layers and 12 full-attention layers, followed
# by a 512-expert/top-10 BF16 MoE in every layer. For head_dim=256, vLLM 0.23
# selects ROCM_ATTN; Qwen3-Next's hybrid KV layout then uses the stride-aware
# Triton attention/cache kernels below rather than ROCM_AITER_FA fallbacks.
KERNEL_IDS=(
  # Gated DeltaNet prefill and its BT=64 triangular solve.
  "vllm.triton.fused_post_conv_kernel"
  "vllm.triton.chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
  "vllm.triton.chunk_fwd_kernel_o"
  "vllm.triton.chunk_scaled_dot_kkt_fwd_kernel"
  "vllm.triton.chunk_local_cumsum_scalar_kernel"
  "vllm.triton.fla_ops_wy_fast.recompute_w_u_fwd_kernel"
  "vllm.triton.solve_tril_bt64_callsite"

  # Gated DeltaNet decode and gated output normalization.
  "aiter.triton.reshape_causal_conv1d_update_single_token_kernel"
  "aiter.triton.fused_rearrange_sigmoid_gated_delta_rule_update_kernel"
  "vllm.triton.layer_norm_fwd_kernel"

  # ROCM_ATTN prefill, decode, hybrid-layout KV-cache write, and RoPE.
  "vllm.triton.attention_ops_prefix_prefill.fwd_kernel"
  "vllm.triton.kernel_paged_attention_2d"
  "vllm.triton.reshape_and_cache_flash"
  "vllm.hip.rotary_embedding"

  # BF16 dense projections.
  "vllm.hip.rocm_unquantized_gemm"

  # BF16 routed MoE: top-k router, sorting, and CK two-stage experts.
  "aiter.hip.topk_softmax"
  "aiter.hip.moe_sorting_fwd"
  "aiter.hip.ck_moe_stage1"
  "aiter.hip.ck_moe_stage2"

  # Runtime-confirmed TP collective dispatch.
  "vllm.hip.custom_all_reduce_callsite"
  "vllm.hip.pynccl_all_reduce_callsite"
)

if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "Docker is required and its daemon must be available." >&2
  exit 1
fi
if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
  echo "Required image is not local: $DOCKER_IMAGE" >&2
  exit 1
fi
ACTUAL_IMAGE_ID="$(docker image inspect "$DOCKER_IMAGE" --format '{{.Id}}')"
if [[ "$ACTUAL_IMAGE_ID" != "$EXPECTED_IMAGE_ID" ]]; then
  echo "Local image does not match the v0.23 trace registry:" >&2
  echo "  expected: $EXPECTED_IMAGE_ID" >&2
  echo "  actual:   $ACTUAL_IMAGE_ID" >&2
  exit 1
fi

# Validate the workload and print the resolved HuggingFace cache path.
HF_CACHE_PATH="$(python3 - "$TRACE_BENCH" <<'PY'
import os
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
bench = data.get("benchmark", data)
envs = bench.get("envs") or {}
errors = []
expected = {
    "framework": "vllm",
    "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "precision": "bf16",
}
for key, value in expected.items():
    if bench.get(key) != value:
        errors.append(f"benchmark.{key} must be {value!r}, got {bench.get(key)!r}")
for key, value in {"TP": 4, "CONC": 16, "ISL": 8192, "OSL": 1024}.items():
    if int(envs.get(key, -1)) != value:
        errors.append(f"benchmark.envs.{key} must be {value}, got {envs.get(key)!r}")
if int(envs.get("MAX_MODEL_LEN", 0)) < int(envs["ISL"]) + int(envs["OSL"]):
    errors.append("MAX_MODEL_LEN must cover ISL + OSL")
if str(envs.get("VLLM_ROCM_USE_AITER", "")).lower() not in {"1", "true"}:
    errors.append("VLLM_ROCM_USE_AITER must be enabled")
profiler = bench.get("profiler") or {}
if bool((profiler.get("torch_profiler") or {}).get("enabled", False)):
    errors.append("torch_profiler must remain disabled for kernel shape tracing")
cache = bench.get("hf_cache_path")
if not cache:
    errors.append("benchmark.hf_cache_path must be explicit")
if errors:
    raise SystemExit("Benchmark preflight failed:\n  " + "\n  ".join(errors))
print(Path(os.path.expandvars(os.path.expanduser(str(cache)))).resolve())
PY
)"
mkdir -p "$HF_CACHE_PATH"
chmod 0755 "$HF_CACHE_PATH"

# BenchmarkConfig.from_dict() receives hf_cache_path verbatim. Materialize a
# run-local config so the documented HF_CACHE_PATH override actually controls
# the Docker bind mount while preserving the checked-in workload definition.
RESOLVED_TRACE_BENCH="$RESULTS_DIR/benchmark_config.resolved.yaml"
python3 - "$TRACE_BENCH" "$RESOLVED_TRACE_BENCH" "$HF_CACHE_PATH" <<'PY'
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
cache = Path(sys.argv[3]).resolve()
data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
benchmark = data.get("benchmark")
if not isinstance(benchmark, dict):
    raise SystemExit(f"Missing benchmark mapping in {source}")
benchmark["hf_cache_path"] = str(cache)
destination.write_text(
    yaml.safe_dump(data, sort_keys=False, width=120),
    encoding="utf-8",
)
PY
TRACE_BENCH="$RESOLVED_TRACE_BENCH"

if [[ "$DRY_RUN" == "0" ]]; then
  docker run --rm \
    --entrypoint /bin/true \
    -v "$HF_CACHE_PATH:/root/.cache/huggingface" \
    "$DOCKER_IMAGE"
fi

# Validate all IDs against the registry tied to the exact image.
python3 - "$APEX_ROOT" "$DOCKER_IMAGE" "${KERNEL_IDS[@]}" <<'PY'
import collections
import sys
from pathlib import Path

root = Path(sys.argv[1])
image = sys.argv[2]
requested = sys.argv[3:]
sys.path.insert(0, str(root / "pipeline"))
from kernel_tracing.registry import load_supported_kernels

entries = load_supported_kernels(
    docker_image=image,
    repo_root=root,
    validate_files=False,
)
by_id = {entry.id: entry for entry in entries}
missing = [kernel_id for kernel_id in requested if kernel_id not in by_id]
if missing:
    raise SystemExit("IDs missing from the registry:\n  " + "\n  ".join(missing))
names = collections.defaultdict(list)
for kernel_id in requested:
    names[by_id[kernel_id].kernel_name].append(kernel_id)
duplicates = {name: ids for name, ids in names.items() if len(ids) > 1}
if duplicates:
    print("Duplicate runtime kernel names (coverage will be name-level):")
    for name, ids in sorted(duplicates.items()):
        print(f"  {name}: {', '.join(ids)}")
print(f"Validated {len(requested)} kernel IDs against {image}")
PY

CMD=(
  python3 workload_optimizer.py trace-kernel
  -r "$RESULTS_DIR"
  --docker-image "$DOCKER_IMAGE"
  --max-records "$MAX_RECORDS"
  --sample-rate "$SAMPLE_RATE"
  --benchmark-timeout "$BENCHMARK_TIMEOUT"
  --framework vllm
  -b "$TRACE_BENCH"
)
for kernel_id in "${KERNEL_IDS[@]}"; do
  CMD+=(--kernel-id "$kernel_id")
done
if [[ "$DRY_RUN" != "0" ]]; then
  CMD+=(--dry-run)
fi
if [[ "$DISABLE_BENCHMARK_CUDA_GRAPH" != "0" ]]; then
  CMD+=(--disable-benchmark-cuda-graph)
fi

echo "Apex root:        $APEX_ROOT"
echo "Magpie root:      $MAGPIE_ROOT"
echo "Results dir:      $RESULTS_DIR"
echo "Benchmark config: $TRACE_BENCH"
echo "HF cache path:    $HF_CACHE_PATH"
echo "Docker image:     $DOCKER_IMAGE"
echo "Kernel IDs:       ${#KERNEL_IDS[@]}"
echo "Max records:      $MAX_RECORDS per worker"
echo "Sample rate:      $SAMPLE_RATE"
echo "Warmup policy:    serving_only post-filter + hard no-cap validation"
echo "Dry run:          $DRY_RUN"

set +e
"${CMD[@]}" 2>&1 | tee "$RESULTS_DIR/trace-kernel.log"
CLI_STATUS=${PIPESTATUS[0]}
set -e
if ((CLI_STATUS != 0)); then
  echo "trace-kernel exited with status $CLI_STATUS" >&2
  exit "$CLI_STATUS"
fi

if [[ "$DRY_RUN" != "0" ]]; then
  echo "Dry-run completed: $RESULTS_DIR"
  exit 0
fi

# Build the authoritative benchmark-client-only dataset. Sampling is accepted
# only if no process reached its cap, so startup/warmup samples cannot crowd out
# serving samples.
python3 - "$APEX_ROOT" "$RESULTS_DIR" "$MAX_RECORDS" "$SAMPLE_RATE" <<'PY'
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

root = Path(sys.argv[1])
results = Path(sys.argv[2])
max_records = int(sys.argv[3])
sample_rate = float(sys.argv[4])
benchmark_path = results / "benchmark" / "benchmark_result.json"
log_path = results / "trace-kernel.log"
trace_config_path = results / "trace_config.json"
trace_result_path = results / "trace_result.json"
for path in (benchmark_path, log_path, trace_config_path, trace_result_path):
    if not path.is_file():
        raise SystemExit(f"Missing required artifact: {path}")

benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
if not benchmark.get("success"):
    raise SystemExit("Benchmark result is not successful")
throughput = benchmark.get("throughput") or {}
duration = float(throughput.get("duration_seconds") or 0)
completed = int(throughput.get("completed_requests") or 0)
if duration <= 0 or completed <= 0:
    raise SystemExit("Benchmark duration/completed_requests are invalid")

pattern = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*"
    r"Magpie\.modes\.benchmark\.benchmarker - INFO - "
    r"Benchmark completed successfully"
)
matches = pattern.findall(log_path.read_text(encoding="utf-8", errors="replace"))
if len(matches) != 1:
    raise SystemExit(
        f"Expected one benchmarker completion timestamp, found {len(matches)}"
    )
# Magpie and this host both use the system local timezone (UTC on this host).
end_dt = datetime.strptime(matches[0], "%Y-%m-%d %H:%M:%S,%f").astimezone()
end_ns = int(end_dt.timestamp() * 1_000_000_000)
start_ns = end_ns - int(duration * 1_000_000_000)

serving = results / "serving_only"
raw_dir = serving / "trace_raw"
raw_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(trace_config_path, serving / "trace_config.json")
(serving / "benchmark").mkdir(exist_ok=True)
shutil.copy2(benchmark_path, serving / "benchmark" / "benchmark_result.json")

full_calls = 0
serving_calls = 0
first_serving_ts_ns = None
last_serving_ts_ns = None
worker_full = Counter()
worker_serving = Counter()
cap_reached = []
window_file = raw_dir / "benchmark_client.jsonl"
with window_file.open("w", encoding="utf-8") as out:
    for path in sorted((results / "trace_raw").glob("*.jsonl")):
        target_calls = 0
        for line in path.open(encoding="utf-8"):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") == "module_import":
                continue
            target_calls += 1
            full_calls += 1
            process = event.get("process") or {}
            worker = (
                f"rank={process.get('rank', 'na')},"
                f"pid={process.get('pid', path.name)}"
            )
            worker_full[worker] += 1
            ts_ns = int(event.get("ts_ns") or 0)
            if start_ns <= ts_ns <= end_ns:
                out.write(json.dumps(event, sort_keys=True) + "\n")
                serving_calls += 1
                first_serving_ts_ns = (
                    ts_ns
                    if first_serving_ts_ns is None
                    else min(first_serving_ts_ns, ts_ns)
                )
                last_serving_ts_ns = (
                    ts_ns
                    if last_serving_ts_ns is None
                    else max(last_serving_ts_ns, ts_ns)
                )
                worker_serving[worker] += 1
        if target_calls >= max_records:
            cap_reached.append({"file": path.name, "calls": target_calls})

if cap_reached:
    raise SystemExit(
        "Trace record cap was reached; warmup may have consumed serving sample "
        f"budget: {cap_reached}"
    )
if serving_calls <= 0:
    raise SystemExit("Serving window contains no trace calls")

sys.path.insert(0, str(root / "pipeline"))
from kernel_tracing.postprocess import postprocess_trace

ranges = postprocess_trace(serving)
shutil.rmtree(raw_dir)

target_doc = json.loads(
    (serving / "target_kernel_tensor_shapes.json").read_text(encoding="utf-8")
)
targets = target_doc.get("targets") or {}
present = sorted(name for name, item in targets.items() if item.get("events"))
missing = sorted(name for name, item in targets.items() if not item.get("events"))

window = {
    "window": "benchmark_client_only",
    "start_ts_ns": start_ns,
    "end_ts_ns": end_ns,
    "duration_seconds": duration,
    "completed_requests": completed,
    "source": (
        "Magpie benchmarker completion timestamp minus "
        "benchmark_result.throughput.duration_seconds"
    ),
    "warmup_excluded": True,
}
(serving / "window.json").write_text(
    json.dumps(window, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
coverage = {
    "sampling_rate": sample_rate,
    "max_records_per_process": max_records,
    "full_trace_calls": full_calls,
    "serving_window_calls": serving_calls,
    "first_serving_event_ts_ns": first_serving_ts_ns,
    "last_serving_event_ts_ns": last_serving_ts_ns,
    "warmup_and_startup_calls_excluded": full_calls - serving_calls,
    "per_worker_full_trace_calls": dict(sorted(worker_full.items())),
    "per_worker_serving_window_calls": dict(sorted(worker_serving.items())),
    "target_count": len(targets),
    "targets_present": len(present),
    "missing_targets": missing,
    "record_cap_reached": False,
    "sampling_acceptance": (
        "accepted: IID sampling plus no per-process cap reached; "
        "warmup events cannot consume serving sample budget"
    ),
    "benchmark_client_duration_seconds": duration,
}
(serving / "coverage_summary.json").write_text(
    json.dumps(coverage, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(
    f"Serving-only postprocess OK: calls={serving_calls}, "
    f"groups={len(ranges.get('groups') or [])}, "
    f"targets={len(present)}/{len(targets)}, cap_reached=False"
)
PY

# Final invariants: every accepted event is in the client window, every worker
# contributed samples, and enough observations remain for a useful distribution.
python3 - "$RESULTS_DIR" "$REQUIRE_ALL" <<'PY'
import json
import sys
from pathlib import Path

results = Path(sys.argv[1])
require_all = sys.argv[2] != "0"
serving = results / "serving_only"
window = json.loads((serving / "window.json").read_text(encoding="utf-8"))
coverage = json.loads(
    (serving / "coverage_summary.json").read_text(encoding="utf-8")
)
shapes = json.loads(
    (serving / "target_kernel_tensor_shapes.json").read_text(encoding="utf-8")
)
if coverage.get("record_cap_reached"):
    raise SystemExit("record cap reached")
if coverage.get("serving_window_calls", 0) < 1000:
    raise SystemExit("too few serving-window calls for a stable distribution")
per_worker = coverage.get("per_worker_serving_window_calls") or {}
if len(per_worker) < 4 or any(int(count) <= 0 for count in per_worker.values()):
    raise SystemExit(f"expected samples from four TP workers, got {per_worker}")
if not window.get("warmup_excluded"):
    raise SystemExit("warmup exclusion is not recorded")
first_ts = int(coverage.get("first_serving_event_ts_ns") or 0)
last_ts = int(coverage.get("last_serving_event_ts_ns") or 0)
if not (
    int(window["start_ts_ns"])
    <= first_ts
    <= last_ts
    <= int(window["end_ts_ns"])
):
    raise SystemExit(
        f"serving event timestamps fall outside the client window: "
        f"{first_ts}..{last_ts}"
    )
targets = shapes.get("targets") or {}
present = [name for name, item in targets.items() if item.get("events")]
missing = [name for name, item in targets.items() if not item.get("events")]
if not present:
    raise SystemExit("no target kernels observed in serving window")
if require_all and missing:
    raise SystemExit(f"REQUIRE_ALL=1 but targets are missing: {missing}")
print(
    f"Final validation OK: {coverage['serving_window_calls']} calls, "
    f"{len(present)}/{len(targets)} target names hit, "
    f"{len(per_worker)} TP workers, warmup excluded"
)
PY

SHARE_ARCHIVE="$(python3 - "$APEX_ROOT" "$RESULTS_DIR" "$TRACE_BENCH" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
results = Path(sys.argv[2])
benchmark_config = Path(sys.argv[3])
sys.path.insert(0, str(root / "pipeline"))

from kernel_tracing.postprocess import write_tensor_shape_share_archive

archive = write_tensor_shape_share_archive(
    results,
    analysis_dir=results / "serving_only",
    benchmark_config_path=benchmark_config,
)
print(archive)
PY
)"

echo "Results folder: $RESULTS_DIR"
echo "Serving-only shapes: $RESULTS_DIR/serving_only/target_kernel_tensor_shapes.json"
echo "Coverage: $RESULTS_DIR/serving_only/coverage_summary.json"
echo "Window: $RESULTS_DIR/serving_only/window.json"
echo "Share archive: $SHARE_ARCHIVE"
