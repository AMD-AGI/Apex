#!/usr/bin/env bash
set -euo pipefail

# This workspace is on root-squashed NFS. The Docker daemon cannot traverse
# results/overlay directories created with the host's default umask 0077, so
# generated bind-mount sources must be at least 0755/0644. Apex separately
# makes trace_raw world-writable for events emitted by container workers.
umask 0022

# Trace the source-derived kernel candidates for:
#   benchmark: pipeline/kernel_tracing/examples/benchmark_vllm_amd_kimi_k2_6_mxfp4.yaml
#   model:     amd/Kimi-K2.6-MXFP4
#   image:     vllm/vllm-openai-rocm:v0.23.0
#
# Static call-chain audit (vLLM 0.23.0 + AITER 0.1.13.post1):
#
#   KimiK25ForConditionalGeneration
#     -> DeepseekV2/DeepseekV3 text model (the benchmark sends text only)
#     -> DeepseekV2MLAAttention
#        -> ROCM_AITER_MLA prefill/decode
#     -> layer 0 dense MLP; layers 1..60 routed MoE
#        -> Quark MXFP4 dense/shared-expert linears
#        -> AITER MXFP4 fused MoE (384 experts, top-8)
#
# The arrays below are the intersection of that call graph with the fixed
# v0.23 trace registry. Runtime trace hits, not static reachability alone,
# determine which shape-dependent candidates this particular run touched.
# Some real GPU kernels cannot be listed because they are not in the registry:
# Kimi-tuned AITER FlyDSL GEMM/MoE launchers, ROCm flash-attention extensions,
# RCCL/vLLM collective C++ kernels, and the plain vLLM SiluAndMul C++ op.
#
# torch_profiler must stay disabled. Apex kernel tracing uses a Python overlay
# and JSONL events; it is independent of torch.profiler.
#
# Basic use:
#   HF_CACHE_PATH=/path/to/huggingface \
#     bash pipeline/kernel_tracing/examples/trace_kimi_k2_6_mxfp4_multi_kernels.sh
#
# Cheap patch/source validation without starting the model:
#   DRY_RUN=1 \
#     bash pipeline/kernel_tracing/examples/trace_kimi_k2_6_mxfp4_multi_kernels.sh
#
# Trace one component (all | mla | moe | linear | norm):
#   KERNEL_GROUP=mla \
#     bash pipeline/kernel_tracing/examples/trace_kimi_k2_6_mxfp4_multi_kernels.sh
#
# Override common knobs without editing this file:
#   RESULTS_DIR=tmp/results_trace_kimi_k2_6_custom \
#   MAX_RECORDS=40000 BENCHMARK_TIMEOUT=10800 \
#     bash pipeline/kernel_tracing/examples/trace_kimi_k2_6_mxfp4_multi_kernels.sh
#
# CUDA Graph is disabled by default for reliable Python-launch coverage. Set
# DISABLE_BENCHMARK_CUDA_GRAPH=0 to preserve the production graph/fusion path;
# the script then selects the graph-specific norm/cache candidates as well.

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

# The paired YAML uses this environment variable for its cache bind mount.
export HF_CACHE_PATH="${HF_CACHE_PATH:-${HF_HOME:-$HOME/.cache/huggingface}}"

# The trace registry and extracted Python sources are tied to this exact tag.
DOCKER_IMAGE="vllm/vllm-openai-rocm:v0.23.0"
EXPECTED_IMAGE_ID="sha256:648be227ec3ee60b566f9def3485d29713f3d76464081e10a5d9ac56d25732cb"
export MAGPIE_RUN_MODE=docker

MAX_RECORDS="${MAX_RECORDS:-300000}"
SAMPLE_RATE="${SAMPLE_RATE:-0.01}"
BENCHMARK_TIMEOUT="${BENCHMARK_TIMEOUT:-7200}"
TRACE_ALL="${TRACE_ALL:-0}"
DRY_RUN="${DRY_RUN:-0}"
DISABLE_BENCHMARK_CUDA_GRAPH="${DISABLE_BENCHMARK_CUDA_GRAPH:-1}"
REQUIRE_ALL="${REQUIRE_ALL:-0}"
KERNEL_GROUP="${KERNEL_GROUP:-all}"

DEFAULT_TRACE_BENCH="$SCRIPT_DIR/benchmark_vllm_amd_kimi_k2_6_mxfp4.yaml"
TRACE_BENCH="${BENCH_CONFIG:-$DEFAULT_TRACE_BENCH}"
if [[ ! -f "$TRACE_BENCH" ]]; then
  echo "Benchmark config not found: $TRACE_BENCH" >&2
  exit 1
fi
TRACE_BENCH="$(readlink -f "$TRACE_BENCH")"

TS="$(date -u +%Y%m%d_%H%M%S)"
DEFAULT_RESULTS_DIR="$APEX_ROOT/tmp/results_trace_kimi_k2_6_mxfp4_${TS}"
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
  echo "Choose a fresh RESULTS_DIR or remove the old directory explicitly." >&2
  exit 1
fi
mkdir -p "$RESULTS_DIR"

# Common MLA path. The two chunked-context entries are reached when a later
# prompt chunk attends to an already-populated KV cache. The FP4 BMM reduce
# kernel is intentionally absent: Kimi's W_K/W_V configs use NUM_KSPLIT=1.
MLA_COMMON_KERNEL_IDS=(
  # vLLM metadata/cache helpers.
  "vllm.triton.expand_page_indices_kernel"
  "vllm.hip.gather_and_maybe_dequant_cache"
  "vllm.triton.merge_attn_states_kernel"

  # AITER prefill metadata + gfx950 persistent-scheduling attention.
  "aiter.hip.get_ps_metadata_v1"
  "aiter.hip.mla_prefill_ps_asm_fwd"

  # AITER decode metadata + two-stage MLA attention.
  "aiter.hip.get_mla_metadata_v1"
  "aiter.hip.mla_decode_stage1_asm_fwd"
  "aiter.hip.mla_reduce_v1"

  # Decode W_K / W_V FP4 batched projections (split-K reduce is not selected).
  "aiter.triton.batched_gemm_a16wfp4_kernel"
)

# With --disable-benchmark-cuda-graph, DeepSeek YaRN RoPE and MLA KV-cache
# insertion remain separate vLLM custom ops.
MLA_EAGER_KERNEL_IDS=(
  "vllm.hip.rotary_embedding"
  "vllm.hip.concat_and_cache_mla"
)

# With production compilation/CUDA Graph, the RoPE + cache insertion fusion
# may replace the separate cache op. Keep the separate entries too because the
# pass is range-dependent and can legitimately fall back.
MLA_GRAPH_KERNEL_IDS=(
  "vllm.hip.rotary_embedding"
  "vllm.hip.concat_and_cache_mla"
  "vllm.hip.concat_and_cache_mla_rope_fused"
)

# Unquantized attention/router/lm-head linears first enter the vLLM custom op.
# Shape dispatch then selects skinny vLLM kernels, Kimi-tuned AITER ASM/Triton,
# or untraceable FlyDSL/BLAS backends. The MXFP4 entries serve dense layer 0
# and every non-fused shared expert. The basic FP4 reduce is shape-dependent.
LINEAR_KERNEL_IDS=(
  "vllm.hip.rocm_unquantized_gemm"
  "vllm.hip.wvsplitkrc"
  "vllm.hip.wvsplitk"

  "aiter.hip.gemm_a16w16_asm"
  "aiter.triton.gemm_a16_w16_kernel"

  "aiter.triton.dynamic_mxfp4_quant_kernel"
  "aiter.triton.gemm_basic_gemm_afp4wfp4.gemm_afp4wfp4_kernel"
  "aiter.triton.gemm_basic_gemm_afp4wfp4.gemm_afp4wfp4_reduce_kernel"
)

# The M32 steady decode and long-prefill paths select different quantization
# branches. CK stage2 is only expected around M=1/ramp boundaries; Kimi's M32,
# M4096 and M8192 tuned expert GEMMs are FlyDSL and therefore not traceable.
# CK stage1 is deliberately excluded: its Kimi config starts at M=16384 while
# this image's normal max_num_batched_tokens is 8192.
MOE_KERNEL_IDS=(
  "aiter.hip.biased_grouped_topk"
  "aiter.hip.moe_sorting_fwd"
  "aiter.hip.fused_dynamic_mxfp4_quant_moe_sort_hip"
  "aiter.hip.dynamic_per_group_scaled_quant_fp4"
  "aiter.hip.mxfp4_moe_sort_hip"
  "aiter.hip.ck_moe_stage2"
)

# Eager mode selects vLLM's ROCm C kernels. Do not replace these with
# vllm.hip.custom_ops.*: the IR dispatcher calls kernels/vllm_c.py directly.
NORM_EAGER_KERNEL_IDS=(
  "vllm.hip.kernels_vllm_c.rms_norm"
  "vllm.hip.kernels_vllm_c.fused_add_rms_norm"
)

# Graph mode enables AITER RMSNorm priority and optional O2 fusion passes.
NORM_GRAPH_KERNEL_IDS=(
  "vllm.hip.kernels_aiter_ops.rms_norm"
  "vllm.hip.kernels_aiter_ops.fused_add_rms_norm"
  # vLLM imports aiter.rms_norm; @compile_ops exposes it as rmsnorm2d_fwd.
  "aiter.hip.rmsnorm2d_fwd"
  # hidden_size=7168 takes the non-CK fused-add branch.
  "aiter.hip.add_rmsnorm"
  "aiter.hip.fused_qk_rmsnorm"
  "aiter.hip.fused_allreduce_rmsnorm"
)

KERNEL_IDS=()
declare -A SEEN_KERNEL_IDS=()
append_kernel_ids() {
  local kernel_id
  for kernel_id in "$@"; do
    if [[ -z "${SEEN_KERNEL_IDS[$kernel_id]+present}" ]]; then
      KERNEL_IDS+=("$kernel_id")
      SEEN_KERNEL_IDS["$kernel_id"]=1
    fi
  done
}

case "$KERNEL_GROUP" in
  all)
    append_kernel_ids "${MLA_COMMON_KERNEL_IDS[@]}"
    append_kernel_ids "${LINEAR_KERNEL_IDS[@]}"
    append_kernel_ids "${MOE_KERNEL_IDS[@]}"
    ;;
  mla)
    append_kernel_ids "${MLA_COMMON_KERNEL_IDS[@]}"
    ;;
  linear)
    append_kernel_ids "${LINEAR_KERNEL_IDS[@]}"
    ;;
  moe)
    append_kernel_ids "${MOE_KERNEL_IDS[@]}"
    ;;
  norm)
    ;;
  *)
    echo "Invalid KERNEL_GROUP=$KERNEL_GROUP (expected all|mla|moe|linear|norm)" >&2
    exit 1
    ;;
esac

if [[ "$KERNEL_GROUP" == "all" || "$KERNEL_GROUP" == "mla" ]]; then
  if [[ "$DISABLE_BENCHMARK_CUDA_GRAPH" != "0" ]]; then
    append_kernel_ids "${MLA_EAGER_KERNEL_IDS[@]}"
  else
    append_kernel_ids "${MLA_GRAPH_KERNEL_IDS[@]}"
  fi
fi

if [[ "$KERNEL_GROUP" == "all" || "$KERNEL_GROUP" == "norm" ]]; then
  if [[ "$DISABLE_BENCHMARK_CUDA_GRAPH" != "0" ]]; then
    append_kernel_ids "${NORM_EAGER_KERNEL_IDS[@]}"
  else
    append_kernel_ids "${NORM_GRAPH_KERNEL_IDS[@]}"
  fi
fi

if ((${#KERNEL_IDS[@]} == 0)); then
  echo "No kernel IDs selected." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for the fixed-image trace." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not available." >&2
  exit 1
fi
if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
  echo "Required image is not local: $DOCKER_IMAGE" >&2
  echo "Run: docker pull $DOCKER_IMAGE" >&2
  exit 1
fi
ACTUAL_IMAGE_ID="$(docker image inspect "$DOCKER_IMAGE" --format '{{.Id}}')"
if [[ "$ACTUAL_IMAGE_ID" != "$EXPECTED_IMAGE_ID" ]]; then
  echo "Local image does not match the v0.23 tracing registry:" >&2
  echo "  expected: $EXPECTED_IMAGE_ID" >&2
  echo "  actual:   $ACTUAL_IMAGE_ID" >&2
  echo "Pull the fixed image again before tracing: docker pull $DOCKER_IMAGE" >&2
  exit 1
fi

# Verify that this is still the intended Kimi workload. In particular, the
# torch profiler remains off and the global AITER switch remains on.
python3 - "$TRACE_BENCH" <<'PY'
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
bench = data.get("benchmark", data)
errors = []
if bench.get("framework") != "vllm":
    errors.append(f"framework must be vllm, got {bench.get('framework')!r}")
if bench.get("model") != "amd/Kimi-K2.6-MXFP4":
    errors.append(
        "model must be amd/Kimi-K2.6-MXFP4, "
        f"got {bench.get('model')!r}"
    )
envs = bench.get("envs") or {}
if str(envs.get("VLLM_ROCM_USE_AITER", "")).lower() not in {"1", "true"}:
    errors.append("benchmark.envs.VLLM_ROCM_USE_AITER must be enabled")
torch_profiler = ((bench.get("profiler") or {}).get("torch_profiler") or {})
if bool(torch_profiler.get("enabled", False)):
    errors.append("profiler.torch_profiler.enabled must remain false")
hf_cache_path = bench.get("hf_cache_path")
if not hf_cache_path:
    errors.append(
        "benchmark.hf_cache_path must be explicit; this host's Docker daemon "
        "cannot traverse the default ~/.cache/huggingface path"
    )
if errors:
    raise SystemExit("Benchmark preflight failed:\n  " + "\n  ".join(errors))
print(
    "Benchmark preflight OK: "
    f"model={bench['model']} TP={envs.get('TP')} "
    f"CONC={envs.get('CONC')} ISL={envs.get('ISL')} OSL={envs.get('OSL')} "
    f"hf_cache_path={hf_cache_path}"
)
PY

HF_CACHE_PATH="$(python3 - "$TRACE_BENCH" <<'PY'
import os
import sys
from pathlib import Path

import yaml

path = Path(sys.argv[1])
data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
bench = data.get("benchmark", data)
print(Path(os.path.expandvars(os.path.expanduser(str(bench["hf_cache_path"])))).resolve())
PY
)"
if [[ ! -d "$HF_CACHE_PATH" ]]; then
  echo "Configured HuggingFace cache is not a directory: $HF_CACHE_PATH" >&2
  exit 1
fi

# Magpie bind-mounts the cache into /root/.cache/huggingface. Test that exact
# mount before spending time extracting/patching sources or starting vLLM.
# This catches root-squashed or non-traversable host paths such as ~/.cache.
if [[ "$DRY_RUN" == "0" ]]; then
  if ! docker run --rm \
    --entrypoint /bin/true \
    -v "$HF_CACHE_PATH:/root/.cache/huggingface" \
    "$DOCKER_IMAGE"; then
    echo "Docker cannot bind-mount the configured HuggingFace cache:" >&2
    echo "  $HF_CACHE_PATH" >&2
    echo "Set benchmark.hf_cache_path to a Docker-accessible directory." >&2
    exit 1
  fi
  echo "HuggingFace cache mount smoke check OK: $HF_CACHE_PATH"
fi

# Verify every source-derived ID against the exact v0.23 registry and guard
# against ambiguous runtime coverage (coverage is keyed by kernel_name).
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
    raise SystemExit(
        "IDs not present in the v0.23 registry:\n  " + "\n  ".join(missing)
    )

names = collections.defaultdict(list)
for kernel_id in requested:
    names[by_id[kernel_id].kernel_name].append(kernel_id)
duplicates = {name: ids for name, ids in names.items() if len(ids) > 1}
if duplicates:
    print(
        "Warning: runtime coverage cannot distinguish these duplicate names:",
        file=sys.stderr,
    )
    for name, ids in sorted(duplicates.items()):
        print(f"  {name}: {', '.join(ids)}", file=sys.stderr)

print(f"Validated {len(requested)} kernel IDs against {image}:")
for index, kernel_id in enumerate(requested, 1):
    entry = by_id[kernel_id]
    print(
        f"  {index:2d}. {kernel_id} "
        f"[{entry.trace_mode}; kernel_name={entry.kernel_name}]"
    )
PY

BENCH_SHA_BEFORE="$(sha256sum "$TRACE_BENCH" | awk '{print $1}')"
verify_benchmark_unchanged() {
  local after
  after="$(sha256sum "$TRACE_BENCH" | awk '{print $1}')"
  if [[ "$after" != "$BENCH_SHA_BEFORE" ]]; then
    echo "ERROR: source benchmark YAML was modified: $TRACE_BENCH" >&2
    return 1
  fi
}
on_exit() {
  local status=$?
  if ! verify_benchmark_unchanged; then
    status=1
  fi
  exit "$status"
}
trap on_exit EXIT

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
if [[ "$TRACE_ALL" != "0" ]]; then
  CMD+=(--trace-all)
fi
if [[ "$DRY_RUN" != "0" ]]; then
  CMD+=(--dry-run)
fi
if [[ "$DISABLE_BENCHMARK_CUDA_GRAPH" != "0" ]]; then
  CMD+=(--disable-benchmark-cuda-graph)
fi

echo
echo "Apex root:        $APEX_ROOT"
echo "Magpie root:      $MAGPIE_ROOT"
echo "Results dir:      $RESULTS_DIR"
echo "Benchmark config: $TRACE_BENCH"
echo "HF cache path:    $HF_CACHE_PATH"
echo "Docker image:     $DOCKER_IMAGE"
echo "Kernel group:     $KERNEL_GROUP"
echo "Kernel IDs:       ${#KERNEL_IDS[@]}"
echo "Max records:      $MAX_RECORDS (shared by all targets in each process)"
echo "Sample rate:      $SAMPLE_RATE"
echo "Benchmark timeout:$BENCHMARK_TIMEOUT"
echo "Trace all:        $TRACE_ALL"
echo "Dry run:          $DRY_RUN"
echo "Disable graph:    $DISABLE_BENCHMARK_CUDA_GRAPH"
echo "Require all hits: $REQUIRE_ALL"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n\n'

set +e
"${CMD[@]}" 2>&1 | tee "$RESULTS_DIR/trace-kernel.log"
CLI_STATUS=${PIPESTATUS[0]}
set -e
if ((CLI_STATUS != 0)); then
  echo "trace-kernel exited with status $CLI_STATUS" >&2
  exit "$CLI_STATUS"
fi

verify_benchmark_unchanged

# trace-kernel may return process status 0 when the benchmark ran but no target
# was observed, so validate artifacts and coverage explicitly.
python3 - \
  "$RESULTS_DIR" \
  "$DOCKER_IMAGE" \
  "$MAX_RECORDS" \
  "$REQUIRE_ALL" \
  "$DRY_RUN" \
  "$HF_CACHE_PATH" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

results_dir = Path(sys.argv[1])
expected_image = sys.argv[2]
max_records = int(sys.argv[3])
require_all = sys.argv[4] != "0"
dry_run = sys.argv[5] != "0"
expected_hf_cache = str(Path(sys.argv[6]).resolve())

trace_config_path = results_dir / "trace_config.json"
manifest_path = results_dir / "patched_files" / "patch_manifest.json"
if not trace_config_path.is_file():
    raise SystemExit(f"Missing trace config: {trace_config_path}")
if not manifest_path.is_file():
    raise SystemExit(f"Missing patch manifest: {manifest_path}")

trace_config = json.loads(trace_config_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if trace_config.get("docker_image") != expected_image:
    raise SystemExit(
        "trace_config.json image mismatch: "
        f"{trace_config.get('docker_image')!r} != {expected_image!r}"
    )

targets = trace_config.get("targets") or []
overlay_count = len(manifest.get("overlay_modules") or {})
print(
    f"Patch validation OK: targets={len(targets)}, "
    f"overlay_modules={overlay_count}"
)

if dry_run:
    patched_files = trace_config.get("patch_results") or []
    if not patched_files:
        raise SystemExit("Dry-run produced no patched files")
    print(f"Dry-run OK: compiled {len(patched_files)} patch results; model not started.")
    raise SystemExit(0)

traced_benchmark_path = results_dir / "benchmark" / "trace_benchmark_config.yaml"
if not traced_benchmark_path.is_file():
    raise SystemExit(f"Missing generated benchmark config: {traced_benchmark_path}")
traced_data = yaml.safe_load(traced_benchmark_path.read_text(encoding="utf-8")) or {}
traced_bench = traced_data.get("benchmark", traced_data)
actual_image = traced_bench.get("docker_image")
if actual_image != expected_image:
    raise SystemExit(
        "Generated benchmark image mismatch: "
        f"{actual_image!r} != {expected_image!r}"
    )
print(f"Generated benchmark image is pinned correctly: {actual_image}")
actual_hf_cache = traced_bench.get("hf_cache_path")
if not actual_hf_cache or str(Path(str(actual_hf_cache)).resolve()) != expected_hf_cache:
    raise SystemExit(
        "Generated benchmark HuggingFace cache mismatch: "
        f"{actual_hf_cache!r} != {expected_hf_cache!r}"
    )
print(f"Generated benchmark HF cache is pinned correctly: {actual_hf_cache}")

trace_result_path = results_dir / "trace_result.json"
if not trace_result_path.is_file():
    raise SystemExit(f"Missing trace result: {trace_result_path}")
result = json.loads(trace_result_path.read_text(encoding="utf-8"))
coverage = result.get("target_events_found") or {}

ids_by_name = defaultdict(list)
for target in targets:
    ids_by_name[str(target.get("kernel_name"))].append(str(target.get("kernel_id")))

hit_names = sorted(name for name, found in coverage.items() if found)
missing_names = sorted(name for name, found in coverage.items() if not found)
print(f"Runtime coverage: {len(hit_names)}/{len(coverage)} unique kernel names hit")
if hit_names:
    print("Hit IDs:")
    for name in hit_names:
        print(f"  {name}: {', '.join(ids_by_name[name])}")
if missing_names:
    print("Shape/branch-dependent candidates not observed:")
    for name in missing_names:
        print(f"  {name}: {', '.join(ids_by_name[name])}")

cap_reached = []
for path in sorted((results_dir / "trace_raw").glob("*.jsonl")):
    valid_events = 0
    target_events = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        valid_events += 1
        if event.get("kind") != "module_import":
            target_events += 1
    if valid_events >= max_records or target_events >= max_records:
        cap_reached.append((path.name, valid_events, target_events))
if cap_reached:
    print(
        "WARNING: trace record cap may have hidden late/decode targets; "
        "increase MAX_RECORDS or run a smaller KERNEL_GROUP:",
        file=sys.stderr,
    )
    for name, valid, target in cap_reached:
        print(
            f"  {name}: valid_events={valid}, target_events={target}, "
            f"cap={max_records}",
            file=sys.stderr,
        )

if not result.get("success", False):
    raise SystemExit(
        "trace_result.success is false; inspect trace-kernel.log and "
        "benchmark/benchmark_result.json"
    )
if require_all and missing_names:
    raise SystemExit(
        f"REQUIRE_ALL=1 but {len(missing_names)} kernel names were not observed"
    )
PY

echo
if [[ "$DRY_RUN" != "0" ]]; then
  echo "Dry-run artifacts:"
  echo "  trace config:   $RESULTS_DIR/trace_config.json"
  echo "  patched files:  $RESULTS_DIR/patched_files/"
  echo "  command log:    $RESULTS_DIR/trace-kernel.log"
else
  echo "Trace artifacts:"
  echo "  trace result:   $RESULTS_DIR/trace_result.json"
  echo "  trace config:   $RESULTS_DIR/trace_config.json"
  echo "  patched files:  $RESULTS_DIR/patched_files/"
  echo "  raw events:     $RESULTS_DIR/trace_raw/"
  echo "  target shapes:  $RESULTS_DIR/target_kernel_tensor_shapes.json"
  echo "  ranges:         $RESULTS_DIR/workload_ranges.json"
  echo "  summary:        $RESULTS_DIR/workload_summary.md"
  echo "  command log:    $RESULTS_DIR/trace-kernel.log"
fi
