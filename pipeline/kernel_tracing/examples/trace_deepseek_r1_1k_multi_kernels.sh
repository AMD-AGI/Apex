#!/usr/bin/env bash
set -euo pipefail

# DeepSeek R1 MXFP4 1K/1K multi-kernel tracing example.
#
# Basic use:
#   HF_CACHE_PATH=/path/to/huggingface \
#     bash pipeline/kernel_tracing/examples/trace_deepseek_r1_1k_multi_kernels.sh
#
# Common knobs:
#   MAX_RECORDS=300000 SAMPLE_RATE=0.01 BENCHMARK_TIMEOUT=7200
#   TRACE_ALL=1 DRY_RUN=1 DISABLE_BENCHMARK_CUDA_GRAPH=0

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

export HF_CACHE_PATH="${HF_CACHE_PATH:-${HF_HOME:-$HOME/.cache/huggingface}}"
export MAGPIE_RUN_MODE="${MAGPIE_RUN_MODE:-docker}"

TS="$(date -u +%Y%m%d_%H%M%S)"
DEFAULT_RESULTS_DIR="$APEX_ROOT/tmp/results_trace_deepseek_r1_1k_multi_kernels_${TS}"
RESULTS_DIR="${RESULTS_DIR:-$DEFAULT_RESULTS_DIR}"
mkdir -p "$RESULTS_DIR"

# Keep the script and its one supported workload paired in this directory.
DEFAULT_TRACE_BENCH="$SCRIPT_DIR/benchmark_sglang_dsr1_isl1024_osl1024.yaml"
TRACE_BENCH="${BENCH_CONFIG:-$DEFAULT_TRACE_BENCH}"
if [[ ! -f "$TRACE_BENCH" ]]; then
  echo "Benchmark config not found: $TRACE_BENCH" >&2
  exit 1
fi

DOCKER_IMAGE="${DOCKER_IMAGE:-lmsysorg/sglang:v0.5.12-rocm720-mi35x}"
MAX_RECORDS="${MAX_RECORDS:-300000}"
SAMPLE_RATE="${SAMPLE_RATE:-0.01}"
BENCHMARK_TIMEOUT="${BENCHMARK_TIMEOUT:-7200}"

# These cover the CSV analysis targets that are present in the fixed-image
# trace registries under pipeline/kernel_tracing/registries/.
# Keep this list explicit: different workload/model analyses often need
# different kernel sets.
KERNEL_IDS=(
  # moe_stage2_ck_mxgemm
  "aiter.hip.ck_moe_stage2"

  # fp4_dense_gemm
  "aiter.triton.gemm_basic_gemm_afp4wfp4.gemm_afp4wfp4_kernel"
  "aiter.triton.triton_gluon_gemm_afp4wfp4.gemm_afp4wfp4_kernel"

  # mla_decode_attention
  "aiter.hip.mla_decode_stage1_asm_fwd"
  "aiter.hip.mla_reduce_v1"

  # moe_dynamic_quant_sort
  "aiter.hip.fused_dynamic_mxfp4_quant_moe_sort_hip"

  # batched_fp4_gemm_attention
  "aiter.triton.batched_gemm_a16wfp4_kernel"

  # moe_sorting
  "aiter.hip.moe_sorting_fwd"

  # fused_rms_mxfp4_quant
  "aiter.triton.fused_rms_mxfp4_quant_kernel"

  # moe_gate_bf16_gemm
  "aiter.hip.gemm_a16w16_asm"

  # fused_flatten_mxfp4_quant
  "aiter.triton.fused_flatten_mxfp4_quant"

  # rmsnorm
  "aiter.hip.add_rmsnorm"

  # moe_topk_routing
  "aiter.hip.biased_grouped_topk"

  # mla_qk_rope_cache_update
  "aiter.triton.fused_qk_rope_cat_and_cache_mla_kernel"

  # moe_append_shared_experts
  "sglang.triton.fused_append_shared_experts_kernel"

  # decode_all_gather
  "sglang.hip.reg_all_gather_into_tensor"
)

KERNEL_ARGS=()
for kernel_id in "${KERNEL_IDS[@]}"; do
  KERNEL_ARGS+=(--kernel-id "$kernel_id")
done

TRACE_ALL_ARGS=()
if [[ "${TRACE_ALL:-0}" != "0" ]]; then
  TRACE_ALL_ARGS+=(--trace-all)
fi

DRY_RUN_ARGS=()
if [[ "${DRY_RUN:-0}" != "0" ]]; then
  DRY_RUN_ARGS+=(--dry-run)
fi

CUDA_GRAPH_ARGS=()
if [[ "${DISABLE_BENCHMARK_CUDA_GRAPH:-1}" != "0" ]]; then
  CUDA_GRAPH_ARGS+=(--disable-benchmark-cuda-graph)
fi

echo "Apex root:       $APEX_ROOT"
echo "Magpie root:     $MAGPIE_ROOT"
echo "Results dir:     $RESULTS_DIR"
echo "Benchmark config:$TRACE_BENCH"
echo "Docker image:    $DOCKER_IMAGE"
echo "HF cache path:   $HF_CACHE_PATH"
echo "Run mode:        $MAGPIE_RUN_MODE"
echo "Framework:       ${FRAMEWORK:-sglang}"
echo "CUDA graph:      disable overlay=${DISABLE_BENCHMARK_CUDA_GRAPH:-1}"
echo "Kernel IDs:      ${#KERNEL_IDS[@]}"
echo "Trace all:       ${TRACE_ALL:-0}"
echo "Dry run:         ${DRY_RUN:-0}"

python3 workload_optimizer.py trace-kernel \
  -r "$RESULTS_DIR" \
  "${KERNEL_ARGS[@]}" \
  "${TRACE_ALL_ARGS[@]}" \
  "${DRY_RUN_ARGS[@]}" \
  "${CUDA_GRAPH_ARGS[@]}" \
  --docker-image "$DOCKER_IMAGE" \
  --max-records "$MAX_RECORDS" \
  --sample-rate "$SAMPLE_RATE" \
  --benchmark-timeout "$BENCHMARK_TIMEOUT" \
  --framework "${FRAMEWORK:-sglang}" \
  -b "$TRACE_BENCH"

echo
echo "Trace artifacts:"
echo "  trace result:   $RESULTS_DIR/trace_result.json"
echo "  target shapes:  $RESULTS_DIR/target_kernel_tensor_shapes.json"
echo "  ranges:         $RESULTS_DIR/workload_ranges.json"
echo "  summary:        $RESULTS_DIR/workload_summary.md"
