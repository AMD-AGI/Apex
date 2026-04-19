#!/usr/bin/env bash
# scripts/run_gluon_agent_eval.sh — Multi-agent Gluon evaluation harness.
#
# Drives Apex's three agent backends (claude, codex, cursor) through three
# scenarios:
#   A  Optimize a Gluon baseline (files/gluon_vector_add_baseline.py)
#   B  Optimize a Triton baseline asking for a Gluon rewrite
#      (files/triton_rms_norm_baseline.py + --rewrite-as gluon)
#   C  Full GPT-OSS-20B e2e with --rewrite-as gluon (top-k 10, 3 iters)
#
# Usage:
#   scripts/run_gluon_agent_eval.sh                       # all scenarios all agents
#   SCENARIOS="A B" scripts/run_gluon_agent_eval.sh       # subset of scenarios
#   AGENTS="claude" scripts/run_gluon_agent_eval.sh       # subset of agents
#   DRY_RUN=1 scripts/run_gluon_agent_eval.sh             # plumb --dry-run
#   ENABLE_ENV_POLICY=1 ENV_POLICY_AGENT=codex scripts/run_gluon_agent_eval.sh
#                                                         # add prompt-driven env-policy task to scenario C
#
# The script never invokes the agent CLIs directly — it only spawns
# `python3 workload_optimizer.py …`, which constructs a natural-language
# prompt and hands it to the configured backend. Agents themselves only
# ever see the rendered task prompt (saved per-iteration under
# <task_dir>/prompts/iter_NNN_user.md).

set -uo pipefail

cd "$(dirname "$0")/.."
APEX_ROOT="$(pwd)"

AGENTS="${AGENTS:-claude codex cursor}"
SCENARIOS="${SCENARIOS:-A B C}"
RESULTS_ROOT="${RESULTS_ROOT:-${APEX_ROOT}/results/gluon_eval}"
GPU="${GPU:-gfx950}"
MAX_ITERS_AB="${MAX_ITERS_AB:-3}"
MAX_TURNS_AB="${MAX_TURNS_AB:-25}"
MAX_ITERS_C="${MAX_ITERS_C:-3}"
MAX_TURNS_C="${MAX_TURNS_C:-25}"
TOP_K_C="${TOP_K_C:-10}"
DRY_RUN="${DRY_RUN:-0}"
TIMEOUT_AB="${TIMEOUT_AB:-90m}"
TIMEOUT_C="${TIMEOUT_C:-12h}"

# Phase 3 enabler: prompt-driven env-policy task. When --enable-env-policy
# is set, after kernel rewrites Apex asks the env-policy agent to propose a
# minimal `VLLM_ROCM_USE_AITER_*` diff via a yaml block, then runs an
# additional benchmark draw with that diff applied so the contribution is
# attributed independently from kernel rewrites.
#   ENABLE_ENV_POLICY=1                      # opt-in, default off
#   ENV_POLICY_AGENT="codex"                 # cursor|codex|claude
ENABLE_ENV_POLICY="${ENABLE_ENV_POLICY:-0}"
ENV_POLICY_AGENT="${ENV_POLICY_AGENT:-}"

MAGPIE_ROOT="${MAGPIE_ROOT:-/home/sirafati/code_combine/Magpie}"
GPTOSS_YAML="${MAGPIE_ROOT}/examples/benchmarks/benchmark_vllm_gptoss_20b.yaml"

mkdir -p "${RESULTS_ROOT}"
echo "Apex Gluon eval starting at $(date -u +%FT%TZ)"
echo "  AGENTS=${AGENTS}"
echo "  SCENARIOS=${SCENARIOS}"
echo "  RESULTS_ROOT=${RESULTS_ROOT}"
echo "  DRY_RUN=${DRY_RUN}"
echo

dry_flag=""
if [[ "${DRY_RUN}" == "1" ]]; then
    dry_flag="--dry-run"
fi

env_policy_flags=""
if [[ "${ENABLE_ENV_POLICY}" == "1" ]]; then
    env_policy_flags="--enable-env-policy"
    if [[ -n "${ENV_POLICY_AGENT}" ]]; then
        env_policy_flags="${env_policy_flags} --env-policy-agent ${ENV_POLICY_AGENT}"
    fi
fi

run_scenario_a() {  # gluon baseline
    local agent="$1"
    local rdir="${RESULTS_ROOT}/${agent}/scenario_a_gluon_baseline"
    mkdir -p "${rdir}"
    echo "[A][${agent}] -> ${rdir}"
    timeout "${TIMEOUT_AB}" python3 workload_optimizer.py optimize-kernel \
        --kernel files/gluon_vector_add_baseline.py \
        --kernel-type gluon \
        --kernel-name vector_add \
        --correctness-mode pytorch \
        --reference files/gluon_vector_add_baseline.py \
        --gpu "${GPU}" \
        --agent-backend "${agent}" \
        -r "${rdir}" \
        --max-iterations "${MAX_ITERS_AB}" \
        --max-turns "${MAX_TURNS_AB}" \
        ${dry_flag} \
        > "${rdir}/run.log" 2>&1
    local rc=$?
    echo "[A][${agent}] exit=${rc}"
    return ${rc}
}

run_scenario_b() {  # triton -> gluon rewrite
    local agent="$1"
    local rdir="${RESULTS_ROOT}/${agent}/scenario_b_triton_to_gluon"
    mkdir -p "${rdir}"
    echo "[B][${agent}] -> ${rdir}"
    timeout "${TIMEOUT_AB}" python3 workload_optimizer.py optimize-kernel \
        --kernel files/triton_rms_norm_baseline.py \
        --kernel-type triton \
        --kernel-name rms_norm \
        --rewrite-as gluon \
        --correctness-mode pytorch \
        --reference files/triton_rms_norm_baseline.py \
        --gpu "${GPU}" \
        --agent-backend "${agent}" \
        -r "${rdir}" \
        --max-iterations "${MAX_ITERS_AB}" \
        --max-turns "${MAX_TURNS_AB}" \
        ${dry_flag} \
        > "${rdir}/run.log" 2>&1
    local rc=$?
    echo "[B][${agent}] exit=${rc}"
    return ${rc}
}

run_scenario_c() {  # full GPT-OSS-20B e2e
    local agent="$1"
    local rdir="${RESULTS_ROOT}/${agent}/scenario_c_gptoss20b_gluon"
    mkdir -p "${rdir}"
    echo "[C][${agent}] -> ${rdir}"
    if [[ ! -f "${GPTOSS_YAML}" ]]; then
        echo "[C][${agent}] SKIP: ${GPTOSS_YAML} not found" | tee "${rdir}/run.log"
        return 2
    fi
    MAGPIE_ROOT="${MAGPIE_ROOT}" timeout "${TIMEOUT_C}" python3 workload_optimizer.py run \
        -r "${rdir}" \
        -b "${GPTOSS_YAML}" \
        --kernel-types triton \
        --rewrite-as gluon \
        --top-k "${TOP_K_C}" \
        --max-iterations "${MAX_ITERS_C}" \
        --max-turns "${MAX_TURNS_C}" \
        --gpu "${GPU}" \
        --agent-backend "${agent}" \
        --leaderboard \
        ${env_policy_flags} \
        ${dry_flag} \
        > "${rdir}/run.log" 2>&1
    local rc=$?
    echo "[C][${agent}] exit=${rc}"
    return ${rc}
}

declare -A SUMMARY=()

for s in ${SCENARIOS}; do
    for a in ${AGENTS}; do
        t0=$(date +%s)
        case "${s}" in
            A) run_scenario_a "${a}"; rc=$? ;;
            B) run_scenario_b "${a}"; rc=$? ;;
            C) run_scenario_c "${a}"; rc=$? ;;
            *) echo "Unknown scenario ${s}"; exit 1 ;;
        esac
        t1=$(date +%s)
        SUMMARY["${a}/${s}"]="rc=${rc} elapsed=$((t1 - t0))s"
    done
done

echo
echo "=========================================="
echo " Apex Gluon eval complete at $(date -u +%FT%TZ)"
echo "=========================================="
for k in "${!SUMMARY[@]}"; do
    echo "  ${k}: ${SUMMARY[${k}]}"
done

echo
echo "Aggregate report:"
python3 scripts/aggregate_gluon_eval_report.py "${RESULTS_ROOT}" \
    || echo "  (aggregate report failed, see traceback above)"
