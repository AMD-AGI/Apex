#!/usr/bin/env bash
# Build Composable Kernel examples for Accordo validation.
# Usage: bash tools/build_ck.sh [--gpu-targets gfx950] [--jobs N]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CK_DIR="${SCRIPT_DIR}/rocm/composable_kernel"

GPU_TARGETS=""
JOBS="$(nproc 2>/dev/null || echo 4)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-targets) GPU_TARGETS="$2"; shift 2 ;;
        --jobs)        JOBS="$2";        shift 2 ;;
        -j)            JOBS="$2";        shift 2 ;;
        -h|--help)
            echo "Usage: bash tools/build_ck.sh [--gpu-targets gfx950] [--jobs N]"
            echo ""
            echo "Builds CK example binaries for Accordo HSA-level validation."
            echo ""
            echo "Options:"
            echo "  --gpu-targets   GPU architecture target (default: auto-detect or gfx950)"
            echo "  --jobs, -j      Parallel build jobs (default: nproc)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$GPU_TARGETS" ]]; then
    if command -v rocminfo &>/dev/null; then
        GPU_TARGETS="$(rocminfo 2>/dev/null | grep -oP 'gfx\w+' | head -1 || true)"
    fi
    if [[ -z "$GPU_TARGETS" ]]; then
        GPU_TARGETS="gfx950"
        echo "[build_ck] Could not auto-detect GPU target, defaulting to ${GPU_TARGETS}"
    else
        echo "[build_ck] Auto-detected GPU target: ${GPU_TARGETS}"
    fi
fi

if [[ ! -d "$CK_DIR" ]]; then
    echo "[build_ck] ERROR: CK directory not found at ${CK_DIR}"
    exit 1
fi

BUILD_DIR="${CK_DIR}/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "[build_ck] Configuring CK for GPU_TARGETS=${GPU_TARGETS}..."
cmake "${CK_DIR}" \
    -DCMAKE_CXX_COMPILER=hipcc \
    -DCMAKE_BUILD_TYPE=Release \
    -DGPU_TARGETS="${GPU_TARGETS}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "[build_ck] Building examples with -j${JOBS}..."
make -j"${JOBS}" examples 2>&1 | tail -20

echo ""
echo "[build_ck] Build complete. Binaries:"
BUILT=0
if [[ -d "${BUILD_DIR}/bin" ]]; then
    for bin in "${BUILD_DIR}/bin"/example_*; do
        if [[ -x "$bin" ]]; then
            echo "  $(basename "$bin")"
            BUILT=$((BUILT + 1))
        fi
    done
fi
echo "[build_ck] ${BUILT} example binaries built in ${BUILD_DIR}/bin/"
