#!/usr/bin/env bash
# setup_files.sh
# Downloads code and documentation into the RL env kernel optimization sandbox.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$SCRIPT_DIR/code"
DOCS_DIR="$SCRIPT_DIR/docs"

clone() {
    local repo="$1"
    local dest="$2"
    if [ -d "$dest/.git" ]; then
        echo "  [skip] $dest already exists"
    else
        echo "  cloning $repo -> $dest"
        git clone --depth=1 "$repo" "$dest"
    fi
}

fetch_page() {
    local url="$1"
    local dest="$2"
    echo "  fetching $url"
    mkdir -p "$(dirname "$dest")"
    curl -fsSL "$url" -o "$dest" 2>/dev/null || echo "    [warn] failed: $url"
}

fetch_site() {
    # Shallow recursive wget of a docs site (depth 3, HTML only, no external links).
    local url="$1"
    local dest="$2"
    echo "  mirroring $url -> $dest"
    mkdir -p "$dest"
    wget \
        --recursive \
        --level=3 \
        --no-parent \
        --no-host-directories \
        --cut-dirs=999 \
        --accept "html,htm" \
        --reject "search*,genindex*,py-modindex*" \
        --convert-links \
        --quiet \
        --directory-prefix="$dest" \
        "$url" 2>/dev/null || echo "    [warn] partial or failed: $url"
}

# ── Code repos ─────────────────────────────────────────────────────────────────

echo ""
echo "=== ROCm libraries ==="
mkdir -p "$CODE_DIR/rocm"

# Monorepo (hipBLAS, rocBLAS, hipFFT, hipSPARSE, rocSPARSE, rocPRIM, rocsolver, MIOpen…)
clone https://github.com/ROCm/rocm-libraries       "$CODE_DIR/rocm/rocm-libraries"

# Individual libraries
clone https://github.com/ROCm/hipBLASLt            "$CODE_DIR/rocm/hipBLASLt"
clone https://github.com/ROCm/rocRAND              "$CODE_DIR/rocm/hipRAND"      # hipRAND lives here
clone https://github.com/ROCm/hipSOLVER            "$CODE_DIR/rocm/hipSOLVER"
clone https://github.com/ROCm/hipSPARSELt          "$CODE_DIR/rocm/hipSPARSELt"
clone https://github.com/ROCm/hipCUB               "$CODE_DIR/rocm/hipCUB"
clone https://github.com/ROCm/hipTensor            "$CODE_DIR/rocm/hipTensor"
clone https://github.com/ROCm/rocPRIM              "$CODE_DIR/rocm/rocPRIM"
clone https://github.com/ROCm/rocThrust            "$CODE_DIR/rocm/rocThrust"
clone https://github.com/ROCm/rocWMMA              "$CODE_DIR/rocm/rocWMMA"
clone https://github.com/ROCm/composable_kernel    "$CODE_DIR/rocm/composable_kernel"
clone https://github.com/ROCm/AMDMIGraphX          "$CODE_DIR/rocm/AMDMIGraphX"
clone https://github.com/ROCm/rccl                 "$CODE_DIR/rocm/rccl"
clone https://github.com/ROCm/rocSHMEM             "$CODE_DIR/rocm/rocSHMEM"
clone https://github.com/AMD-AGI/Magpie            "$CODE_DIR/rocm/Magpie"

echo ""
echo "=== AITER ==="
clone https://github.com/ROCm/aiter               "$CODE_DIR/aiter"

echo ""
echo "=== Triton ==="
clone https://github.com/triton-lang/triton        "$CODE_DIR/triton"

echo ""
echo "=== SGLang & vLLM ==="
clone https://github.com/sgl-project/sglang        "$CODE_DIR/sglang"
clone https://github.com/vllm-project/vllm         "$CODE_DIR/vllm"

# ── Documentation ──────────────────────────────────────────────────────────────

echo ""
echo "=== ROCm documentation ==="
fetch_site \
    "https://rocm.docs.amd.com/en/latest/" \
    "$DOCS_DIR/rocm"

echo ""
echo "=== HIP documentation ==="
fetch_site \
    "https://rocm.docs.amd.com/projects/HIP/en/latest/" \
    "$DOCS_DIR/hip"

# Individual key pages as fallback
for page in \
    "understand/programming_model.html" \
    "understand/performance_optimization.html" \
    "understand/hardware_implementation.html" \
    "how-to/performance_guidelines.html" \
    "how-to/hip_cpp_language_extensions.html"
do
    fetch_page \
        "https://rocm.docs.amd.com/projects/HIP/en/latest/$page" \
        "$DOCS_DIR/hip-pages/$(basename "$page")"
done

echo ""
echo "=== Composable Kernel (CK) documentation ==="
fetch_site \
    "https://rocm.docs.amd.com/projects/composable_kernel/en/latest/" \
    "$DOCS_DIR/ck"

echo ""
echo "=== AMD Instinct GPU documentation (MI300/MI350 series) ==="
fetch_site \
    "https://instinct.docs.amd.com/" \
    "$DOCS_DIR/amd-instinct"

# ISA and whitepaper PDFs for each CDNA generation
echo ""
echo "=== AMD GPU ISA PDFs ==="
mkdir -p "$DOCS_DIR/isa"

# CDNA4 / MI350 series
fetch_page \
    "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna4-architecture-white-paper.pdf" \
    "$DOCS_DIR/isa/amd-cdna4-whitepaper.pdf"

# CDNA3 / MI300 series (most relevant for MI355X class)
fetch_page \
    "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna3-whitepaper.pdf" \
    "$DOCS_DIR/isa/amd-cdna3-whitepaper.pdf"

# GPU ISA reference (CDNA3)
fetch_page \
    "https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/amd-gpu-isa-overview.pdf" \
    "$DOCS_DIR/isa/amd-gpu-isa-overview.pdf"

echo ""
echo "=== HIP/ROCm optimization tutorials ==="
mkdir -p "$DOCS_DIR/tutorials"

# ROCm performance blog posts (GitHub)
fetch_page \
    "https://gpuopen.com/learn/amdgpu-optimization-series/" \
    "$DOCS_DIR/tutorials/gpuopen-optimization-series.html"

fetch_page \
    "https://rocm.docs.amd.com/en/latest/how-to/system-optimization/index.html" \
    "$DOCS_DIR/tutorials/rocm-system-optimization.html"

fetch_page \
    "https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/index.html" \
    "$DOCS_DIR/tutorials/rocm-for-ai.html"

echo ""
echo "=== Triton documentation ==="
fetch_site \
    "https://triton-lang.org/main/" \
    "$DOCS_DIR/triton"

# Individual Triton doc pages as fallback
for page in \
    "getting-started/tutorials/index.html" \
    "python-api/triton.language.html" \
    "programming-guide/chapter-1/introduction.html" \
    "programming-guide/chapter-3/debugging.html"
do
    fetch_page \
        "https://triton-lang.org/main/$page" \
        "$DOCS_DIR/triton-pages/$(basename "$page")"
done

echo ""
echo "Done. Layout:"
echo "  $CODE_DIR/  — cloned source repositories"
echo "  $DOCS_DIR/  — downloaded documentation"
