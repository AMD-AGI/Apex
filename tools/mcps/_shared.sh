#!/usr/bin/env bash
# _shared.sh — Common helpers for MCP setup scripts.
# Sourced by individual setup.sh files; not meant to run standalone.

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_DIR="$TOOLS_DIR/rocm"
DOC_DIR="$TOOLS_DIR/doc"
JSONS_DIR="$TOOLS_DIR/jsons"
ROCM_JSON="$JSONS_DIR/rocm.json"

# ── Clone ROCm repos from rocm.json ─────────────────────────────────────────

clone_rocm_repos() {
    echo "=== Cloning ROCm repositories ==="

    if [[ ! -f "$ROCM_JSON" ]]; then
        echo "ERROR: $ROCM_JSON not found"; return 1
    fi

    command -v jq >/dev/null 2>&1 || { echo "ERROR: 'jq' is required (apt install jq)"; return 1; }

    mkdir -p "$ROCM_DIR"

    local repos
    repos=$(jq -r '.rocm_libraries[].github' "$ROCM_JSON" | sort -u)

    local total cloned=0 skipped=0
    total=$(echo "$repos" | wc -l | tr -d ' ')
    echo "  $total repositories to process"

    while IFS= read -r repo_url; do
        [[ -z "$repo_url" ]] && continue
        local name
        name=$(basename "$repo_url" .git)
        local target="$ROCM_DIR/$name"

        if [[ -d "$target/.git" ]]; then
            skipped=$((skipped + 1))
        else
            echo "  cloning $name ..."
            git clone --depth 1 -q "$repo_url" "$target" 2>/dev/null && cloned=$((cloned + 1)) || echo "  [warn] failed: $name"
        fi
    done <<< "$repos"

    echo "  [ok] cloned=$cloned  skipped=$skipped"
}

# ── Download documentation PDFs ──────────────────────────────────────────────

download_docs() {
    echo "=== Downloading documentation PDFs ==="
    mkdir -p "$DOC_DIR"

    local -a PDF_URLS=(
        "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf"
        "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf"
        "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-datasheet.pdf"
        "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/instinct-mi325x-platform-datasheet.pdf"
        "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi350x-gpu-brochure.pdf"
        "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/product-briefs/amd-instinct-mi350x-platform-brochure.pdf"
        "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf"
        "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf"
        "https://rocm.docs.amd.com/_/downloads/HIP/en/docs-6.1.2/pdf/"
        "https://rocm.docs.amd.com/_/downloads/HIPCC/en/latest/pdf/"
        "https://rocm.docs.amd.com/_/downloads/composable_kernel/en/latest/pdf/"
        "https://rocm.docs.amd.com/_/downloads/rocWMMA/en/latest/pdf/"
        "https://rocm.docs.amd.com/_/downloads/hipBLASLt/en/docs-6.4.2/pdf/"
        "https://rocm.docs.amd.com/_/downloads/rocBLAS/en/docs-6.4.0/pdf/"
        "https://rocm.docs.amd.com/_/downloads/Tensile/en/docs-6.3.0/pdf/"
        "https://rocm.docs.amd.com/_/downloads/hipTensor/en/docs-7.0.1/pdf/"
        "https://rocm.docs.amd.com/_/downloads/hipCUB/en/docs-6.4.3/pdf/"
        "https://rocm.docs.amd.com/_/downloads/rocprofiler/en/latest/pdf/"
        "https://rocm.docs.amd.com/_/downloads/rocprofiler-sdk/en/docs-6.4.2/pdf/"
        "https://rocm.docs.amd.com/_/downloads/roctracer/en/latest/pdf/"
        "https://rocm.docs.amd.com/_/downloads/amdsmi/en/latest/pdf/"
        "https://rocm.docs.amd.com/_/downloads/rocm_smi_lib/en/docs-7.0.1/pdf/"
        "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf"
        "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf"
        "https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/instinct-mi100-cdna1-shader-instruction-set-architecture.pdf"
        "https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf"
    )

    local ua="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    local downloaded=0 skipped=0

    for url in "${PDF_URLS[@]}"; do
        local base
        base="$(basename "${url%%\?*}")"
        [[ -z "$base" || "$base" == "/" ]] && base="$(echo -n "$url" | md5sum | cut -c1-16)"
        [[ "$base" != *.pdf ]] && base="${base}.pdf"
        local outfile="$DOC_DIR/$base"

        if [[ -s "$outfile" ]]; then
            skipped=$((skipped + 1))
        else
            curl -L --retry 2 --connect-timeout 30 --max-time 300 -s -A "$ua" -o "$outfile" "$url" 2>/dev/null
            [[ -s "$outfile" ]] && downloaded=$((downloaded + 1)) || rm -f "$outfile"
        fi
    done

    echo "  [ok] downloaded=$downloaded  skipped=$skipped"
}
