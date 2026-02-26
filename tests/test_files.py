"""
test_files.py — Content and existence tests for files/ directory artifacts.

These tests do NOT run setup_files.sh (that would clone many large repos).
They verify:
  - The setup script itself is correct and executable
  - The two markdown best-practices files exist and have required sections
"""

import os
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT  = Path(__file__).parent.parent
FILES_DIR  = REPO_ROOT / "files"
SETUP_SH   = FILES_DIR / "setup_files.sh"
HIP_MD     = FILES_DIR / "hip_best_practices.md"
TRITON_MD  = FILES_DIR / "triton_best_practices.md"


# ── setup_files.sh ────────────────────────────────────────────────────────────

class TestSetupScript:
    def test_exists(self):
        assert SETUP_SH.exists(), f"{SETUP_SH} not found"

    def test_executable(self):
        assert os.access(SETUP_SH, os.X_OK), f"{SETUP_SH} is not executable"

    def test_shebang(self):
        first_line = SETUP_SH.read_text().splitlines()[0]
        assert first_line.startswith("#!"), "setup_files.sh must have a shebang line"

    @pytest.mark.parametrize("repo_url", [
        "https://github.com/ROCm/rocm-libraries",
        "https://github.com/ROCm/hipBLASLt",
        "https://github.com/ROCm/composable_kernel",
        "https://github.com/ROCm/AMDMIGraphX",
        "https://github.com/ROCm/rccl",
        "https://github.com/AMD-AGI/Magpie",
        "https://github.com/ROCm/aiter",
        "https://github.com/triton-lang/triton",
        "https://github.com/sgl-project/sglang",
        "https://github.com/vllm-project/vllm",
    ])
    def test_contains_repo_url(self, repo_url):
        content = SETUP_SH.read_text()
        assert repo_url in content, \
            f"{SETUP_SH.name} missing repo URL: {repo_url}"

    @pytest.mark.parametrize("doc_site", [
        "rocm.docs.amd.com",
        "instinct.docs.amd.com",
        "triton-lang.org",
    ])
    def test_contains_doc_site(self, doc_site):
        content = SETUP_SH.read_text()
        assert doc_site in content, \
            f"{SETUP_SH.name} missing documentation site: {doc_site}"

    def test_has_error_handling(self):
        content = SETUP_SH.read_text()
        assert "set -e" in content, "setup_files.sh should use 'set -e' for error handling"

    def test_creates_code_and_docs_dirs(self):
        content = SETUP_SH.read_text()
        assert "code" in content
        assert "docs" in content


# ── hip_best_practices.md ─────────────────────────────────────────────────────

class TestHIPBestPractices:
    def test_exists(self):
        assert HIP_MD.exists(), f"{HIP_MD} not found"

    def test_non_empty(self):
        assert HIP_MD.stat().st_size > 1000, "HIP best practices file seems too small"

    @pytest.mark.parametrize("section", [
        "Coalescing",
        "Occupancy",
        "LDS",
        "Register",
        "Atomic",
        "Profil",          # profiling / rocprof
        "MFMA",            # CDNA-specific
    ])
    def test_contains_section(self, section):
        content = HIP_MD.read_text()
        assert section.lower() in content.lower(), \
            f"HIP best practices missing section about: {section}"

    def test_contains_code_blocks(self):
        content = HIP_MD.read_text()
        assert "```" in content, "HIP best practices should contain code examples"

    def test_contains_rocm_link(self):
        content = HIP_MD.read_text()
        assert "rocm.docs.amd.com" in content or "rocm" in content.lower()

    def test_has_checklist(self):
        content = HIP_MD.read_text()
        assert "- [ ]" in content, "HIP best practices should have a quick checklist"

    def test_mentions_target_gpu(self):
        content = HIP_MD.read_text()
        # Should mention MI300 or MI350 or CDNA
        assert any(k in content for k in ["MI300", "MI350", "MI355", "CDNA"])


# ── triton_best_practices.md ──────────────────────────────────────────────────

class TestTritonBestPractices:
    def test_exists(self):
        assert TRITON_MD.exists(), f"{TRITON_MD} not found"

    def test_non_empty(self):
        assert TRITON_MD.stat().st_size > 1000, "Triton best practices file seems too small"

    @pytest.mark.parametrize("section", [
        "autotune",
        "tl.dot",
        "mask",
        "reduction",
        "constexpr",
        "AMD",
        "debug",
    ])
    def test_contains_section(self, section):
        content = TRITON_MD.read_text()
        assert section.lower() in content.lower(), \
            f"Triton best practices missing content about: {section}"

    def test_contains_code_blocks(self):
        content = TRITON_MD.read_text()
        assert "```" in content

    def test_has_checklist(self):
        content = TRITON_MD.read_text()
        assert "- [ ]" in content

    def test_mentions_mfma_or_tensor_cores(self):
        content = TRITON_MD.read_text()
        assert any(k in content for k in ["MFMA", "mfma", "matrix core", "tl.dot"])

    def test_mentions_autotuning(self):
        content = TRITON_MD.read_text()
        assert "@triton.autotune" in content or "autotune" in content.lower()
