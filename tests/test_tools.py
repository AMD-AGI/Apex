# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_tools.py — Tests for tools/setup_tools.sh and the MCP server modules.

Verifies:
  - The setup script is correct and executable
  - All MCP server Python modules are valid and have the right structure
  - Each MCP has a working individual setup.sh
  - MCP tool schemas are present
"""

import ast
import os
from pathlib import Path

import pytest

REPO_ROOT   = Path(__file__).parent.parent
TOOLS_DIR   = REPO_ROOT / "tools"
MCPS_DIR    = TOOLS_DIR / "mcps"
SETUP_SH    = TOOLS_DIR / "setup_tools.sh"
SKILLS_DIR  = TOOLS_DIR / "skills"

RAG_DIR     = MCPS_DIR / "rag_tool"
FUSION_DIR  = MCPS_DIR / "fusion_advisor"
GPU_DIR     = MCPS_DIR / "gpu_info"
SOURCE_DIR  = MCPS_DIR / "source_finder"
MAGPIE_DIR  = MCPS_DIR / "magpie"
JSONS_DIR   = TOOLS_DIR / "jsons"
SHARED_SH   = MCPS_DIR / "_shared.sh"

EXPECTED_SKILLS = [
    "aiter-reflection",
    "gpu-architecture-fundamentals",
    "hip-kernel-optimization",
    "kernel-exp-history",
    "mi300-cdna3-architecture",
    "mi300-hip-programming-insights",
    "mi300-hip-vs-nvidia",
    "pytorch-kernel-optimization",
    "rocprof-compute",
    "skill-creator",
    "triton-hip-reference-kernel-search",
    "triton-kernel-optimization",
    "triton-kernel-reflection-prompts",
]

ALL_MCP_DIRS = [RAG_DIR, FUSION_DIR, GPU_DIR, SOURCE_DIR]
ALL_MCP_DIRS_WITH_MAGPIE = ALL_MCP_DIRS + [MAGPIE_DIR]


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_functions(source: str) -> set[str]:
    tree = ast.parse(source)
    return {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _get_classes(source: str) -> set[str]:
    tree = ast.parse(source)
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}


# ── setup_tools.sh ────────────────────────────────────────────────────────────

class TestSetupToolsScript:
    def test_exists(self):
        assert SETUP_SH.exists(), f"{SETUP_SH} not found"

    def test_executable(self):
        assert os.access(SETUP_SH, os.X_OK)

    def test_shebang(self):
        assert SETUP_SH.read_text().splitlines()[0].startswith("#!")

    def test_installs_magpie(self):
        content = SETUP_SH.read_text()
        assert "AMD-AGI/Magpie" in content

    def test_has_error_handling(self):
        content = SETUP_SH.read_text()
        assert "set -e" in content

    def test_supports_claude_flag(self):
        content = SETUP_SH.read_text()
        assert "--claude" in content

    def test_supports_cursor_flag(self):
        content = SETUP_SH.read_text()
        assert "--cursor" in content

    def test_supports_skip_repos(self):
        content = SETUP_SH.read_text()
        assert "--skip-repos" in content

    def test_supports_skip_docs(self):
        content = SETUP_SH.read_text()
        assert "--skip-docs" in content

    def test_registers_all_mcps(self):
        content = SETUP_SH.read_text()
        for name in ["magpie", "kernel-rag", "fusion-advisor", "gpu-info", "source-finder"]:
            assert name in content, f"MCP '{name}' not registered in setup_tools.sh"

    def test_clones_rocm_repos(self):
        content = SETUP_SH.read_text()
        assert "clone_rocm_repos" in content

    def test_downloads_docs(self):
        content = SETUP_SH.read_text()
        assert "download_docs" in content

    def test_sources_shared_helpers(self):
        content = SETUP_SH.read_text()
        assert "_shared.sh" in content

    def test_syncs_skills_for_claude(self):
        content = SETUP_SH.read_text()
        assert ".claude/skills" in content

    def test_syncs_skills_for_cursor(self):
        content = SETUP_SH.read_text()
        assert ".cursor/skills" in content

    def test_uses_rsync_for_skills(self):
        content = SETUP_SH.read_text()
        assert "rsync" in content


# ── Individual setup.sh scripts ──────────────────────────────────────────────

class TestIndividualSetupScripts:
    @pytest.mark.parametrize("mcp_dir", ALL_MCP_DIRS_WITH_MAGPIE, ids=lambda d: d.name)
    def test_setup_exists(self, mcp_dir):
        assert (mcp_dir / "setup.sh").exists()

    @pytest.mark.parametrize("mcp_dir", ALL_MCP_DIRS_WITH_MAGPIE, ids=lambda d: d.name)
    def test_setup_executable(self, mcp_dir):
        assert os.access(mcp_dir / "setup.sh", os.X_OK)

    @pytest.mark.parametrize("mcp_dir", ALL_MCP_DIRS_WITH_MAGPIE, ids=lambda d: d.name)
    def test_setup_shebang(self, mcp_dir):
        first_line = (mcp_dir / "setup.sh").read_text().splitlines()[0]
        assert first_line.startswith("#!")

    @pytest.mark.parametrize("mcp_dir", ALL_MCP_DIRS_WITH_MAGPIE, ids=lambda d: d.name)
    def test_setup_supports_claude(self, mcp_dir):
        content = (mcp_dir / "setup.sh").read_text()
        assert "--claude" in content

    @pytest.mark.parametrize("mcp_dir", ALL_MCP_DIRS_WITH_MAGPIE, ids=lambda d: d.name)
    def test_setup_supports_cursor(self, mcp_dir):
        content = (mcp_dir / "setup.sh").read_text()
        assert "--cursor" in content

    @pytest.mark.parametrize("mcp_dir", ALL_MCP_DIRS, ids=lambda d: d.name)
    def test_setup_references_server(self, mcp_dir):
        content = (mcp_dir / "setup.sh").read_text()
        assert "server.py" in content

    @pytest.mark.parametrize("mcp_dir", ALL_MCP_DIRS, ids=lambda d: d.name)
    def test_pyproject_exists(self, mcp_dir):
        assert (mcp_dir / "pyproject.toml").exists()


# ── Shared helpers & data ────────────────────────────────────────────────────

class TestSharedInfrastructure:
    def test_shared_helper_exists(self):
        assert SHARED_SH.exists()

    def test_shared_helper_has_clone_function(self):
        content = SHARED_SH.read_text()
        assert "clone_rocm_repos" in content

    def test_shared_helper_has_download_function(self):
        content = SHARED_SH.read_text()
        assert "download_docs" in content

    def test_shared_helper_reads_rocm_json(self):
        content = SHARED_SH.read_text()
        assert "rocm.json" in content

    def test_jsons_dir_exists(self):
        assert JSONS_DIR.exists()

    def test_rocm_json_exists(self):
        assert (JSONS_DIR / "rocm.json").exists()

    def test_hip_sheet_exists(self):
        assert (JSONS_DIR / "hip_sheet.json").exists()

    def test_triton_sheet_exists(self):
        assert (JSONS_DIR / "triton_sheet.json").exists()

    def test_rocm_json_valid(self):
        import json
        data = json.loads((JSONS_DIR / "rocm.json").read_text())
        assert "rocm_libraries" in data
        assert len(data["rocm_libraries"]) > 0

    def test_rocm_json_has_github_urls(self):
        import json
        data = json.loads((JSONS_DIR / "rocm.json").read_text())
        for lib in data["rocm_libraries"]:
            assert "github" in lib, f"Missing 'github' in {lib.get('name', '?')}"
            assert lib["github"].startswith("https://github.com/"), f"Bad URL: {lib['github']}"

    def test_rag_setup_clones_repos(self):
        content = (RAG_DIR / "setup.sh").read_text()
        assert "clone_rocm_repos" in content

    def test_rag_setup_downloads_docs(self):
        content = (RAG_DIR / "setup.sh").read_text()
        assert "download_docs" in content

    def test_source_finder_setup_clones_repos(self):
        content = (SOURCE_DIR / "setup.sh").read_text()
        assert "clone_rocm_repos" in content


# ── Magpie MCP ───────────────────────────────────────────────────────────────

class TestMagpieMCP:
    def test_setup_exists(self):
        assert (MAGPIE_DIR / "setup.sh").exists()

    def test_setup_clones_magpie(self):
        content = (MAGPIE_DIR / "setup.sh").read_text()
        assert "AMD-AGI/Magpie" in content

    def test_setup_uses_ssh(self):
        content = (MAGPIE_DIR / "setup.sh").read_text()
        assert "git@github.com:" in content

    def test_setup_installs_editable(self):
        content = (MAGPIE_DIR / "setup.sh").read_text()
        assert "pip install" in content

    def test_setup_registers_magpie_mcp(self):
        content = (MAGPIE_DIR / "setup.sh").read_text()
        assert "magpie" in content


# ── MCP structure (common to all) ────────────────────────────────────────────

class TestMCPCommonStructure:
    @pytest.mark.parametrize("mcp_dir", ALL_MCP_DIRS, ids=lambda d: d.name)
    def test_server_exists(self, mcp_dir):
        assert (mcp_dir / "server.py").exists()

    @pytest.mark.parametrize("mcp_dir", ALL_MCP_DIRS, ids=lambda d: d.name)
    def test_server_valid_python(self, mcp_dir):
        source = (mcp_dir / "server.py").read_text()
        ast.parse(source)

    @pytest.mark.parametrize("mcp_dir", ALL_MCP_DIRS, ids=lambda d: d.name)
    def test_server_uses_mcp_sdk(self, mcp_dir):
        source = (mcp_dir / "server.py").read_text()
        assert "from mcp.server" in source
        assert "from mcp.server.stdio" in source

    @pytest.mark.parametrize("mcp_dir", ALL_MCP_DIRS, ids=lambda d: d.name)
    def test_server_defines_list_tools(self, mcp_dir):
        source = (mcp_dir / "server.py").read_text()
        assert "list_tools" in _get_functions(source)

    @pytest.mark.parametrize("mcp_dir", ALL_MCP_DIRS, ids=lambda d: d.name)
    def test_server_defines_call_tool(self, mcp_dir):
        source = (mcp_dir / "server.py").read_text()
        assert "call_tool" in _get_functions(source)

    @pytest.mark.parametrize("mcp_dir", ALL_MCP_DIRS, ids=lambda d: d.name)
    def test_server_defines_main(self, mcp_dir):
        source = (mcp_dir / "server.py").read_text()
        assert "main" in _get_functions(source)

    @pytest.mark.parametrize("mcp_dir", ALL_MCP_DIRS, ids=lambda d: d.name)
    def test_server_has_init(self, mcp_dir):
        assert (mcp_dir / "__init__.py").exists()


# ── RAG tool: index.py ────────────────────────────────────────────────────────

class TestRAGIndex:
    @pytest.fixture(scope="class")
    def index_source(self):
        return (RAG_DIR / "index.py").read_text()

    def test_parses_as_valid_python(self, index_source):
        ast.parse(index_source)

    def test_defines_build_index_function(self, index_source):
        assert "build_index" in _get_functions(index_source)

    def test_defines_chunk_text_function(self, index_source):
        assert "chunk_text" in _get_functions(index_source)

    def test_uses_chromadb(self, index_source):
        assert "chromadb" in index_source

    def test_uses_sentence_transformers(self, index_source):
        assert "sentence-transformers" in index_source or "SentenceTransformer" in index_source

    def test_has_argparse_cli(self, index_source):
        assert "argparse" in index_source

    def test_indexes_code_and_docs(self, index_source):
        assert "code" in index_source
        assert "docs" in index_source


# ── RAG tool: server.py ───────────────────────────────────────────────────────

class TestRAGServer:
    @pytest.fixture(scope="class")
    def server_source(self):
        return (RAG_DIR / "server.py").read_text()

    def test_defines_search_tool(self, server_source):
        assert "search_kernel_optimization" in server_source

    def test_defines_snippet_tool(self, server_source):
        assert "get_optimization_snippet" in server_source

    def test_defines_playbook_tool(self, server_source):
        assert "get_optimization_playbook" in server_source

    def test_defines_analyze_tool(self, server_source):
        assert "analyze_kernel_for_optimization" in server_source

    def test_tool_schemas_present(self, server_source):
        assert "query" in server_source

    def test_has_optimization_patterns(self, server_source):
        assert "pattern" in server_source.lower()

    def test_supports_hip_and_triton(self, server_source):
        assert "hip" in server_source.lower()
        assert "triton" in server_source.lower()


# ── Fusion Advisor: server.py ─────────────────────────────────────────────────

class TestFusionAdvisorServer:
    @pytest.fixture(scope="class")
    def server_source(self):
        return (FUSION_DIR / "server.py").read_text()

    def test_defines_fusion_patterns(self, server_source):
        assert "FUSION_PATTERNS" in server_source

    def test_defines_library_fused_kernels(self, server_source):
        assert "LIBRARY_FUSED_KERNELS" in server_source

    def test_detect_tool(self, server_source):
        assert "detect_fusion_opportunities" in server_source

    def test_generate_tool(self, server_source):
        assert "generate_fused_kernel" in server_source

    def test_validate_tool(self, server_source):
        assert "validate_fusion" in server_source

    def test_magpie_integration(self, server_source):
        assert "parse_magpie_output" in server_source

    def test_memory_savings_tool(self, server_source):
        assert "calculate_memory_savings" in server_source

    def test_library_check_tool(self, server_source):
        assert "check_library_fusion" in server_source

    def test_benchmark_tool(self, server_source):
        assert "benchmark_compare" in server_source

    def test_has_kernel_dataclasses(self, server_source):
        classes = _get_classes(server_source)
        assert "KernelNode" in classes
        assert "FusionOpportunity" in classes

    def test_covers_known_fusion_patterns(self, server_source):
        for pattern in ["elementwise_chain", "gemm_epilogue", "attention_block",
                        "norm_activation", "residual_norm"]:
            assert pattern in server_source, f"Missing fusion pattern: {pattern}"

    def test_covers_library_backends(self, server_source):
        for lib in ["ck", "hipblaslt", "vllm", "triton"]:
            assert lib in server_source.lower(), f"Missing library backend: {lib}"


# ── GPU Info: server.py ───────────────────────────────────────────────────────

class TestGPUInfoServer:
    @pytest.fixture(scope="class")
    def server_source(self):
        return (GPU_DIR / "server.py").read_text()

    def test_defines_arch_specs(self, server_source):
        assert "ARCH_SPECS" in server_source

    def test_get_gpu_info_tool(self, server_source):
        assert "get_gpu_info" in server_source

    def test_optimization_hints_tool(self, server_source):
        assert "get_arch_optimization_hints" in server_source

    def test_set_target_arch_tool(self, server_source):
        assert "set_target_arch" in server_source

    def test_list_architectures_tool(self, server_source):
        assert "list_supported_architectures" in server_source

    def test_has_gpu_detector_class(self, server_source):
        assert "GPUDetector" in _get_classes(server_source)

    def test_covers_cdna_architectures(self, server_source):
        for arch in ["gfx942", "gfx90a", "gfx908"]:
            assert arch in server_source, f"Missing CDNA arch: {arch}"

    def test_covers_mi300(self, server_source):
        assert "MI300" in server_source

    def test_covers_mfma_instructions(self, server_source):
        assert "mfma_f32_32x32x8_f16" in server_source

    def test_has_kernel_type_hints(self, server_source):
        for ktype in ["gemm", "reduction", "elementwise", "attention", "moe"]:
            assert ktype in server_source, f"Missing kernel type hint: {ktype}"


# ── Source Finder: server.py ──────────────────────────────────────────────────

class TestSourceFinderServer:
    @pytest.fixture(scope="class")
    def server_source(self):
        return (SOURCE_DIR / "server.py").read_text()

    def test_find_kernel_source_tool(self, server_source):
        assert "find_kernel_source" in server_source

    def test_demangle_tool(self, server_source):
        assert "demangle_kernel_name" in server_source

    def test_identify_origin_tool(self, server_source):
        assert "identify_kernel_origin" in server_source

    def test_find_alternative_tool(self, server_source):
        assert "find_library_alternative" in server_source

    def test_decode_tensile_tool(self, server_source):
        assert "decode_tensile_kernel" in server_source

    def test_classify_tool(self, server_source):
        assert "classify_kernel" in server_source

    def test_hotspots_tool(self, server_source):
        assert "identify_hotspots" in server_source

    def test_ck_template_tool(self, server_source):
        assert "find_ck_template" in server_source

    def test_has_ck_templates_db(self, server_source):
        assert "CK_TEMPLATES" in server_source

    def test_has_kernel_patterns(self, server_source):
        assert "KERNEL_PATTERNS" in server_source

    def test_has_library_alternatives(self, server_source):
        assert "LIBRARY_ALTERNATIVES" in server_source

    def test_covers_known_origins(self, server_source):
        for origin in ["tensile", "composable_kernel", "pytorch", "triton", "vllm", "aiter"]:
            assert origin in server_source, f"Missing kernel origin: {origin}"

    def test_covers_ck_operation_types(self, server_source):
        for op in ["gemm", "attention", "moe", "normalization", "reduction"]:
            assert op in server_source, f"Missing CK operation type: {op}"


# ── Skills ────────────────────────────────────────────────────────────────────

class TestSkills:
    def test_skills_dir_exists(self):
        assert SKILLS_DIR.exists(), f"{SKILLS_DIR} not found"

    def test_all_expected_skills_present(self):
        actual = sorted(d.name for d in SKILLS_DIR.iterdir() if d.is_dir())
        for skill in EXPECTED_SKILLS:
            assert skill in actual, f"Missing skill: {skill}"

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_has_skill_md(self, skill_name):
        skill_md = SKILLS_DIR / skill_name / "SKILL.md"
        assert skill_md.exists(), f"{skill_md} not found"

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_md_has_frontmatter(self, skill_name):
        content = (SKILLS_DIR / skill_name / "SKILL.md").read_text()
        assert content.startswith("---"), f"{skill_name}/SKILL.md missing YAML frontmatter"

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_md_has_name_field(self, skill_name):
        content = (SKILLS_DIR / skill_name / "SKILL.md").read_text()
        assert "name:" in content, f"{skill_name}/SKILL.md missing 'name:' field"

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_md_has_description_field(self, skill_name):
        content = (SKILLS_DIR / skill_name / "SKILL.md").read_text()
        assert "description:" in content, f"{skill_name}/SKILL.md missing 'description:' field"

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_name_is_hyphen_case(self, skill_name):
        assert skill_name == skill_name.lower(), f"Skill name not lowercase: {skill_name}"
        assert "_" not in skill_name, f"Skill name uses underscores instead of hyphens: {skill_name}"

    def test_skill_count(self):
        actual = [d.name for d in SKILLS_DIR.iterdir() if d.is_dir()]
        assert len(actual) >= len(EXPECTED_SKILLS), (
            f"Expected at least {len(EXPECTED_SKILLS)} skills, found {len(actual)}"
        )

    def test_kernel_exp_history_has_references(self):
        refs = SKILLS_DIR / "kernel-exp-history" / "references"
        assert refs.exists()
        assert (refs / "kernel_exp_dataclass.py").exists()

    def test_rocprof_compute_has_references(self):
        refs = SKILLS_DIR / "rocprof-compute" / "references"
        assert refs.exists()

    def test_skill_creator_has_scripts(self):
        scripts = SKILLS_DIR / "skill-creator" / "scripts"
        assert scripts.exists()
        assert (scripts / "init_skill.py").exists()
        assert (scripts / "quick_validate.py").exists()

    def test_triton_hip_search_has_references(self):
        refs = SKILLS_DIR / "triton-hip-reference-kernel-search" / "references"
        assert refs.exists()
        assert (refs / "SEARCH.md").exists()

    def test_mi300_skills_have_references(self):
        for name in ["mi300-cdna3-architecture", "mi300-hip-programming-insights", "mi300-hip-vs-nvidia"]:
            refs = SKILLS_DIR / name / "references"
            assert refs.exists(), f"{name} missing references/"
