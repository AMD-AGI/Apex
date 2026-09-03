"""Executable architecture contract for the clean-cut Apex source tree."""

from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"
APEX_ROOT = SOURCE_ROOT / "apex"
MAX_MODULE_LINES = 600
MAX_FUNCTION_LINES = 80
README_SECTIONS = (
    "Purpose",
    "Public API",
    "Invariants",
    "Dependencies",
    "Failure semantics",
    "Tests",
    "Provenance",
)
LAYER = {
    "core": 0,
    "intake": 1,
    "knowledge": 1,
    "orchestration": 1,
    "ports": 1,
    "storage": 1,
    "context": 2,
    "diagnostics": 2,
    "execution": 2,
    "evaluation": 3,
    "runtime": 3,
    "benchmark": 4,
    "delivery": 4,
    "rl": 5,
    "reporting": 6,
    "optimization": 7,
    "mcp": 8,
    "bootstrap": 9,
    "cli": 10,
}


def _python_files() -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(APEX_ROOT.rglob("*.py"))
        if "__pycache__" not in path.parts
    )


def _package_dirs() -> tuple[Path, ...]:
    return tuple(
        path.parent for path in _python_files() if path.name == "__init__.py"
    )


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _source_domain(path: Path) -> str:
    relative = path.relative_to(APEX_ROOT)
    if len(relative.parts) == 1:
        return "bootstrap" if path.stem == "bootstrap" else "root"
    return relative.parts[0]


def _target_domain(module: str) -> str | None:
    if module == "apex":
        return "root"
    if not module.startswith("apex."):
        return None
    return module.split(".", 2)[1]


def _apex_imports(tree: ast.Module) -> tuple[tuple[int, str, tuple[str, ...]], ...]:
    found: list[tuple[int, str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("apex"):
                    found.append((node.lineno, alias.name, ()))
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("apex"):
                found.append(
                    (node.lineno, node.module, tuple(alias.name for alias in node.names))
                )
    return tuple(found)


def test_every_package_has_a_substantive_readme() -> None:
    violations: list[str] = []
    for package in _package_dirs():
        readme = package / "README.md"
        if not readme.is_file():
            violations.append(f"{package.relative_to(ROOT)}: missing README.md")
            continue
        text = readme.read_text(encoding="utf-8")
        lines = text.splitlines()
        if len([line for line in text.splitlines() if line.strip()]) < 18:
            violations.append(f"{readme.relative_to(ROOT)}: fewer than 18 content lines")
        for section in README_SECTIONS:
            marker = f"## {section}"
            if marker not in lines:
                violations.append(f"{readme.relative_to(ROOT)}: missing {marker}")
                continue
            start = lines.index(marker) + 1
            body = []
            for line in lines[start:]:
                if line.startswith("## "):
                    break
                if line.strip():
                    body.append(line)
            if not body:
                violations.append(f"{readme.relative_to(ROOT)}: empty {marker}")
    assert not violations, "\n" + "\n".join(sorted(violations))


def test_modules_and_functions_stay_reviewable() -> None:
    violations: list[str] = []
    for path in _python_files():
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > MAX_MODULE_LINES:
            violations.append(
                f"{path.relative_to(ROOT)}: {line_count}>{MAX_MODULE_LINES} lines"
            )
        for node in ast.walk(_tree(path)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                span = (node.end_lineno or node.lineno) - node.lineno + 1
                if span > MAX_FUNCTION_LINES:
                    violations.append(
                        f"{path.relative_to(ROOT)}:{node.lineno} "
                        f"{node.name} {span}>{MAX_FUNCTION_LINES} lines"
                    )
    assert not violations, "\n" + "\n".join(sorted(violations))


def test_cross_package_imports_follow_the_dag_and_public_boundaries() -> None:
    violations: list[str] = []
    edges: set[tuple[str, str]] = set()
    for path in _python_files():
        source = _source_domain(path)
        for line, module, names in _apex_imports(_tree(path)):
            target = _target_domain(module)
            if target is None or target == source or target == "root":
                continue
            if source == "root":
                continue
            if source not in LAYER or target not in LAYER:
                violations.append(
                    f"{path.relative_to(ROOT)}:{line}: unknown layer {source}->{target}"
                )
                continue
            edges.add((source, target))
            if LAYER[target] >= LAYER[source]:
                violations.append(
                    f"{path.relative_to(ROOT)}:{line}: non-downward import {source}->{target}"
                )
            module_tail = module.split(".")[2:]
            private_parts = [part for part in module_tail if part.startswith("_")]
            private_names = [name for name in names if name.startswith("_")]
            if private_parts or private_names:
                violations.append(
                    f"{path.relative_to(ROOT)}:{line}: private cross-package import "
                    f"{module} {private_names}"
                )
    assert not violations, "\n" + "\n".join(sorted(violations))
    assert all(LAYER[target] < LAYER[source] for source, target in edges)


def test_module_bodies_are_declarative_at_import_time() -> None:
    """Reject executable module bodies; the subprocess test catches dynamics."""

    allowed_nodes = (
        ast.Expr,
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )
    safe_calls = {"TypeVar", "frozenset", "re.compile", "sorted", "str.strip"}
    violations: list[str] = []
    for path in _python_files():
        for node in _tree(path).body:
            if isinstance(node, ast.If) and _is_main_guard(node.test):
                continue
            if not isinstance(node, allowed_nodes):
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: "
                    f"executable {type(node).__name__}"
                )
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                for call in (item for item in ast.walk(value) if isinstance(item, ast.Call)):
                    name = _call_name(call.func)
                    if name not in safe_calls:
                        violations.append(
                            f"{path.relative_to(ROOT)}:{call.lineno}: "
                            f"module-level call {name}"
                        )
    assert not violations, "\n" + "\n".join(sorted(violations))


def _is_main_guard(value: ast.expr) -> bool:
    return (
        isinstance(value, ast.Compare)
        and isinstance(value.left, ast.Name)
        and value.left.id == "__name__"
        and len(value.ops) == 1
        and isinstance(value.ops[0], ast.Eq)
        and len(value.comparators) == 1
        and isinstance(value.comparators[0], ast.Constant)
        and value.comparators[0].value == "__main__"
    )


def _call_name(value: ast.expr) -> str:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        if isinstance(value.value, ast.Name):
            return f"{value.value.id}.{value.attr}"
        if isinstance(value.value, ast.Constant) and isinstance(value.value.value, str):
            return f"str.{value.attr}"
    return ast.unparse(value)


@pytest.mark.parametrize(
    "package",
    [
        ".".join(("apex", *path.relative_to(APEX_ROOT).parts))
        .removesuffix(".__init__")
        .replace("/__init__", "")
        for path in _package_dirs()
    ],
)
def test_package_all_is_explicit_unique_and_resolvable(package: str) -> None:
    module = importlib.import_module(package)
    exports = getattr(module, "__all__", None)
    assert isinstance(exports, (list, tuple)), f"{package} must define __all__"
    assert all(isinstance(name, str) and name and not name.startswith("_") for name in exports)
    assert len(exports) == len(set(exports)), f"{package} has duplicate exports"
    missing = [name for name in exports if not hasattr(module, name)]
    assert not missing, f"{package} cannot resolve exports: {missing}"


def test_importing_all_modules_has_no_observable_side_effects(tmp_path: Path) -> None:
    modules = []
    for path in _python_files():
        relative = path.relative_to(SOURCE_ROOT).with_suffix("")
        parts = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        modules.append(".".join(parts))
    script = """
import importlib, json, os, pathlib, sys, threading
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
before_env = dict(os.environ)
before_threads = {thread.ident for thread in threading.enumerate()}
for name in json.loads(sys.argv[3]):
    importlib.import_module(name)
result = {
    'environment_changed': before_env != dict(os.environ),
    'new_threads': sorted(str(thread.name) for thread in threading.enumerate()
                          if thread.ident not in before_threads),
    'created_paths': sorted(path.as_posix() for path in pathlib.Path('.').rglob('*')),
}
print(json.dumps(result, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script, str(SOURCE_ROOT), str(tmp_path), json.dumps(modules)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 1, f"imports wrote to stdout: {result.stdout!r}"
    observed = json.loads(lines[0])
    assert observed == {
        "created_paths": [],
        "environment_changed": False,
        "new_threads": [],
    }
    assert result.stderr == "", f"imports wrote to stderr: {result.stderr!r}"


def test_clean_cut_deletion_inventory_and_zero_reference_gate() -> None:
    inventory_path = ROOT / "deletion_inventory.yaml"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    entries = inventory["entries"]
    paths = [entry["path"] for entry in entries]
    assert inventory["schema_version"] == 1
    assert inventory["policy"]["migration"] == "clean_cut_no_legacy_reader"
    assert len(paths) == len(set(paths)) >= 100
    assert all(entry["status"] == "deleted" for entry in entries)
    assert all(entry["replacement_owner"].startswith("PR-") for entry in entries)
    assert not [relative for relative in paths if (ROOT / relative).exists()]

    forbidden = (
        "workload_optimizer",
        "knowledge_base.json",
        "TrajectoryV2",
        "from pipeline",
        "from graders",
        "from prompts",
        "from agents",
        "import pipeline",
        "import graders",
        "import prompts",
        "import agents",
    )
    production = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (*_python_files(), ROOT / "main.py")
    )
    assert not [token for token in forbidden if token in production]


def test_e2e_production_has_no_model_filename_or_literal_config_hash_routing() -> None:
    """Keep workload identity opaque; behavior comes from resolved capabilities."""

    model_markers = (
        "qwen",
        "deepseek",
        "glm",
        "kimi",
        "minimax",
        "mini-max",
        "gpt-oss",
    )
    forbidden_names = {"QWEN_CONFIG_SHA256", "QWEN_MODEL_ID", "QWEN_MODEL_REVISION"}
    violations: list[str] = []
    for path in _python_files():
        tree = _tree(path)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        for name in sorted(names & forbidden_names):
            violations.append(f"{path.relative_to(ROOT)}: legacy routing name {name}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            rendered = ast.unparse(node).lower()
            strings = tuple(
                item.value.lower()
                for item in ast.walk(node)
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
            if any(marker in value for marker in model_markers for value in strings):
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: model-name comparison"
                )
            if any("benchmark_" in value and value.endswith((".yaml", ".yml")) for value in strings):
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: benchmark-filename comparison"
                )
            literal_digests = tuple(
                value
                for value in strings
                if len(value) == 64 and all(character in "0123456789abcdef" for character in value)
            )
            if "config" in rendered and literal_digests:
                violations.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: literal config-hash comparison"
                )
    assert not violations, "\n" + "\n".join(sorted(violations))
