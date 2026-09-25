"""Canonical portable argv for the local release evidence gate."""

from __future__ import annotations


CPU_GATE_PYTEST_ARGV = (
    ".venv/bin/pytest",
    "-q",
    "-p",
    "no:cacheprovider",
    "--import-mode=importlib",
    "tests/unit",
    "tests/contract",
    "tests/integration",
    "tests/architecture",
    "tests/test_bootstrap_dependencies.py",
)
CPU_GATE_COMPILEALL_ARGV = (
    ".venv/bin/python",
    "-m",
    "compileall",
    "-q",
    "src/apex",
    "main.py",
    "scripts",
)
CPU_GATE_SCAN_ARGV = (
    "rg",
    "-n",
    "shell" + "=True|os" + r"\.system",
    "src/apex",
)


__all__ = [
    "CPU_GATE_COMPILEALL_ARGV",
    "CPU_GATE_PYTEST_ARGV",
    "CPU_GATE_SCAN_ARGV",
]
