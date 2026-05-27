"""Static patching for Python-visible custom op wrappers."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from .patch_triton import PatchResult, _import_insert_index


def _docstring_insert_line(node: ast.FunctionDef) -> int:
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return getattr(node.body[0], "end_lineno", node.body[0].lineno)
    return node.body[0].lineno - 1 if node.body else node.lineno


def _apply_insertions(source: str, tree: ast.Module, insertions: list[tuple[int, str]]) -> str:
    lines = source.splitlines(keepends=True)
    if "apex_kernel_tracing_runtime" not in source:
        insertions.append((
            _import_insert_index(tree),
            "from apex_kernel_tracing_runtime import apex_trace_event",
        ))
    for line_idx, text in sorted(insertions, key=lambda x: x[0], reverse=True):
        indent = ""
        if 0 <= line_idx < len(lines):
            indent = lines[line_idx][:len(lines[line_idx]) - len(lines[line_idx].lstrip())]
        lines.insert(line_idx, textwrap.indent(text, indent) + "\n")
    return "".join(lines)


def _wrapper_event_text(kind: str, kernel_name: str, line: int, wrapper: str) -> str:
    return (
        "apex_trace_event(\n"
        f"    kind={kind!r},\n"
        f"    kernel_name={kernel_name!r},\n"
        "    source_file=__file__,\n"
        f"    line={line},\n"
        "    args=(),\n"
        "    kwargs=locals(),\n"
        "    grid=None,\n"
        "    extra={\n"
        f"        'wrapper': {wrapper!r},\n"
        "        'patch_strategy': 'static',\n"
        "    },\n"
        ")"
    )


def _aiter_compile_ops_event_text(kind: str, line: int, wrapper: str, ffi_type: str) -> str:
    return (
        "apex_trace_event(\n"
        f"    kind={kind!r},\n"
        "    kernel_name=loadName,\n"
        "    source_file=__file__,\n"
        f"    line={line},\n"
        "    args=args,\n"
        "    kwargs=kwargs,\n"
        "    grid=None,\n"
        "    extra={\n"
        f"        'wrapper': {wrapper!r},\n"
        "        'module_name': _md_name,\n"
        "        'load_name': loadName,\n"
        f"        'ffi_type': {ffi_type!r},\n"
        "        'original_func': getattr(func, '__name__', ''),\n"
        "        'develop': develop,\n"
        "        'patch_strategy': 'static-central',\n"
        "    },\n"
        ")"
    )


class _WrapperCollector(ast.NodeVisitor):
    def __init__(self, kernel_name: str, kind: str):
        self.kernel_name = kernel_name
        self.kind = kind
        self.insertions: list[tuple[int, str]] = []
        self.events: list[dict] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == self.kernel_name:
            self.insertions.append(
                (
                    _docstring_insert_line(node),
                    _wrapper_event_text(self.kind, self.kernel_name, node.lineno, node.name),
                )
            )
            self.events.append({
                "kind": self.kind,
                "kernel_name": self.kernel_name,
                "line": node.lineno,
                "wrapper": node.name,
            })
        self.generic_visit(node)


class _AiterCompileOpsCollector(ast.NodeVisitor):
    def __init__(self, kind: str):
        self.kind = kind
        self.function_stack: list[str] = []
        self.insertions: list[tuple[int, str]] = []
        self.events: list[dict] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.function_stack.append(node.name)
        if self.function_stack[-3:] == ["compile_ops", "decorator", "ctypes_wrapper"]:
            self.insertions.append((
                _docstring_insert_line(node),
                _aiter_compile_ops_event_text(
                    self.kind,
                    node.lineno,
                    "compile_ops.ctypes_wrapper",
                    "ctypes",
                ),
            ))
            self.events.append({
                "kind": self.kind,
                "kernel_name": "<loadName>",
                "line": node.lineno,
                "wrapper": "compile_ops.ctypes_wrapper",
            })
        elif self.function_stack[-3:] == ["compile_ops", "decorator", "wrapper"]:
            self.insertions.append((
                _docstring_insert_line(node),
                _aiter_compile_ops_event_text(
                    self.kind,
                    node.lineno,
                    "compile_ops.pybind_wrapper",
                    "pybind",
                ),
            ))
            self.events.append({
                "kind": self.kind,
                "kernel_name": "<loadName>",
                "line": node.lineno,
                "wrapper": "compile_ops.pybind_wrapper",
            })
        self.generic_visit(node)
        self.function_stack.pop()


def patch_wrapper_entry_file(
    *,
    source_path: Path,
    output_path: Path,
    kernel_name: str,
    trace_kind: str,
    module_name: str,
    package_rel_path: str,
) -> PatchResult:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    collector = _WrapperCollector(kernel_name, trace_kind)
    collector.visit(tree)
    if not collector.events:
        raise ValueError(f"No Python wrapper function {kernel_name!r} found in {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _apply_insertions(source, tree, collector.insertions),
        encoding="utf-8",
    )
    return PatchResult(
        source_path=source_path,
        patched_path=output_path,
        module_name=module_name,
        package_rel_path=package_rel_path,
        events=collector.events,
    )


def patch_aiter_compile_ops_file(
    *,
    source_path: Path,
    output_path: Path,
    trace_kind: str,
    module_name: str,
    package_rel_path: str,
) -> PatchResult:
    """Patch aiter.jit.core.compile_ops as the HIP Python tracing boundary."""
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    collector = _AiterCompileOpsCollector(trace_kind)
    collector.visit(tree)
    if not collector.events:
        raise ValueError(f"No aiter compile_ops wrapper found in {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _apply_insertions(source, tree, collector.insertions),
        encoding="utf-8",
    )
    return PatchResult(
        source_path=source_path,
        patched_path=output_path,
        module_name=module_name,
        package_rel_path=package_rel_path,
        events=collector.events,
    )
