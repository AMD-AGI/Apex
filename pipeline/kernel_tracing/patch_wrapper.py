"""Static patching for Python-visible custom op wrappers."""

from __future__ import annotations

import ast
import re
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


def _before_call_return_line(node: ast.FunctionDef, callee_names: set[str]) -> int:
    for child in ast.walk(node):
        if not isinstance(child, ast.Return) or not isinstance(child.value, ast.Call):
            continue
        func = child.value.func
        if isinstance(func, ast.Name) and func.id in callee_names:
            return child.lineno - 1
    return _docstring_insert_line(node)


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
        "_apex_trace_load_name = fc_name if fc_name is not None else getattr(func, '__name__', '')\n"
        "try:\n"
        "    _apex_trace_develop = develop\n"
        "except NameError:\n"
        "    _apex_trace_develop = False\n"
        "_apex_trace_skip = False\n"
        "try:\n"
        "    _apex_trace_skip = bool(getattr(getattr(torch, 'compiler', None), 'is_compiling', lambda: False)())\n"
        "except Exception:\n"
        "    _apex_trace_skip = False\n"
        "try:\n"
        "    _apex_trace_skip = _apex_trace_skip or bool(getattr(getattr(torch, '_dynamo', None), 'is_compiling', lambda: False)())\n"
        "except Exception:\n"
        "    pass\n"
        "if not _apex_trace_skip:\n"
        "    apex_trace_event(\n"
        f"        kind={kind!r},\n"
        "        kernel_name=_apex_trace_load_name,\n"
        "        source_file=__file__,\n"
        f"        line={line},\n"
        "        args=args,\n"
        "        kwargs=kwargs,\n"
        "        grid=None,\n"
        "        extra={\n"
        f"            'wrapper': {wrapper!r},\n"
        "            'module_name': _md_name,\n"
        "            'load_name': _apex_trace_load_name,\n"
        f"            'ffi_type': {ffi_type!r},\n"
        "            'original_func': getattr(func, '__name__', ''),\n"
        "            'develop': _apex_trace_develop,\n"
        "            'patch_strategy': 'static-central',\n"
        "        },\n"
        "    )"
    )


class _WrapperCollector(ast.NodeVisitor):
    def __init__(self, kernel_name: str, kind: str, *, trace_all: bool = False):
        self.kernel_name = kernel_name
        self.kind = kind
        self.trace_all = trace_all
        self.function_depth = 0
        self.insertions: list[tuple[int, str]] = []
        self.events: list[dict] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        should_trace = node.name == self.kernel_name
        if self.trace_all and self.function_depth == 0:
            should_trace = True
        if should_trace:
            event_kernel_name = node.name if self.trace_all else self.kernel_name
            self.insertions.append(
                (
                    _docstring_insert_line(node),
                    _wrapper_event_text(self.kind, event_kernel_name, node.lineno, node.name),
                )
            )
            self.events.append({
                "kind": self.kind,
                "kernel_name": event_kernel_name,
                "line": node.lineno,
                "wrapper": node.name,
            })
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef):
        if not self.trace_all:
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
                _before_call_return_line(node, {"ctypes_caller"}),
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
                _before_call_return_line(node, {"op"}),
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
    trace_all: bool = False,
) -> PatchResult:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    collector = _WrapperCollector(kernel_name, trace_kind, trace_all=trace_all)
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


def _harden_aiter_compile_ops_source(source: str) -> str:
    """Keep aiter tracing overlays from crashing on uncheckable annotations."""
    pattern = re.compile(
        r"(?P<indent>[ \t]+)sub_t = typing\.get_args\(expected_type\)\n"
        r"\n"
        r"(?P=indent)if origin is None:\n"
        r"(?P=indent)    if not isinstance\(arg, expected_type\) and not \(\n"
    )

    def replace(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}sub_t = typing.get_args(expected_type)\n"
            "\n"
            f"{indent}if expected_type is inspect._empty:\n"
            f"{indent}    continue\n"
            "\n"
            f"{indent}if origin is None:\n"
            f"{indent}    if not isinstance(expected_type, type):\n"
            f"{indent}        continue\n"
            f"{indent}    if not isinstance(arg, expected_type) and not (\n"
        )

    return pattern.sub(replace, source)


def patch_aiter_compile_ops_file(
    *,
    source_path: Path,
    output_path: Path,
    trace_kind: str,
    module_name: str,
    package_rel_path: str,
) -> PatchResult:
    """Patch aiter.jit.core.compile_ops as the HIP Python tracing boundary."""
    source = _harden_aiter_compile_ops_source(source_path.read_text(encoding="utf-8"))
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
