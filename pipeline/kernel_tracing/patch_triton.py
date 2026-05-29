"""Static AST patching for Triton launch sites."""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class PatchResult:
    source_path: Path
    patched_path: Path
    module_name: str
    package_rel_path: str
    events: list[dict]
    strategy: str = "static"


def _target_name(expr: ast.AST) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


def _call_target_name(call: ast.Call) -> str | None:
    if not isinstance(call.func, ast.Subscript):
        return None
    return _target_name(call.func.value)


def find_triton_launches(source: str, kernel_name: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    launches: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = _call_target_name(node)
        if target == kernel_name:
            launches.append((getattr(node, "lineno", 0), target))
    return launches


def has_triton_launch(source: str, kernel_name: str) -> bool:
    return bool(find_triton_launches(source, kernel_name))


def _insert_import(body: list[ast.stmt]) -> list[ast.stmt]:
    for stmt in body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "apex_kernel_tracing_runtime":
            return body

    idx = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            idx = 1
    while (
        idx < len(body)
        and isinstance(body[idx], ast.ImportFrom)
        and body[idx].module == "__future__"
    ):
        idx += 1
    imp = ast.ImportFrom(
        module="apex_kernel_tracing_runtime",
        names=[ast.alias(name="apex_trace_event")],
        level=0,
    )
    return body[:idx] + [imp] + body[idx:]


def _dict_from_keywords(keywords: Iterable[ast.keyword]) -> ast.Dict:
    keys: list[ast.expr | None] = []
    values: list[ast.expr] = []
    for kw in keywords:
        if kw.arg is None:
            keys.append(None)
            values.append(kw.value)
        else:
            keys.append(ast.Constant(kw.arg))
            values.append(kw.value)
    return ast.Dict(keys=keys, values=values)


def _event_stmt(call: ast.Call, kernel_name: str, wrapper: str | None) -> ast.Expr:
    assert isinstance(call.func, ast.Subscript)
    extra_keys: list[ast.expr | None] = [
        ast.Constant("wrapper"),
        ast.Constant("patch_strategy"),
    ]
    extra_vals: list[ast.expr] = [
        ast.Constant(wrapper or ""),
        ast.Constant("static"),
    ]
    event = ast.Call(
        func=ast.Name(id="apex_trace_event", ctx=ast.Load()),
        args=[],
        keywords=[
            ast.keyword(arg="kind", value=ast.Constant("triton_launch")),
            ast.keyword(arg="kernel_name", value=ast.Constant(kernel_name)),
            ast.keyword(arg="source_file", value=ast.Name(id="__file__", ctx=ast.Load())),
            ast.keyword(arg="line", value=ast.Constant(getattr(call, "lineno", 0))),
            ast.keyword(arg="args", value=ast.List(elts=list(call.args), ctx=ast.Load())),
            ast.keyword(arg="kwargs", value=_dict_from_keywords(call.keywords)),
            ast.keyword(arg="grid", value=call.func.slice),
            ast.keyword(arg="extra", value=ast.Dict(keys=extra_keys, values=extra_vals)),
        ],
    )
    return ast.Expr(value=event)


class _LaunchPatcher(ast.NodeTransformer):
    def __init__(self, kernel_name: str):
        self.kernel_name = kernel_name
        self.function_stack: list[str] = []
        self.events: list[dict] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.function_stack.append(node.name)
        node.body = self._patch_body(node.body)
        self.generic_visit(node)
        self.function_stack.pop()
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.function_stack.append(node.name)
        node.body = self._patch_body(node.body)
        self.generic_visit(node)
        self.function_stack.pop()
        return node

    def _matching_calls(self, stmt: ast.stmt) -> list[ast.Call]:
        matches: list[ast.Call] = []
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and _call_target_name(node) == self.kernel_name:
                matches.append(node)
        return matches

    def _patch_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for stmt in body:
            matches = self._matching_calls(stmt)
            for call in matches:
                wrapper = self.function_stack[-1] if self.function_stack else ""
                out.append(_event_stmt(call, self.kernel_name, wrapper))
                self.events.append({
                    "kind": "triton_launch",
                    "kernel_name": self.kernel_name,
                    "line": getattr(call, "lineno", 0),
                    "wrapper": wrapper,
                })
            out.append(stmt)
        return out


def patch_triton_launch_file(
    *,
    source_path: Path,
    output_path: Path,
    kernel_name: str,
    module_name: str,
    package_rel_path: str,
) -> PatchResult:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    patcher = _LaunchPatcher(kernel_name)
    tree.body = _insert_import(tree.body)
    tree = patcher.visit(tree)
    ast.fix_missing_locations(tree)
    if not patcher.events:
        raise ValueError(f"No Triton launch for {kernel_name!r} found in {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ast.unparse(tree) + "\n", encoding="utf-8")
    return PatchResult(
        source_path=source_path,
        patched_path=output_path,
        module_name=module_name,
        package_rel_path=package_rel_path,
        events=patcher.events,
    )


class _LaunchCollector(ast.NodeVisitor):
    def __init__(self, kernel_name: str):
        self.kernel_name = kernel_name
        self.function_stack: list[str] = []
        self.insertions: list[tuple[int, str]] = []
        self.events: list[dict] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.function_stack.append(node.name)
        self._collect_from_body(node.body)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.function_stack.append(node.name)
        self._collect_from_body(node.body)
        self.generic_visit(node)
        self.function_stack.pop()

    def _collect_from_body(self, body: list[ast.stmt]) -> None:
        for stmt in body:
            for node in self._matching_calls_for_insertion(stmt):
                wrapper = self.function_stack[-1] if self.function_stack else ""
                event = _event_stmt(node, self.kernel_name, wrapper)
                self.insertions.append((stmt.lineno - 1, ast.unparse(event)))
                self.events.append({
                    "kind": "triton_launch",
                    "kernel_name": self.kernel_name,
                    "line": getattr(node, "lineno", 0),
                    "wrapper": wrapper,
                })

    def _matching_calls_for_insertion(self, stmt: ast.stmt) -> list[ast.Call]:
        # Compound statements own nested bodies. Their children are collected
        # when visit_If/visit_For/etc. recurses into those bodies, otherwise a
        # launch inside an if branch would be logged before the branch itself.
        if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With,
                             ast.AsyncWith, ast.Try, ast.FunctionDef,
                             ast.AsyncFunctionDef, ast.ClassDef)):
            return []
        matches: list[ast.Call] = []
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and _call_target_name(node) == self.kernel_name:
                matches.append(node)
        return matches

    def visit_If(self, node: ast.If):
        self._collect_from_body(node.body)
        self._collect_from_body(node.orelse)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self._collect_from_body(node.body)
        self._collect_from_body(node.orelse)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor):
        self._collect_from_body(node.body)
        self._collect_from_body(node.orelse)
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        self._collect_from_body(node.body)
        self._collect_from_body(node.orelse)
        self.generic_visit(node)

    def visit_With(self, node: ast.With):
        self._collect_from_body(node.body)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith):
        self._collect_from_body(node.body)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try):
        self._collect_from_body(node.body)
        self._collect_from_body(node.orelse)
        self._collect_from_body(node.finalbody)
        for handler in node.handlers:
            self._collect_from_body(handler.body)
        self.generic_visit(node)


def _import_insert_index(tree: ast.Module) -> int:
    body_idx = 0
    line_idx = 0
    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            line_idx = getattr(body[0], "end_lineno", body[0].lineno)
            body_idx = 1
    while (
        body_idx < len(body)
        and isinstance(body[body_idx], ast.ImportFrom)
        and body[body_idx].module == "__future__"
    ):
        line_idx = getattr(body[body_idx], "end_lineno", body[body_idx].lineno)
        body_idx += 1
    return line_idx


def patch_triton_launch_file(
    *,
    source_path: Path,
    output_path: Path,
    kernel_name: str,
    module_name: str,
    package_rel_path: str,
) -> PatchResult:
    """Patch by inserting log calls while preserving original source text."""
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    collector = _LaunchCollector(kernel_name)
    collector.visit(tree)
    if not collector.events:
        raise ValueError(f"No Triton launch for {kernel_name!r} found in {source_path}")

    lines = source.splitlines(keepends=True)
    insertions = list(collector.insertions)
    if "apex_kernel_tracing_runtime" not in source:
        insertions.append((
            _import_insert_index(tree),
            "from apex_kernel_tracing_runtime import apex_trace_event",
        ))

    for line_idx, text in sorted(insertions, key=lambda x: x[0], reverse=True):
        indent = ""
        if 0 <= line_idx < len(lines):
            indent = lines[line_idx][:len(lines[line_idx]) - len(lines[line_idx].lstrip())]
        block = textwrap.indent(text, indent) + "\n"
        lines.insert(line_idx, block)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(lines), encoding="utf-8")
    return PatchResult(
        source_path=source_path,
        patched_path=output_path,
        module_name=module_name,
        package_rel_path=package_rel_path,
        events=collector.events,
    )
