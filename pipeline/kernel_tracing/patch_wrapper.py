"""Static patching for Python-visible custom op wrappers."""

from __future__ import annotations

import ast
from pathlib import Path

from .patch_triton import PatchResult, _insert_import


def _event_stmt(kind: str, kernel_name: str, line: int, wrapper: str) -> ast.Expr:
    return ast.Expr(value=ast.Call(
        func=ast.Name(id="apex_trace_event", ctx=ast.Load()),
        args=[],
        keywords=[
            ast.keyword(arg="kind", value=ast.Constant(kind)),
            ast.keyword(arg="kernel_name", value=ast.Constant(kernel_name)),
            ast.keyword(arg="source_file", value=ast.Name(id="__file__", ctx=ast.Load())),
            ast.keyword(arg="line", value=ast.Constant(line)),
            ast.keyword(arg="args", value=ast.Tuple(elts=[], ctx=ast.Load())),
            ast.keyword(arg="kwargs", value=ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[])),
            ast.keyword(arg="grid", value=ast.Constant(None)),
            ast.keyword(arg="extra", value=ast.Dict(
                keys=[ast.Constant("wrapper"), ast.Constant("patch_strategy")],
                values=[ast.Constant(wrapper), ast.Constant("static")],
            )),
        ],
    ))


def _insert_after_docstring(body: list[ast.stmt], stmt: ast.stmt) -> list[ast.stmt]:
    idx = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            idx = 1
    return body[:idx] + [stmt] + body[idx:]


class _WrapperPatcher(ast.NodeTransformer):
    def __init__(self, kernel_name: str, kind: str):
        self.kernel_name = kernel_name
        self.kind = kind
        self.events: list[dict] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == self.kernel_name:
            node.body = _insert_after_docstring(
                node.body,
                _event_stmt(self.kind, self.kernel_name, node.lineno, node.name),
            )
            self.events.append({
                "kind": self.kind,
                "kernel_name": self.kernel_name,
                "line": node.lineno,
                "wrapper": node.name,
            })
        self.generic_visit(node)
        return node


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
    patcher = _WrapperPatcher(kernel_name, trace_kind)
    tree.body = _insert_import(tree.body)
    tree = patcher.visit(tree)
    ast.fix_missing_locations(tree)
    if not patcher.events:
        raise ValueError(f"No Python wrapper function {kernel_name!r} found in {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ast.unparse(tree) + "\n", encoding="utf-8")
    return PatchResult(
        source_path=source_path,
        patched_path=output_path,
        module_name=module_name,
        package_rel_path=package_rel_path,
        events=patcher.events,
    )
