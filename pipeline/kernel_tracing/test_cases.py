"""Compatibility view over the trace-kernel whitelist for tests."""

from __future__ import annotations

from dataclasses import dataclass

from .registry import load_supported_kernels


@dataclass(frozen=True)
class TraceTestCase:
    repo: str
    kind: str
    target: str
    file: str
    patch_path: str
    static_expected: bool


TRACE_TEST_CASES: list[TraceTestCase] = [
    TraceTestCase(
        repo=entry.repo,
        kind=entry.kernel_type,
        target=entry.kernel_name,
        file=entry.kernel_file,
        patch_path=entry.trace_mode,
        static_expected=entry.patch_strategy == "static",
    )
    for entry in load_supported_kernels()
]
