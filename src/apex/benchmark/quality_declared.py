"""Exact quality output set declared by an evaluator execution receipt."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .evaluator_execution import LmEvalExecutionReceipt
from .quality_artifacts import load_declared_quality_artifacts


@dataclass(frozen=True, slots=True)
class DeclaredQualityArtifacts:
    """Authority-declared paths and bytes; no workspace discovery is involved."""

    receipt: LmEvalExecutionReceipt
    result_paths: tuple[Path, ...]
    sample_paths: tuple[Path, ...]
    result_receipts: tuple[Mapping[str, Any], ...]
    sample_receipts: tuple[Mapping[str, Any], ...]
    result_document: Mapping[str, Any]

    @property
    def all_paths(self) -> tuple[Path, ...]:
        return (*self.result_paths, *self.sample_paths)


def load_declared_quality_outputs(
    root: Path, value: Mapping[str, Any]
) -> DeclaredQualityArtifacts:
    """Rehash exact receipt locators and parse result bytes from the same fd read."""

    receipt = LmEvalExecutionReceipt.from_mapping(value)
    results = tuple(item.to_dict() for item in receipt.result_artifacts)
    samples = tuple(item.to_dict() for item in receipt.sample_artifacts)
    loaded = load_declared_quality_artifacts(root, (*results, *samples))
    by_path = {item.relative_path: item for item in loaded}
    result_items = tuple(by_path[str(item["path"])] for item in results)
    sample_items = tuple(by_path[str(item["path"])] for item in samples)
    if len(result_items) != 1:
        raise ValueError("Evaluator receipt must declare one result artifact")
    try:
        document = json.loads(result_items[0].content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Evaluator result artifact is invalid JSON") from error
    if not isinstance(document, Mapping) or not isinstance(
        document.get("results"), Mapping
    ):
        raise ValueError("Evaluator result artifact lacks a results mapping")
    return DeclaredQualityArtifacts(
        receipt=receipt,
        result_paths=tuple(item.path for item in result_items),
        sample_paths=tuple(item.path for item in sample_items),
        result_receipts=results,
        sample_receipts=samples,
        result_document=document,
    )


__all__ = ["DeclaredQualityArtifacts", "load_declared_quality_outputs"]
