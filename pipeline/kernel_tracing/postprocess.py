"""Post-processing for trace JSONL events."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _iter_events(trace_raw_dir: Path):
    for path in sorted(trace_raw_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _tensor_items(event: dict):
    for section in ("args", "kwargs"):
        values = event.get(section, {})
        if not isinstance(values, dict):
            continue
        for name, value in values.items():
            if isinstance(value, dict) and value.get("type") == "tensor":
                yield name, value


def _signature(event: dict) -> tuple:
    sig: list[tuple[str, Any]] = [
        ("kind", event.get("kind")),
        ("kernel_name", event.get("kernel_name")),
    ]
    for name, tensor in _tensor_items(event):
        sig.append((f"{name}.dtype", tensor.get("dtype")))
        sig.append((f"{name}.layout", tensor.get("layout")))
    kwargs = event.get("kwargs", {})
    if isinstance(kwargs, dict):
        for key, value in sorted(kwargs.items()):
            if isinstance(value, (bool, int, float, str)):
                if key.isupper() or key in {"causal", "top_k", "num_experts", "block_size"}:
                    sig.append((key, value))
    return tuple(sig)


def _update_shape_ranges(ranges: dict, name: str, shape: list[int]) -> None:
    for idx, dim in enumerate(shape):
        key = f"{name}.shape.{idx}"
        cur = ranges.setdefault(key, {"min": dim, "max": dim})
        cur["min"] = min(cur["min"], dim)
        cur["max"] = max(cur["max"], dim)


def postprocess_trace(results_dir: Path) -> dict:
    trace_raw_dir = results_dir / "trace_raw"
    events = list(_iter_events(trace_raw_dir))
    raw_jsonl = results_dir / "trace_raw.jsonl"
    with raw_jsonl.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    groups: dict[tuple, dict] = {}
    total_target = 0
    for event in events:
        if event.get("kind") == "module_import":
            continue
        total_target += 1
        sig = _signature(event)
        group = groups.setdefault(sig, {
            "count": 0,
            "signature": dict(sig),
            "shape_ranges": {},
            "shape_frequency": Counter(),
            "source_lines": Counter(),
            "examples": [],
        })
        group["count"] += 1
        group["source_lines"][f"{event.get('source_file')}:{event.get('line')}"] += 1
        for name, tensor in _tensor_items(event):
            shape = tensor.get("shape") or []
            if shape:
                _update_shape_ranges(group["shape_ranges"], name, shape)
                group["shape_frequency"][f"{name}:{shape}"] += 1
        if len(group["examples"]) < 3:
            group["examples"].append(event)

    out_groups = []
    for group in sorted(groups.values(), key=lambda g: g["count"], reverse=True):
        out_groups.append({
            "count": group["count"],
            "percent": round((100.0 * group["count"] / total_target), 3) if total_target else 0.0,
            "signature": group["signature"],
            "shape_ranges": group["shape_ranges"],
            "top_shapes": group["shape_frequency"].most_common(10),
            "source_lines": group["source_lines"].most_common(10),
            "examples": group["examples"],
        })

    result = {
        "schema_version": 1,
        "total_events": len(events),
        "total_calls": total_target,
        "module_imports": sum(1 for e in events if e.get("kind") == "module_import"),
        "groups": out_groups,
    }
    (results_dir / "workload_ranges.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_summary(results_dir, result)
    return result


def _write_summary(results_dir: Path, result: dict) -> None:
    lines = [
        "# Workload Trace Summary",
        "",
        f"- Total events: {result['total_events']}",
        f"- Total traced calls: {result['total_calls']}",
        f"- Module import events: {result['module_imports']}",
        "",
        "## Top Groups",
        "",
    ]
    for idx, group in enumerate(result["groups"][:10], 1):
        lines.extend([
            f"### {idx}. count={group['count']} percent={group['percent']}",
            "",
            "**Signature**",
            "",
            "| Field | Value |",
            "|---|---|",
        ])
        for key, value in group["signature"].items():
            lines.append(f"| `{key}` | `{value}` |")
        lines.extend([
            "",
            "**Shape Ranges**",
            "",
            "| Tensor | Dim | Min | Max |",
            "|---|---:|---:|---:|",
        ])
        for key, value in sorted(group["shape_ranges"].items()):
            tensor, _, dim = key.rpartition(".shape.")
            lines.append(
                f"| `{tensor}` | {dim} | {value.get('min')} | {value.get('max')} |"
            )
        lines.append("")
    (results_dir / "workload_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
