"""Post-processing for trace JSONL events."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SIGNATURE_SCALAR_KEYS = {
    "block_size",
    "cache_dtype",
    "causal",
    "head_dim",
    "kv_block_size",
    "num_experts",
    "num_heads",
    "num_kv_heads",
    "num_query_heads",
    "page_size",
    "sliding_window",
    "top_k",
}


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
                if key.isupper() or key in SIGNATURE_SCALAR_KEYS:
                    sig.append((key, value))
    return tuple(sig)


def _update_shape_ranges(ranges: dict, name: str, shape: list[int]) -> None:
    for idx, dim in enumerate(shape):
        key = f"{name}.shape.{idx}"
        cur = ranges.setdefault(key, {"min": dim, "max": dim})
        cur["min"] = min(cur["min"], dim)
        cur["max"] = max(cur["max"], dim)


def _read_json_object(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _target_names_from_trace_config(results_dir: Path, result: dict) -> list[str]:
    trace_config = _read_json_object(results_dir / "trace_config.json")
    names: list[str] = []
    seen: set[str] = set()

    targets = trace_config.get("targets")
    if isinstance(targets, list):
        for target in targets:
            if not isinstance(target, dict):
                continue
            name = str(target.get("kernel_name") or "").strip()
            if name and name not in seen:
                names.append(name)
                seen.add(name)

    name = str(trace_config.get("kernel_name") or "").strip()
    if name and name not in seen:
        names.append(name)
        seen.add(name)

    if names:
        return names

    for group in result.get("groups", []):
        signature = group.get("signature") or {}
        name = str(signature.get("kernel_name") or "").strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _compact_benchmark_result(results_dir: Path) -> dict:
    benchmark = _read_json_object(results_dir / "benchmark" / "benchmark_result.json")
    throughput = benchmark.get("throughput")
    if not isinstance(throughput, dict):
        throughput = {}

    data = {
        "completed_requests": throughput.get("completed_requests")
        or benchmark.get("completed_requests"),
        "duration": throughput.get("duration_seconds")
        or throughput.get("duration")
        or benchmark.get("duration"),
        "execution_time": benchmark.get("execution_time"),
        "output_throughput": throughput.get("output_throughput")
        or benchmark.get("output_throughput"),
        "request_throughput": throughput.get("request_throughput")
        or benchmark.get("request_throughput"),
        "success": benchmark.get("success"),
        "total_throughput": throughput.get("total_token_throughput")
        or throughput.get("total_throughput")
        or benchmark.get("total_throughput"),
        "workspace_dir": benchmark.get("workspace_dir"),
    }
    return {key: value for key, value in data.items() if value is not None}


def _compact_trace_settings(results_dir: Path) -> dict:
    trace_config = _read_json_object(results_dir / "trace_config.json")
    data = {
        "benchmark_config": trace_config.get("benchmark_config"),
        "max_records": trace_config.get("max_records"),
        "sample_rate": trace_config.get("sample_rate"),
        "trace_all": trace_config.get("trace_all"),
    }
    return {key: value for key, value in data.items() if value is not None}


def _compact_top_shapes(top_shapes: Any) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    if not isinstance(top_shapes, list):
        return compact
    for item in top_shapes:
        if isinstance(item, dict):
            shape = item.get("shape")
            count = item.get("count")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            shape, count = item
        else:
            continue
        compact.append({"shape": str(shape), "count": count})
    return compact


def _compact_trace_result(
    result: dict,
    target_names: list[str],
    targets: dict[str, dict],
    trace_result: dict | None,
) -> dict:
    if isinstance(trace_result, dict) and trace_result:
        return {
            "any_event_found": trace_result.get("any_event_found"),
            "any_target_event_found": trace_result.get("any_target_event_found"),
            "missing_kernel_names": trace_result.get("missing_kernel_names") or [],
            "partial_coverage": trace_result.get("partial_coverage"),
            "success": trace_result.get("success"),
            "target_event_found": trace_result.get("target_event_found"),
        }

    missing = [name for name in target_names if not targets.get(name, {}).get("events")]
    any_event_found = bool(result.get("total_calls"))
    any_target_event_found = (
        any(bool(targets.get(name, {}).get("events")) for name in target_names)
        if target_names
        else any_event_found
    )
    target_event_found = any_event_found and not missing
    partial_coverage = any_target_event_found and not target_event_found
    return {
        "any_event_found": any_event_found,
        "any_target_event_found": any_target_event_found,
        "missing_kernel_names": missing,
        "partial_coverage": partial_coverage,
        "success": any_event_found and any_target_event_found,
        "target_event_found": target_event_found,
    }


def write_target_kernel_tensor_shapes(
    results_dir: Path,
    result: dict,
    trace_result: dict | None = None,
) -> dict:
    """Write a target-kernel-oriented tensor shape summary JSON."""
    target_names = _target_names_from_trace_config(results_dir, result)
    groups_by_kernel: dict[str, list[dict]] = defaultdict(list)
    for group in result.get("groups", []):
        signature = group.get("signature") or {}
        kernel_name = str(signature.get("kernel_name") or "").strip()
        if kernel_name:
            groups_by_kernel[kernel_name].append(group)

    targets: dict[str, dict] = {}
    for name in target_names:
        compact_groups = []
        events = 0
        for group in groups_by_kernel.get(name, []):
            events += int(group.get("count") or 0)
            compact_groups.append({
                "count": group.get("count"),
                "matched_kernel_name": name,
                "shape_ranges": group.get("shape_ranges") or {},
                "signature": group.get("signature") or {},
                "source_lines": group.get("source_lines") or [],
                "top_shapes": _compact_top_shapes(group.get("top_shapes")),
            })
        targets[name] = {
            "events": events,
            "group_count": len(compact_groups),
            "groups": compact_groups,
        }

    output = {
        "benchmark": _compact_benchmark_result(results_dir),
        "results_dir": str(results_dir.resolve()),
        "settings": _compact_trace_settings(results_dir),
        "targets": targets,
        "trace_result": _compact_trace_result(
            result,
            target_names,
            targets,
            trace_result,
        ),
        "workload_ranges": {
            "groups": len(result.get("groups", [])),
            "module_imports": result.get("module_imports"),
            "total_calls": result.get("total_calls"),
            "total_events": result.get("total_events"),
        },
    }
    (results_dir / "target_kernel_tensor_shapes.json").write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


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
    write_target_kernel_tensor_shapes(results_dir, result)
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
