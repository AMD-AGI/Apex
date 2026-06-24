"""Runtime file generation for patched tracing code."""

from __future__ import annotations

from pathlib import Path

from .serializer import runtime_serializer_source


RUNTIME_SOURCE = r'''
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

''' + runtime_serializer_source() + r'''
_LOCK = threading.Lock()
_COUNT = 0


def _enabled():
    enabled = os.environ.get("APEX_TRACE_ENABLED")
    if enabled is not None:
        return enabled not in ("", "0", "false", "False")
    # Some model servers sanitize worker-process environments. The patch
    # manifest is enough signal that this process belongs to a trace run.
    return bool(
        os.environ.get("APEX_TRACE_PATCH_MANIFEST")
        or Path("/apex_trace/patched_files/patch_manifest.json").exists()
    )

def _contains_trace_unsafe_proxy(value, depth=0):
    if depth > 3:
        return False
    if _is_trace_unsafe_proxy(value):
        return True
    if isinstance(value, dict):
        return any(_contains_trace_unsafe_proxy(v, depth + 1) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_trace_unsafe_proxy(v, depth + 1) for v in value)
    return False


def _rank_info():
    rank_env = os.environ.get("APEX_TRACE_RANK_ENV", "RANK,LOCAL_RANK,LOCAL_WORLD_SIZE")
    out = {"pid": os.getpid()}
    for name in [x.strip() for x in rank_env.split(",") if x.strip()]:
        if name in os.environ:
            out[name.lower()] = os.environ[name]
    return out


def _output_file():
    out_dir_raw = os.environ.get("APEX_TRACE_OUTPUT_DIR", "")
    if out_dir_raw:
        out_dir = Path(out_dir_raw)
    else:
        manifest = os.environ.get("APEX_TRACE_PATCH_MANIFEST", "")
        if manifest:
            out_dir = Path(manifest).parent.parent / "trace_raw"
        elif Path("/apex_trace/patched_files/patch_manifest.json").exists():
            out_dir = Path("/apex_trace/trace_raw")
        else:
            out_dir = Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    return out_dir / f"trace_pid{os.getpid()}_rank{rank}.jsonl"


def _target_kernel_names():
    raw = os.environ.get("APEX_TRACE_KERNEL_NAMES", "")
    if not raw:
        raw = os.environ.get("APEX_TRACE_KERNEL_NAME", "")
    return {name.strip() for name in raw.split(",") if name.strip()}


def _apex_trace_event_impl(kind, kernel_name, source_file, line, args=None, kwargs=None, grid=None, extra=None):
    global _COUNT
    if not _enabled():
        return
    is_diagnostic_event = kind == "module_import"
    targets = _target_kernel_names()
    if not is_diagnostic_event and targets and kernel_name not in targets and kernel_name != "":
        return
    kind_filter = os.environ.get("APEX_TRACE_KIND", "")
    if not is_diagnostic_event and kind_filter and kind != kind_filter:
        return
    if not is_diagnostic_event and (
        _contains_trace_unsafe_proxy(args)
        or _contains_trace_unsafe_proxy(kwargs)
        or _contains_trace_unsafe_proxy(grid)
    ):
        return

    # Diagnostic import events tell Apex whether the overlay was actually used.
    # They are intentionally exempt from sampling and max-record throttling.
    if not is_diagnostic_event:
        try:
            max_records = int(os.environ.get("APEX_TRACE_MAX_RECORDS", "100000"))
        except ValueError:
            max_records = 100000
        try:
            sample_rate = float(os.environ.get("APEX_TRACE_SAMPLE_RATE", "1.0"))
        except ValueError:
            sample_rate = 1.0
        if sample_rate < 1.0 and random.random() > sample_rate:
            return
        with _LOCK:
            if _COUNT >= max_records:
                return
            _COUNT += 1
    event = {
        "schema_version": 1,
        "ts_ns": time.time_ns(),
        "kind": kind,
        "kernel_name": kernel_name,
        "source_file": source_file,
        "line": int(line or 0),
        "process": _rank_info(),
        "grid": serialize_value(grid),
        "args": serialize_args(args or ()),
        "kwargs": serialize_value(kwargs or {}),
        "extra": serialize_value(extra or {}),
    }
    with _output_file().open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def apex_trace_event(kind, kernel_name, source_file, line, args=None, kwargs=None, grid=None, extra=None):
    try:
        _apex_trace_event_impl(kind, kernel_name, source_file, line, args, kwargs, grid, extra)
    except Exception:
        return
'''


def write_runtime_file(patched_files_dir: Path) -> Path:
    patched_files_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = patched_files_dir / "apex_kernel_tracing_runtime.py"
    runtime_path.write_text(RUNTIME_SOURCE.lstrip(), encoding="utf-8")
    return runtime_path
