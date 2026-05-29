"""Runtime file generation for patched tracing code."""

from __future__ import annotations

from pathlib import Path


RUNTIME_SOURCE = r'''
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path

_LOCK = threading.Lock()
_COUNT = 0


def _enabled():
    return os.environ.get("APEX_TRACE_ENABLED", "0") not in ("", "0", "false", "False")


def _is_tensor(value):
    if _is_trace_unsafe_proxy(value):
        return False
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and hasattr(value, "stride")
        and callable(getattr(value, "stride", None))
    )


def _is_trace_unsafe_proxy(value):
    typ = type(value)
    mod = getattr(typ, "__module__", "")
    name = getattr(typ, "__name__", "")
    markers = (
        "torch.fx",
        "proxy_tensor",
        "fake_tensor",
        "torch._subclasses",
    )
    if any(marker in mod for marker in markers):
        return True
    return "Proxy" in name or "FakeTensor" in name


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


def _hash_ptr(ptr):
    salt = os.environ.get("APEX_TRACE_HASH_SALT", "apex-trace")
    return hashlib.sha256(f"{salt}:{ptr}".encode()).hexdigest()[:16]


def _serialize(value, depth=0):
    if depth > 3:
        return {"type": type(value).__name__, "truncated": True}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if _is_trace_unsafe_proxy(value):
        typ = type(value)
        return {
            "type": getattr(typ, "__name__", "proxy"),
            "module": getattr(typ, "__module__", ""),
            "skipped": "torch_tracing_proxy",
        }
    if _is_tensor(value):
        out = {
            "type": "tensor",
            "shape": [int(x) for x in list(getattr(value, "shape", []))],
            "dtype": str(getattr(value, "dtype", "")),
            "device": str(getattr(value, "device", "")),
            "layout": str(getattr(value, "layout", "")),
            "requires_grad": bool(getattr(value, "requires_grad", False)),
        }
        try:
            out["stride"] = [int(x) for x in list(value.stride())]
        except Exception:
            out["stride"] = []
        try:
            out["is_contiguous"] = bool(value.is_contiguous())
        except Exception:
            out["is_contiguous"] = False
        try:
            out["numel"] = int(value.numel())
        except Exception:
            pass
        try:
            out["element_size"] = int(value.element_size())
        except Exception:
            pass
        try:
            out["data_ptr_hash"] = _hash_ptr(value.data_ptr())
        except Exception:
            pass
        return out
    if isinstance(value, dict):
        items = list(value.items())[:32]
        out = {str(k): _serialize(v, depth + 1) for k, v in items}
        if len(value) > 32:
            out["_truncated"] = len(value) - 32
        return out
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_serialize(v, depth + 1) for v in list(value)[:32]]}
    if isinstance(value, list):
        return [_serialize(v, depth + 1) for v in value[:32]]
    if callable(value):
        return {"type": "callable", "repr": repr(value)[:200]}
    out = {"type": type(value).__name__, "repr": repr(value)[:200]}
    for attr in ("shape", "dtype", "device"):
        if hasattr(value, attr):
            try:
                out[attr] = str(getattr(value, attr))
            except Exception:
                pass
    return out


def _serialize_args(args):
    if isinstance(args, (list, tuple)):
        return {f"arg{i}": _serialize(v) for i, v in enumerate(args)}
    return _serialize(args)


def _rank_info():
    rank_env = os.environ.get("APEX_TRACE_RANK_ENV", "RANK,LOCAL_RANK,LOCAL_WORLD_SIZE")
    out = {"pid": os.getpid()}
    for name in [x.strip() for x in rank_env.split(",") if x.strip()]:
        if name in os.environ:
            out[name.lower()] = os.environ[name]
    return out


def _output_file():
    out_dir = Path(os.environ.get("APEX_TRACE_OUTPUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    return out_dir / f"trace_pid{os.getpid()}_rank{rank}.jsonl"


def _apex_trace_event_impl(kind, kernel_name, source_file, line, args=None, kwargs=None, grid=None, extra=None):
    global _COUNT
    if not _enabled():
        return
    is_diagnostic_event = kind == "module_import"
    target = os.environ.get("APEX_TRACE_KERNEL_NAME", "")
    if not is_diagnostic_event and target and kernel_name not in ("", target):
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
        "grid": _serialize(grid),
        "args": _serialize_args(args or ()),
        "kwargs": _serialize(kwargs or {}),
        "extra": _serialize(extra or {}),
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
