"""Safe JSON serialization for tracing runtime values."""

from __future__ import annotations

import hashlib
import inspect
import os
import textwrap
from collections.abc import Mapping, Sequence
from typing import Any, Callable


_MAX_DEPTH = 3
_MAX_ITEMS = 32


def _is_tensor(value: Any) -> bool:
    if _is_trace_unsafe_proxy(value):
        return False
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and hasattr(value, "stride")
        and callable(getattr(value, "stride", None))
    )


def _is_trace_unsafe_proxy(value: Any) -> bool:
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


def _hash_ptr(ptr: Any) -> str:
    salt = os.environ.get("APEX_TRACE_HASH_SALT", "apex-trace")
    raw = f"{salt}:{ptr}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def serialize_value(value: Any, *, depth: int = 0, max_depth: int = _MAX_DEPTH) -> Any:
    """Serialize host-side metadata without reading tensor contents."""
    if depth > max_depth:
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
        out: dict[str, Any] = {
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

    if isinstance(value, Mapping):
        items = list(value.items())[:_MAX_ITEMS]
        out = {
            str(k): serialize_value(v, depth=depth + 1, max_depth=max_depth)
            for k, v in items
        }
        if len(value) > _MAX_ITEMS:
            out["_truncated"] = len(value) - _MAX_ITEMS
        return out

    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [
                serialize_value(v, depth=depth + 1, max_depth=max_depth)
                for v in list(value)[:_MAX_ITEMS]
            ],
        }

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [
            serialize_value(v, depth=depth + 1, max_depth=max_depth)
            for v in list(value)[:_MAX_ITEMS]
        ]

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


def serialize_args(args: Any) -> Any:
    if isinstance(args, (list, tuple)):
        return {
            f"arg{i}": serialize_value(v)
            for i, v in enumerate(args)
        }
    return serialize_value(args)


def runtime_serializer_source() -> str:
    """Return serializer code to embed in the generated standalone runtime."""
    blocks: list[str | Callable[..., Any]] = [
        f"_MAX_DEPTH = {_MAX_DEPTH}",
        f"_MAX_ITEMS = {_MAX_ITEMS}",
        _is_trace_unsafe_proxy,
        _is_tensor,
        _hash_ptr,
        serialize_value,
        serialize_args,
    ]
    rendered: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            rendered.append(block)
        else:
            rendered.append(textwrap.dedent(inspect.getsource(block)).strip())
    return "\n\n".join(rendered) + "\n"
