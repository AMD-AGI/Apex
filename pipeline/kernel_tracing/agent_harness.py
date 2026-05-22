"""Constrained Agent fallback for complex tracing patches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentPatchRequest:
    results_dir: Path
    apex_root: Path
    kernel_name: str
    kernel_file: Path
    trace_mode: str
    agent_backend: str = "claude"
    agent_model: str | None = None
    agent_max_turns: int = 8


def _build_system_prompt() -> str:
    return (
        "You are generating a temporary tracing patch for Apex. "
        "Only add imports and apex_trace_event(...) calls. Do not change numerical "
        "logic, launch parameters, branch conditions, return values, benchmark "
        "configuration, or source files outside the requested results directory."
    )


def _build_prompt(req: AgentPatchRequest) -> str:
    try:
        source = req.kernel_file.read_text(encoding="utf-8")
    except Exception:
        source = ""
    snippet = "\n".join(source.splitlines()[:220])
    manifest_path = req.results_dir / "agent_patch" / "agent_patch_manifest.json"
    return f"""
Generate a tracing patch for this target.

Target kernel/op: {req.kernel_name}
Trace mode: {req.trace_mode}
Source file: {req.kernel_file}
Allowed output root: {req.results_dir}
Required manifest path: {manifest_path}

Runtime API:
from apex_kernel_tracing_runtime import apex_trace_event

apex_trace_event(
    kind=<triton_launch or wrapper kind>,
    kernel_name={req.kernel_name!r},
    source_file=__file__,
    line=<source line>,
    args=<positional args if available>,
    kwargs=<keyword args or locals()>,
    grid=<triton grid or None>,
    extra={{"patch_strategy": "agent"}},
)

Write patched files only under:
  {req.results_dir}/patched_files/overlay/

Write JSON manifest exactly at:
  {manifest_path}

Manifest fields:
  strategy, backend, target, patched_files, expected_events, notes

Source snippet:
```python
{snippet}
```
"""


def run_agent_patch_fallback(req: AgentPatchRequest) -> dict:
    """Run the existing Apex agent backend and validate the manifest exists."""
    from agents.backends import resolve_default_model, run_agent_task

    manifest = req.results_dir / "agent_patch" / "agent_patch_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    model = req.agent_model or resolve_default_model(req.agent_backend)
    _messages, written = run_agent_task(
        prompt=_build_prompt(req),
        cwd=req.apex_root,
        model=model,
        max_turns=req.agent_max_turns,
        agent=req.agent_backend,
        system_prompt=_build_system_prompt(),
        solution_path=manifest,
    )
    if not written or not manifest.exists():
        raise RuntimeError("Agent fallback did not produce agent_patch_manifest.json")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    validate_agent_manifest(data, req.results_dir)
    return data


def validate_agent_manifest(data: dict, results_dir: Path) -> None:
    patched_root = (results_dir / "patched_files").resolve()
    agent_root = (results_dir / "agent_patch").resolve()
    for item in data.get("patched_files", []):
        patched = Path(item.get("patched_file", "")).resolve()
        if patched_root not in patched.parents and patched != patched_root:
            raise ValueError(f"Agent patched file outside patched tree: {patched}")
        if patched.exists() and "apex_trace_event" not in patched.read_text(encoding="utf-8"):
            raise ValueError(f"Agent patched file lacks apex_trace_event: {patched}")
    manifest_path = (agent_root / "agent_patch_manifest.json").resolve()
    if agent_root not in manifest_path.parents and manifest_path != agent_root:
        raise ValueError("Invalid agent manifest path")
