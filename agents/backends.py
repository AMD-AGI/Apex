#!/usr/bin/env python3
"""
Shared agent backend runner for repo eval scripts.

Supports:
  - Codex CLI (`codex exec`) [default backend]
  - Claude Code via `claude-agent-sdk`
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_AGENT = "codex"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_CODEX_MODEL = "gpt-5.3-codex"


def resolve_default_model(agent: str) -> str | None:
    agent = agent.lower()
    if agent == "claude":
        return DEFAULT_CLAUDE_MODEL
    if agent == "codex":
        return _read_codex_config_model() or DEFAULT_CODEX_MODEL
    raise ValueError(f"Unsupported agent backend: {agent}")


def model_display_name(model: str | None, agent: str) -> str:
    if model:
        return model
    return "(default)"


def run_agent_task(
    *,
    prompt: str,
    cwd: Path,
    model: str | None,
    max_turns: int,
    agent: str,
    system_prompt: str | None,
    solution_path: Path,
) -> tuple[list, bool]:
    agent = agent.lower()
    if agent == "claude":
        if not model:
            model = DEFAULT_CLAUDE_MODEL
        return _run_claude_task(
            prompt=prompt,
            cwd=cwd,
            model=model,
            max_turns=max_turns,
            system_prompt=system_prompt,
            solution_path=solution_path,
        )
    if agent == "codex":
        return _run_codex_task(
            prompt=prompt,
            cwd=cwd,
            model=model,
            max_turns=max_turns,
            system_prompt=system_prompt,
            solution_path=solution_path,
        )
    raise ValueError(f"Unsupported agent backend: {agent}")


def _run_claude_task(
    *,
    prompt: str,
    cwd: Path,
    model: str,
    max_turns: int,
    system_prompt: str | None,
    solution_path: Path,
) -> tuple[list, bool]:
    try:
        from claude_agent_sdk import query, ClaudeAgentOptions  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "claude-agent-sdk not installed. Run: pip install claude-agent-sdk"
        ) from e

    async def _run() -> tuple[list, bool]:
        from claude_agent_sdk import query, ClaudeAgentOptions

        options = ClaudeAgentOptions(
            cwd=str(cwd),
            model=model,
            max_turns=max_turns,
            permission_mode="bypassPermissions",
            system_prompt=system_prompt,
        )

        trajectory = []
        async for message in query(prompt=prompt, options=options):
            trajectory.append(message)
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "name"):
                        keys = list(block.input.keys()) if hasattr(block, "input") else []
                        print(f"    tool: {block.name}({keys})")
            if hasattr(message, "num_turns"):
                cost = getattr(message, "total_cost_usd", 0.0) or 0.0
                print(f"  result: turns={message.num_turns}, cost=${cost:.4f}")

        return trajectory, solution_path.exists()

    # Avoid nested-session guard when launched programmatically.
    _cc = os.environ.pop("CLAUDECODE", None)
    try:
        return asyncio.run(_run())
    finally:
        if _cc is not None:
            os.environ["CLAUDECODE"] = _cc


def _run_codex_task(
    *,
    prompt: str,
    cwd: Path,
    model: str | None,
    max_turns: int,
    system_prompt: str | None,
    solution_path: Path,
) -> tuple[list, bool]:
    if not _command_exists("codex"):
        raise RuntimeError("codex CLI not installed or not in PATH.")

    combined_prompt = _build_codex_prompt(prompt, system_prompt, max_turns)
    cmd = [
        "codex",
        "exec",
        "--json",
        "--color",
        "never",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(cwd),
    ]
    if model:
        cmd += ["-m", model]
    cmd.append(combined_prompt)

    env = os.environ.copy()
    for key in (
        "CODEX_THREAD_ID",
        "CODEX_CI",
        "CODEX_INTERNAL_ORIGINATOR_OVERRIDE",
    ):
        env.pop(key, None)

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)

    stderr = (proc.stderr or "").strip()
    if stderr:
        for line in stderr.splitlines():
            line = line.strip()
            if line:
                print(f"    codex-stderr: {line[:200]}")

    trajectory: list[dict] = []
    usage = None
    final_error = None
    for raw_line in (proc.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        trajectory.append(event)
        etype = event.get("type")
        if etype == "item.completed":
            item = event.get("item") or {}
            item_type = item.get("type")
            if item_type in {"function_call", "tool_call"}:
                name = item.get("name") or item.get("tool_name") or "tool"
                print(f"    tool: {name}")
            elif item_type == "agent_message":
                text = (item.get("text") or "").strip()
                if text:
                    print(f"    text: {text.splitlines()[0][:100]}...")
            elif item_type == "error":
                msg = item.get("message") or ""
                if msg:
                    print(f"    error: {str(msg)[:200]}")
        elif etype == "turn.completed":
            usage = event.get("usage")
        elif etype in {"turn.failed", "error"}:
            err = event.get("error") or {}
            final_error = err.get("message") if isinstance(err, dict) else event.get("message")

    if usage:
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        print(f"  result: turns=1+, tokens(in={in_tok}, out={out_tok})")

    if proc.returncode != 0 and final_error:
        print(f"  codex error: {final_error}")

    return trajectory, solution_path.exists()


def _build_codex_prompt(prompt: str, system_prompt: str | None, max_turns: int) -> str:
    parts = []
    if system_prompt:
        parts.append("System guidance:")
        parts.append(system_prompt.strip())
        parts.append("")
    parts.append(f"Try to finish efficiently (target <= {max_turns} turns if possible).")
    parts.append("")
    parts.append(prompt.rstrip())
    return "\n".join(parts)


def _command_exists(name: str) -> bool:
    return any(
        (Path(path) / name).exists()
        for path in os.environ.get("PATH", "").split(os.pathsep)
        if path
    )


def _read_codex_config_model() -> str | None:
    """Best-effort parse of ~/.codex/config.toml for top-level `model = \"...\"`."""
    config_path = Path.home() / ".codex" / "config.toml"
    try:
        text = config_path.read_text()
    except OSError:
        return None
    m = re.search(r'(?m)^\s*model\s*=\s*"([^"]+)"\s*$', text)
    return m.group(1) if m else None
