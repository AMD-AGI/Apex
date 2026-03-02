#!/usr/bin/env python3
"""
Shared agent backend runner for repo eval scripts.

Supports two backends (selectable via --agent-backend):

  - claude (default): Claude Code via `claude-code-sdk`.
    Auth: Max plan (claude auth login) or ANTHROPIC_API_KEY.
    Gets access to MCP tools (Magpie, GPU info, RAG, etc.) from mcp_config.json.
    Default model: claude-sonnet-4-6.

  - codex: OpenAI Codex via `codex exec` CLI.
    Auth: OPENAI_API_KEY or `codex login`.
    MCP tools configured globally via `codex mcp add` (same tools as Claude).
    Default model: gpt-5.3-codex. Requires Node.js 18+.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_AGENT = "claude"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_CODEX_MODEL = "gpt-5.3-codex"

REPO_ROOT = Path(__file__).resolve().parent.parent
MCP_CONFIG_PATH = REPO_ROOT / "mcp_config.json"


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


def _load_mcp_config() -> dict | None:
    """Load MCP server config for Claude Code from mcp_config.json."""
    if MCP_CONFIG_PATH.exists():
        with open(MCP_CONFIG_PATH) as f:
            return json.load(f)
    return None


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
        from claude_code_sdk import query, ClaudeCodeOptions  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "claude-code-sdk not installed. Run: pip install claude-code-sdk"
        ) from e

    mcp_cfg = _load_mcp_config()
    mcp_servers = None
    if mcp_cfg and "mcpServers" in mcp_cfg:
        mcp_servers = mcp_cfg["mcpServers"]
        print(f"    MCPs: {', '.join(mcp_servers.keys())}")

    async def _run() -> tuple[list, bool]:
        from claude_code_sdk import query, ClaudeCodeOptions

        options = ClaudeCodeOptions(
            cwd=str(cwd),
            model=model,
            max_turns=max_turns,
            permission_mode="bypassPermissions",
            system_prompt=system_prompt,
        )

        if mcp_servers is not None:
            options.mcp_servers = mcp_servers

        trajectory = []
        try:
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
        except Exception as e:
            err_str = str(e)
            if trajectory and ("exit code" in err_str or "Command failed" in err_str):
                print(f"    [warn] CLI exited non-zero after {len(trajectory)} messages "
                      f"(likely max-turns reached): {err_str[:120]}")
            else:
                print(f"    [error] Claude SDK error: {err_str[:200]}")
                raise

        return trajectory, solution_path.exists()

    # Remove env vars that prevent Claude Code SDK from spawning a fresh session
    saved_env = {}
    for key in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
        val = os.environ.pop(key, None)
        if val is not None:
            saved_env[key] = val

    try:
        return asyncio.run(_run())
    finally:
        os.environ.update(saved_env)


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
        "--skip-git-repo-check",
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

    # Stream output line-by-line for long-running tasks
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, bufsize=1,
    )

    trajectory: list[dict] = []
    usage = None
    final_error = None
    total_in_tokens = 0
    total_out_tokens = 0
    turn_count = 0

    try:
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line or not line.startswith("{"):
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
                elif item_type == "mcp_call":
                    name = item.get("server_label", "") + "::" + (item.get("name") or "")
                    print(f"    mcp: {name}")
            elif etype == "turn.completed":
                usage = event.get("usage")
                turn_count += 1
                if usage:
                    total_in_tokens += usage.get("input_tokens", 0)
                    total_out_tokens += usage.get("output_tokens", 0)
            elif etype in {"turn.failed", "error"}:
                err = event.get("error") or {}
                final_error = err.get("message") if isinstance(err, dict) else event.get("message")
    finally:
        proc.wait()

    stderr = (proc.stderr.read() or "").strip()
    if stderr:
        for sline in stderr.splitlines()[:10]:
            sline = sline.strip()
            if sline:
                print(f"    codex-stderr: {sline[:200]}")

    if turn_count > 0:
        print(f"  result: turns={turn_count}, tokens(in={total_in_tokens}, out={total_out_tokens})")

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
