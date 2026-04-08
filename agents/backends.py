#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
Shared agent backend runner for repo eval scripts.

Supports three backends (selectable via --agent-backend):

  - claude (default): Claude Code via `claude-code-sdk`.
    Auth: Max plan (claude auth login) or ANTHROPIC_API_KEY.
    Gets access to MCP tools (Magpie, GPU info, RAG, etc.) from mcp_config.json.
    Default model: claude-sonnet-4-6.

  - codex: OpenAI Codex via `codex exec` CLI.
    Auth: OPENAI_API_KEY or `codex login`.
    MCP tools configured globally via `codex mcp add` (same tools as Claude).
    Default model: gpt-5.3-codex. Requires Node.js 18+.

  - cursor: Cursor Agent via `cursor-agent` CLI.
    Auth: cursor-agent login or CURSOR_API_KEY.
    MCP tools configured via workspace settings.
    Default model: auto.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

DEFAULT_AGENT = "claude"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_CODEX_MODEL = "gpt-5.3-codex"
DEFAULT_CURSOR_MODEL = "auto"

REPO_ROOT = Path(__file__).resolve().parent.parent
MCP_CONFIG_PATH = REPO_ROOT / "mcp_config.json"


def resolve_default_model(agent: str) -> str | None:
    agent = agent.lower()
    if agent == "claude":
        return DEFAULT_CLAUDE_MODEL
    if agent == "codex":
        return _read_codex_config_model() or DEFAULT_CODEX_MODEL
    if agent == "cursor":
        return DEFAULT_CURSOR_MODEL
    raise ValueError(f"Unsupported agent backend: {agent}")


def model_display_name(model: str | None, agent: str) -> str:
    if model:
        return model
    return "(default)"


def _load_mcp_config() -> dict | None:
    """Load MCP server config and resolve relative paths against REPO_ROOT."""
    if not MCP_CONFIG_PATH.exists():
        return None
    with open(MCP_CONFIG_PATH) as f:
        cfg = json.load(f)
    servers = cfg.get("mcpServers", {})
    for _name, srv in servers.items():
        for i, arg in enumerate(srv.get("args", [])):
            if arg.startswith("./"):
                srv["args"][i] = str(REPO_ROOT / arg)
        env = srv.get("env", {})
        for k, v in env.items():
            if isinstance(v, str) and v.startswith("./"):
                env[k] = str(REPO_ROOT / v)
        cmd = srv.get("command", "")
        if cmd == "python3":
            srv["command"] = sys.executable or "python3"
    return cfg


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
    if agent == "cursor":
        return _run_cursor_task(
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
            permission_mode="acceptEdits",
            allowed_tools=[
                "Bash", "Read", "Edit", "Write", "MultiEdit",
                "Glob", "Grep", "Agent",
                "mcp__magpie__analyze", "mcp__magpie__compare",
                "mcp__magpie__hardware_spec", "mcp__magpie__suggest_optimizations",
                "mcp__magpie__discover_kernels", "mcp__magpie__create_kernel_config",
                "mcp__gpu-info__get_gpu_info", "mcp__gpu-info__get_arch_optimization_hints",
                "mcp__source-finder__find_kernel_source", "mcp__source-finder__classify_kernel",
                "mcp__source-finder__find_ck_template", "mcp__source-finder__identify_kernel_origin",
                "mcp__source-finder__suggest_optimization_approach",
                "mcp__kernel-rag__search_kernel_optimization", "mcp__kernel-rag__search_gpu_documentation",
                "mcp__kernel-rag__get_optimization_snippet", "mcp__kernel-rag__analyze_kernel_for_optimization",
                "mcp__kernel-rag__get_optimization_playbook", "mcp__kernel-rag__get_gpu_specs",
                "mcp__fusion-advisor__detect_fusion_opportunities",
                "mcp__fusion-advisor__generate_fused_kernel", "mcp__fusion-advisor__estimate_fusion_benefit",
                "mcp__fusion-advisor__check_library_fusion",
            ],
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
            if "MessageParseError" in type(e).__name__ or "Unknown message type" in err_str:
                print(f"    [warn] Ignoring unknown message type: {err_str[:120]}")
            elif trajectory and ("exit code" in err_str or "Command failed" in err_str):
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


_CURSOR_TIMEOUT_SECONDS = 600
_CURSOR_INITIAL_OUTPUT_TIMEOUT = 30


def _find_cursor_agent_binary() -> str | None:
    """Locate the cursor-agent standalone binary (preferred over VS Code wrapper)."""
    if _command_exists("cursor-agent"):
        return "cursor-agent"
    local_bin = Path.home() / ".local" / "bin" / "cursor-agent"
    if local_bin.exists():
        return str(local_bin)
    if _command_exists("cursor"):
        return "cursor"
    return None


def _run_cursor_task(
    *,
    prompt: str,
    cwd: Path,
    model: str | None,
    max_turns: int,
    system_prompt: str | None,
    solution_path: Path,
) -> tuple[list, bool]:
    import select
    import time

    cursor_bin = _find_cursor_agent_binary()
    if not cursor_bin:
        raise RuntimeError(
            "cursor-agent CLI not found. Install: "
            "curl https://cursor.com/install -fsS | bash  OR  "
            "npm install -g @anthropic-ai/cursor-agent"
        )

    combined_prompt = _build_codex_prompt(prompt, system_prompt, max_turns)
    cmd = [
        cursor_bin,
        "-p", "--force", "--trust",
        "--approve-mcps",
        "--output-format", "stream-json",
        "--workspace", str(cwd),
    ]
    if model:
        cmd += ["--model", model]
    cmd.append(combined_prompt)

    env = os.environ.copy()

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True, env=env, bufsize=1,
    )

    if proc.stdin:
        proc.stdin.close()

    trajectory: list[dict] = []
    stderr_lines: list[str] = []
    session_id = None
    duration_ms = None
    final_error = None

    def _read_stderr():
        try:
            for line in proc.stderr:
                line = line.strip()
                if line:
                    stderr_lines.append(line)
        except Exception:
            pass

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    first_output_received = False
    start_time = time.monotonic()

    try:
        while True:
            if not first_output_received:
                elapsed = time.monotonic() - start_time
                if elapsed > _CURSOR_INITIAL_OUTPUT_TIMEOUT:
                    final_error = (
                        f"cursor-agent produced no output after {_CURSOR_INITIAL_OUTPUT_TIMEOUT}s. "
                        "Check authentication: run 'cursor-agent login' or set CURSOR_API_KEY."
                    )
                    proc.kill()
                    break
                ready, _, _ = select.select([proc.stdout], [], [], 2.0)
                if not ready:
                    continue

            raw_line = proc.stdout.readline()
            if not raw_line:
                break
            first_output_received = True
            line = raw_line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            trajectory.append(event)
            etype = event.get("type")
            subtype = event.get("subtype")

            if etype == "system" and subtype == "init":
                session_id = event.get("session_id")
                model_name = event.get("model", "")
                print(f"    cursor session: {session_id or 'n/a'}, model: {model_name}")

            elif etype == "tool_call":
                if subtype == "started":
                    tc = event.get("tool_call", {})
                    tool_name = _extract_cursor_tool_name(tc)
                    print(f"    tool: {tool_name}")

            elif etype == "assistant":
                msg = event.get("message", {})
                content = msg.get("content", [])
                for block in content:
                    text = block.get("text", "").strip()
                    if text:
                        first_line = text.splitlines()[0][:100]
                        print(f"    text: {first_line}...")
                        break

            elif etype == "thinking":
                pass

            elif etype == "result":
                duration_ms = event.get("duration_ms")
                is_error = event.get("is_error", False)
                if is_error:
                    final_error = event.get("result", "unknown error")

    finally:
        proc.wait()

    stderr_thread.join(timeout=5)

    if stderr_lines:
        for sline in stderr_lines[:10]:
            print(f"    cursor-stderr: {sline[:200]}")
            if "Authentication required" in sline or "CURSOR_API_KEY" in sline:
                final_error = sline
            if "Cannot use this model" in sline:
                final_error = sline

    if duration_ms is not None:
        print(f"  result: duration={duration_ms}ms, events={len(trajectory)}")

    if proc.returncode != 0 and final_error:
        print(f"  cursor error: {final_error}")
    elif final_error and not trajectory:
        print(f"  cursor error: {final_error}")

    return trajectory, solution_path.exists()


def _extract_cursor_tool_name(tool_call: dict) -> str:
    """Extract a human-readable tool name from a Cursor NDJSON tool_call object."""
    if "function" in tool_call:
        return tool_call["function"].get("name", "tool")
    for key in tool_call:
        if key.endswith("ToolCall"):
            return key.replace("ToolCall", "")
    return "tool"


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
