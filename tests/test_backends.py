# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Tests for agents/backends.py — retry logic, turn tracking, agent summary."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "agents"))

import backends


def _make_mock_proc(returncode=0, stdout_lines=None, stderr_text=""):
    """Create a mock subprocess.Popen that streams stdout lines and returns stderr."""
    proc = MagicMock()
    proc.returncode = returncode
    if stdout_lines is None:
        stdout_lines = []
    proc.stdout = iter(stdout_lines)
    proc.stderr = MagicMock()
    proc.stderr.read.return_value = stderr_text
    proc.wait.return_value = returncode
    return proc


class TestCodexRetry:
    def test_codex_retries_on_transient_failure(self, tmp_path):
        solution = tmp_path / "solution.py"
        call_count = {"n": 0}

        def _mock_popen(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_mock_proc(returncode=1, stderr_text="connection reset")
            solution.write_text("# optimized")
            return _make_mock_proc(returncode=0)

        with patch.object(backends, "_command_exists", return_value=True), \
             patch("subprocess.Popen", side_effect=_mock_popen), \
             patch.object(backends, "_time") as mock_time:
            mock_time.sleep = MagicMock()
            trajectory, found = backends._run_codex_task(
                prompt="optimize", cwd=tmp_path, model=None,
                max_turns=5, system_prompt=None, solution_path=solution,
            )
        assert call_count["n"] == 2
        assert found is True

    def test_codex_no_retry_on_auth_error(self, tmp_path):
        solution = tmp_path / "solution.py"

        def _mock_popen(*args, **kwargs):
            return _make_mock_proc(
                returncode=1,
                stderr_text="authentication failed: invalid API key",
            )

        with patch.object(backends, "_command_exists", return_value=True), \
             patch("subprocess.Popen", side_effect=_mock_popen):
            trajectory, found = backends._run_codex_task(
                prompt="optimize", cwd=tmp_path, model=None,
                max_turns=5, system_prompt=None, solution_path=solution,
            )
        assert found is False

    def test_codex_max_retries_exhausted(self, tmp_path):
        solution = tmp_path / "solution.py"
        call_count = {"n": 0}

        def _mock_popen(*args, **kwargs):
            call_count["n"] += 1
            return _make_mock_proc(returncode=1, stderr_text="server error 500")

        with patch.object(backends, "_command_exists", return_value=True), \
             patch("subprocess.Popen", side_effect=_mock_popen), \
             patch.object(backends, "_time") as mock_time:
            mock_time.sleep = MagicMock()
            trajectory, found = backends._run_codex_task(
                prompt="optimize", cwd=tmp_path, model=None,
                max_turns=5, system_prompt=None, solution_path=solution,
            )
        assert call_count["n"] == 3  # 1 initial + 2 retries
        assert found is False

    def test_codex_clears_stale_solution_before_retry(self, tmp_path):
        solution = tmp_path / "solution.py"
        solution.write_text("# pre-existing")
        call_count = {"n": 0}

        def _mock_popen(*args, **kwargs):
            call_count["n"] += 1
            return _make_mock_proc(returncode=1, stderr_text="crash")

        with patch.object(backends, "_command_exists", return_value=True), \
             patch("subprocess.Popen", side_effect=_mock_popen), \
             patch.object(backends, "_time") as mock_time:
            mock_time.sleep = MagicMock()
            trajectory, found = backends._run_codex_task(
                prompt="optimize", cwd=tmp_path, model=None,
                max_turns=5, system_prompt=None, solution_path=solution,
            )
        assert call_count["n"] == 3  # stale file cleared, all retries attempted
        assert found is False


def _make_cursor_mock_proc(returncode=0, stdout_lines=None, stderr_lines_iter=None):
    """Create a mock for cursor that handles select.select and threading."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdin = MagicMock()
    if stdout_lines is None:
        stdout_lines = []

    stdout_mock = MagicMock()
    stdout_mock.readline = MagicMock(side_effect=list(stdout_lines) + [""])
    proc.stdout = stdout_mock

    if stderr_lines_iter is None:
        stderr_lines_iter = iter([])
    proc.stderr = MagicMock()
    proc.stderr.__iter__ = MagicMock(return_value=stderr_lines_iter)
    proc.wait.return_value = returncode
    proc.kill = MagicMock()
    return proc


class TestCursorRetry:
    def test_cursor_retries_on_transient_failure(self, tmp_path):
        solution = tmp_path / "solution.py"
        call_count = {"n": 0}

        def _mock_popen(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _make_cursor_mock_proc(returncode=1, stderr_lines_iter=iter(["connection reset\n"]))
            solution.write_text("# optimized")
            return _make_cursor_mock_proc(returncode=0)

        def _mock_select(rlist, wlist, xlist, timeout=None):
            return (rlist, [], [])

        import select as _sel_mod
        with patch.object(backends, "_find_cursor_agent_binary", return_value="cursor-agent"), \
             patch("subprocess.Popen", side_effect=_mock_popen), \
             patch.object(_sel_mod, "select", side_effect=_mock_select), \
             patch.object(backends, "_time") as mock_time:
            mock_time.sleep = MagicMock()
            trajectory, found = backends._run_cursor_task(
                prompt="optimize", cwd=tmp_path, model=None,
                max_turns=5, system_prompt=None, solution_path=solution,
            )
        assert call_count["n"] == 2
        assert found is True

    def test_cursor_no_retry_on_auth_error(self, tmp_path):
        solution = tmp_path / "solution.py"

        def _mock_popen(*args, **kwargs):
            return _make_cursor_mock_proc(
                returncode=1,
                stderr_lines_iter=iter(["Authentication required\n"]),
            )

        def _mock_select(rlist, wlist, xlist, timeout=None):
            return (rlist, [], [])

        import select as _sel_mod
        with patch.object(backends, "_find_cursor_agent_binary", return_value="cursor-agent"), \
             patch("subprocess.Popen", side_effect=_mock_popen), \
             patch.object(_sel_mod, "select", side_effect=_mock_select):
            trajectory, found = backends._run_cursor_task(
                prompt="optimize", cwd=tmp_path, model=None,
                max_turns=5, system_prompt=None, solution_path=solution,
            )
        assert found is False

    def test_cursor_max_retries_exhausted(self, tmp_path):
        solution = tmp_path / "solution.py"
        call_count = {"n": 0}

        def _mock_popen(*args, **kwargs):
            call_count["n"] += 1
            return _make_cursor_mock_proc(returncode=1, stderr_lines_iter=iter(["server error\n"]))

        def _mock_select(rlist, wlist, xlist, timeout=None):
            return (rlist, [], [])

        import select as _sel_mod
        with patch.object(backends, "_find_cursor_agent_binary", return_value="cursor-agent"), \
             patch("subprocess.Popen", side_effect=_mock_popen), \
             patch.object(_sel_mod, "select", side_effect=_mock_select), \
             patch.object(backends, "_time") as mock_time:
            mock_time.sleep = MagicMock()
            trajectory, found = backends._run_cursor_task(
                prompt="optimize", cwd=tmp_path, model=None,
                max_turns=5, system_prompt=None, solution_path=solution,
            )
        assert call_count["n"] == 3
        assert found is False


class TestAgentSummaryEvent:
    """Verify _agent_summary is appended to trajectory."""

    def test_codex_appends_summary_event(self, tmp_path):
        solution = tmp_path / "solution.py"
        turn_event = json.dumps({
            "type": "turn.completed",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }) + "\n"

        def _mock_popen(*args, **kwargs):
            solution.write_text("# done")
            return _make_mock_proc(returncode=0, stdout_lines=[turn_event])

        with patch.object(backends, "_command_exists", return_value=True), \
             patch("subprocess.Popen", side_effect=_mock_popen):
            trajectory, found = backends._run_codex_task(
                prompt="optimize", cwd=tmp_path, model=None,
                max_turns=5, system_prompt=None, solution_path=solution,
            )
        summary = [e for e in trajectory if isinstance(e, dict) and e.get("type") == "_agent_summary"]
        assert len(summary) == 1
        assert summary[0]["turns"] == 1
        assert summary[0]["input_tokens"] == 100
        assert summary[0]["output_tokens"] == 50

    def test_cursor_appends_summary_event(self, tmp_path):
        solution = tmp_path / "solution.py"
        result_event = json.dumps({
            "type": "result", "duration_ms": 5000, "is_error": False,
        }) + "\n"

        def _mock_popen(*args, **kwargs):
            solution.write_text("# done")
            return _make_cursor_mock_proc(
                returncode=0,
                stdout_lines=[result_event],
            )

        def _mock_select(rlist, wlist, xlist, timeout=None):
            return (rlist, [], [])

        import select as _sel_mod
        with patch.object(backends, "_find_cursor_agent_binary", return_value="cursor-agent"), \
             patch("subprocess.Popen", side_effect=_mock_popen), \
             patch.object(_sel_mod, "select", side_effect=_mock_select):
            trajectory, found = backends._run_cursor_task(
                prompt="optimize", cwd=tmp_path, model=None,
                max_turns=5, system_prompt=None, solution_path=solution,
            )
        summary = [e for e in trajectory if isinstance(e, dict) and e.get("type") == "_agent_summary"]
        assert len(summary) == 1
        assert "duration_ms" in summary[0]

    def test_summary_event_zero_on_failure(self, tmp_path):
        solution = tmp_path / "solution.py"

        def _mock_popen(*args, **kwargs):
            return _make_mock_proc(returncode=1, stderr_text="crash")

        with patch.object(backends, "_command_exists", return_value=True), \
             patch("subprocess.Popen", side_effect=_mock_popen), \
             patch.object(backends, "_time") as mock_time:
            mock_time.sleep = MagicMock()
            trajectory, found = backends._run_codex_task(
                prompt="optimize", cwd=tmp_path, model=None,
                max_turns=5, system_prompt=None, solution_path=solution,
            )
        summary = [e for e in trajectory if isinstance(e, dict) and e.get("type") == "_agent_summary"]
        assert len(summary) == 1
        assert summary[0]["turns"] == 0
