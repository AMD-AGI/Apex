"""Bounded subprocess execution with concurrent pipe draining and group cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from apex.core import ContractError


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Complete normalized result of a supervised process group."""

    argv: tuple[str, ...]
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    duration_seconds: float
    budget_exceeded: bool = False


class _BoundedText:
    def __init__(self, limit_bytes: int) -> None:
        self._limit = limit_bytes
        self._parts: list[bytes] = []
        self._size = 0
        self.truncated = False

    def add(self, text: str) -> None:
        content = text.encode("utf-8", errors="replace")
        remaining = self._limit - self._size
        if remaining > 0:
            kept = content[:remaining]
            self._parts.append(kept)
            self._size += len(kept)
        if len(content) > max(remaining, 0):
            self.truncated = True

    def text(self) -> str:
        value = b"".join(self._parts).decode("utf-8", errors="replace")
        return value + ("\n[apex output truncated]\n" if self.truncated else "")


class SubprocessSupervisor:
    """Run argv without a shell and always terminate the complete process group."""

    def __init__(self, *, max_output_bytes: int = 16 * 1024 * 1024, kill_grace_seconds: float = 2.0) -> None:
        if max_output_bytes <= 0 or kill_grace_seconds < 0:
            raise ValueError("invalid supervisor limits")
        self._max_output_bytes = max_output_bytes
        self._kill_grace_seconds = kill_grace_seconds

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: int,
        stdin_text: str | None = None,
        stdout_budget: Callable[[str], bool] | None = None,
    ) -> ProcessResult:
        command = tuple(argv)
        self._validate(command, cwd, timeout_seconds)
        started = time.monotonic()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        stdout = _BoundedText(self._max_output_bytes)
        stderr = _BoundedText(self._max_output_bytes)
        budget_stop = threading.Event()
        readers = self._start_readers(
            process,
            stdout,
            stderr,
            stdout_budget=stdout_budget,
            budget_stop=budget_stop,
        )
        if stdin_text is not None and process.stdin is not None:
            try:
                process.stdin.write(stdin_text)
                process.stdin.close()
            except BrokenPipeError:
                pass

        timed_out = False
        if stdout_budget is None:
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate_group(process)
        else:
            timed_out = self._wait_with_budget(
                process,
                budget_stop=budget_stop,
                timeout_seconds=timeout_seconds,
            )
        self._finish_readers(process.pid, readers)
        return ProcessResult(
            argv=command,
            exit_code=process.returncode,
            timed_out=timed_out,
            stdout=stdout.text(),
            stderr=stderr.text(),
            stdout_truncated=stdout.truncated,
            stderr_truncated=stderr.truncated,
            duration_seconds=time.monotonic() - started,
            budget_exceeded=budget_stop.is_set(),
        )

    @staticmethod
    def _validate(argv: tuple[str, ...], cwd: Path, timeout_seconds: int) -> None:
        if not argv or any(not isinstance(arg, str) or not arg for arg in argv):
            raise ContractError("subprocess argv must contain non-empty strings", "invalid_subprocess")
        if not cwd.is_absolute() or not cwd.is_dir():
            raise ContractError("subprocess cwd must be an absolute directory", "invalid_subprocess_cwd")
        if timeout_seconds <= 0:
            raise ContractError("subprocess timeout must be positive", "invalid_subprocess_timeout")

    @staticmethod
    def _start_readers(
        process: subprocess.Popen[str],
        stdout: _BoundedText,
        stderr: _BoundedText,
        *,
        stdout_budget: Callable[[str], bool] | None,
        budget_stop: threading.Event,
    ) -> tuple[threading.Thread, threading.Thread]:
        assert process.stdout is not None and process.stderr is not None

        def drain(
            pipe: object,
            target: _BoundedText,
            observer: Callable[[str], bool] | None = None,
        ) -> None:
            try:
                for line in pipe:  # type: ignore[union-attr]
                    target.add(line)
                    if observer is not None and not budget_stop.is_set():
                        try:
                            exceeded = observer(line)
                        except Exception:  # pragma: no cover - defensive fail-closed boundary
                            exceeded = True
                        if exceeded:
                            budget_stop.set()
            finally:
                pipe.close()  # type: ignore[union-attr]

        threads = (
            threading.Thread(
                target=drain,
                args=(process.stdout, stdout, stdout_budget),
                daemon=True,
            ),
            threading.Thread(target=drain, args=(process.stderr, stderr), daemon=True),
        )
        for thread in threads:
            thread.start()
        return threads

    def _wait_with_budget(
        self,
        process: subprocess.Popen[str],
        *,
        budget_stop: threading.Event,
        timeout_seconds: int,
    ) -> bool:
        deadline = time.monotonic() + timeout_seconds
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate_group(process)
                return True
            if budget_stop.wait(timeout=min(remaining, 0.05)):
                self._terminate_group(process)
                return False
        return False

    def _terminate_group(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=self._kill_grace_seconds)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    def _finish_readers(self, process_group: int, readers: tuple[threading.Thread, ...]) -> None:
        for reader in readers:
            reader.join(timeout=0.05)
        if not any(reader.is_alive() for reader in readers):
            return
        self._signal_group(process_group, signal.SIGTERM)
        for reader in readers:
            reader.join(timeout=max(self._kill_grace_seconds, 0.1))
        if any(reader.is_alive() for reader in readers):
            self._signal_group(process_group, signal.SIGKILL)
            for reader in readers:
                reader.join(timeout=1.0)

    @staticmethod
    def _signal_group(process_group: int, sig: signal.Signals) -> None:
        try:
            os.killpg(process_group, sig)
        except ProcessLookupError:
            pass
