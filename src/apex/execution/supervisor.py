"""Bounded subprocess execution with concurrent pipe draining and group cleanup."""

from __future__ import annotations

import hashlib
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
    observer_stopped: bool = False
    observer_termination_started: bool = False
    observer_suspend_sent: bool = False
    suspension_verified: bool = False
    cleanup_succeeded: bool = True
    discarded_stdout_lines: int = 0
    discarded_stdout_bytes: int = 0
    discarded_stdout_sha256: str | None = None


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


class _DiscardedTail:
    """Digest stdout received after the observer's terminal boundary."""

    def __init__(self) -> None:
        self.lines = 0
        self.bytes = 0
        self._digest = hashlib.sha256()

    def add(self, text: str) -> None:
        content = text.encode("utf-8", errors="replace")
        self.lines += 1
        self.bytes += len(content)
        self._digest.update(content)

    @property
    def sha256(self) -> str | None:
        return self._digest.hexdigest() if self.lines else None


@dataclass(slots=True)
class _BoundarySuspension:
    sent: bool = False
    verified: bool = False


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
        discarded_stdout = _DiscardedTail()
        suspension = _BoundarySuspension()
        observer_stop = threading.Event()
        readers = self._start_readers(
            process,
            stdout,
            stderr,
            discarded_stdout,
            suspension,
            stdout_budget=stdout_budget,
            observer_stop=observer_stop,
        )
        if stdin_text is not None and process.stdin is not None:
            try:
                process.stdin.write(stdin_text)
                process.stdin.close()
            except BrokenPipeError:
                pass

        timed_out = False
        observer_termination_started = False
        if stdout_budget is None:
            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate_group(process)
        else:
            timed_out, observer_termination_started = self._wait_with_budget(
                process,
                observer_stop=observer_stop,
                suspension=suspension,
                timeout_seconds=timeout_seconds,
            )
        readers_finished = self._finish_readers(process.pid, readers)
        group_reaped = self._reap_process_group(process.pid)
        return ProcessResult(
            argv=command,
            exit_code=process.returncode,
            timed_out=timed_out,
            stdout=stdout.text(),
            stderr=stderr.text(),
            stdout_truncated=stdout.truncated,
            stderr_truncated=stderr.truncated,
            duration_seconds=time.monotonic() - started,
            observer_stopped=observer_stop.is_set(),
            observer_termination_started=observer_termination_started,
            observer_suspend_sent=suspension.sent,
            suspension_verified=suspension.verified,
            cleanup_succeeded=readers_finished and group_reaped,
            discarded_stdout_lines=discarded_stdout.lines,
            discarded_stdout_bytes=discarded_stdout.bytes,
            discarded_stdout_sha256=discarded_stdout.sha256,
        )

    @staticmethod
    def _validate(argv: tuple[str, ...], cwd: Path, timeout_seconds: int) -> None:
        if not argv or any(not isinstance(arg, str) or not arg for arg in argv):
            raise ContractError("subprocess argv must contain non-empty strings", "invalid_subprocess")
        if not cwd.is_absolute() or not cwd.is_dir():
            raise ContractError("subprocess cwd must be an absolute directory", "invalid_subprocess_cwd")
        if timeout_seconds <= 0:
            raise ContractError("subprocess timeout must be positive", "invalid_subprocess_timeout")

    def _start_readers(
        self,
        process: subprocess.Popen[str],
        stdout: _BoundedText,
        stderr: _BoundedText,
        discarded_stdout: _DiscardedTail,
        suspension: _BoundarySuspension,
        *,
        stdout_budget: Callable[[str], bool] | None,
        observer_stop: threading.Event,
    ) -> tuple[threading.Thread, threading.Thread]:
        assert process.stdout is not None and process.stderr is not None

        def drain(
            pipe: object,
            target: _BoundedText,
            observer: Callable[[str], bool] | None = None,
            discarded: _DiscardedTail | None = None,
        ) -> None:
            try:
                for line in pipe:  # type: ignore[union-attr]
                    if observer is not None and observer_stop.is_set():
                        assert discarded is not None
                        discarded.add(line)
                        continue
                    target.add(line)
                    if observer is not None:
                        try:
                            should_stop = observer(line)
                        except Exception:  # pragma: no cover - defensive fail-closed boundary
                            should_stop = True
                        if should_stop:
                            suspension.sent, suspension.verified = self._suspend_group(
                                process.pid
                            )
                            observer_stop.set()
            finally:
                pipe.close()  # type: ignore[union-attr]

        threads = (
            threading.Thread(
                target=drain,
                args=(process.stdout, stdout, stdout_budget, discarded_stdout),
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
        observer_stop: threading.Event,
        suspension: _BoundarySuspension,
        timeout_seconds: int,
    ) -> tuple[bool, bool]:
        deadline = time.monotonic() + timeout_seconds
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate_group(process)
                return True, False
            if observer_stop.wait(timeout=min(remaining, 0.05)):
                if suspension.sent:
                    self._kill_suspended_group(process)
                else:
                    self._terminate_group(process)
                return False, True
        return False, False

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

    @staticmethod
    def _kill_suspended_group(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait()

    def _suspend_group(self, process_group: int) -> tuple[bool, bool]:
        try:
            os.killpg(process_group, signal.SIGSTOP)
        except ProcessLookupError:
            return False, False
        deadline = time.monotonic() + max(min(self._kill_grace_seconds, 0.5), 0.1)
        while time.monotonic() < deadline:
            if self._group_is_suspended(process_group):
                return True, True
            time.sleep(0.001)
        return True, self._group_is_suspended(process_group)

    @staticmethod
    def _group_is_suspended(process_group: int) -> bool:
        states: list[str] = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                suffix = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                state, group = suffix[0], int(suffix[2])
            except (FileNotFoundError, IndexError, OSError, ValueError):
                continue
            if group == process_group:
                states.append(state)
        return bool(states) and any(state in {"T", "t"} for state in states) and all(
            state in {"T", "t", "Z", "X", "x"} for state in states
        )

    def _finish_readers(
        self, process_group: int, readers: tuple[threading.Thread, ...]
    ) -> bool:
        for reader in readers:
            reader.join(timeout=0.05)
        if not any(reader.is_alive() for reader in readers):
            return True
        self._signal_group(process_group, signal.SIGTERM)
        for reader in readers:
            reader.join(timeout=max(self._kill_grace_seconds, 0.1))
        if any(reader.is_alive() for reader in readers):
            self._signal_group(process_group, signal.SIGKILL)
            for reader in readers:
                reader.join(timeout=1.0)
        return not any(reader.is_alive() for reader in readers)

    @staticmethod
    def _signal_group(process_group: int, sig: signal.Signals) -> None:
        try:
            os.killpg(process_group, sig)
        except ProcessLookupError:
            pass

    def _reap_process_group(self, process_group: int) -> bool:
        """Verify no same-group descendant survives after the leader exits."""

        if not self._group_exists(process_group):
            return True
        self._signal_group(process_group, signal.SIGTERM)
        deadline = time.monotonic() + self._kill_grace_seconds
        while self._group_exists(process_group) and time.monotonic() < deadline:
            time.sleep(0.01)
        if not self._group_exists(process_group):
            return True
        self._signal_group(process_group, signal.SIGKILL)
        deadline = time.monotonic() + 1.0
        while self._group_exists(process_group) and time.monotonic() < deadline:
            time.sleep(0.01)
        return not self._group_exists(process_group)

    @staticmethod
    def _group_exists(process_group: int) -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
