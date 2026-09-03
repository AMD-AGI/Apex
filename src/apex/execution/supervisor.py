"""Bounded subprocess execution and authoritative agent-process containment."""

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
from apex.ports import AgentProcessContainmentReceipt

from .containment import (
    ActivePidNamespace,
    abort_prepared_namespace,
    establish_pid_namespace,
    finalize_pid_namespace,
    prepare_pid_namespace,
)


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Complete normalized result of a supervised process."""

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
    process_containment: AgentProcessContainmentReceipt | None = None
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


class SubprocessSupervisor:
    """Run argv without a shell and contain untrusted agent descendants."""

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
        require_pid_namespace: bool = False,
    ) -> ProcessResult:
        command = tuple(argv)
        self._validate(command, cwd, timeout_seconds)
        if stdout_budget is not None and not require_pid_namespace:
            raise ContractError(
                "Streaming agent budgets require PID namespace containment",
                "agent_process_containment_required",
            )
        started = time.monotonic()
        process, boundary = self._start_process(
            command,
            cwd=cwd,
            environment=environment,
            stdin_text=stdin_text,
            require_pid_namespace=require_pid_namespace,
        )
        stdout = _BoundedText(self._max_output_bytes)
        stderr = _BoundedText(self._max_output_bytes)
        discarded_stdout = _DiscardedTail()
        observer_stop = threading.Event()
        readers = self._start_readers(
            process,
            stdout,
            stderr,
            discarded_stdout,
            boundary,
            stdout_budget=stdout_budget,
            observer_stop=observer_stop,
        )
        if stdin_text is not None and process.stdin is not None:
            try:
                process.stdin.write(stdin_text)
            except BrokenPipeError:
                pass
            finally:
                process.stdin.close()

        (
            timed_out,
            observer_termination_started,
            containment,
            cleanup_succeeded,
        ) = self._complete_process(
            process,
            boundary=boundary,
            observer_stop=observer_stop,
            readers=readers,
            timeout_seconds=timeout_seconds,
        )
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
            process_containment=containment,
            cleanup_succeeded=cleanup_succeeded,
            discarded_stdout_lines=discarded_stdout.lines,
            discarded_stdout_bytes=discarded_stdout.bytes,
            discarded_stdout_sha256=discarded_stdout.sha256,
        )

    def _complete_process(
        self,
        process: subprocess.Popen[str],
        *,
        boundary: ActivePidNamespace | None,
        observer_stop: threading.Event,
        readers: tuple[threading.Thread, threading.Thread],
        timeout_seconds: int,
    ) -> tuple[bool, bool, AgentProcessContainmentReceipt | None, bool]:
        if boundary is None:
            timed_out = self._wait_uncontained(process, timeout_seconds)
            readers_finished = self._finish_readers(
                process.pid, readers, contained=False
            )
            cleanup = readers_finished and self._reap_process_group(process.pid)
            return timed_out, False, None, cleanup
        timed_out, observer_started = self._wait_contained(
            process,
            observer_stop=observer_stop,
            timeout_seconds=timeout_seconds,
        )
        if not timed_out and not observer_started:
            # The wrapper can exit after writing the boundary but before the
            # stdout reader has evaluated that line. Drain it before freezing
            # the reason so transcript and containment evidence cannot diverge.
            readers_finished = self._finish_readers(
                process.pid, readers, contained=True
            )
            observer_started = observer_stop.is_set()
            reason = (
                "stdout_budget_boundary"
                if observer_started
                else "natural_exit"
                if readers_finished
                else "stdout_observer_unresolved"
            )
            receipt = finalize_pid_namespace(
                process,
                boundary,
                termination_reason=reason,
                terminate=observer_started or not readers_finished,
                timeout_seconds=max(self._kill_grace_seconds, 1.0),
            )
            return (
                False,
                observer_started,
                receipt,
                readers_finished and receipt.namespace_empty_verified,
            )
        reason = (
            "timeout"
            if timed_out
            else "stdout_budget_boundary"
        )
        receipt = finalize_pid_namespace(
            process,
            boundary,
            termination_reason=reason,
            terminate=timed_out or observer_started,
            timeout_seconds=max(self._kill_grace_seconds, 1.0),
        )
        readers_finished = self._finish_readers(process.pid, readers, contained=True)
        return (
            timed_out,
            observer_started,
            receipt,
            readers_finished and receipt.namespace_empty_verified,
        )

    @staticmethod
    def _start_process(
        command: tuple[str, ...],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        stdin_text: str | None,
        require_pid_namespace: bool,
    ) -> tuple[subprocess.Popen[str], ActivePidNamespace | None]:
        prepared = prepare_pid_namespace(command) if require_pid_namespace else None
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                prepared.argv if prepared is not None else command,
                cwd=cwd,
                env=dict(environment),
                stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
                pass_fds=prepared.pass_fds if prepared is not None else (),
            )
            if prepared is None:
                return process, None
            prepared.release_child_fds()
            return process, establish_pid_namespace(prepared, process)
        except Exception:
            if prepared is not None:
                abort_prepared_namespace(prepared, process)
            raise

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
        boundary: ActivePidNamespace | None,
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
                            if boundary is None:
                                raise RuntimeError("stream boundary lacks process containment")
                            observer_stop.set()
                            try:
                                boundary.terminate_now()
                            except Exception:
                                # The supervisor observes the boundary and repeats
                                # exact-pidfd teardown while producing fail-closed
                                # containment evidence.
                                return
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

    @staticmethod
    def _wait_contained(
        process: subprocess.Popen[str],
        *,
        observer_stop: threading.Event,
        timeout_seconds: int,
    ) -> tuple[bool, bool]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            if observer_stop.is_set():
                return False, True
            if process.poll() is not None:
                return False, observer_stop.is_set()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True, False
            if observer_stop.wait(timeout=min(remaining, 0.05)):
                return False, True

    def _wait_uncontained(
        self, process: subprocess.Popen[str], timeout_seconds: int
    ) -> bool:
        try:
            process.wait(timeout=timeout_seconds)
            return False
        except subprocess.TimeoutExpired:
            self._terminate_group(process)
            return True

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

    def _finish_readers(
        self,
        process_group: int,
        readers: tuple[threading.Thread, ...],
        *,
        contained: bool,
    ) -> bool:
        for reader in readers:
            reader.join(timeout=0.05)
        if not any(reader.is_alive() for reader in readers):
            return True
        if contained:
            for reader in readers:
                reader.join(timeout=max(self._kill_grace_seconds, 1.0))
            return not any(reader.is_alive() for reader in readers)
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
