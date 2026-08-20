"""Blocking Unix-socket handoff from Magpie to the evaluator sidecar authority."""

from __future__ import annotations

import json
import os
import socket
import stat
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from apex.core import ConfigurationError, canonical_json_bytes, sha256_json

from .evaluator_execution import (
    LmEvalExecutionReceipt,
    validate_receipt_against_contract,
)
from .evaluator_preparation import PreparedLmEvalExecution


REQUEST_SCHEMA = "apex.evaluator-handoff-request/v1"
RESPONSE_SCHEMA = "apex.evaluator-handoff-response/v1"
RECEIPT_SCHEMA = "apex.evaluator-handoff-receipt/v1"
MAX_MESSAGE_BYTES = 64 * 1024


class EvaluatorSidecarAuthority(Protocol):
    """Run an independently observed sidecar from frozen preparation inputs."""

    def execute(
        self, prepared: PreparedLmEvalExecution
    ) -> LmEvalExecutionReceipt: ...


@dataclass(frozen=True, slots=True)
class CompletedEvaluatorHandoff:
    """Verified execution plus the ordering receipt released to Magpie."""

    execution_receipt: LmEvalExecutionReceipt
    handoff_receipt_path: Path
    handoff_receipt_sha256: str


@dataclass(slots=True)
class EvaluatorHandoffSession:
    """Mutable run-scoped state owned only by the handoff authority."""

    prepared: PreparedLmEvalExecution
    authority: EvaluatorSidecarAuthority
    listener: socket.socket
    started_ns: int
    abort: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    request: Mapping[str, Any] | None = None
    request_received_ns: int | None = None
    sidecar_started_ns: int | None = None
    sidecar_finished_ns: int | None = None
    handoff_released_ns: int | None = None
    execution_receipt: LmEvalExecutionReceipt | None = None
    error: str | None = None
    error_reason_code: str | None = None


class EvaluatorHandoffBarrier:
    """Accept exactly one post-throughput request and block until sidecar exit."""

    def start(
        self,
        prepared: PreparedLmEvalExecution,
        authority: EvaluatorSidecarAuthority,
    ) -> EvaluatorHandoffSession:
        listener = _bind_listener(prepared.inferencex_projection.handoff_socket)
        session = EvaluatorHandoffSession(
            prepared=prepared,
            authority=authority,
            listener=listener,
            started_ns=time.monotonic_ns(),
        )
        session.thread = threading.Thread(
            target=_serve,
            args=(session,),
            name=f"apex-evaluator-handoff-{prepared.contract.run_id}",
            daemon=True,
        )
        session.thread.start()
        return session

    def complete(
        self, session: EvaluatorHandoffSession
    ) -> CompletedEvaluatorHandoff:
        if session.thread is None:
            raise _invalid("Evaluator handoff listener did not start")
        session.thread.join(timeout=5)
        if session.thread.is_alive():
            self.abort(session, reason="handoff_incomplete_after_magpie_exit")
        if session.error is not None or session.execution_receipt is None:
            raise ConfigurationError(
                session.error or "Evaluator handoff did not execute",
                session.error_reason_code or "evaluator_handoff_invalid",
            )
        value = _handoff_receipt(session)
        path = _write_new(
            session.prepared.authority_root / "handoff_receipt.json", value
        )
        return CompletedEvaluatorHandoff(
            session.execution_receipt,
            path,
            sha256_json(value),
        )

    def abort(self, session: EvaluatorHandoffSession, *, reason: str) -> None:
        session.abort.set()
        if session.error is None:
            session.error = reason
            session.error_reason_code = reason
        _wake(session.prepared.inferencex_projection.handoff_socket)
        if session.thread is not None:
            session.thread.join(timeout=5)
            if session.thread.is_alive():
                raise _invalid("Evaluator handoff did not stop after abort")


def _serve(session: EvaluatorHandoffSession) -> None:
    connection: socket.socket | None = None
    try:
        connection = _accept(session)
        if connection is None:
            return
        request = _read_request(connection)
        _validate_request(request, session.prepared)
        session.request = request
        session.request_received_ns = time.monotonic_ns()
        session.sidecar_started_ns = time.monotonic_ns()
        execution = session.authority.execute(session.prepared)
        session.sidecar_finished_ns = time.monotonic_ns()
        mismatch = validate_receipt_against_contract(
            execution, session.prepared.contract
        )
        if mismatch:
            raise _invalid(mismatch)
        _write_new(
            session.prepared.authority_root / "execution_receipt.json",
            execution.to_dict(),
        )
        session.execution_receipt = execution
        _send_response(connection, 0)
        session.handoff_released_ns = time.monotonic_ns()
    except Exception as error:
        session.error = f"{type(error).__name__}:{error}"
        session.error_reason_code = getattr(
            error, "reason_code", "evaluator_handoff_invalid"
        )
        if connection is not None:
            try:
                _send_response(connection, 125)
            except OSError:
                pass
    finally:
        if connection is not None:
            connection.close()
        session.listener.close()
        _remove_socket(session.prepared.inferencex_projection.handoff_socket)
        session.done.set()


def _bind_listener(path: Path) -> socket.socket:
    if path.exists() or path.is_symlink():
        raise _invalid("Evaluator handoff socket already exists")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    directory: int | None = None
    try:
        directory = _socket_directory(path)
        listener.bind(_descriptor_socket_path(directory, path.name))
        os.chmod(path, 0o600)
        listener.listen(1)
        listener.settimeout(0.1)
    except Exception:
        listener.close()
        _remove_socket(path)
        raise
    finally:
        if directory is not None:
            os.close(directory)
    return listener


def _accept(session: EvaluatorHandoffSession) -> socket.socket | None:
    deadline = time.monotonic() + session.prepared.contract.timeout_seconds + 60
    while not session.abort.is_set() and time.monotonic() < deadline:
        try:
            connection, _ = session.listener.accept()
            connection.settimeout(5)
            return connection
        except socket.timeout:
            continue
    if session.error is None:
        session.error = "evaluator_handoff_request_timeout"
    return None


def _read_request(connection: socket.socket) -> Mapping[str, Any]:
    payload = bytearray()
    while not payload.endswith(b"\n") and len(payload) <= MAX_MESSAGE_BYTES:
        chunk = connection.recv(min(4096, MAX_MESSAGE_BYTES + 1 - len(payload)))
        if not chunk:
            break
        payload.extend(chunk)
    if not payload.endswith(b"\n") or len(payload) > MAX_MESSAGE_BYTES:
        raise _invalid("Evaluator handoff request is not bounded")
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise _invalid("Evaluator handoff request is invalid JSON") from error
    if not isinstance(value, Mapping):
        raise _invalid("Evaluator handoff request is invalid")
    return value


def _validate_request(
    value: Mapping[str, Any], prepared: PreparedLmEvalExecution
) -> None:
    handoff = json.loads(
        prepared.inferencex_projection.handoff_contract_path.read_text(
            encoding="utf-8"
        )
    )
    contract = prepared.contract
    expected_argv = [
        "--framework", "lm-eval",
        "--port", str(contract.endpoint_port),
        "--concurrent-requests", str(contract.concurrent_requests),
    ]
    if (
        set(value) != {
            "schema", "run_id", "execution_contract_sha256", "nonce", "argv"
        }
        or value.get("schema") != REQUEST_SCHEMA
        or value.get("run_id") != contract.run_id
        or value.get("execution_contract_sha256") != contract.sha256
        or value.get("nonce") != handoff.get("nonce")
        or value.get("argv") != expected_argv
    ):
        raise _invalid("Evaluator handoff request differs from its contract")


def _send_response(connection: socket.socket, exit_code: int) -> None:
    payload = canonical_json_bytes(
        {"schema": RESPONSE_SCHEMA, "exit_code": exit_code}
    ) + b"\n"
    connection.sendall(payload)


def _handoff_receipt(session: EvaluatorHandoffSession) -> dict[str, object]:
    values = (
        session.request_received_ns,
        session.sidecar_started_ns,
        session.sidecar_finished_ns,
        session.handoff_released_ns,
    )
    if (
        session.request is None
        or session.execution_receipt is None
        or any(value is None for value in values)
        or tuple(values) != tuple(sorted(values))
    ):
        raise _invalid("Evaluator handoff ordering is invalid")
    return {
        "schema": RECEIPT_SCHEMA,
        "verified": True,
        "request_sha256": sha256_json(dict(session.request)),
        "execution_receipt_sha256": session.execution_receipt.sha256,
        "ordering_ns": {
            "listener_started": session.started_ns,
            "request_received": session.request_received_ns,
            "sidecar_started": session.sidecar_started_ns,
            "sidecar_finished": session.sidecar_finished_ns,
            "handoff_released": session.handoff_released_ns,
        },
    }


def _write_new(path: Path, value: Mapping[str, Any]) -> Path:
    payload = canonical_json_bytes(value) + b"\n"
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o400
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise _invalid("Cannot write evaluator handoff artifact")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path.resolve(strict=True)


def _wake(path: Path) -> None:
    try:
        directory = _socket_directory(path)
    except OSError:
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(0.2)
            client.connect(_descriptor_socket_path(directory, path.name))
    except OSError:
        pass
    finally:
        os.close(directory)


def _socket_directory(path: Path) -> int:
    return os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )


def _descriptor_socket_path(directory: int, name: str) -> str:
    if not name or "/" in name or name in {".", ".."}:
        raise _invalid("Evaluator handoff socket name is unsafe")
    return f"/proc/self/fd/{directory}/{name}"


def _remove_socket(path: Path) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISSOCK(observed.st_mode):
        path.unlink()


def _invalid(message: str) -> ConfigurationError:
    return ConfigurationError(message, "evaluator_handoff_invalid")


__all__ = [
    "CompletedEvaluatorHandoff",
    "EvaluatorHandoffBarrier",
    "EvaluatorHandoffSession",
    "EvaluatorSidecarAuthority",
]
