"""Bounded host broker from a sidecar Unix socket to one verified TCP listener."""

from __future__ import annotations

import os
import re
import socket
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path

from apex.core import ConfigurationError, sha256_json


SCHEMA = "apex.evaluator-serving-broker-receipt/v1"
SOCKET_NAME = "serving.sock"
MAX_CONNECTIONS = 512
_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class EvaluatorServingBrokerReceipt:
    """Observed broker activity bound to the preverified serving listener."""

    serving_port: int
    listener_receipt_sha256: str
    max_connections: int
    connection_count: int

    def __post_init__(self) -> None:
        if (
            not 1 <= self.serving_port <= 65535
            or not _DIGEST.fullmatch(self.listener_receipt_sha256)
            or not 1 <= self.max_connections <= MAX_CONNECTIONS
            or not 1 <= self.connection_count
        ):
            raise ValueError("Evaluator serving broker receipt is invalid")

    @property
    def sha256(self) -> str:
        return sha256_json(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "target": {"host": "127.0.0.1", "port": self.serving_port},
            "unix_socket": SOCKET_NAME,
            "listener_receipt_sha256": self.listener_receipt_sha256,
            "max_connections": self.max_connections,
            "connection_count": self.connection_count,
            "verified": True,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_sha256": self.sha256}


@dataclass(slots=True)
class EvaluatorServingBrokerSession:
    root: Path
    socket_path: Path
    serving_port: int
    listener_receipt_sha256: str
    max_connections: int
    listener: socket.socket
    stop: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    workers: list[threading.Thread] = field(default_factory=list)
    thread: threading.Thread | None = None
    connection_count: int = 0
    error: str | None = None


class EvaluatorServingBroker:
    """Expose only one loopback serving listener to a no-network sidecar."""

    def start(
        self,
        root: Path,
        *,
        serving_port: int,
        listener_receipt_sha256: str,
        max_connections: int,
    ) -> EvaluatorServingBrokerSession:
        if (
            not 1 <= serving_port <= 65535
            or not _DIGEST.fullmatch(listener_receipt_sha256)
            or not 1 <= max_connections <= MAX_CONNECTIONS
        ):
            raise _invalid("Evaluator serving broker contract is invalid")
        try:
            root.mkdir(mode=0o700, exist_ok=False)
        except OSError as error:
            raise _invalid("Cannot create evaluator serving broker root") from error
        socket_path = root / SOCKET_NAME
        listener = _bind(socket_path, max_connections)
        session = EvaluatorServingBrokerSession(
            root.resolve(strict=True),
            socket_path,
            serving_port,
            listener_receipt_sha256,
            max_connections,
            listener,
        )
        session.thread = threading.Thread(
            target=_accept_loop,
            args=(session,),
            name="apex-evaluator-serving-broker",
            daemon=True,
        )
        session.thread.start()
        return session

    def stop(
        self, session: EvaluatorServingBrokerSession
    ) -> EvaluatorServingBrokerReceipt:
        self._shutdown(session)
        try:
            return EvaluatorServingBrokerReceipt(
                session.serving_port,
                session.listener_receipt_sha256,
                session.max_connections,
                session.connection_count,
            )
        except ValueError as error:
            raise _invalid("Evaluator serving broker saw no traffic") from error

    def abort(self, session: EvaluatorServingBrokerSession) -> None:
        """Close an uncommitted broker without claiming it relayed traffic."""

        self._shutdown(session)

    @staticmethod
    def _shutdown(session: EvaluatorServingBrokerSession) -> None:
        session.stop.set()
        _wake(session.socket_path)
        if session.thread is not None:
            session.thread.join(timeout=5)
        with session.lock:
            workers = tuple(session.workers)
        for worker in workers:
            worker.join(timeout=5)
        session.listener.close()
        _remove_socket(session.socket_path)
        if (
            session.error
            or session.thread is None
            or session.thread.is_alive()
            or any(worker.is_alive() for worker in workers)
        ):
            raise _invalid(session.error or "Evaluator serving broker did not stop")


def _accept_loop(session: EvaluatorServingBrokerSession) -> None:
    capacity = threading.BoundedSemaphore(session.max_connections)
    while not session.stop.is_set():
        try:
            connection, _ = session.listener.accept()
        except socket.timeout:
            continue
        except OSError as error:
            if not session.stop.is_set():
                session.error = f"serving_broker_accept_failed:{error}"
            return
        if session.stop.is_set():
            connection.close()
            return
        if not capacity.acquire(blocking=False):
            connection.close()
            session.error = "serving_broker_capacity_exceeded"
            return
        worker = threading.Thread(
            target=_serve_connection,
            args=(session, connection, capacity),
            daemon=True,
        )
        with session.lock:
            session.workers.append(worker)
            session.connection_count += 1
        worker.start()


def _serve_connection(
    session: EvaluatorServingBrokerSession,
    connection: socket.socket,
    capacity: threading.BoundedSemaphore,
) -> None:
    upstream: socket.socket | None = None
    try:
        upstream = socket.create_connection(
            ("127.0.0.1", session.serving_port), timeout=30
        )
        upstream.settimeout(None)
        connection.settimeout(None)
        reverse = threading.Thread(
            target=_pump, args=(connection, upstream), daemon=True
        )
        reverse.start()
        _pump(upstream, connection)
        reverse.join(timeout=5)
        if reverse.is_alive():
            session.error = "serving_broker_relay_did_not_stop"
    except OSError as error:
        session.error = f"serving_broker_connection_failed:{error}"
    finally:
        connection.close()
        if upstream is not None:
            upstream.close()
        capacity.release()


def _pump(source: socket.socket, target: socket.socket) -> None:
    try:
        while True:
            payload = source.recv(1024 * 1024)
            if not payload:
                break
            target.sendall(payload)
    except OSError:
        pass
    finally:
        try:
            target.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _bind(path: Path, backlog: int) -> socket.socket:
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    directory = os.open(
        path.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        listener.bind(f"/proc/self/fd/{directory}/{path.name}")
        os.chmod(path, 0o600)
        listener.listen(backlog)
        listener.settimeout(0.1)
        return listener
    except Exception:
        listener.close()
        _remove_socket(path)
        raise
    finally:
        os.close(directory)


def _wake(path: Path) -> None:
    try:
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(0.2)
            client.connect(f"/proc/self/fd/{directory}/{path.name}")
    except OSError:
        pass
    finally:
        os.close(directory)


def _remove_socket(path: Path) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISSOCK(observed.st_mode):
        path.unlink()


def _invalid(message: str) -> ConfigurationError:
    return ConfigurationError(message, "evaluator_serving_broker_invalid")


__all__ = [
    "EvaluatorServingBroker",
    "EvaluatorServingBrokerReceipt",
    "EvaluatorServingBrokerSession",
]
