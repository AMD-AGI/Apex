from __future__ import annotations

import os
import socket
import threading
from pathlib import Path

import pytest

from apex.benchmark.evaluator_serving_broker import EvaluatorServingBroker
from apex.core import ConfigurationError


class _EchoServer:
    def __init__(self) -> None:
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(4)
        self.port = int(self.listener.getsockname()[1])
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        try:
            connection, _ = self.listener.accept()
            with connection:
                while True:
                    payload = connection.recv(4096)
                    if not payload:
                        return
                    connection.sendall(payload)
        except OSError:
            pass

    def close(self) -> None:
        self.listener.close()
        self.thread.join(timeout=5)


def _connect(path: Path) -> socket.socket:
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(f"/proc/self/fd/{directory}/{path.name}")
    finally:
        os.close(directory)
    return client


def test_relays_only_to_frozen_loopback_listener_and_receipts_use(tmp_path: Path) -> None:
    echo = _EchoServer()
    broker = EvaluatorServingBroker()
    root = tmp_path / ("long-" * 20) / "broker"
    root.parent.mkdir(parents=True)
    session = broker.start(
        root,
        serving_port=echo.port,
        listener_receipt_sha256="1" * 64,
        max_connections=16,
    )
    try:
        with _connect(session.socket_path) as client:
            client.sendall(b"hello")
            client.shutdown(socket.SHUT_WR)
            assert client.recv(5) == b"hello"
        receipt = broker.stop(session)
    finally:
        echo.close()

    assert receipt.serving_port == echo.port
    assert receipt.listener_receipt_sha256 == "1" * 64
    assert receipt.connection_count == 1
    assert receipt.to_dict()["target"] == {
        "host": "127.0.0.1",
        "port": echo.port,
    }
    assert not session.socket_path.exists()


def test_rejects_unexercised_broker(tmp_path: Path) -> None:
    broker = EvaluatorServingBroker()
    session = broker.start(
        tmp_path / "broker",
        serving_port=8888,
        listener_receipt_sha256="1" * 64,
        max_connections=16,
    )

    with pytest.raises(ConfigurationError, match="saw no traffic"):
        broker.stop(session)


def test_abort_closes_unexercised_broker_without_a_receipt(tmp_path: Path) -> None:
    broker = EvaluatorServingBroker()
    session = broker.start(
        tmp_path / "broker-abort",
        serving_port=8888,
        listener_receipt_sha256="1" * 64,
        max_connections=16,
    )

    broker.abort(session)

    assert not session.socket_path.exists()


@pytest.mark.parametrize(
    "values",
    [
        {"serving_port": 0, "listener_receipt_sha256": "1" * 64, "max_connections": 1},
        {"serving_port": 8888, "listener_receipt_sha256": "bad", "max_connections": 1},
        {"serving_port": 8888, "listener_receipt_sha256": "1" * 64, "max_connections": 0},
    ],
)
def test_rejects_invalid_broker_contract(tmp_path: Path, values) -> None:
    with pytest.raises(ConfigurationError, match="contract is invalid"):
        EvaluatorServingBroker().start(tmp_path / "broker", **values)
