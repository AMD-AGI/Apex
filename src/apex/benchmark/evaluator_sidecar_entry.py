"""Network-isolated sidecar entrypoint with a bounded local-to-Unix proxy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import selectors
import socket
import socketserver
import subprocess
import sys
import threading
from pathlib import Path


MAX_PROXY_CONNECTIONS = 512
MAX_PROBE_BYTES = 1024 * 1024


class _ProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True

    def __init__(self, address, handler, *, unix_socket: str, limit: int) -> None:
        self.unix_socket = unix_socket
        self.capacity = threading.BoundedSemaphore(limit)
        self.request_queue_size = min(limit, 512)
        super().__init__(address, handler)


class _ProxyHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server = self.server
        assert isinstance(server, _ProxyServer)
        if not server.capacity.acquire(timeout=5):
            return
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
                upstream.settimeout(30)
                upstream.connect(server.unix_socket)
                upstream.settimeout(None)
                _relay(self.request, upstream)
        finally:
            server.capacity.release()


def _relay(left: socket.socket, right: socket.socket) -> None:
    selector = selectors.DefaultSelector()
    for source in (left, right):
        source.setblocking(False)
        selector.register(source, selectors.EVENT_READ)
    peers = {left: right, right: left}
    active = {left, right}
    try:
        while active:
            events = selector.select(timeout=60)
            if not events:
                break
            for key, _ in events:
                source = key.fileobj
                assert isinstance(source, socket.socket)
                try:
                    payload = source.recv(1024 * 1024)
                except BlockingIOError:
                    continue
                if not payload:
                    selector.unregister(source)
                    active.discard(source)
                    try:
                        peers[source].shutdown(socket.SHUT_WR)
                    except OSError:
                        pass
                    continue
                _send_nonblocking(peers[source], payload)
    finally:
        selector.close()


def _send_nonblocking(target: socket.socket, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        try:
            sent = target.send(view)
        except BlockingIOError:
            continue
        if sent <= 0:
            raise ConnectionError("proxy peer closed while writing")
        view = view[sent:]


def _runtime_probe(path: Path) -> None:
    import lm_eval

    module = Path(lm_eval.__file__).resolve(strict=True)
    payload = {
        "schema": "apex.lm-eval-runtime-probe/v1",
        "python": {
            "implementation": sys.implementation.name,
            "version": list(sys.version_info[:3]),
            "executable": sys.executable,
        },
        "lm_eval": {
            "version": importlib.metadata.version("lm-eval"),
            "module_path": str(module),
            "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        },
        "python_path": sys.path,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    if len(encoded) > MAX_PROBE_BYTES:
        raise RuntimeError("runtime probe exceeds its bound")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o400
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("cannot write runtime probe")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unix-socket", required=True)
    parser.add_argument("--proxy-port", required=True, type=int)
    parser.add_argument("--max-connections", required=True, type=int)
    parser.add_argument("--runtime-probe", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if (
        not 1 <= args.proxy_port <= 65535
        or not 1 <= args.max_connections <= MAX_PROXY_CONNECTIONS
        or not args.command
        or args.command[0] != "--"
        or len(args.command) < 2
    ):
        parser.error("invalid sidecar entrypoint contract")
    command = args.command[1:]
    server = _ProxyServer(
        ("127.0.0.1", args.proxy_port),
        _ProxyHandler,
        unix_socket=args.unix_socket,
        limit=args.max_connections,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _runtime_probe(args.runtime_probe)
        completed = subprocess.run(command, check=False, shell=False)
        return int(completed.returncode)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
