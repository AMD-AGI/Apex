"""Run-scoped InferenceX projection for the Apex evaluator handoff."""

from __future__ import annotations

import os
import re
import secrets
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from apex.core import ConfigurationError, canonical_json_bytes, sha256_file, sha256_json

from .evaluator_execution import LmEvalExecutionContract


PROJECTION_SCHEMA = "apex.evaluator-inferencex-projection/v1"
HANDOFF_SCHEMA = "apex.evaluator-handoff-contract/v1"
SOCKET_RELATIVE_PATH = ".apex/evaluator_handoff.sock"
_CLIENT_RELATIVE_PATH = "benchmarks/apex_evaluator_handoff.py"
_LIBRARY_RELATIVE_PATH = "benchmarks/benchmark_lib.sh"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class EvaluatorInferenceXProjectionReceipt:
    """Exact source and overlay identities for one private execution tree."""

    inferencex_commit: str
    inferencex_tree: str
    magpie_commit: str
    magpie_tree: str
    source_manifest_sha256: str
    magpie_scripts_manifest_sha256: str
    handoff_contract_sha256: str
    overlay_sha256: str
    projection_manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            not all(
                _COMMIT.fullmatch(value)
                for value in (
                    self.inferencex_commit,
                    self.inferencex_tree,
                    self.magpie_commit,
                    self.magpie_tree,
                )
            )
            or not all(
                _DIGEST.fullmatch(value)
                for value in (
                    self.source_manifest_sha256,
                    self.magpie_scripts_manifest_sha256,
                    self.handoff_contract_sha256,
                    self.overlay_sha256,
                    self.projection_manifest_sha256,
                )
            )
        ):
            raise ValueError("Evaluator InferenceX projection receipt is invalid")

    @property
    def sha256(self) -> str:
        return sha256_json(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema": PROJECTION_SCHEMA,
            "inferencex": {
                "commit": self.inferencex_commit,
                "tree": self.inferencex_tree,
                "source_manifest_sha256": self.source_manifest_sha256,
            },
            "magpie": {
                "commit": self.magpie_commit,
                "tree": self.magpie_tree,
                "scripts_manifest_sha256": self.magpie_scripts_manifest_sha256,
            },
            "handoff_contract_sha256": self.handoff_contract_sha256,
            "overlay_sha256": self.overlay_sha256,
            "projection_manifest_sha256": self.projection_manifest_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "receipt_sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class PreparedInferenceXProjection:
    """Private mutable tree plus immutable pre-launch receipts."""

    root: Path
    handoff_socket: Path
    handoff_contract_path: Path
    receipt: EvaluatorInferenceXProjectionReceipt


def materialize_inferencex_projection(
    source_root: Path,
    magpie_root: Path,
    destination: Path,
    *,
    inferencex_commit: str,
    inferencex_tree: str,
    magpie_commit: str,
    magpie_tree: str,
    execution_contract: LmEvalExecutionContract,
    nonce: str | None = None,
) -> PreparedInferenceXProjection:
    """Copy pinned sources, install the handoff overlay, and bind the result."""

    source = _source_root(source_root, "InferenceX")
    magpie = _source_root(magpie_root, "Magpie")
    if destination.exists() or destination.is_symlink():
        raise _invalid("Evaluator InferenceX projection already exists")
    source_manifest = _tree_manifest(source)
    scripts = _magpie_scripts(magpie)
    try:
        shutil.copytree(source, destination, symlinks=True, ignore=_copy_ignore)
        _copy_magpie_scripts(scripts, destination / "benchmarks")
        handoff = _write_handoff_contract(
            destination, execution_contract, nonce or secrets.token_hex(32)
        )
        overlay = _install_overlay(destination)
        projection_manifest = _tree_manifest(destination)
    except Exception:
        _remove_projection(destination)
        raise
    receipt = EvaluatorInferenceXProjectionReceipt(
        inferencex_commit=inferencex_commit,
        inferencex_tree=inferencex_tree,
        magpie_commit=magpie_commit,
        magpie_tree=magpie_tree,
        source_manifest_sha256=source_manifest,
        magpie_scripts_manifest_sha256=_paths_manifest(scripts, magpie),
        handoff_contract_sha256=sha256_file(handoff),
        overlay_sha256=overlay,
        projection_manifest_sha256=projection_manifest,
    )
    return PreparedInferenceXProjection(
        root=destination.resolve(strict=True),
        handoff_socket=destination / SOCKET_RELATIVE_PATH,
        handoff_contract_path=handoff,
        receipt=receipt,
    )


def verify_inferencex_projection(
    projection: PreparedInferenceXProjection,
) -> None:
    """Reject any content drift after Magpie has used the private tree."""

    if projection.handoff_socket.exists():
        raise _invalid("Evaluator handoff socket was not cleaned up")
    if _tree_manifest(projection.root) != projection.receipt.projection_manifest_sha256:
        raise _invalid("Evaluator InferenceX projection changed during execution")
    if sha256_file(projection.handoff_contract_path) != projection.receipt.handoff_contract_sha256:
        raise _invalid("Evaluator handoff contract changed during execution")


def _source_root(path: Path, label: str) -> Path:
    try:
        observed = path.lstat()
        selected = path.resolve(strict=True)
    except OSError as error:
        raise _invalid(f"{label} source root is unavailable") from error
    if path.is_symlink() or not stat.S_ISDIR(observed.st_mode):
        raise _invalid(f"{label} source root is unsafe")
    return selected


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in {".git", "__pycache__"} or name.endswith(".pyc")}


def _magpie_scripts(root: Path) -> tuple[Path, ...]:
    selected = root / "Magpie" / "scripts" / "benchmark"
    scripts = tuple(sorted(selected.glob("*.sh")))
    if not scripts or any(path.is_symlink() or not path.is_file() for path in scripts):
        raise _invalid("Pinned Magpie benchmark scripts are unavailable")
    return scripts


def _copy_magpie_scripts(scripts: tuple[Path, ...], destination: Path) -> None:
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    for source in scripts:
        target = destination / source.name
        shutil.copyfile(source, target)
        target.chmod(0o755)


def _write_handoff_contract(
    root: Path, contract: LmEvalExecutionContract, nonce: str
) -> Path:
    if len(nonce) != 64 or set(nonce) - set("0123456789abcdef"):
        raise _invalid("Evaluator handoff nonce is invalid")
    value = {
        "schema": HANDOFF_SCHEMA,
        "run_id": contract.run_id,
        "execution_contract_sha256": contract.sha256,
        "serving_port": contract.endpoint_port,
        "concurrent_requests": contract.concurrent_requests,
        "timeout_seconds": contract.timeout_seconds,
        "nonce": nonce,
    }
    directory = root / ".apex"
    directory.mkdir(mode=0o700)
    return _write_new(directory / "handoff_contract.json", canonical_json_bytes(value) + b"\n")


def _install_overlay(root: Path) -> str:
    client = _write_new(root / _CLIENT_RELATIVE_PATH, _CLIENT_SOURCE.encode("utf-8"))
    client.chmod(0o500)
    library = root / _LIBRARY_RELATIVE_PATH
    if library.is_symlink() or not library.is_file():
        raise _invalid("InferenceX benchmark library is unavailable")
    with library.open("ab") as output:
        output.write(_SHELL_OVERLAY.encode("utf-8"))
        output.flush()
        os.fsync(output.fileno())
    library.chmod(0o755)
    return sha256_json(
        {
            "client_sha256": sha256_file(client),
            "shell_overlay_sha256": sha256_json(_SHELL_OVERLAY),
        }
    )


def _write_new(path: Path, payload: bytes) -> Path:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o400)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise _invalid("Cannot write evaluator projection artifact")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path.resolve(strict=True)


def _tree_manifest(root: Path) -> str:
    entries: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        parts = PurePosixPath(relative).parts
        if (
            relative == SOCKET_RELATIVE_PATH
            or ".git" in parts
            or "__pycache__" in parts
            or relative.endswith(".pyc")
        ):
            continue
        observed = path.lstat()
        if stat.S_ISDIR(observed.st_mode):
            continue
        if stat.S_ISLNK(observed.st_mode):
            target = os.readlink(path)
            _validate_link(relative, target)
            entries.append({"path": relative, "kind": "symlink", "target": target})
        elif stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1:
            entries.append(
                {
                    "path": relative,
                    "kind": "file",
                    "mode": stat.S_IMODE(observed.st_mode),
                    "size_bytes": observed.st_size,
                    "sha256": sha256_file(path),
                }
            )
        else:
            raise _invalid("Evaluator projection contains an unsafe entry")
    return sha256_json(entries)


def _paths_manifest(paths: tuple[Path, ...], root: Path) -> str:
    return sha256_json(
        [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ]
    )


def _validate_link(relative: str, target: str) -> None:
    selected = PurePosixPath(relative).parent / target
    if PurePosixPath(target).is_absolute() or ".." in selected.parts:
        raise _invalid("Evaluator projection contains an unsafe symlink")


def _remove_projection(path: Path) -> None:
    if not path.exists():
        return
    shutil.rmtree(path)


def _invalid(message: str) -> ConfigurationError:
    return ConfigurationError(message, "evaluator_inferencex_projection_invalid")


_SHELL_OVERLAY = r'''

# Modified by Apex: replace InferenceX's online evaluator with a blocking,
# run-scoped handoff to the independent Apex evaluator authority.
run_eval() {
    python3 /opt/InferenceX/benchmarks/apex_evaluator_handoff.py "$@"
}

append_lm_eval_summary() {
    return 0
}
'''


_CLIENT_SOURCE = r'''#!/usr/bin/env python3
import json
import socket
import sys
from pathlib import Path

contract_path = Path("/opt/InferenceX/.apex/handoff_contract.json")
socket_path = "/opt/InferenceX/.apex/evaluator_handoff.sock"
contract = json.loads(contract_path.read_text(encoding="utf-8"))
request = {
    "schema": "apex.evaluator-handoff-request/v1",
    "run_id": contract["run_id"],
    "execution_contract_sha256": contract["execution_contract_sha256"],
    "nonce": contract["nonce"],
    "argv": sys.argv[1:],
}
payload = json.dumps(request, sort_keys=True, separators=(",", ":")).encode() + b"\n"
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.settimeout(int(contract["timeout_seconds"]) + 60)
    client.connect(socket_path)
    client.sendall(payload)
    response = b""
    while not response.endswith(b"\n") and len(response) <= 65536:
        chunk = client.recv(65536 - len(response) + 1)
        if not chunk:
            break
        response += chunk
value = json.loads(response)
if value.get("schema") != "apex.evaluator-handoff-response/v1":
    raise SystemExit(125)
raise SystemExit(int(value.get("exit_code", 125)))
'''


__all__ = [
    "EvaluatorInferenceXProjectionReceipt",
    "PreparedInferenceXProjection",
    "SOCKET_RELATIVE_PATH",
    "materialize_inferencex_projection",
    "verify_inferencex_projection",
]
