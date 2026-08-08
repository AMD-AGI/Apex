#!/usr/bin/env python3
"""Fresh-checkout launcher for ``apex.runtime.dependencies``.

This shim uses only the standard library: it creates/reuses the selected venv,
installs Apex itself there when necessary, and then replaces this process with
the authoritative runtime dependency CLI. It never modifies ``sys.path``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


APEX_IMPORT_PROBE = r"""
import apex
import pathlib
print(pathlib.Path(apex.__file__).resolve())
"""

BUILD_TOOL_VERSIONS = {
    "packaging": "26.3",
    "setuptools": "83.0.0",
    "wheel": "0.47.0",
}
BUILD_TOOL_PROBE = r"""
import importlib.metadata
import json
import setuptools.build_meta
import sys

expected = json.loads(sys.argv[1])
observed = {name: importlib.metadata.version(name) for name in expected}
print(json.dumps({
    "versions": observed,
    "build_editable": hasattr(setuptools.build_meta, "build_editable"),
}, sort_keys=True))
"""


class LauncherError(RuntimeError):
    """The Apex runtime CLI could not be prepared safely."""


def _run(
    argv: Sequence[str], *, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            list(argv),
            env=dict(env) if env else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise LauncherError(f"cannot execute {argv[0]!r}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise LauncherError(
            f"command failed ({result.returncode}): {' '.join(argv)}"
            + (f"\n{detail[-2000:]}" if detail else "")
        )
    return result


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _prepare_venv(venv: Path, base_python: str) -> Path:
    python = venv / "bin" / "python"
    if python.is_file():
        return python
    if venv.exists() and any(venv.iterdir()):
        raise LauncherError(
            f"venv path exists but has no bin/python; refusing to overwrite: {venv}"
        )
    venv.parent.mkdir(parents=True, exist_ok=True)
    _run((base_python, "-m", "venv", str(venv)))
    if not python.is_file():
        raise LauncherError(f"venv creation did not produce {python}")
    return python


def _runtime_is_installed(python: Path, apex_root: Path) -> bool:
    result = subprocess.run(
        (str(python), "-c", APEX_IMPORT_PROBE),
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    try:
        imported = Path(result.stdout.strip()).resolve()
        imported.relative_to((apex_root / "src" / "apex").resolve())
    except (OSError, ValueError):
        return False
    return True


def _build_tools_ready(python: Path) -> bool:
    result = subprocess.run(
        (
            str(python),
            "-c",
            BUILD_TOOL_PROBE,
            json.dumps(BUILD_TOOL_VERSIONS, sort_keys=True),
        ),
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    try:
        observed = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    return bool(
        observed.get("versions") == BUILD_TOOL_VERSIONS
        and observed.get("build_editable") is True
    )


def _prepare_build_tools(python: Path, *, offline: bool) -> None:
    """Install the exact backend needed by old system pip without isolation."""

    if _build_tools_ready(python):
        return
    requirements = tuple(
        f"{name}=={version}" for name, version in BUILD_TOOL_VERSIONS.items()
    )
    if offline:
        raise LauncherError(
            "offline setup requires locked build tools in the selected venv: "
            + ", ".join(requirements)
        )
    _run(
        (
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            *requirements,
        ),
        env=_clean_env(),
    )
    if not _build_tools_ready(python):
        raise LauncherError("locked Python build tools failed verification")


def _install_runtime(
    python: Path, apex_root: Path, *, offline: bool
) -> None:
    argv = [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-build-isolation",
        "--no-deps",
        "--editable",
        str(apex_root),
    ]
    if offline:
        argv[5:5] = ["--no-index"]
    _run(argv, env=_clean_env())
    if not _runtime_is_installed(python, apex_root):
        raise LauncherError(
            f"Apex runtime import does not resolve inside {apex_root / 'src' / 'apex'}"
        )


def _launcher_options(
    argv: Sequence[str], apex_root: Path
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--venv", type=Path, default=apex_root / ".venv")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--json", action="store_true")
    options, _ = parser.parse_known_args(argv)
    return options


def _emit_error(error: str, *, json_output: bool) -> None:
    if json_output:
        print(
            json.dumps(
                {
                    "schema": "apex.dependencies.receipt/v1",
                    "status": "error",
                    "error": error,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"dependency launcher failed: {error}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    forwarded = list(argv if argv is not None else sys.argv[1:])
    apex_root = Path(__file__).resolve().parents[1]
    options = _launcher_options(forwarded, apex_root)
    venv = options.venv.expanduser().resolve()
    try:
        python = _prepare_venv(venv, options.python)
        _prepare_build_tools(python, offline=options.offline)
        if not _runtime_is_installed(python, apex_root):
            _install_runtime(python, apex_root, offline=options.offline)
        env = _clean_env()
        os.execve(
            str(python),
            (str(python), "-m", "apex.runtime.dependencies", *forwarded),
            env,
        )
    except LauncherError as exc:
        _emit_error(str(exc), json_output=options.json)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
