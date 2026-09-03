"""Deterministic offline builder for the locked lm-eval runtime CAS."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from apex.execution import (
    DOCKER_RUNTIME_ENVIRONMENT_KEYS,
    SubprocessSupervisor,
    build_subprocess_environment,
)

from .lm_eval_lock import DownloadLock, LmEvalRuntimeLock, RUNTIME_SCHEMA, WheelLock
from .lm_eval_runtime import (
    LmEvalRuntimeReceipt,
    canonical_json,
    collect_runtime_files,
    default_lm_eval_runtime_root,
    verify_lm_eval_runtime,
)
from .repositories import BootstrapError


_SMOKE = """
import importlib.metadata as metadata
import json
import lm_eval
import sys
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.tasks import TaskManager

expected = json.loads(sys.argv[1])
for name, version in expected.items():
    if metadata.version(name) != version:
        raise SystemExit(f"base distribution mismatch: {name}")
tasks = TaskManager(include_path="/inferencex/utils/evals").match_tasks(["gsm8k"])
if not tasks or metadata.version("lm_eval") != "0.4.9.2":
    raise SystemExit("lm-eval identity/task discovery failed")
if not str(lm_eval.__file__).startswith("/runtime/site-packages/"):
    raise SystemExit("lm_eval imported outside isolated runtime")
print(json.dumps({"abi": sys.implementation.cache_tag, "tasks": tasks,
                  "model": LocalChatCompletion.__name__}, sort_keys=True))
""".strip()

_RESTORE_OWNERSHIP = """
import os
import sys

uid, gid = int(sys.argv[1]), int(sys.argv[2])
for raw_root in sys.argv[3:]:
    for current, directories, filenames in os.walk(raw_root):
        os.chown(current, uid, gid, follow_symlinks=False)
        for name in directories + filenames:
            os.chown(os.path.join(current, name), uid, gid, follow_symlinks=False)
""".strip()


class LmEvalRuntimePreparer:
    """Build one exact runtime from verified sources and publish it read-only."""

    def __init__(
        self,
        lock: LmEvalRuntimeLock,
        *,
        apex_root: Path,
        inferencex_root: Path,
        runtime_root: Path | None = None,
        artifact_cache: Path | None = None,
        offline: bool = False,
        supervisor: SubprocessSupervisor | None = None,
    ) -> None:
        self.lock = lock
        self.apex_root = apex_root.resolve()
        self.inferencex_root = inferencex_root.resolve()
        self.runtime_root = (
            runtime_root.resolve()
            if runtime_root is not None
            else default_lm_eval_runtime_root(self.apex_root, lock)
        )
        self.artifact_cache = (
            artifact_cache.resolve()
            if artifact_cache is not None
            else self.apex_root / ".cache" / "apex-runtime" / "downloads"
        )
        self.offline = offline
        self.supervisor = supervisor or SubprocessSupervisor(max_output_bytes=2 * 1024 * 1024)

    def prepare(self) -> LmEvalRuntimeReceipt:
        if self.runtime_root.exists():
            return verify_lm_eval_runtime(self.runtime_root, self.lock)
        self._verify_inferencex()
        self._verify_base_image()
        self.runtime_root.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix="apex-lm-eval-"))
        temporary.chmod(0o755)
        try:
            candidate = temporary / "runtime"
            wheelhouse = temporary / "wheelhouse"
            candidate.mkdir()
            wheelhouse.mkdir()
            self._materialize_wheels(temporary, wheelhouse)
            self._install(wheelhouse, candidate)
            self._smoke(candidate)
            self._write_manifest(candidate)
            return self._publish(candidate)
        finally:
            _remove_tree(temporary)

    def _verify_inferencex(self) -> None:
        if not self.inferencex_root.is_dir() or self.inferencex_root.is_symlink():
            raise BootstrapError("InferenceX runtime source must be a real directory")
        commit = self._run(
            ("git", "-C", str(self.inferencex_root), "rev-parse", "HEAD"), timeout=60
        ).strip()
        tree = self._run(
            ("git", "-C", str(self.inferencex_root), "rev-parse", "HEAD^{tree}"), timeout=60
        ).strip()
        expected = self.lock.identity
        if commit != expected["inferencex_commit"] or tree != expected["inferencex_tree"]:
            raise BootstrapError("InferenceX source differs from the lm-eval runtime lock")

    def _verify_base_image(self) -> None:
        image_id = self._run(
            (
                "docker", "image", "inspect", "--format", "{{.Id}}",
                self.lock.base_image,
            ),
            timeout=120,
        ).strip()
        if image_id != self.lock.identity["base_image_id"]:
            raise BootstrapError("lm-eval base image ID differs from its lock")

    def _materialize_wheels(self, temporary: Path, wheelhouse: Path) -> None:
        source_archive = self._artifact(self.lock.source)
        source = temporary / "lm-eval-source"
        _extract_source(source_archive, source)
        self._verify_source_tree(source)
        for wheel in self.lock.wheels:
            if wheel.download is not None:
                shutil.copyfile(self._artifact(wheel.download), wheelhouse / wheel.filename)
            elif wheel.name.casefold().replace("_", "-") == "lm-eval":
                self._build_source_wheel(source, wheelhouse)
            else:
                assert wheel.build_source is not None
                sources = temporary / "build-sources"
                sources.mkdir(exist_ok=True)
                local_source = sources / wheel.build_source.filename
                shutil.copyfile(self._artifact(wheel.build_source), local_source)
                self._build_archive_wheel(local_source, wheelhouse)
            _verify_file(wheelhouse / wheel.filename, wheel.sha256, field=wheel.filename)

    def _verify_source_tree(self, source: Path) -> None:
        git_directory = source / ".git"
        try:
            self._run(("git", "init", "--quiet", str(source)), timeout=60)
            self._run(("git", "-C", str(source), "add", "-f", "--all"), timeout=120)
            tree = self._run(("git", "-C", str(source), "write-tree"), timeout=120).strip()
            if tree != self.lock.identity["lm_eval_tree"]:
                raise BootstrapError("lm-eval source archive has the wrong Git tree")
        finally:
            _remove_tree(git_directory)
        pyproject = (source / "pyproject.toml").read_text(encoding="utf-8")
        version = f'version = "{self.lock.identity["lm_eval_version"]}"'
        if version not in pyproject:
            raise BootstrapError("lm-eval source version differs from its lock")

    def _build_source_wheel(self, source: Path, wheelhouse: Path) -> None:
        mounts = ((source, "/source", "rw"), (wheelhouse, "/wheelhouse", "rw"))
        try:
            self._docker_python(
                mounts=mounts,
                env={"SOURCE_DATE_EPOCH": str(self.lock.source_date_epoch)},
                arguments=(
                    "-m", "pip", "wheel", "--disable-pip-version-check", "--no-index",
                    "--no-deps", "--no-build-isolation", "--wheel-dir", "/wheelhouse", "/source",
                ),
                timeout=900,
                as_root=True,
            )
        finally:
            self._restore_ownership(mounts, ("/source", "/wheelhouse"))

    def _build_archive_wheel(self, source: Path, wheelhouse: Path) -> None:
        mounts = ((source.parent, "/artifacts", "ro"), (wheelhouse, "/wheelhouse", "rw"))
        try:
            self._docker_python(
                mounts=mounts,
                env={"SOURCE_DATE_EPOCH": str(self.lock.source_date_epoch)},
                arguments=(
                    "-m", "pip", "wheel", "--disable-pip-version-check", "--no-index",
                    "--no-deps", "--no-build-isolation", "--wheel-dir", "/wheelhouse",
                    f"/artifacts/{source.name}",
                ),
                timeout=600,
                as_root=True,
            )
        finally:
            self._restore_ownership(mounts, ("/wheelhouse",))

    def _restore_ownership(
        self,
        mounts: Sequence[tuple[Path, str, str]],
        targets: Sequence[str],
    ) -> None:
        self._docker_python(
            mounts=mounts,
            env={},
            arguments=(
                "-c", _RESTORE_OWNERSHIP, str(os.getuid()), str(os.getgid()), *targets,
            ),
            timeout=120,
            as_root=True,
        )

    def _install(self, wheelhouse: Path, candidate: Path) -> None:
        site_packages = candidate / "site-packages"
        site_packages.mkdir()
        self._docker_python(
            mounts=((wheelhouse, "/wheelhouse", "ro"), (candidate, "/runtime", "rw")),
            env={},
            arguments=(
                "-m", "pip", "install", "--disable-pip-version-check", "--no-index",
                "--no-deps", "--no-compile", "--target", "/runtime/site-packages",
                "--find-links", "/wheelhouse", *(wheel.requirement for wheel in self.lock.wheels),
            ),
            timeout=900,
        )

    def _smoke(self, candidate: Path) -> None:
        inferencex = candidate.parent / "inferencex-smoke"
        evaluations = self.inferencex_root / "utils" / "evals"
        if evaluations.is_symlink() or not evaluations.is_dir():
            raise BootstrapError("InferenceX evaluation task directory is invalid")
        shutil.copytree(evaluations, inferencex / "utils" / "evals")
        result = self._docker_python(
            mounts=((candidate, "/runtime", "ro"), (inferencex, "/inferencex", "ro")),
            env={
                "PYTHONPATH": "/runtime/site-packages", "PYTHONNOUSERSITE": "1",
                "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            arguments=("-c", _SMOKE, json.dumps(dict(self.lock.base_distributions), sort_keys=True)),
            timeout=180,
        )
        try:
            observed = json.loads(result)
        except json.JSONDecodeError as error:
            raise BootstrapError("lm-eval runtime smoke emitted invalid JSON") from error
        if observed.get("abi") != self.lock.identity["python_abi"]:
            raise BootstrapError("lm-eval runtime Python ABI differs from its lock")

    def _write_manifest(self, candidate: Path) -> None:
        site_packages = candidate / "site-packages"
        _normalize_read_only(site_packages)
        files = collect_runtime_files(site_packages)
        tree_digest = hashlib.sha256(canonical_json(files)).hexdigest()
        runtime_digest = hashlib.sha256(
            canonical_json({"identity": dict(self.lock.identity), "files": files})
        ).hexdigest()
        if tree_digest != self.lock.installed_tree_sha256 or runtime_digest != self.lock.runtime_sha256:
            raise BootstrapError(
                "built lm-eval runtime does not match locked tree digests: "
                f"tree expected {self.lock.installed_tree_sha256}, observed {tree_digest}; "
                f"runtime expected {self.lock.runtime_sha256}, observed {runtime_digest}"
            )
        manifest = {
            "schema": RUNTIME_SCHEMA, "runtime_sha256": runtime_digest,
            "site_packages": "site-packages", "identity": dict(self.lock.identity), "files": files,
        }
        path = candidate / "lm_eval_runtime_manifest.json"
        path.write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        path.chmod(0o444)
        candidate.chmod(0o555)

    def _publish(self, candidate: Path) -> LmEvalRuntimeReceipt:
        staging = Path(
            tempfile.mkdtemp(prefix=".lm-eval-staging-", dir=self.runtime_root.parent)
        )
        try:
            shutil.copytree(candidate, staging, dirs_exist_ok=True)
            verify_lm_eval_runtime(staging, self.lock)
            if not self.runtime_root.exists():
                try:
                    staging.rename(self.runtime_root)
                except OSError:
                    if not self.runtime_root.exists():
                        raise
            return verify_lm_eval_runtime(self.runtime_root, self.lock)
        finally:
            _remove_tree(staging)

    def _artifact(self, artifact: DownloadLock) -> Path:
        self.artifact_cache.mkdir(parents=True, exist_ok=True)
        target = self.artifact_cache / artifact.filename
        if target.exists():
            _verify_download(target, artifact)
            return target
        if self.offline:
            raise BootstrapError(f"offline lm-eval artifact is missing: {artifact.filename}")
        _download_artifact(artifact, target)
        return target

    def _docker_python(
        self,
        *,
        mounts: Sequence[tuple[Path, str, str]],
        env: Mapping[str, str],
        arguments: Sequence[str],
        timeout: int,
        as_root: bool = False,
    ) -> str:
        argv = ["docker", "run", "--rm", "--network=none"]
        if not as_root:
            argv.extend(("--user", f"{os.getuid()}:{os.getgid()}"))
        argv.extend(("--entrypoint", "python3"))
        for key, value in sorted(env.items()):
            argv.extend(("-e", f"{key}={value}"))
        for source, target, mode in mounts:
            argv.extend(("-v", f"{source.resolve()}:{target}:{mode}"))
        argv.extend((self.lock.base_image, *arguments))
        return self._run(tuple(argv), timeout=timeout)

    def _run(self, argv: Sequence[str], *, timeout: int) -> str:
        environment = build_subprocess_environment(inherit=DOCKER_RUNTIME_ENVIRONMENT_KEYS)
        result = self.supervisor.run(
            argv, cwd=self.apex_root, environment=environment, timeout_seconds=timeout
        )
        if result.exit_code != 0 or result.timed_out:
            detail = (result.stderr or result.stdout)[-2000:]
            raise BootstrapError(f"lm-eval runtime command failed: {' '.join(argv)}\n{detail}")
        return result.stdout.strip()


def _download_artifact(artifact: DownloadLock, target: Path) -> None:
    if target.is_symlink():
        raise BootstrapError(f"refusing artifact-cache symlink: {target}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with urllib.request.urlopen(artifact.url, timeout=120) as response, temporary.open("xb") as out:
            remaining = artifact.size_bytes
            while remaining:
                block = response.read(min(1024 * 1024, remaining + 1))
                if not block or len(block) > remaining:
                    raise BootstrapError(f"download size mismatch for {artifact.filename}")
                out.write(block)
                remaining -= len(block)
            if response.read(1) or remaining:
                raise BootstrapError(f"download size mismatch for {artifact.filename}")
        _verify_download(temporary, artifact)
        os.replace(temporary, target)
    except BootstrapError:
        raise
    except OSError as error:
        raise BootstrapError(
            f"cannot download locked artifact {artifact.filename}: {error}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _verify_file(path: Path, expected: str, *, field: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise BootstrapError(f"{field} is not a regular file")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise BootstrapError(
            f"{field} digest mismatch: expected {expected}, observed {observed}"
        )


def _verify_download(path: Path, artifact: DownloadLock) -> None:
    if path.stat().st_size != artifact.size_bytes:
        raise BootstrapError(f"{artifact.filename} size mismatch")
    _verify_file(path, artifact.sha256, field=artifact.filename)


def _extract_source(archive: Path, destination: Path) -> None:
    destination.mkdir()
    directory_times: list[tuple[Path, int]] = []
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        roots = {PurePosixPath(item.name).parts[0] for item in members if item.name}
        if len(roots) != 1:
            raise BootstrapError("lm-eval source archive must have one root")
        for member in members:
            parts = PurePosixPath(member.name).parts[1:]
            if not parts:
                if member.isdir():
                    destination.chmod(0o700)
                    directory_times.append((destination, member.mtime))
                continue
            if ".." in parts or member.issym() or member.islnk():
                raise BootstrapError("lm-eval source archive contains an unsafe entry")
            target = destination.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                target.chmod(0o700)
                directory_times.append((target, member.mtime))
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = source.extractfile(member)
                if stream is None:
                    raise BootstrapError("lm-eval source archive file is unreadable")
                with stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output)
                target.chmod(0o700 if stat.S_IMODE(member.mode) & 0o111 else 0o600)
                os.utime(target, (member.mtime, member.mtime))
            else:
                raise BootstrapError("lm-eval source archive contains a special entry")
    for directory, modified in reversed(directory_times):
        os.utime(directory, (modified, modified))


def _normalize_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise BootstrapError(f"lm-eval runtime contains a symlink: {path}")
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), reverse=True):
        path.chmod(0o555)
    root.chmod(0o555)


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for current, directories, files in os.walk(path, topdown=False):
        for name in files:
            (Path(current) / name).chmod(0o600)
        for name in directories:
            (Path(current) / name).chmod(0o700)
    path.chmod(0o700)
    shutil.rmtree(path)


__all__ = ["LmEvalRuntimePreparer"]
