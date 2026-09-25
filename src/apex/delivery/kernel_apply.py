"""Explicit, exact-baseline application of a verified kernel bundle."""

from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from apex.core import ContractError, IntegrityError, sha256_bytes, sha256_file
from apex.execution import SubprocessSupervisor, build_subprocess_environment

from .kernel_bundle import KernelBundle, load_and_verify_kernel_bundle


@dataclass(frozen=True, slots=True)
class KernelApplyReceipt:
    """Evidence left after an explicit, verified workspace mutation."""

    task_id: str
    bundle_digest: str
    workspace: str
    changed_files: tuple[str, ...]
    applied_file_hashes: Mapping[str, str]
    applied: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_verified_kernel_bundle(
    bundle_dir: Path,
    workspace: Path,
    *,
    expected_digest: str | None = None,
    supervisor: SubprocessSupervisor | None = None,
) -> KernelApplyReceipt:
    """Apply only to an exact clean Git baseline, rolling back on failure."""

    bundle = load_and_verify_kernel_bundle(
        bundle_dir, expected_digest=expected_digest
    )
    root = _clean_git_root(workspace, supervisor)
    baseline, candidates, patch_payload = _bundle_contract(bundle)
    _verify_patch_targets(root, patch_payload, bundle.changed_files, supervisor)
    _verify_baseline(root, baseline, bundle.changed_files)
    _git(
        root,
        ("apply", "--check", "--whitespace=nowarn"),
        supervisor,
        stdin_text=patch_payload,
    )
    backups = _backups(root, bundle.changed_files)
    try:
        _git(
            root,
            ("apply", "--whitespace=nowarn"),
            supervisor,
            stdin_text=patch_payload,
        )
        observed = _verify_applied(root, candidates, bundle.changed_files)
    except Exception:
        _restore(root, backups)
        raise
    return KernelApplyReceipt(
        task_id=bundle.task_id,
        bundle_digest=bundle.digest,
        workspace=str(root),
        changed_files=bundle.changed_files,
        applied_file_hashes=observed,
    )


def _clean_git_root(
    workspace: Path, supervisor: SubprocessSupervisor | None
) -> Path:
    supplied = Path(workspace)
    if not supplied.is_absolute() or supplied.is_symlink():
        raise ContractError("Apply workspace must be an absolute directory", "invalid_apply_workspace")
    root = supplied.resolve(strict=True)
    if not root.is_dir():
        raise ContractError("Apply workspace is not a directory", "invalid_apply_workspace")
    top = _git(root, ("rev-parse", "--show-toplevel"), supervisor).strip()
    if Path(top).resolve(strict=True) != root:
        raise ContractError("Apply workspace must be the Git root", "invalid_apply_workspace")
    status_text = _git(
        root,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        supervisor,
    )
    if status_text:
        raise ContractError("Apply workspace is not clean", "dirty_apply_workspace")
    return root


def _bundle_contract(
    bundle: KernelBundle,
) -> tuple[Mapping[str, str], Mapping[str, str], str]:
    manifest = bundle.manifest
    baseline_value = manifest.get("baseline")
    baseline = baseline_value.get("file_hashes") if isinstance(baseline_value, Mapping) else None
    candidates = manifest.get("candidate_file_hashes")
    patches_value = manifest.get("patches")
    if not isinstance(baseline, Mapping) or not isinstance(candidates, Mapping):
        raise IntegrityError("Bundle source hashes are missing", "invalid_bundle_manifest")
    if set(candidates) != set(bundle.changed_files):
        raise IntegrityError("Candidate hashes differ from changed files", "invalid_bundle_manifest")
    hashes = (_hash_mapping(baseline), _hash_mapping(candidates))
    if not isinstance(patches_value, list) or not patches_value:
        raise IntegrityError("Bundle patches are missing", "invalid_bundle_manifest")
    payload = _validated_patch_payload(bundle, patches_value)
    return hashes[0], hashes[1], payload


def _validated_patch_payload(bundle: KernelBundle, entries: list[object]) -> str:
    parts: list[str] = []
    for item in entries:
        if not isinstance(item, Mapping):
            raise IntegrityError("Bundle patch entry is invalid", "invalid_bundle_manifest")
        relative = _safe_relative(str(item.get("path", "")))
        path = bundle.path.joinpath(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise IntegrityError("Bundle patch is unsafe", "unsafe_bundle_patch")
        content = path.read_bytes()
        expected = str(item.get("sha256", "")).removeprefix("sha256:")
        if sha256_bytes(content) != expected:
            raise IntegrityError("Bundle patch digest changed", "bundle_patch_digest_mismatch")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise IntegrityError("Bundle patch is not UTF-8", "invalid_bundle_patch") from error
        parts.append(text if text.endswith("\n") else f"{text}\n")
    return "".join(parts)


def _hash_mapping(value: Mapping[object, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_path, raw_digest in value.items():
        path = _safe_relative(str(raw_path))
        digest = str(raw_digest).removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise IntegrityError("Bundle source digest is invalid", "invalid_bundle_manifest")
        result[path] = digest
    return result


def _verify_patch_targets(
    root: Path,
    patch_payload: str,
    changed_files: tuple[str, ...],
    supervisor: SubprocessSupervisor | None,
) -> None:
    summary = _git(
        root,
        ("apply", "--summary"),
        supervisor,
        stdin_text=patch_payload,
    )
    if summary:
        raise IntegrityError("Patch changes file identity or mode", "unsupported_bundle_patch")
    output = _git(
        root,
        ("apply", "--numstat", "-z"),
        supervisor,
        stdin_text=patch_payload,
    )
    targets: list[str] = []
    for record in output.split("\0"):
        if not record:
            continue
        fields = record.split("\t", 2)
        if len(fields) != 3:
            raise IntegrityError("Git returned malformed patch targets", "invalid_bundle_patch")
        if fields[0] == "-" or fields[1] == "-":
            raise IntegrityError("Binary bundle patches are unsupported", "unsupported_bundle_patch")
        targets.append(_safe_relative(fields[2]))
    if len(targets) != len(set(targets)) or set(targets) != set(changed_files):
        raise IntegrityError("Patch targets differ from changed files", "bundle_patch_target_mismatch")


def _verify_baseline(
    root: Path, baseline: Mapping[str, str], changed_files: tuple[str, ...]
) -> None:
    for relative in changed_files:
        if relative not in baseline:
            raise IntegrityError("Changed file lacks a baseline hash", "invalid_bundle_manifest")
        path = _regular_source(root, relative)
        if sha256_file(path) != baseline[relative]:
            raise IntegrityError("Workspace differs from bundle baseline", "bundle_baseline_mismatch")


def _verify_applied(
    root: Path, candidates: Mapping[str, str], changed_files: tuple[str, ...]
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative in changed_files:
        digest = sha256_file(_regular_source(root, relative))
        if digest != candidates[relative]:
            raise IntegrityError("Applied source differs from bundle", "bundle_apply_hash_mismatch")
        observed[relative] = digest
    return observed


def _backups(root: Path, changed_files: tuple[str, ...]) -> dict[str, tuple[bytes, int]]:
    return {
        relative: (
            _regular_source(root, relative).read_bytes(),
            stat.S_IMODE(_regular_source(root, relative).stat().st_mode),
        )
        for relative in changed_files
    }


def _restore(root: Path, backups: Mapping[str, tuple[bytes, int]]) -> None:
    for relative, (content, mode) in backups.items():
        path = _regular_source(root, relative)
        descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(mode)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _regular_source(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(_safe_relative(relative)).parts)
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise IntegrityError("Workspace source is unsafe", "unsafe_apply_source")
    if path.resolve().parent != root and root not in path.resolve().parents:
        raise IntegrityError("Workspace source escapes root", "unsafe_apply_source")
    return path


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise IntegrityError("Bundle path is unsafe", "unsafe_bundle_path")
    return path.as_posix()


def _git(
    root: Path,
    arguments: tuple[str, ...],
    supervisor: SubprocessSupervisor | None,
    *,
    stdin_text: str | None = None,
) -> str:
    runner = supervisor or SubprocessSupervisor(max_output_bytes=4 * 1024 * 1024)
    environment = build_subprocess_environment(
        fixed={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    result = runner.run(
        ("git", "-C", str(root), *arguments),
        cwd=root,
        environment=environment,
        timeout_seconds=120,
        stdin_text=stdin_text,
    )
    if result.timed_out or result.exit_code != 0 or result.stdout_truncated or result.stderr_truncated:
        raise IntegrityError("Git bundle apply failed", "bundle_apply_failed")
    return result.stdout.strip()


__all__ = ["KernelApplyReceipt", "apply_verified_kernel_bundle"]
