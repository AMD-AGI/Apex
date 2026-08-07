"""Capture and independently replay exact repository source patches."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from apex.core import ContractError, IntegrityError, sha256_bytes, sha256_file
from apex.execution import SubprocessSupervisor
from apex.runtime import canonical_repository

from .e2e_models import SourceFileChange, SourceRepositoryLock, safe_bundle_path


@dataclass(frozen=True, slots=True)
class CapturedRepositoryPatch:
    """Patch bytes and their exact source lock."""

    lock: SourceRepositoryLock
    content: bytes


@dataclass(frozen=True, slots=True)
class _RepositoryIdentity:
    base: Path
    candidate: Path
    base_commit: str
    base_tree: str
    base_url: str


@dataclass(frozen=True, slots=True)
class RepositoryApplyReceipt:
    """Independent apply/reverse/reapply evidence for one clean checkout."""

    repository_id: str
    base_commit: str
    base_tree: str
    patched_tree: str
    patch_sha256: str
    before_blobs_verified: bool
    after_blobs_verified: bool
    apply_check_passed: bool
    reverse_check_passed: bool
    reverse_restored_clean_base: bool
    reapplied_for_build: bool

    @property
    def verified(self) -> bool:
        return all(
            (
                self.before_blobs_verified,
                self.after_blobs_verified,
                self.apply_check_passed,
                self.reverse_check_passed,
                self.reverse_restored_clean_base,
                self.reapplied_for_build,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "verified": self.verified}


def _environment(**extra: str) -> dict[str, str]:
    value = os.environ.copy()
    value.pop("PYTHONPATH", None)
    value.update(extra)
    return value


class _Git:
    def __init__(self, supervisor: SubprocessSupervisor | None = None) -> None:
        self.supervisor = supervisor or SubprocessSupervisor(max_output_bytes=64 * 1024 * 1024)

    def run(
        self,
        root: Path,
        *args: str,
        environment: Mapping[str, str] | None = None,
        reason: str = "git_command_failed",
        timeout: int = 120,
    ) -> str:
        result = self.supervisor.run(
            ("git", *args),
            cwd=root,
            environment=environment or _environment(),
            timeout_seconds=timeout,
        )
        if result.exit_code != 0 or result.timed_out or result.stdout_truncated or result.stderr_truncated:
            detail = (result.stderr or result.stdout).strip()[-2000:]
            raise IntegrityError(
                f"Git evidence command failed: {' '.join(args)}" + (f"\n{detail}" if detail else ""),
                reason,
            )
        return result.stdout


def _ensure_git_root(root: Path, git: _Git) -> Path:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir() or git.run(resolved, "rev-parse", "--is-inside-work-tree").strip() != "true":
        raise ContractError("Source path is not a Git worktree", "invalid_source_repository")
    top = Path(git.run(resolved, "rev-parse", "--show-toplevel").strip()).resolve()
    if top != resolved:
        raise ContractError("Source path must be the Git worktree root", "invalid_source_repository")
    return resolved


def _tree_entry(git: _Git, root: Path, revision: str, path: str, env: Mapping[str, str] | None = None) -> tuple[str, str] | None:
    if revision == ":":
        raw = git.run(root, "ls-files", "-s", "-z", "--", path, environment=env)
        if not raw:
            return None
        record = raw.rstrip("\x00")
        prefix, recorded_path = record.split("\t", 1)
        mode, blob, stage = prefix.split(" ", 2)
        if recorded_path != path or stage != "0":
            raise IntegrityError("Index contains an ambiguous source entry", "invalid_source_index")
        return mode, blob
    raw = git.run(root, "ls-tree", "-z", revision, "--", path)
    if not raw:
        return None
    record = raw.rstrip("\x00")
    prefix, recorded_path = record.split("\t", 1)
    mode, object_type, blob = prefix.split(" ", 2)
    if recorded_path != path or object_type != "blob":
        reason = "submodule_boundary" if object_type == "commit" else "invalid_source_tree"
        raise IntegrityError("Source entry is not a regular blob", reason)
    return mode, blob


def _content_sha256(root: Path, path: str) -> str:
    target = root.joinpath(*safe_bundle_path(path).split("/"))
    if target.is_symlink() or not target.is_file():
        raise IntegrityError(f"Source is not a regular file: {path}", "unsupported_source_mode")
    content = target.read_bytes()
    try:
        content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise IntegrityError(f"Binary source is unsupported: {path}", "binary_source_unsupported") from error
    if b"\x00" in content:
        raise IntegrityError(f"Binary source is unsupported: {path}", "binary_source_unsupported")
    return sha256_bytes(content)


def _name_status(raw: str) -> list[tuple[str, str, str | None]]:
    tokens = raw.split("\x00")
    if tokens and tokens[-1] == "":
        tokens.pop()
    output: list[tuple[str, str, str | None]] = []
    index = 0
    while index < len(tokens):
        status = tokens[index]
        index += 1
        if not status:
            raise IntegrityError("Empty Git name-status record", "invalid_source_diff")
        code = status[0]
        if code in {"R", "C"}:
            if index + 1 >= len(tokens):
                raise IntegrityError("Truncated Git rename record", "invalid_source_diff")
            old, new = tokens[index], tokens[index + 1]
            index += 2
            if code == "C":
                # A copy is semantically an addition; preserving it as a rename
                # would make reverse verification lie about the original path.
                output.append(("A", new, None))
            else:
                output.append(("R", old, new))
        else:
            if index >= len(tokens):
                raise IntegrityError("Truncated Git name-status record", "invalid_source_diff")
            output.append((code, tokens[index], None))
            index += 1
    return output


def capture_repository_patch(
    *,
    repository_id: str,
    base_root: Path,
    candidate_root: Path,
    patch_path: str,
    order: int,
    dependencies: Sequence[str],
    editable_allowlist: Sequence[str],
    build_recipe_sha256: str,
    accepted_candidate_id: str,
    anchor_generation: int,
    license_id: str,
    runtime_component: str,
    supervisor: SubprocessSupervisor | None = None,
) -> CapturedRepositoryPatch:
    """Freeze a candidate worktree without mutating its real Git index."""

    git = _Git(supervisor)
    identity = _capture_identity(git, base_root, candidate_root)
    patched_tree, changes, patch = _capture_index_patch(git, identity)
    lock = SourceRepositoryLock(
        repository_id=repository_id,
        url=identity.base_url,
        base_commit=identity.base_commit,
        base_tree=identity.base_tree,
        patched_tree=patched_tree,
        patch_path=safe_bundle_path(patch_path, field="patch_path"),
        patch_sha256=sha256_bytes(patch),
        order=order,
        dependencies=tuple(dependencies),
        editable_allowlist=tuple(editable_allowlist),
        changes=changes,
        build_recipe_sha256=build_recipe_sha256,
        accepted_candidate_id=accepted_candidate_id,
        anchor_generation=anchor_generation,
        clean_base=True,
        license_id=license_id,
        runtime_component=runtime_component,
    )
    return CapturedRepositoryPatch(lock, patch)


def _capture_identity(git: _Git, base_root: Path, candidate_root: Path) -> _RepositoryIdentity:
    base = _ensure_git_root(base_root, git)
    candidate = _ensure_git_root(candidate_root, git)
    base_commit = git.run(base, "rev-parse", "HEAD").strip()
    base_tree = git.run(base, "rev-parse", "HEAD^{tree}").strip()
    if git.run(base, "status", "--porcelain=v1", "--untracked-files=all"):
        raise IntegrityError("Bundle base repository must be clean", "dirty_source_base")
    if git.run(candidate, "rev-parse", "HEAD").strip() != base_commit:
        raise IntegrityError(
            "Candidate and source base commits differ",
            "repository_commit_mismatch",
        )
    if git.run(candidate, "rev-parse", "HEAD^{tree}").strip() != base_tree:
        raise IntegrityError(
            "Candidate and source base trees differ",
            "repository_tree_mismatch",
        )
    base_url = git.run(base, "remote", "get-url", "origin").strip()
    candidate_url = git.run(candidate, "remote", "get-url", "origin").strip()
    if canonical_repository(base_url) != canonical_repository(candidate_url):
        raise IntegrityError(
            "Candidate repository origin differs from base",
            "repository_origin_mismatch",
        )
    return _RepositoryIdentity(base, candidate, base_commit, base_tree, base_url)


def _capture_index_patch(
    git: _Git,
    identity: _RepositoryIdentity,
) -> tuple[str, tuple[SourceFileChange, ...], bytes]:
    descriptor, index_name = tempfile.mkstemp(prefix="apex-git-index-")
    os.close(descriptor)
    Path(index_name).unlink()
    env = _environment(GIT_INDEX_FILE=index_name)
    try:
        git.run(identity.candidate, "read-tree", "HEAD", environment=env)
        git.run(identity.candidate, "add", "-A", "--", ".", environment=env)
        patched_tree = git.run(identity.candidate, "write-tree", environment=env).strip()
        status = git.run(
            identity.candidate,
            "diff",
            "--cached",
            "--name-status",
            "-z",
            "--find-renames=50%",
            "HEAD",
            "--",
            environment=env,
        )
        records = _name_status(status)
        if not records:
            raise ContractError("Candidate contains no source change", "no_changed_files")
        changes = _collect_source_changes(git, identity, records, env)
        patch_text = git.run(
            identity.candidate,
            "diff",
            "--cached",
            "--binary",
            "--full-index",
            "--find-renames=50%",
            "--no-ext-diff",
            "--no-textconv",
            "HEAD",
            "--",
            environment=env,
        )
        patch = patch_text.encode("utf-8")
        if not patch or b"\x00" in patch or b"\xef\xbf\xbd" in patch:
            raise IntegrityError("Patch is empty or not UTF-8 source", "invalid_source_patch")
    finally:
        Path(index_name).unlink(missing_ok=True)
    return patched_tree, changes, patch


def _collect_source_changes(
    git: _Git,
    identity: _RepositoryIdentity,
    records: Sequence[tuple[str, str, str | None]],
    env: Mapping[str, str],
) -> tuple[SourceFileChange, ...]:
    changes: list[SourceFileChange] = []
    kinds = {"A": "added", "M": "modified", "D": "deleted", "R": "renamed"}
    for code, first, second in records:
        old_path = first if code in {"M", "D", "R"} else None
        new_path = second if code == "R" else (first if code in {"M", "A"} else None)
        old_entry = (
            _tree_entry(git, identity.base, "HEAD", old_path) if old_path else None
        )
        new_entry = (
            _tree_entry(git, identity.candidate, ":", new_path, env)
            if new_path
            else None
        )
        if old_path and old_entry is None or new_path and new_entry is None:
            raise IntegrityError(
                "Git diff entry lacks a blob identity",
                "invalid_source_diff",
            )
        if code not in kinds:
            raise IntegrityError(
                f"Unsupported Git source status: {code}",
                "unsupported_source_change",
            )
        changes.append(
            SourceFileChange(
                kind=kinds[code],
                old_path=old_path,
                new_path=new_path,
                before_blob=old_entry[1] if old_entry else None,
                after_blob=new_entry[1] if new_entry else None,
                before_sha256=(
                    _content_sha256(identity.base, old_path) if old_path else None
                ),
                after_sha256=(
                    _content_sha256(identity.candidate, new_path) if new_path else None
                ),
                old_mode=old_entry[0] if old_entry else None,
                new_mode=new_entry[0] if new_entry else None,
            )
        )
    return tuple(changes)


class CleanPatchMaterializer:
    """Clone exact locks, prove apply/reverse, then leave patched build roots."""

    def __init__(self, supervisor: SubprocessSupervisor | None = None) -> None:
        self._git = _Git(supervisor)

    def materialize(
        self,
        *,
        bundle_root: Path,
        locks: Sequence[SourceRepositoryLock],
        destination: Path,
        source_overrides: Mapping[str, Path] | None = None,
    ) -> tuple[dict[str, Path], tuple[RepositoryApplyReceipt, ...]]:
        _prepare_replay_destination(destination)
        roots: dict[str, Path] = {}
        receipts: list[RepositoryApplyReceipt] = []
        overrides = dict(source_overrides or {})
        ordered = validate_lock_order(locks)
        try:
            for lock in ordered:
                source = overrides.get(lock.repository_id)
                clone_source = str(source.resolve(strict=True)) if source else lock.url
                if source:
                    self._validate_override(source, lock)
                target = destination / lock.repository_id
                self._git.run(
                    destination,
                    "clone",
                    "--no-checkout",
                    "--no-hardlinks",
                    clone_source,
                    str(target),
                    timeout=600,
                    reason="source_materialization_failed",
                )
                self._git.run(target, "checkout", "--detach", lock.base_commit)
                self._git.run(target, "remote", "set-url", "origin", lock.url)
                self._verify_base(target, lock)
                patch = (bundle_root / lock.patch_path).resolve(strict=True)
                if bundle_root.resolve() not in patch.parents or patch.is_symlink() or not patch.is_file():
                    raise IntegrityError("Patch path escapes bundle", "unsafe_bundle_path")
                if sha256_file(patch) != lock.patch_sha256:
                    raise IntegrityError("Patch digest mismatch", "bundle_patch_digest_mismatch")
                self._verify_changes(target, lock, before=True)
                self._git.run(target, "apply", "--check", "--index", "--whitespace=nowarn", str(patch), reason="patch_apply_check_failed")
                self._git.run(target, "apply", "--index", "--whitespace=nowarn", str(patch), reason="patch_apply_failed")
                self._verify_changes(target, lock, before=False)
                patched_tree = self._git.run(target, "write-tree").strip()
                if patched_tree != lock.patched_tree:
                    raise IntegrityError("Applied tree differs from source lock", "repository_tree_mismatch")
                self._git.run(target, "apply", "--check", "--reverse", "--index", str(patch), reason="patch_reverse_check_failed")
                self._git.run(target, "apply", "--reverse", "--index", str(patch), reason="patch_reverse_failed")
                restored = (
                    self._git.run(target, "status", "--porcelain=v1", "--untracked-files=all") == ""
                    and self._git.run(target, "write-tree").strip() == lock.base_tree
                    and self._git.run(target, "rev-parse", "HEAD").strip() == lock.base_commit
                )
                if not restored:
                    raise IntegrityError("Patch reverse did not restore the exact clean base", "patch_reverse_mismatch")
                self._git.run(target, "apply", "--check", "--index", "--whitespace=nowarn", str(patch), reason="patch_reapply_check_failed")
                self._git.run(target, "apply", "--index", "--whitespace=nowarn", str(patch), reason="patch_reapply_failed")
                self._verify_changes(target, lock, before=False)
                if self._git.run(target, "write-tree").strip() != lock.patched_tree:
                    raise IntegrityError("Reapplied tree differs from source lock", "repository_tree_mismatch")
                roots[lock.repository_id] = target
                receipts.append(
                    RepositoryApplyReceipt(
                        repository_id=lock.repository_id,
                        base_commit=lock.base_commit,
                        base_tree=lock.base_tree,
                        patched_tree=patched_tree,
                        patch_sha256=lock.patch_sha256,
                        before_blobs_verified=True,
                        after_blobs_verified=True,
                        apply_check_passed=True,
                        reverse_check_passed=True,
                        reverse_restored_clean_base=True,
                        reapplied_for_build=True,
                    )
                )
            return roots, tuple(receipts)
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise

    def _validate_override(self, source: Path, lock: SourceRepositoryLock) -> None:
        root = _ensure_git_root(source, self._git)
        if canonical_repository(self._git.run(root, "remote", "get-url", "origin").strip()) != canonical_repository(lock.url):
            raise IntegrityError("Source override origin differs from lock", "repository_origin_mismatch")
        self._verify_base(root, lock)

    def _verify_base(self, root: Path, lock: SourceRepositoryLock) -> None:
        if self._git.run(root, "rev-parse", "HEAD").strip() != lock.base_commit:
            raise IntegrityError("Clean checkout commit differs from source lock", "repository_commit_mismatch")
        if self._git.run(root, "rev-parse", "HEAD^{tree}").strip() != lock.base_tree:
            raise IntegrityError("Clean checkout tree differs from source lock", "repository_tree_mismatch")
        if self._git.run(root, "status", "--porcelain=v1", "--untracked-files=all"):
            raise IntegrityError("Source checkout is dirty", "dirty_source_base")

    def _verify_changes(self, root: Path, lock: SourceRepositoryLock, *, before: bool) -> None:
        for change in lock.changes:
            path = change.old_path if before else change.new_path
            expected_blob = change.before_blob if before else change.after_blob
            expected_sha = change.before_sha256 if before else change.after_sha256
            expected_mode = change.old_mode if before else change.new_mode
            if path is None:
                absent_path = change.new_path if before else change.old_path
                if absent_path and root.joinpath(*absent_path.split("/")).exists():
                    raise IntegrityError("Added/deleted path state is incorrect", "source_blob_mismatch")
                continue
            entry = _tree_entry(self._git, root, ":", path)
            if entry is None:
                # Before application, HEAD is the relevant source; after
                # application, changes are staged in the index.
                entry = _tree_entry(self._git, root, "HEAD", path) if before else None
            if entry != (expected_mode, expected_blob):
                raise IntegrityError(f"Git blob/mode mismatch for {path}", "source_blob_mismatch")
            if _content_sha256(root, path) != expected_sha:
                raise IntegrityError(f"Source byte mismatch for {path}", "source_byte_mismatch")

        if not before:
            raw = self._git.run(
                root,
                "diff",
                "--cached",
                "--name-status",
                "-z",
                "--find-renames=50%",
                "HEAD",
                "--",
            )
            actual = {
                (
                    {"A": "added", "M": "modified", "D": "deleted", "R": "renamed"}.get(code, code),
                    first,
                    second,
                )
                for code, first, second in _name_status(raw)
            }
            expected = {
                (
                    change.kind,
                    change.old_path if change.kind != "added" else change.new_path,
                    change.new_path if change.kind == "renamed" else None,
                )
                for change in lock.changes
            }
            if actual != expected:
                raise IntegrityError(
                    "Applied patch changed files not declared by its source lock",
                    "source_change_set_mismatch",
                )


def _prepare_replay_destination(destination: Path) -> None:
    if destination.exists():
        raise ContractError(
            "Clean replay destination already exists",
            "replay_destination_exists",
        )
    destination.mkdir(parents=True)


def validate_lock_order(locks: Sequence[SourceRepositoryLock]) -> tuple[SourceRepositoryLock, ...]:
    if not locks:
        raise ContractError("At least one repository patch is required", "config_only_candidate")
    ordered = tuple(sorted(locks, key=lambda item: item.order))
    if tuple(item.order for item in ordered) != tuple(range(len(ordered))):
        raise ContractError("Repository patch order must be contiguous from zero", "invalid_patch_order")
    identities = [item.repository_id for item in ordered]
    if len(set(identities)) != len(identities):
        raise ContractError("Repository identities must be unique", "invalid_source_lock")
    completed: set[str] = set()
    known = set(identities)
    for lock in ordered:
        if any(dependency not in known for dependency in lock.dependencies):
            raise ContractError("Repository dependency is missing", "invalid_patch_order")
        if any(dependency not in completed for dependency in lock.dependencies):
            raise ContractError("Repository dependency appears after its consumer", "invalid_patch_order")
        completed.add(lock.repository_id)
    return ordered


__all__ = [
    "CapturedRepositoryPatch",
    "CleanPatchMaterializer",
    "RepositoryApplyReceipt",
    "capture_repository_patch",
    "validate_lock_order",
]
