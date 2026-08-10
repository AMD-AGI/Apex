"""Deterministic Docker image construction for formal Python/Triton delivery."""

from __future__ import annotations

import io
import json
import os
import re
import stat
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

from apex.core import (
    ContractError,
    IntegrityError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from apex.delivery import (
    BuildRecipeLock,
    BuildStepReceipt,
    BuiltArtifact,
    DerivedImageIdentity,
    LoadedArtifact,
    LoadedByteEngagementReceipt,
)
from apex.execution import (
    DOCKER_RUNTIME_ENVIRONMENT_KEYS,
    ProcessResult,
    SubprocessSupervisor,
    build_subprocess_environment,
)

from .source_image_sbom import write_source_sbom


_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_MARKER = "__APEX_FORMAL_SOURCE_V1__"
_OVERLAY_ROOT = "opt/apex/python"
_SITE_ROOT = "opt/apex"
_ORIGINAL_SITE = "/usr/local/lib/python3.12/dist-packages"


class CommandPort(Protocol):
    """Small injectable argv-only process boundary."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: int,
        stdin_text: str | None = None,
    ) -> ProcessResult: ...


@dataclass(frozen=True, slots=True)
class SourceInventoryEntry:
    repository_id: str
    relative_path: str
    sha256: str
    mode: int
    content: bytes


@dataclass(frozen=True, slots=True)
class SourceImageBuild:
    image: DerivedImageIdentity
    sbom_path: Path
    artifacts: tuple[BuiltArtifact, ...]
    step_receipts: tuple[BuildStepReceipt, ...]
    build_document: Mapping[str, Any]
    probe_document: Mapping[str, Any]


class DockerPythonSourceImageBuilder:
    """Bake exact source packages into a reproducible immutable image layer."""

    def __init__(
        self,
        *,
        parent_locator: str,
        parent_image_id: str,
        source_date_epoch: int,
        commands: CommandPort | None = None,
    ) -> None:
        if "@sha256:" not in parent_locator or not _IMAGE_ID.fullmatch(parent_image_id):
            raise ContractError("Formal parent image is not immutable", "invalid_image_identity")
        if source_date_epoch <= 0:
            raise ContractError("SOURCE_DATE_EPOCH is invalid", "invalid_build_recipe")
        self.parent_locator = parent_locator
        self.parent_image_id = parent_image_id
        self.source_date_epoch = source_date_epoch
        self.commands = commands or SubprocessSupervisor(max_output_bytes=32 * 1024 * 1024)

    def build(
        self,
        *,
        recipe: BuildRecipeLock,
        repository_roots: Mapping[str, Path],
        source_stack_sha256: str,
        output_dir: Path,
    ) -> SourceImageBuild:
        root = _new_directory(output_dir)
        self._validate_parent(recipe, root)
        steps = self._run_recipe_steps(recipe, repository_roots)
        inventory, changed = self._inventory(repository_roots)
        manifest = _source_manifest(
            self.commands,
            recipe,
            repository_roots,
            source_stack_sha256,
            inventory,
        )
        layer = root / "source-layer.tar"
        _write_source_layer(layer, inventory, manifest, self.source_date_epoch)
        dockerfile = root / "Dockerfile"
        _write_once(dockerfile, self._dockerfile(recipe, source_stack_sha256))
        os.utime(dockerfile, (self.source_date_epoch, self.source_date_epoch))
        image_id, build_result = self._build_image(root, dockerfile)
        sbom = write_source_sbom(
            root,
            recipe,
            repository_roots,
            source_stack_sha256,
            inventory,
            self.source_date_epoch,
        )
        image = DerivedImageIdentity(image_id, self.parent_image_id, image_id, sha256_file(sbom))
        artifacts = _built_artifacts(changed, source_stack_sha256)
        probe = self._probe_image(image_id, manifest, repository_roots, root)
        document = _build_document(
            recipe,
            image,
            self.parent_locator,
            layer,
            dockerfile,
            build_result,
            artifacts,
            steps,
        )
        return SourceImageBuild(image, sbom, artifacts, steps, document, probe)

    def engage(
        self,
        *,
        bundle_digest: str,
        image: DerivedImageIdentity,
        source_stack_sha256: str,
        artifacts: Sequence[BuiltArtifact],
        cwd: Path,
    ) -> LoadedByteEngagementReceipt:
        specifications = [_module_spec(item) for item in artifacts]
        payload = self._python_probe(
            image.image_digest,
            _ENGAGEMENT_PROBE,
            (json.dumps(specifications, sort_keys=True, separators=(",", ":")),),
            cwd=cwd,
        )
        observed = payload.get("artifacts")
        if not isinstance(observed, list) or len(observed) != len(artifacts):
            raise IntegrityError("Import probe is incomplete", "loaded_byte_engagement_failed")
        loaded = tuple(_loaded_artifact(expected, value) for expected, value in zip(artifacts, observed, strict=True))
        return LoadedByteEngagementReceipt(
            bundle_digest,
            image.image_digest,
            source_stack_sha256,
            True,
            loaded,
        )

    def _validate_parent(self, recipe: BuildRecipeLock, cwd: Path) -> None:
        if recipe.parent_image_digest != self.parent_image_id:
            raise IntegrityError("Recipe uses another parent image", "untrusted_build_recipe")
        result = self._docker(("docker", "image", "inspect", self.parent_locator), cwd, 60)
        try:
            observed = str(json.loads(result.stdout)[0]["Id"])
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise IntegrityError("Parent image inspection failed", "image_inspection_failed") from error
        if observed != self.parent_image_id:
            raise IntegrityError("Parent image tag drifted", "image_identity_mismatch")

    def _run_recipe_steps(
        self, recipe: BuildRecipeLock, roots: Mapping[str, Path]
    ) -> tuple[BuildStepReceipt, ...]:
        receipts = []
        for index, step in enumerate(recipe.steps):
            root = roots.get(step.repository_id)
            if root is None:
                raise IntegrityError("Recipe source root is missing", "invalid_build_recipe")
            cwd = root if step.cwd == "." else root.joinpath(*step.cwd.split("/"))
            result = self.commands.run(
                step.argv,
                cwd=cwd.resolve(strict=True),
                environment=_git_environment(dict(step.environment)),
                timeout_seconds=step.timeout_seconds,
            )
            receipt = _step_receipt(index, step.repository_id, step.cwd, step.argv, result)
            receipts.append(receipt)
            if not receipt.verified or result.stdout_truncated or result.stderr_truncated:
                raise IntegrityError("Fixed source validation failed", "source_build_failed")
        return tuple(receipts)

    def _inventory(
        self, roots: Mapping[str, Path]
    ) -> tuple[tuple[SourceInventoryEntry, ...], tuple[SourceInventoryEntry, ...]]:
        inventory: list[SourceInventoryEntry] = []
        changed: list[SourceInventoryEntry] = []
        for repository_id, root in sorted(roots.items()):
            tracked = _tracked_entries(self.commands, root, repository_id)
            inventory.extend(tracked)
            changed_paths = _changed_python_paths(self.commands, root, repository_id)
            by_path = {item.relative_path: item for item in tracked}
            if not changed_paths or any(path not in by_path for path in changed_paths):
                raise IntegrityError("Changed source inventory is incomplete", "source_build_failed")
            changed.extend(by_path[path] for path in changed_paths)
        return tuple(inventory), tuple(changed)

    def _dockerfile(self, recipe: BuildRecipeLock, source_stack: str) -> bytes:
        lines = (
            f"FROM {self.parent_locator}",
            f"ARG SOURCE_DATE_EPOCH={self.source_date_epoch}",
            "ADD source-layer.tar /",
            "ENV PYTHONPATH=/opt/apex:/opt/apex/python",
            f"ENV AITER_META_DIR={_ORIGINAL_SITE}/aiter_meta",
            f'LABEL apex.parent.image.id="{self.parent_image_id}"',
            f'LABEL apex.source.stack.sha256="{source_stack}"',
            f'LABEL apex.build.recipe.sha256="{recipe.computed_sha256}"',
        )
        return ("\n".join(lines) + "\n").encode("utf-8")

    def _build_image(self, root: Path, dockerfile: Path) -> tuple[str, ProcessResult]:
        iidfile = root / "derived-image.id"
        argv = _buildx_argv(root, dockerfile, iidfile, self.source_date_epoch)
        result = self._docker(argv, root, 1800)
        if not iidfile.is_file() or iidfile.is_symlink():
            raise IntegrityError("Buildx emitted no immutable image ID", "source_build_failed")
        image_id = iidfile.read_text(encoding="utf-8").strip()
        if not _IMAGE_ID.fullmatch(image_id):
            raise IntegrityError("Buildx image ID is invalid", "source_build_failed")
        inspect = self._docker(("docker", "image", "inspect", image_id), root, 60)
        if str(json.loads(inspect.stdout)[0].get("Id")) != image_id:
            raise IntegrityError("Derived image inspection drifted", "image_identity_mismatch")
        return image_id, result

    def _probe_image(
        self,
        image_id: str,
        manifest: Mapping[str, Any],
        roots: Mapping[str, Path],
        cwd: Path,
    ) -> Mapping[str, Any]:
        payload = self._python_probe(
            image_id,
            _IMAGE_PROBE,
            (
                sha256_json(manifest),
                json.dumps(sorted(roots), separators=(",", ":")),
            ),
            cwd=cwd,
        )
        if payload.get("manifest_sha256") != sha256_json(manifest):
            raise IntegrityError("Image source manifest differs", "source_build_receipt_mismatch")
        packages = payload.get("packages")
        if not isinstance(packages, Mapping) or set(packages) != set(roots):
            raise IntegrityError("Image package probe is incomplete", "source_build_receipt_mismatch")
        for name, value in packages.items():
            expected = f"/opt/apex/python/{name}/__init__.py"
            if not isinstance(value, Mapping) or value.get("module_file") != expected:
                raise IntegrityError(
                    "Image imported a package outside the baked source tree",
                    "source_build_receipt_mismatch",
                )
        return payload

    def _python_probe(
        self, image_id: str, script: str, args: tuple[str, ...], *, cwd: Path
    ) -> Mapping[str, Any]:
        result = self._docker(
            ("docker", "run", "--rm", "--network=none", "--entrypoint", "python3", image_id, "-c", script, *args),
            cwd,
            300,
        )
        marked = [line[len(_MARKER) :] for line in result.stdout.splitlines() if line.startswith(_MARKER)]
        if len(marked) != 1:
            raise IntegrityError("Image probe emitted no unique receipt", "loaded_byte_probe_failed")
        try:
            value = json.loads(marked[0])
        except json.JSONDecodeError as error:
            raise IntegrityError("Image probe receipt is invalid", "loaded_byte_probe_failed") from error
        if not isinstance(value, Mapping):
            raise IntegrityError("Image probe receipt is not an object", "loaded_byte_probe_failed")
        return value

    def _docker(self, argv: tuple[str, ...], cwd: Path, timeout: int) -> ProcessResult:
        result = self.commands.run(
            argv,
            cwd=cwd.resolve(strict=True),
            environment=build_subprocess_environment(inherit=DOCKER_RUNTIME_ENVIRONMENT_KEYS),
            timeout_seconds=timeout,
        )
        if result.exit_code != 0 or result.timed_out or result.stdout_truncated or result.stderr_truncated:
            raise IntegrityError("Formal Docker command failed", "container_command_failed")
        return result


def _new_directory(path: Path) -> Path:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise IntegrityError("Source image output must be a new absolute path", "immutable_delivery_artifact")
    path.mkdir(parents=True)
    return path.resolve()


def _git_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    return build_subprocess_environment(
        extra,
        fixed={
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
        },
    )


def _git(commands: CommandPort, root: Path, argv: tuple[str, ...]) -> str:
    result = commands.run(argv, cwd=root, environment=_git_environment(), timeout_seconds=120)
    if result.exit_code != 0 or result.timed_out or result.stdout_truncated or result.stderr_truncated:
        raise IntegrityError("Source inventory command failed", "source_build_failed")
    return result.stdout


def _tracked_entries(
    commands: CommandPort, root: Path, repository_id: str
) -> tuple[SourceInventoryEntry, ...]:
    raw = _git(commands, root, ("git", "ls-files", "-s", "-z", "--", f"{repository_id}/"))
    entries = []
    for record in raw.rstrip("\0").split("\0") if raw else ():
        metadata, relative = record.split("\t", 1)
        mode, _, stage = metadata.split(" ", 2)
        path = root.joinpath(*PurePosixPath(relative).parts)
        details = path.lstat()
        if stage != "0" or mode not in {"100644", "100755"} or not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise IntegrityError("Source inventory contains an unsafe entry", "source_build_failed")
        content = path.read_bytes()
        entries.append(SourceInventoryEntry(repository_id, relative, sha256_bytes(content), 0o755 if mode == "100755" else 0o644, content))
    if not entries:
        raise IntegrityError("Source package inventory is empty", "source_build_failed")
    return tuple(entries)


def _changed_python_paths(commands: CommandPort, root: Path, repository_id: str) -> tuple[str, ...]:
    raw = _git(commands, root, ("git", "diff", "--name-status", "--no-renames", "HEAD", "--", f"{repository_id}/"))
    paths = []
    for line in raw.splitlines():
        status_name, relative = line.split("\t", 1)
        path = PurePosixPath(relative)
        if status_name != "M" or path.suffix != ".py" or path.parts[0] != repository_id:
            raise IntegrityError("Formal profile supports modified Python/Triton files only", "unsupported_delivery")
        paths.append(relative)
    return tuple(sorted(paths))


def _source_manifest(commands, recipe, roots, source_stack, inventory) -> dict[str, Any]:
    repositories = []
    for name, root in sorted(roots.items()):
        commit = _git(commands, root, ("git", "rev-parse", "HEAD")).strip()
        tree = _git(commands, root, ("git", "rev-parse", "HEAD^{tree}")).strip()
        selected = [item for item in inventory if item.repository_id == name]
        repositories.append(
            {
                "repository_id": name,
                "base_commit": commit,
                "base_tree": tree,
                "source_content_sha256": sha256_json(
                    [(item.relative_path, item.sha256, item.mode) for item in selected]
                ),
            }
        )
    return {
        "schema": "apex.python-source-image/v1",
        "source_stack_sha256": source_stack,
        "recipe_sha256": recipe.computed_sha256,
        "repositories": repositories,
        "files": [
            {"path": item.relative_path, "sha256": item.sha256, "mode": item.mode}
            for item in inventory
        ],
    }


def _write_source_layer(path, inventory, manifest, epoch) -> None:
    site = _sitecustomize(tuple(sorted({item.repository_id for item in inventory})))
    site_root = PurePosixPath(_SITE_ROOT)
    overlay_root = PurePosixPath(_OVERLAY_ROOT)
    files = [
        (site_root / "sitecustomize.py", site, 0o644),
        (site_root / "source-manifest.json", canonical_json_bytes(manifest) + b"\n", 0o644),
    ]
    files.extend(
        (overlay_root / item.relative_path, item.content, item.mode)
        for item in inventory
    )
    directories = {
        PurePosixPath(*path.parts[:index])
        for path, _, _ in files
        for index in range(1, len(path.parts))
    }
    with tarfile.open(path, "w", format=tarfile.GNU_FORMAT) as archive:
        for directory in sorted(directories, key=lambda item: item.as_posix()):
            info = _tar_info(f"{directory.as_posix()}/", 0o755, epoch, 0)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        for target, content, mode in sorted(files, key=lambda item: item[0].as_posix()):
            archive.addfile(_tar_info(target.as_posix(), mode, epoch, len(content)), io.BytesIO(content))
    os.utime(path, (epoch, epoch))


def _tar_info(name: str, mode: int, epoch: int, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.mode, info.mtime, info.size = mode, epoch, size
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def _sitecustomize(repositories: tuple[str, ...]) -> bytes:
    roots = repr(repositories)
    source = f'''import importlib.abc, importlib.machinery, importlib.util, pathlib, sys
_OVERLAY = pathlib.Path("/opt/apex/python")
_ORIGINAL = pathlib.Path("{_ORIGINAL_SITE}")
_ROOTS = frozenset({roots})
class _ApexSourceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        parts = fullname.split(".")
        if parts[0] not in _ROOTS:
            return None
        base = _OVERLAY.joinpath(*parts)
        package = base / "__init__.py"
        if package.is_file():
            loader = importlib.machinery.SourceFileLoader(fullname, str(package))
            locations = [str(base), str(_ORIGINAL.joinpath(*parts))]
            return importlib.util.spec_from_file_location(fullname, package, loader=loader, submodule_search_locations=locations)
        module = base.with_suffix(".py")
        if module.is_file():
            loader = importlib.machinery.SourceFileLoader(fullname, str(module))
            return importlib.util.spec_from_file_location(fullname, module, loader=loader)
        return None
sys.meta_path.insert(0, _ApexSourceFinder())
'''
    return source.encode("utf-8")


def _built_artifacts(changed, source_stack) -> tuple[BuiltArtifact, ...]:
    return tuple(
        BuiltArtifact(
            item.repository_id,
            f"/{PurePosixPath(_OVERLAY_ROOT) / item.relative_path}",
            item.sha256,
            None,
            source_stack,
        )
        for item in changed
    )


def _step_receipt(index, repository_id, cwd, argv, result) -> BuildStepReceipt:
    return BuildStepReceipt(index, repository_id, cwd, sha256_json(list(argv)), result.exit_code, result.timed_out, sha256_bytes(result.stdout.encode()), sha256_bytes(result.stderr.encode()))


def _buildx_argv(root, dockerfile, iidfile, epoch) -> tuple[str, ...]:
    return (
        "docker", "buildx", "build", "--no-cache", "--pull=false", "--network=none",
        "--provenance=false", "--sbom=false", "--build-arg", f"SOURCE_DATE_EPOCH={epoch}",
        "--output", "type=docker,rewrite-timestamp=true", "--iidfile", str(iidfile),
        "--file", str(dockerfile), str(root),
    )


def _build_document(
    recipe, image, parent_locator, layer, dockerfile, result, artifacts, steps
) -> dict[str, Any]:
    return {
        "schema": "apex.e2e-python-source-build/v1",
        "recipe_sha256": recipe.computed_sha256,
        "parent_locator": parent_locator,
        "image": image.to_dict(),
        "source_layer_sha256": sha256_file(layer),
        "dockerfile_sha256": sha256_file(dockerfile),
        "build_argv_sha256": sha256_json(list(result.argv)),
        "stdout_sha256": sha256_bytes(result.stdout.encode()),
        "stderr_sha256": sha256_bytes(result.stderr.encode()),
        "artifacts": [item.to_dict() for item in artifacts],
        "fixed_recipe_step_receipts": [item.to_dict() for item in steps],
    }


def _module_spec(artifact: BuiltArtifact) -> dict[str, str]:
    relative = PurePosixPath(artifact.runtime_path).relative_to(
        PurePosixPath("/") / _OVERLAY_ROOT
    )
    parts = relative.with_suffix("").parts
    module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
    return {"component": artifact.component, "module": module, "path": artifact.runtime_path, "sha256": artifact.sha256}


def _loaded_artifact(expected: BuiltArtifact, value: object) -> LoadedArtifact:
    if not isinstance(value, Mapping):
        raise IntegrityError("Import receipt is invalid", "loaded_byte_engagement_failed")
    specification = _module_spec(expected)
    return LoadedArtifact(
        expected.component,
        expected.runtime_path,
        expected.sha256,
        str(value.get("sha256", "")),
        None,
        None,
        "python_import",
        str(value.get("module", "")),
        value.get("loaded") is True
        and value.get("module") == specification["module"]
        and value.get("path") == expected.runtime_path,
    )


def _write_once(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise IntegrityError("Immutable source-image artifact exists", "immutable_delivery_artifact")
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


_IMAGE_PROBE = f'''import hashlib, importlib, importlib.metadata, json, pathlib, sys
manifest_path = pathlib.Path("/opt/apex/source-manifest.json")
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
expected_manifest, roots_json = sys.argv[1], sys.argv[2]
canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
if hashlib.sha256(canonical).hexdigest() != expected_manifest:
    raise SystemExit("manifest_mismatch")
for item in manifest["files"]:
    path = pathlib.Path("/opt/apex/python") / item["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise SystemExit("source_byte_mismatch")
packages = {{}}
for root in json.loads(roots_json):
    module = importlib.import_module(root)
    packages[root] = {{"module_file": str(pathlib.Path(module.__file__).resolve()), "version": importlib.metadata.version("amd-aiter" if root == "aiter" else root)}}
print("{_MARKER}" + json.dumps({{"manifest_sha256": expected_manifest, "packages": packages}}, sort_keys=True, separators=(",", ":")))
'''


_ENGAGEMENT_PROBE = f'''import hashlib, importlib, json, pathlib, sys
results = []
for item in json.loads(sys.argv[1]):
    module = importlib.import_module(item["module"])
    path = pathlib.Path(module.__file__).resolve()
    expected = pathlib.Path(item["path"]).resolve()
    results.append({{"module": item["module"], "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "loaded": path == expected}})
print("{_MARKER}" + json.dumps({{"artifacts": results}}, sort_keys=True, separators=(",", ":")))
'''


__all__ = [
    "CommandPort",
    "DockerPythonSourceImageBuilder",
    "SourceImageBuild",
    "SourceInventoryEntry",
]
