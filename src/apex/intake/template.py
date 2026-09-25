"""Attributed, fail-closed kernel-template manifests."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from apex.core import ContractError, sha256_file, sha256_json, validate_identifier


_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_IMMUTABLE_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")
_STATUSES = {"pending", "reviewed"}
_LANGUAGES = {"triton", "hip"}


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field} must be an object", "invalid_template_manifest")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ContractError(
            f"{field} fields do not match the template schema",
            "invalid_template_manifest",
            {"field": field, "missing": sorted(expected - observed), "extra": sorted(observed - expected)},
        )


def _strings(value: Any, field: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractError(f"{field} must be a list", "invalid_template_manifest")
    result = tuple(str(item) for item in value)
    if (not allow_empty and not result) or any(not item for item in result):
        raise ContractError(f"{field} is incomplete", "invalid_template_manifest")
    return result


def _relative(value: str, field: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ContractError(f"{field} is unsafe", "unsafe_template_path")
    if any(part in {"", "."} for part in path.parts):
        raise ContractError(f"{field} is unsafe", "unsafe_template_path")
    return path.as_posix()


def _digest(value: Any, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    result = str(value)
    if not _HEX_64.fullmatch(result):
        raise ContractError(f"{field} is not a SHA-256 digest", "invalid_template_manifest")
    return result


@dataclass(frozen=True, slots=True)
class TemplateFile:
    path: str
    sha256: str
    size: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TemplateFile":
        _exact_keys(value, {"path", "sha256", "size"}, "snapshot file")
        size = int(value["size"])
        if size < 0:
            raise ContractError("Template file size is invalid", "invalid_template_manifest")
        return cls(
            _relative(str(value["path"]), "snapshot file path"),
            str(_digest(value["sha256"], "snapshot file digest")),
            size,
        )


@dataclass(frozen=True, slots=True)
class TemplateRuntime:
    hardware: str
    gpu_arch: str
    mutable_reference: str
    immutable_locator: str | None
    image_id: str | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TemplateRuntime":
        _exact_keys(
            value,
            {"hardware", "gpu_arch", "mutable_reference", "immutable_locator", "image_id"},
            "runtime",
        )
        locator = value["immutable_locator"]
        image_id = value["image_id"]
        if locator is not None and not _IMMUTABLE_IMAGE.fullmatch(str(locator)):
            raise ContractError("Template image locator is mutable", "invalid_template_image")
        if image_id is not None and not _IMAGE_ID.fullmatch(str(image_id)):
            raise ContractError("Template image ID is invalid", "invalid_template_image")
        return cls(
            hardware=str(value["hardware"]),
            gpu_arch=str(value["gpu_arch"]),
            mutable_reference=str(value["mutable_reference"]),
            immutable_locator=str(locator) if locator is not None else None,
            image_id=str(image_id) if image_id is not None else None,
        )


@dataclass(frozen=True, slots=True)
class TemplateSource:
    container_root: str
    workspace_subdir: str
    language: str
    editable_files: tuple[str, ...]
    target_functions: tuple[str, ...]
    baseline_tree_sha256: str | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TemplateSource":
        _exact_keys(
            value,
            {
                "container_root", "workspace_subdir", "language", "editable_files",
                "target_functions", "baseline_tree_sha256",
            },
            "source",
        )
        root = PurePosixPath(str(value["container_root"]))
        language = str(value["language"])
        if not root.is_absolute() or ".." in root.parts:
            raise ContractError("Container source root is unsafe", "unsafe_template_path")
        if language not in _LANGUAGES:
            raise ContractError("Template language is unsupported", "invalid_template_manifest")
        return cls(
            container_root=root.as_posix(),
            workspace_subdir=_relative(str(value["workspace_subdir"]), "workspace subdirectory"),
            language=language,
            editable_files=tuple(
                _relative(item, "editable file")
                for item in _strings(value["editable_files"], "editable_files")
            ),
            target_functions=_strings(value["target_functions"], "target_functions"),
            baseline_tree_sha256=_digest(
                value["baseline_tree_sha256"], "baseline tree digest", optional=True
            ),
        )


@dataclass(frozen=True, slots=True)
class TemplateEvaluator:
    adapter_id: str
    measurement_method_sha256: str
    harness_files: tuple[str, ...]
    compile_argv: tuple[str, ...]
    correctness_argv: tuple[str, ...]
    performance_argv: tuple[str, ...]
    measurement_argv: tuple[str, ...]
    recipe_id: str
    recipe_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TemplateEvaluator":
        expected = {
            "adapter_id", "measurement_method_sha256", "harness_files", "compile_argv",
            "correctness_argv", "performance_argv", "measurement_argv", "recipe_id",
            "recipe_sha256",
        }
        _exact_keys(value, expected, "evaluator")
        adapter_id = str(value["adapter_id"])
        recipe_id = str(value["recipe_id"])
        validate_identifier(adapter_id, field_name="template evaluator adapter ID")
        validate_identifier(recipe_id, field_name="template recipe ID")
        return cls(
            adapter_id=adapter_id,
            measurement_method_sha256=str(
                _digest(value["measurement_method_sha256"], "measurement method digest")
            ),
            harness_files=tuple(
                _relative(item, "harness file")
                for item in _strings(value["harness_files"], "harness_files")
            ),
            compile_argv=_strings(value["compile_argv"], "compile_argv"),
            correctness_argv=_strings(value["correctness_argv"], "correctness_argv"),
            performance_argv=_strings(value["performance_argv"], "performance_argv"),
            measurement_argv=_strings(value["measurement_argv"], "measurement_argv"),
            recipe_id=recipe_id,
            recipe_sha256=str(_digest(value["recipe_sha256"], "recipe digest")),
        )


@dataclass(frozen=True, slots=True)
class ReviewedKernelTemplate:
    root: Path
    template_id: str
    showcase_id: str
    status: str
    manifest_sha256: str
    upstream: Mapping[str, Any]
    runtime: TemplateRuntime
    source: TemplateSource
    evaluator: TemplateEvaluator | None
    blockers: tuple[str, ...]
    snapshot_files: tuple[TemplateFile, ...]

    @property
    def materializable(self) -> bool:
        return self.status == "reviewed" and not self.blockers

    def require_materializable(self) -> None:
        if not self.materializable:
            raise ContractError(
                "Kernel template is attributed but not ready for a formal campaign",
                "template_not_materializable",
                {"template_id": self.template_id, "blockers": list(self.blockers)},
            )


def load_kernel_template(root: Path) -> ReviewedKernelTemplate:
    """Load one exact manifest and verify every attributed snapshot byte."""

    selected = root.expanduser()
    if selected.is_symlink():
        raise ContractError("Kernel template cannot be a symlink", "unsafe_template_path")
    try:
        resolved = selected.resolve(strict=True)
    except OSError as error:
        raise ContractError("Kernel template does not exist", "template_missing") from error
    if not resolved.is_dir():
        raise ContractError("Kernel template must be a directory", "invalid_template_manifest")
    manifest_path = resolved / "template" / "template_manifest.json"
    value = _load_manifest_document(manifest_path)
    expected = {
        "schema", "template_id", "showcase_id", "status", "manifest_sha256", "upstream",
        "runtime", "source", "evaluator", "blockers", "snapshot_files",
    }
    _exact_keys(value, expected, "template manifest")
    if value["schema"] != "apex.kernel-template/v1":
        raise ContractError("Unsupported kernel template schema", "invalid_template_manifest")
    observed_digest = str(value["manifest_sha256"])
    unsigned = dict(value)
    unsigned.pop("manifest_sha256")
    if not _HEX_64.fullmatch(observed_digest) or sha256_json(unsigned) != observed_digest:
        raise ContractError("Kernel template manifest digest is invalid", "template_manifest_mismatch")
    template_id = str(value["template_id"])
    showcase_id = str(value["showcase_id"])
    validate_identifier(template_id, field_name="kernel template ID")
    validate_identifier(showcase_id, field_name="kernel showcase ID")
    status = str(value["status"])
    if status not in _STATUSES:
        raise ContractError("Kernel template status is invalid", "invalid_template_manifest")
    upstream = _validate_upstream(_mapping(value["upstream"], "upstream"))
    runtime = TemplateRuntime.from_mapping(_mapping(value["runtime"], "runtime"))
    source = TemplateSource.from_mapping(_mapping(value["source"], "source"))
    evaluator_value = value["evaluator"]
    evaluator = (
        TemplateEvaluator.from_mapping(_mapping(evaluator_value, "evaluator"))
        if evaluator_value is not None else None
    )
    blockers = _strings(value["blockers"], "blockers", allow_empty=True)
    files = tuple(
        TemplateFile.from_mapping(_mapping(item, "snapshot file"))
        for item in value["snapshot_files"]
    )
    _validate_snapshot_files(resolved, files)
    _validate_readiness(status, blockers, runtime, source, evaluator)
    _validate_registry(template_id, showcase_id, status, observed_digest)
    return ReviewedKernelTemplate(
        resolved, template_id, showcase_id, status, observed_digest, upstream,
        runtime, source, evaluator, blockers, files,
    )


def _load_manifest_document(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError("Kernel template manifest is missing", "template_manifest_missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError("Kernel template manifest cannot be decoded", "invalid_template_manifest") from error
    return _mapping(value, "template manifest")


def _validate_upstream(value: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = {"repository", "commit", "tree", "imported_at_utc", "source_path", "files"}
    _exact_keys(value, expected, "upstream")
    if not _HEX_40.fullmatch(str(value["commit"])) or not _HEX_40.fullmatch(str(value["tree"])):
        raise ContractError("Upstream Git identity is invalid", "invalid_template_manifest")
    _relative(str(value["source_path"]), "upstream source path")
    files = value["files"]
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)) or not files:
        raise ContractError("Upstream file provenance is incomplete", "invalid_template_manifest")
    expected_file = {"original_path", "imported_path", "sha256", "size"}
    for item in files:
        record = _mapping(item, "upstream file")
        _exact_keys(record, expected_file, "upstream file")
        _relative(str(record["original_path"]), "upstream original path")
        _relative(str(record["imported_path"]), "upstream imported path")
        _digest(record["sha256"], "upstream file digest")
        if int(record["size"]) < 0:
            raise ContractError("Upstream file size is invalid", "invalid_template_manifest")
    return dict(value)


def _validate_snapshot_files(root: Path, files: tuple[TemplateFile, ...]) -> None:
    if not files or len({item.path for item in files}) != len(files):
        raise ContractError("Template snapshot files are incomplete", "invalid_template_manifest")
    for item in files:
        path = root.joinpath(*item.path.split("/"))
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ContractError("Template snapshot file is missing", "template_snapshot_mismatch") from error
        metadata = os.lstat(path)
        if path.is_symlink() or not resolved.is_relative_to(root) or not resolved.is_file():
            raise ContractError("Template snapshot file is unsafe", "unsafe_template_path")
        if metadata.st_nlink != 1 or metadata.st_size != item.size or sha256_file(path) != item.sha256:
            raise ContractError("Template snapshot file changed", "template_snapshot_mismatch")


def _validate_readiness(
    status: str,
    blockers: tuple[str, ...],
    runtime: TemplateRuntime,
    source: TemplateSource,
    evaluator: TemplateEvaluator | None,
) -> None:
    ready = (
        runtime.immutable_locator is not None
        and runtime.image_id is not None
        and source.baseline_tree_sha256 is not None
        and evaluator is not None
    )
    if status == "reviewed" and (blockers or not ready):
        raise ContractError("Reviewed template is missing executable proof", "invalid_template_manifest")
    if status == "pending" and not blockers:
        raise ContractError("Pending template must declare blockers", "invalid_template_manifest")
    if evaluator and set(evaluator.harness_files).intersection(source.editable_files):
        raise ContractError("Template harness is agent-editable", "invalid_template_manifest")


def _validate_registry(
    template_id: str, showcase_id: str, status: str, manifest_sha256: str
) -> None:
    try:
        registry_path = Path(__file__).resolve().parent / "data" / "kernel_template_registry.json"
        document = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(
            "Kernel template registry is unavailable", "template_registry_unavailable"
        ) from error
    if document.get("schema") != "apex.kernel-template-registry/v1":
        raise ContractError(
            "Kernel template registry schema is invalid", "template_registry_unavailable"
        )
    expected = {
        (str(item.get("template_id")), str(item.get("manifest_sha256"))): item
        for item in document.get("entries", [])
        if isinstance(item, Mapping)
    }
    entry = expected.get((template_id, manifest_sha256))
    if entry is None or entry.get("showcase_id") != showcase_id or entry.get("status") != status:
        raise ContractError(
            "Kernel template manifest is not in the reviewed registry",
            "template_not_registered",
            {"template_id": template_id, "manifest_sha256": manifest_sha256},
        )


__all__ = [
    "ReviewedKernelTemplate",
    "TemplateEvaluator",
    "TemplateFile",
    "TemplateRuntime",
    "TemplateSource",
    "load_kernel_template",
]
