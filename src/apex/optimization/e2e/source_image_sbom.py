"""Reproducible SPDX inventory for formal source-baked images."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from apex.core import IntegrityError, canonical_json_bytes, sha256_json
from apex.delivery import BuildRecipeLock


class SourceInventoryItem(Protocol):
    repository_id: str
    relative_path: str
    sha256: str
    content: bytes


def write_source_sbom(
    root: Path,
    recipe: BuildRecipeLock,
    repositories: Mapping[str, Path],
    source_stack: str,
    inventory: Sequence[SourceInventoryItem],
    epoch: int,
) -> Path:
    """Write a deterministic SPDX-2.3 document for every baked source byte."""

    namespace = sha256_json(
        {"recipe": recipe.computed_sha256, "source_stack": source_stack}
    )
    files, file_ids = _files(inventory)
    package_ids = {
        name: f"SPDXRef-Package-{name}" for name in sorted(repositories)
    }
    packages = [
        _package(name, package_ids[name], inventory)
        for name in sorted(repositories)
    ]
    relationships = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": package_id,
        }
        for package_id in package_ids.values()
    ]
    relationships.extend(
        {
            "spdxElementId": package_ids[item.repository_id],
            "relationshipType": "CONTAINS",
            "relatedSpdxElement": file_id,
        }
        for item, file_id in zip(inventory, file_ids, strict=True)
    )
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "apex-qwen-python-source-image",
        "documentNamespace": f"urn:apex:spdx:{namespace}",
        "creationInfo": {
            "created": datetime.fromtimestamp(epoch, timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "creators": ["Tool: Apex deterministic source-image builder"],
        },
        "documentDescribes": list(package_ids.values()),
        "packages": packages,
        "files": files,
        "relationships": relationships,
        "apex": {
            "parent_image_id": recipe.parent_image_digest,
            "source_stack_sha256": source_stack,
            "recipe_sha256": recipe.computed_sha256,
        },
    }
    path = root / "image.spdx.json"
    _write_once(path, canonical_json_bytes(document) + b"\n")
    return path.resolve()


def _files(
    inventory: Sequence[SourceInventoryItem],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    values = []
    identifiers = []
    for index, item in enumerate(inventory):
        identifier = f"SPDXRef-File-{index}"
        identifiers.append(identifier)
        values.append(
            {
                "SPDXID": identifier,
                "fileName": f"/opt/apex/python/{item.relative_path}",
                "checksums": [
                    {"algorithm": "SHA1", "checksumValue": _sha1(item.content)},
                    {"algorithm": "SHA256", "checksumValue": item.sha256},
                ],
                "licenseConcluded": "NOASSERTION",
                "copyrightText": "NOASSERTION",
            }
        )
    return values, tuple(identifiers)


def _package(
    name: str,
    identifier: str,
    inventory: Sequence[SourceInventoryItem],
) -> dict[str, Any]:
    hashes = sorted(
        _sha1(item.content) for item in inventory if item.repository_id == name
    )
    verification = _sha1("".join(hashes).encode("ascii"))
    return {
        "SPDXID": identifier,
        "name": name,
        "downloadLocation": "NOASSERTION",
        "filesAnalyzed": True,
        "packageVerificationCode": {
            "packageVerificationCodeValue": verification
        },
        "licenseConcluded": "NOASSERTION",
        "licenseDeclared": "NOASSERTION",
        "copyrightText": "NOASSERTION",
    }


def _sha1(content: bytes) -> str:
    return hashlib.sha1(content, usedforsecurity=False).hexdigest()


def _write_once(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise IntegrityError(
            "Immutable source-image artifact exists", "immutable_delivery_artifact"
        )
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


__all__ = ["SourceInventoryItem", "write_source_sbom"]
