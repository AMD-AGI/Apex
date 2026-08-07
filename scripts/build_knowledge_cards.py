#!/usr/bin/env python3
"""Build attributed Apex knowledge cards from an exact local GEAK commit."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from apex.core import canonical_json_bytes
from apex.knowledge import (
    CardSnapshot,
    PinnedSourceManifest,
    SourceSnapshot,
    archive_pinned_sources,
    build_card_snapshot,
    default_geak_source_pin,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic advisory cards; no upstream content is executed."
    )
    parser.add_argument("--geak-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--package-catalog",
        type=Path,
        help="Also publish/check the generated cards.json shipped inside the Apex wheel.",
    )
    parser.add_argument("--pin", type=Path, help="Override pin for hermetic development fixtures.")
    parser.add_argument(
        "--check", action="store_true", help="Fail unless existing outputs are byte exact."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pin = _load_pin(args.pin)
    sources = archive_pinned_sources(args.geak_root, pin)
    cards = build_card_snapshot(sources)
    documents = _release_documents(pin, sources, cards)
    package_documents = _package_documents(pin, sources, cards)
    if args.check:
        _check_outputs(args.output_dir, documents)
        if args.package_catalog is not None:
            _check_package_outputs(args.package_catalog, package_documents)
    else:
        _write_outputs(args.output_dir, documents)
        if args.package_catalog is not None:
            _write_package_outputs(args.package_catalog, package_documents)
    print(json.dumps(_summary(sources.to_manifest(), cards.cards_document()), sort_keys=True))
    return 0


def _load_pin(path: Path | None) -> PinnedSourceManifest:
    if path is None:
        return default_geak_source_pin()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"unable to read source pin: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit("source pin must be a JSON object")
    return PinnedSourceManifest.from_mapping(value)


def _json_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _release_documents(
    pin: PinnedSourceManifest,
    sources: SourceSnapshot,
    cards: CardSnapshot,
) -> dict[str, bytes]:
    prefix = f"upstream/geak/{pin.git_sha}"
    manifest = sources.to_manifest()
    raw_files = tuple(item for item in sources.files if _include_raw_source(item.path))
    upstream = {
        "schema_version": 1,
        "source_repo": pin.repository,
        "source_commit": pin.git_sha,
        "source_paths": [estate.path for estate in pin.estates],
        "license": pin.license,
        "imported_at": "2026-08-07T00:00:00Z",
        "snapshot_manifest": "cards/source_manifest.json",
        "local_modifications": False,
        "nested_source_audit": "pending",
        "raw_release_policy": (
            "All ordinary perf/kernel/e2e knowledge is present. Nested expert/analysis "
            "skill bundles and executable templates remain manifest-only pending a "
            "separate source/license audit."
        ),
        "raw_included_files": len(raw_files),
        "manifested_files": len(sources.files),
    }
    digest_lines = "".join(
        f"{item.content_sha256}  {item.path}\n" for item in raw_files
    ).encode("utf-8")
    documents: dict[str, bytes] = {
        "source_pin.json": _json_bytes(pin.to_dict()),
        "UPSTREAM.json": _json_bytes(upstream),
        "LICENSE.upstream": sources.license_content,
        "MANIFEST.sha256": digest_lines,
        "cards/source_manifest.json": _json_bytes(manifest),
        "cards/cards.json": _json_bytes(cards.cards_document()),
        "cards/capability_index.json": _json_bytes(cards.capability_index()),
        "cards/ATTRIBUTION.md": _attribution(pin, len(cards.cards)).encode("utf-8"),
        "capability_index.yaml": yaml.safe_dump(
            cards.capability_index(), sort_keys=True, allow_unicode=True
        ).encode("utf-8"),
        f"{prefix}/UPSTREAM.json": _json_bytes(upstream),
        f"{prefix}/MANIFEST.sha256": digest_lines,
        f"{prefix}/LICENSE.upstream": sources.license_content,
    }
    for item in raw_files:
        documents[f"{prefix}/{item.path}"] = item.content
    return documents


def _include_raw_source(path: str) -> bool:
    """Release inert source prose while quarantining unresolved nested code estates."""

    excluded_prefixes = (
        "perf_knowledge/expert_skills/",
        "e2e_workflow/knowledge/analysis_skills/",
    )
    if path.startswith(excluded_prefixes):
        return False
    return not path.endswith((".py", ".sh"))


def _package_documents(
    pin: PinnedSourceManifest,
    sources: SourceSnapshot,
    cards: CardSnapshot,
) -> dict[str, bytes]:
    """Return every GEAK-derived wheel payload plus its redistribution notices."""

    return {
        "cards.json": _json_bytes(cards.cards_document()),
        "LICENSE.GEAK-Apache-2.0": sources.license_content,
        "THIRD_PARTY_NOTICES.md": _package_notice(pin, len(cards.cards)).encode("utf-8"),
    }


def _package_notice(pin: PinnedSourceManifest, card_count: int) -> str:
    return (
        "# Third-party notices for packaged Apex knowledge\n\n"
        f"The packaged catalog contains {card_count} modified, normalized advisory "
        f"excerpts derived from {pin.repository} at immutable commit `{pin.git_sha}`.\n\n"
        f"Upstream license: `{pin.license}`. The complete license text is distributed "
        "beside this notice as `LICENSE.GEAK-Apache-2.0`. Exact source paths and "
        "content digests remain available in the Apex source distribution under "
        "`tools/perf_knowledge/cards/source_manifest.json`. The cards are inert text "
        "and are not executable optimization policy.\n"
    )


def _attribution(pin: PinnedSourceManifest, card_count: int) -> str:
    return (
        "# Generated knowledge attribution\n\n"
        f"These {card_count} advisory cards were transformed from "
        f"[{pin.repository}]({pin.repository}) "
        f"at commit `{pin.git_sha}` under `{pin.license}`.\n\n"
        "The cards are modified, normalized excerpts and are not runtime decisions. "
        "Embedded commands are inert text. Exact source paths and SHA-256 digests are "
        "recorded in `source_manifest.json`.\n\n"
        "`LICENSE.upstream` is an exact copy of the pinned upstream Apache-2.0 license.\n\n"
        "Executable sources, nested expert/analysis bundles, templates, and archived "
        "material are manifest-only and excluded from cards "
        "until separately audited. Apex does not copy or execute those sources in this build.\n"
    )


def _check_outputs(output_dir: Path, documents: dict[str, bytes]) -> None:
    mismatches = []
    for name, expected in documents.items():
        path = output_dir / name
        try:
            observed = path.read_bytes()
        except OSError:
            observed = b""
        if observed != expected:
            mismatches.append(name)
    if mismatches:
        raise SystemExit(f"knowledge outputs differ: {', '.join(sorted(mismatches))}")


def _check_file(path: Path, expected: bytes) -> None:
    try:
        observed = path.read_bytes()
    except OSError:
        observed = b""
    if observed != expected:
        raise SystemExit(f"packaged knowledge catalog differs: {path}")


def _check_package_outputs(catalog_path: Path, documents: dict[str, bytes]) -> None:
    for name, expected in sorted(documents.items()):
        path = catalog_path if name == "cards.json" else catalog_path.parent / name
        _check_file(path, expected)


def _write_package_outputs(catalog_path: Path, documents: dict[str, bytes]) -> None:
    for name, content in sorted(documents.items()):
        path = catalog_path if name == "cards.json" else catalog_path.parent / name
        _atomic_write(path, content)


def _write_outputs(output_dir: Path, documents: dict[str, bytes]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in sorted(documents.items()):
        _atomic_write(output_dir / name, content)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _summary(source_manifest: dict[str, Any], cards: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_manifest_sha256": source_manifest["manifest_sha256"],
        "card_snapshot_sha256": cards["snapshot_sha256"],
        "source_files": source_manifest["summary"]["file_count"],
        "cards": len(cards["cards"]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
