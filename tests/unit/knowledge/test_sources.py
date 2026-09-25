from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex.core import IntegrityError, sha256_bytes
from apex.knowledge import (
    PinnedSourceManifest,
    archive_pinned_sources,
    default_geak_source_pin,
)


def test_default_pin_matches_reviewable_manifest() -> None:
    pin_path = Path("src/apex/knowledge/geak_source_pin.json")
    observed = PinnedSourceManifest.from_mapping(json.loads(pin_path.read_text(encoding="utf-8")))

    assert observed == default_geak_source_pin()
    assert sum(item.expected_files for item in observed.estates) == 733
    assert sum(item.expected_bytes for item in observed.estates) == 4_520_163


def test_archive_reads_committed_bytes_and_records_exclusions(pinned_geak_fixture) -> None:
    root, pin, _ = pinned_geak_fixture
    changed = root / "perf_knowledge/operators/rms_norm/overview.md"
    changed.write_text("# Dirty working tree text\n", encoding="utf-8")

    snapshot = archive_pinned_sources(root, pin)
    by_path = {item.path: item for item in snapshot.files}

    committed = b"# RMS norm\nUse a fused pass.\n"
    assert by_path["perf_knowledge/operators/rms_norm/overview.md"].content_sha256 == sha256_bytes(
        committed
    )
    assert by_path["perf_knowledge/tools/check.py"].exclusion_reason == (
        "executable_source_requires_separate_audit"
    )
    assert by_path["perf_knowledge/expert_skills/unsafe/skill.md"].exclusion_reason == (
        "nested_expert_skill_requires_separate_audit"
    )
    assert snapshot.to_manifest() == snapshot.to_manifest()
    assert snapshot.to_manifest()["summary"]["card_eligible_files"] == 3


def test_wrong_revision_or_estate_size_fails_closed(pinned_geak_fixture) -> None:
    root, pin, _ = pinned_geak_fixture
    wrong_revision = PinnedSourceManifest(
        repository=pin.repository,
        git_sha="f" * 40,
        license=pin.license,
        license_path=pin.license_path,
        license_sha256=pin.license_sha256,
        transform_version=pin.transform_version,
        estates=pin.estates,
    )
    with pytest.raises(IntegrityError) as revision:
        archive_pinned_sources(root, wrong_revision)
    assert revision.value.reason_code == "source_revision_mismatch"

    bad_estate = pin.estates[0]
    bad = PinnedSourceManifest(
        repository=pin.repository,
        git_sha=pin.git_sha,
        license=pin.license,
        license_path=pin.license_path,
        license_sha256=pin.license_sha256,
        transform_version=pin.transform_version,
        estates=(
            type(bad_estate)(bad_estate.estate_id, bad_estate.path, bad_estate.expected_files, 1),
            *pin.estates[1:],
        ),
    )
    with pytest.raises(IntegrityError) as aggregate:
        archive_pinned_sources(root, bad)
    assert aggregate.value.reason_code == "source_estate_mismatch"
