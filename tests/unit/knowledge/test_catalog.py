from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex.core import IntegrityError, canonical_json_bytes, sha256_json
from apex.knowledge import load_knowledge_catalog
from apex import bootstrap as application_bootstrap


def _catalog(path: Path, cards: list[dict[str, object]]) -> Path:
    value = {
        "schema_version": 1,
        "source_manifest_sha256": "a" * 64,
        "transform_version": "fixture_v1",
        "cards": cards,
    }
    value["snapshot_sha256"] = sha256_json(value)
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    return path


def test_catalog_loads_only_canonical_digest_bound_cards(tmp_path: Path, card_factory) -> None:
    card = card_factory(claim="Use a fused reduction")
    path = _catalog(tmp_path / "cards.json", [card.to_dict()])

    catalog = load_knowledge_catalog(path)

    assert catalog.cards == (card,)
    assert catalog.source_manifest_sha256 == "a" * 64
    assert len(catalog.file_sha256) == 64


def test_catalog_rejects_snapshot_or_serialization_tampering(tmp_path: Path, card_factory) -> None:
    card = card_factory(claim="Use a fused reduction")
    path = _catalog(tmp_path / "cards.json", [card.to_dict()])
    value = json.loads(path.read_text(encoding="utf-8"))
    value["cards"][0]["claim"] = "tampered"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    with pytest.raises(IntegrityError) as mismatch:
        load_knowledge_catalog(path)
    assert mismatch.value.reason_code == "knowledge_snapshot_mismatch"

    path = _catalog(tmp_path / "noncanonical.json", [card.to_dict()])
    value = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(IntegrityError) as noncanonical:
        load_knowledge_catalog(path)
    assert noncanonical.value.reason_code == "noncanonical_knowledge_catalog"


def test_default_packaged_catalog_matches_attributed_generated_snapshot() -> None:
    packaged = application_bootstrap._default_knowledge_catalog()
    generated = Path("tools/perf_knowledge/cards/cards.json").resolve(strict=True)

    assert packaged.is_file()
    assert packaged.read_bytes() == generated.read_bytes()
    assert len(load_knowledge_catalog(packaged).cards) == 663

    package_data = packaged.parent
    assert (package_data / "LICENSE.GEAK-Apache-2.0").read_bytes() == Path(
        "tools/perf_knowledge/LICENSE.upstream"
    ).read_bytes()
    notice = (package_data / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "6fa40c36b68bad9d543ae551b95bd3d169865744" in notice
    assert "Apache-2.0" in notice
