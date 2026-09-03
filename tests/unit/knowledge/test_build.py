from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from apex.knowledge import archive_pinned_sources, build_card_snapshot


def test_builder_normalizes_without_executing_sources(pinned_geak_fixture, tmp_path: Path) -> None:
    root, pin, pin_path = pinned_geak_fixture
    snapshot = archive_pinned_sources(root, pin)
    built = build_card_snapshot(snapshot)

    assert len(built.cards) == 3
    assert {card.kind.value for card in built.cards} == {"fact", "anti_pattern", "experience"}
    assert all(card.status.value == "imported_unverified" for card in built.cards)
    assert all(not card.executable for card in built.cards)
    assert built.cards_document() == built.cards_document()
    assert built.capability_index() == built.capability_index()

    output = tmp_path / "output"
    package_catalog = tmp_path / "package" / "cards.json"
    command = [
        sys.executable,
        "scripts/build_knowledge_cards.py",
        "--geak-root",
        str(root),
        "--output-dir",
        str(output),
        "--package-catalog",
        str(package_catalog),
        "--pin",
        str(pin_path),
    ]
    environment = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    first = subprocess.run(command, check=True, capture_output=True, text=True, env=environment)
    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    checked = subprocess.run(
        [*command, "--check"], check=True, capture_output=True, text=True, env=environment
    )

    assert before == {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert json.loads(first.stdout) == json.loads(checked.stdout)
    assert package_catalog.read_bytes() == (output / "cards/cards.json").read_bytes()
    assert (package_catalog.parent / "LICENSE.GEAK-Apache-2.0").read_bytes() == (
        root / "LICENSE.md"
    ).read_bytes()
    package_notice = package_catalog.parent / "THIRD_PARTY_NOTICES.md"
    assert pin.git_sha in package_notice.read_text(encoding="utf-8")
    assert "modified, normalized advisory excerpts" in package_notice.read_text(
        encoding="utf-8"
    )
    assert "sitecustomize.py" in (output / "cards/source_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "must not execute" not in (output / "cards/cards.json").read_text(
        encoding="utf-8"
    )
    assert (output / "LICENSE.upstream").read_bytes() == (root / "LICENSE.md").read_bytes()
    assert not any("expert_skills" in path for path in before if "/upstream/" in path)
    assert not any(path.endswith((".py", ".sh")) for path in before if "/upstream/" in path)
    assert any(path.endswith("overview.md") for path in before)
