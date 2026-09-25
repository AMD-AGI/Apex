from __future__ import annotations

import json
from pathlib import Path

from apex.cli import app


def test_showcase_cli_exports_verifies_shows_and_lists(
    canonical_run, tmp_path: Path, capsys
) -> None:
    root = tmp_path / "exports" / "cli-showcase"
    export_code = app.main(
        [
            "showcase", "export",
            "--run-root", str(canonical_run["root"]),
            "--run-id", canonical_run["run_id"],
            "--id", "cli-showcase",
            "--output", str(root),
        ]
    )
    exported = json.loads(capsys.readouterr().out)

    assert export_code == 0
    assert exported["status"] == "pending"
    assert exported["showcase_id"] == "cli-showcase"

    assert app.main(["showcase", "verify", "--path", str(root)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["schema"] == "apex.showcase-verification/v2"
    assert verified["showcase_id"] == "cli-showcase"
    assert verified["status"] == "pending"
    assert verified["reproduction_verified"] is True
    assert len(verified["verification_receipt_sha256"]) == 64

    assert app.main(["showcase", "show", "--path", str(root)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["schema"] == "apex.showcase/v1"
    assert shown["showcase_id"] == "cli-showcase"

    assert app.main(["showcase", "list", "--root", str(root.parent)]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert [item["showcase_id"] for item in listing["entries"]] == ["cli-showcase"]
