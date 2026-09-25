from __future__ import annotations

import json
import subprocess
from pathlib import Path

from apex.cli import app


def _git(workspace: Path, *arguments: str) -> None:
    subprocess.run(
        ("git", *arguments),
        cwd=workspace,
        check=True,
        capture_output=True,
    )


def test_kernel_dry_run_emits_reproducible_unverified_contract_preview(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "kernel.py").write_text(
        "def kernel(x): return x\n", encoding="utf-8"
    )
    _git(workspace, "init", "--quiet")
    _git(workspace, "config", "user.email", "apex@example.invalid")
    _git(workspace, "config", "user.name", "Apex Test")
    _git(
        workspace,
        "remote",
        "add",
        "origin",
        "https://example.invalid/apex/preview.git",
    )
    _git(workspace, "add", "kernel.py")
    _git(workspace, "commit", "--quiet", "-m", "baseline")
    results = tmp_path / "results"
    descriptor = tmp_path / "task.json"
    descriptor.write_text(
        json.dumps(
            {
                "task_id": "preview-task",
                "workspace": str(workspace),
                "results_dir": str(results),
                "instructions": "Optimize kernel",
                "language": "triton",
                "editable_files": ["kernel.py"],
                "target_functions": ["kernel"],
                "commands": {
                    phase: {"argv": ["true"]}
                    for phase in ("compile", "correctness", "performance")
                },
            }
        ),
        encoding="utf-8",
    )
    result_path = results / "contract-preview.json"

    status = app.main(
        [
            "optimize",
            "kernel",
            "--task-spec",
            str(descriptor),
            "--dry-run",
            "--result-json",
            str(result_path),
            "--json",
        ]
    )

    assert status == 0
    output = json.loads(result_path.read_text())
    assert output["status"] == "evaluation_contract_preview"
    assert len(output["evaluation_contract_draft_digest"]) == 64
    assert output["evaluation_contract"]["status"] == "unverified"
    assert output["evaluation_contract"]["unverified_reason"] == (
        "evaluation_authority_missing"
    )
    assert output["evaluation_contract"]["draft"]["repository"]["remote"] == (
        "example.invalid/apex/preview"
    )
    assert not (results / "runs").exists()
