from __future__ import annotations

import os
from pathlib import Path

import pytest

from apex.core import IntegrityError
from apex.intake import TaskResolver, TaskSpec
from apex.optimization.kernel import CandidateWorkspace


def _resolved(tmp_path: Path):
    workspace = tmp_path / "workspace"
    (workspace / "source").mkdir(parents=True)
    (workspace / "source" / "kernel.py").write_text("value = 1\n", encoding="utf-8")
    (workspace / "harness.py").write_text("assert True\n", encoding="utf-8")
    task = TaskSpec.from_mapping(
        {
            "task_id": "task",
            "workspace": str(workspace),
            "results_dir": str(tmp_path / "results"),
            "instructions": "Optimize value",
            "language": "triton",
            "editable_files": ["source/kernel.py"],
            "target_functions": ["kernel"],
            "commands": {
                "compile": {"argv": ["true"]},
                "correctness": {"argv": ["true"]},
                "performance": {"argv": ["true"]},
            },
        }
    )
    return TaskResolver().resolve(task)


def test_freeze_accepts_only_editable_content_change(tmp_path: Path) -> None:
    resolved = _resolved(tmp_path)
    candidate = CandidateWorkspace.create(resolved, destination=tmp_path / "candidate")
    (candidate.root / "source" / "kernel.py").write_text("value = 2\n", encoding="utf-8")

    frozen = candidate.freeze()

    assert frozen.changed_files == ("source/kernel.py",)


def test_freeze_rejects_harness_edit(tmp_path: Path) -> None:
    resolved = _resolved(tmp_path)
    candidate = CandidateWorkspace.create(resolved, destination=tmp_path / "candidate")
    (candidate.root / "harness.py").write_text("assert False\n", encoding="utf-8")

    with pytest.raises(IntegrityError) as raised:
        candidate.freeze()

    assert raised.value.reason_code == "undeclared_agent_edit"


def test_freeze_rejects_source_mode_change(tmp_path: Path) -> None:
    resolved = _resolved(tmp_path)
    candidate = CandidateWorkspace.create(resolved, destination=tmp_path / "candidate")
    source = candidate.root / "source" / "kernel.py"
    os.chmod(source, 0o755)

    with pytest.raises(IntegrityError) as raised:
        candidate.freeze()

    assert raised.value.reason_code == "source_mode_change_forbidden"
