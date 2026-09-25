from __future__ import annotations

import os
import py_compile
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
    assert frozen.root == candidate.root
    assert frozen.root.name == "candidate.frozen"


def test_freeze_rebuilds_clean_projection_without_agent_bytecode(tmp_path: Path) -> None:
    resolved = _resolved(tmp_path)
    candidate = CandidateWorkspace.create(resolved, destination=tmp_path / "candidate")
    (candidate.root / "source" / "kernel.py").write_text("value = 2\n", encoding="utf-8")
    poison = candidate.root / "sitecustomize.py"
    poison.write_text("raise RuntimeError('agent bytecode executed')\n", encoding="utf-8")
    py_compile.compile(
        str(poison),
        cfile=str(candidate.root / "sitecustomize.pyc"),
        doraise=True,
    )
    poison.unlink()
    cache = candidate.root / "source" / "__pycache__"
    cache.mkdir()
    (cache / "kernel.pyc").write_bytes(b"untrusted-bytecode")

    frozen = candidate.freeze()

    assert frozen.changed_files == ("source/kernel.py",)
    assert not (frozen.root / "sitecustomize.pyc").exists()
    assert not (frozen.root / "source" / "__pycache__").exists()
    assert (frozen.root / "source" / "kernel.py").read_text() == "value = 2\n"


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
