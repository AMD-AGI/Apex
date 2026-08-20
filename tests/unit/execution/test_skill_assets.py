from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from apex.core import IntegrityError
from apex.execution import load_kernel_skill_package


def test_packaged_kernel_skills_have_exact_identity() -> None:
    package = load_kernel_skill_package()

    assert tuple(sorted(package.skill_paths)) == (
        "amd-hip-kernel-optimization",
        "amd-kernel-debugging",
        "amd-kernel-optimization",
    )
    assert package.root.name == "apex-amd-kernel"
    assert len(package.digest) == 64
    assert all(path.is_file() for path in package.skill_paths.values())


def test_kernel_skill_tampering_fails_closed(tmp_path: Path) -> None:
    source = load_kernel_skill_package().root
    copied = tmp_path / "apex-amd-kernel"
    shutil.copytree(source, copied)
    skill = copied / "skills" / "amd-kernel-debugging" / "SKILL.md"
    skill.write_text(skill.read_text() + "\nTODO: weaken evidence\n")

    with pytest.raises(IntegrityError) as error:
        load_kernel_skill_package(copied)

    assert error.value.reason_code == "skill_asset_invalid"


def test_kernel_skill_reference_tampering_changes_package_digest(tmp_path: Path) -> None:
    source = load_kernel_skill_package().root
    copied = tmp_path / "apex-amd-kernel"
    shutil.copytree(source, copied)
    before = load_kernel_skill_package(copied).digest
    reference = (
        copied
        / "skills"
        / "amd-hip-kernel-optimization"
        / "references"
        / "hip2hip-patterns.md"
    )
    reference.write_text(reference.read_text() + "\nDiagnostic note.\n")

    assert load_kernel_skill_package(copied).digest != before
