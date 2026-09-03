from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import ContractError
from apex.runtime import FormalResultsRootValidator, formal_results_validator


def test_rejects_result_inside_protected_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "Apex"
    checkout.mkdir()
    validator = FormalResultsRootValidator((checkout,))

    with pytest.raises(ContractError) as caught:
        validator.validate(checkout / "tmp" / "refactor")

    assert caught.value.reason_code == "formal_results_overlap"


def test_rejects_results_ancestor_of_protected_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "src" / "Apex"
    checkout.mkdir(parents=True)
    validator = FormalResultsRootValidator((checkout,))

    with pytest.raises(ContractError) as caught:
        validator.validate(tmp_path / "src")

    assert caught.value.reason_code == "formal_results_overlap"


def test_rejects_existing_ancestor_symlink(tmp_path: Path) -> None:
    checkout = tmp_path / "Apex"
    checkout.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    link = tmp_path / "results-link"
    link.symlink_to(external, target_is_directory=True)
    validator = FormalResultsRootValidator((checkout,))

    with pytest.raises(ContractError) as caught:
        validator.validate(link / "campaign")

    assert caught.value.reason_code == "unsafe_formal_results_root"


def test_accepts_external_new_campaign_root(tmp_path: Path) -> None:
    checkout = tmp_path / "Apex"
    checkout.mkdir()
    external = tmp_path / "apex-results"
    validator = formal_results_validator(apex_root=checkout)

    selected = validator.validate(external / "campaign-1", require_new=True)

    assert selected == external / "campaign-1"
    assert not selected.exists()


def test_rejects_existing_root_when_new_campaign_is_required(tmp_path: Path) -> None:
    checkout = tmp_path / "Apex"
    checkout.mkdir()
    existing = tmp_path / "apex-results" / "campaign-1"
    existing.mkdir(parents=True)
    validator = FormalResultsRootValidator((checkout,))

    with pytest.raises(ContractError) as caught:
        validator.validate(existing, require_new=True)

    assert caught.value.reason_code == "formal_results_root_exists"


def test_factory_protects_dependency_source_and_workspace_roots(
    tmp_path: Path,
) -> None:
    roots = tuple(tmp_path / name for name in ("apex", "magpie", "vllm", "task"))
    for root in roots:
        root.mkdir()
    validator = formal_results_validator(
        apex_root=roots[0],
        dependency_roots=(roots[1],),
        source_roots=(roots[2],),
        workspace_roots=(roots[3],),
    )

    for root in roots:
        with pytest.raises(ContractError) as caught:
            validator.validate(root / "ignored-results")
        assert caught.value.reason_code == "formal_results_overlap"
