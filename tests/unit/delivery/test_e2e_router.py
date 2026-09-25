from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import apex.delivery.e2e_router as router_module
from apex.core import ContractError
from apex.delivery import E2EBundleVerifierRouter, E2EVerifierProfile


class RecordingVerifier:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def verify(self, **kwargs):
        self.calls.append(kwargs)
        return "verified"


def _candidate(recipe_sha256: str):
    return SimpleNamespace(recipe=SimpleNamespace(computed_sha256=recipe_sha256))


def test_router_selects_exact_recipe_and_composes_profile_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = RecordingVerifier()
    factory_calls: list[str] = []
    recipe = "a" * 64
    router = E2EBundleVerifierRouter(
        (
            E2EVerifierProfile(
                "reviewed-profile-v1",
                frozenset({recipe}),
                lambda: factory_calls.append("called") or verifier,
            ),
        )
    )
    bundle = (tmp_path / "bundle").resolve()
    results = (tmp_path / "results").resolve()
    monkeypatch.setattr(
        router_module,
        "load_and_verify_e2e_bundle",
        lambda path, expected_digest=None: _candidate(recipe),
    )

    outcome = router.verify(
        bundle_dir=bundle,
        results_dir=results,
        expected_digest="b" * 64,
        source_overrides={"runtime": tmp_path},
    )

    assert outcome == "verified"
    assert factory_calls == ["called"]
    assert router.profile_ids == ("reviewed-profile-v1",)
    assert verifier.calls == [
        {
            "bundle_dir": bundle,
            "results_dir": results,
            "expected_digest": "b" * 64,
            "source_overrides": {"runtime": tmp_path},
        }
    ]


def test_router_rejects_unknown_recipe_without_composing_a_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    router = E2EBundleVerifierRouter(
        (
            E2EVerifierProfile(
                "reviewed-profile-v1",
                frozenset({"a" * 64}),
                lambda: calls.append("called") or RecordingVerifier(),
            ),
        )
    )
    monkeypatch.setattr(
        router_module,
        "load_and_verify_e2e_bundle",
        lambda path, expected_digest=None: _candidate("b" * 64),
    )

    with pytest.raises(ContractError) as error:
        router.verify(
            bundle_dir=(tmp_path / "bundle").resolve(),
            results_dir=(tmp_path / "results").resolve(),
        )

    assert error.value.reason_code == "e2e_verifier_profile_unavailable"
    assert error.value.details == {"recipe_sha256": "b" * 64}
    assert calls == []


def test_router_rejects_ambiguous_recipe_ownership() -> None:
    recipe = "a" * 64

    def profile(name: str) -> E2EVerifierProfile:
        return E2EVerifierProfile(name, frozenset({recipe}), RecordingVerifier)

    with pytest.raises(ContractError) as error:
        E2EBundleVerifierRouter((profile("one"), profile("two")))

    assert error.value.reason_code == "duplicate_e2e_verifier_recipe"


def test_router_rejects_relative_paths_before_bundle_inspection(tmp_path: Path) -> None:
    router = E2EBundleVerifierRouter(
        (
            E2EVerifierProfile(
                "reviewed-profile-v1",
                frozenset({"a" * 64}),
                RecordingVerifier,
            ),
        )
    )

    with pytest.raises(ContractError) as error:
        router.verify(bundle_dir=Path("bundle"), results_dir=tmp_path.resolve())

    assert error.value.reason_code == "invalid_bundle_path"
