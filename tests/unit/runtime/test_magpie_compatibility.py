from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from apex.core import sha256_json
from apex.runtime import (
    BootstrapError,
    CorpusFile,
    MagpieCompatibilityEntry,
    MagpieCorpusManifest,
    build_magpie_compatibility_ledger,
    load_magpie_compatibility_ledger,
    verify_magpie_compatibility_ledger,
)


def _corpus() -> MagpieCorpusManifest:
    placeholder = MagpieCorpusManifest(
        "https://example.invalid/Magpie.git",
        "1" * 40,
        "2" * 40,
        "3" * 40,
        (CorpusFile("examples/benchmarks/a.yaml", "4" * 64),),
        "",
    )
    return replace(placeholder, manifest_sha256=sha256_json(placeholder.payload()))


def _entry(*, status: str = "config_compatible") -> MagpieCompatibilityEntry:
    return MagpieCompatibilityEntry(
        path="examples/benchmarks/a.yaml",
        config_sha256="4" * 64,
        framework="vllm",
        run_mode="docker",
        precision="fp8",
        lifecycle="one_shot",
        image_status="mutable_locator",
        model_identity_sha256="5" * 64,
        compatibility_status=status,
    )


def _ledger(status: str = "config_compatible"):
    corpus = _corpus()
    return build_magpie_compatibility_ledger(
        magpie_commit=corpus.commit,
        benchmark_tree=corpus.benchmark_tree,
        corpus_manifest_sha256=corpus.manifest_sha256,
        entries=(_entry(status=status),),
    )


def test_ledger_round_trip_binds_every_corpus_path_and_hash(tmp_path: Path) -> None:
    ledger = _ledger()
    path = tmp_path / "compatibility.json"
    path.write_text(
        json.dumps(ledger.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loaded = load_magpie_compatibility_ledger(path)
    verified = verify_magpie_compatibility_ledger(loaded, _corpus())

    assert verified["summary"] == {
        "config_count": 1,
        "config_compatible": 1,
        "capability_upgrade_required": 0,
        "workflow_qualified": 0,
        "formal_delivery_qualified": 0,
    }
    assert verified["entries"][0]["reward_policy_id"] == (
        "e2e_throughput_qos_v1"
    )
    assert verified["entries"][0]["workflow_qualification"] == "not_claimed"


def test_ledger_rejects_tamper_corpus_drift_and_upgrade_gap(tmp_path: Path) -> None:
    value = _ledger().to_dict()
    value["entries"][0]["precision"] = "bf16"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(BootstrapError, match="digest differs"):
        load_magpie_compatibility_ledger(path)

    with pytest.raises(BootstrapError, match="frozen corpus"):
        verify_magpie_compatibility_ledger(
            _ledger(), replace(_corpus(), benchmark_tree="6" * 40)
        )

    with pytest.raises(BootstrapError, match="capability upgrade"):
        verify_magpie_compatibility_ledger(
            _ledger("capability_upgrade_required"), _corpus()
        )
