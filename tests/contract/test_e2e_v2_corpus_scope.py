from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from apex.core import ContractError
from apex.optimization.e2e.run_contracts import require_docker_one_shot_contract
from apex.runtime import (
    MagpieConfigResolutionEntryEvidence,
    build_magpie_config_resolution_evidence,
)


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_magpie_corpus_has_exact_docker_only_v2_partition() -> None:
    ledger = json.loads(
        (ROOT / "scripts" / "magpie_compatibility_ledger.json").read_text(
            encoding="utf-8"
        )
    )
    supported: list[str] = []
    rejected: dict[str, str] = {}
    for entry in ledger["entries"]:
        resolved = SimpleNamespace(
            plan={
                "identity": {"run_mode": entry["run_mode"]},
                "lifecycle": entry["lifecycle"],
            }
        )
        try:
            require_docker_one_shot_contract(resolved)
        except ContractError as error:
            rejected[entry["path"]] = error.reason_code
        else:
            supported.append(entry["path"])

    assert len(ledger["entries"]) == 27
    assert len(supported) == 21
    assert len(rejected) == 6
    assert set(rejected.values()) == {"e2e_docker_only"}

    resolution = build_magpie_config_resolution_evidence(
        magpie_commit=ledger["magpie_commit"],
        corpus_manifest_sha256=ledger["corpus_manifest_sha256"],
        plan_schema="apex.magpie-main-resolved-plan/v1",
        capability_schema="apex.magpie-main-capability-receipt/v1",
        result_schema="apex.magpie-main-result-contract/v1",
        entries=tuple(
            MagpieConfigResolutionEntryEvidence(
                entry["path"],
                entry["config_sha256"],
                f"{index + 1:064x}",
                f"{index + 101:064x}",
                entry["compatibility_status"],
                entry["run_mode"],
                entry["lifecycle"],
            )
            for index, entry in enumerate(ledger["entries"])
        ),
    )
    assert [item.path for item in resolution.e2e_v2_entries()] == supported
