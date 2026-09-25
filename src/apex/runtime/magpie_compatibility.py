"""Self-digested compatibility ledger for the frozen Magpie config corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from apex.core import sha256_json

from .magpie_corpus import MagpieCorpusManifest
from .repositories import BootstrapError


SCHEMA = "apex.magpie-config-compatibility/v1"
REWARD_POLICY = "e2e_throughput_qos_v1"
REQUIRED_REWARD_METRICS = (
    "total_token_throughput",
    "ttft_p99",
    "tpot_p99",
    "quality",
)
_COMPATIBILITY = {"capability_upgrade_required", "config_compatible"}
_IMAGE_STATUS = {
    "immutable",
    "mutable_locator",
    "not_applicable",
    "runtime_selection_required",
}
_LIFECYCLE = {"cleanup", "one_shot", "reuse"}


@dataclass(frozen=True, slots=True)
class MagpieCompatibilityEntry:
    path: str
    config_sha256: str
    framework: str
    run_mode: str
    precision: str
    lifecycle: str
    image_status: str
    model_identity_sha256: str
    compatibility_status: str
    required_reward_metrics: tuple[str, ...] = REQUIRED_REWARD_METRICS
    reward_policy_id: str = REWARD_POLICY
    workflow_qualification: str = "not_claimed"
    formal_delivery_qualification: str = "not_claimed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "config_sha256": self.config_sha256,
            "framework": self.framework,
            "run_mode": self.run_mode,
            "precision": self.precision,
            "lifecycle": self.lifecycle,
            "image_status": self.image_status,
            "model_identity_sha256": self.model_identity_sha256,
            "compatibility_status": self.compatibility_status,
            "required_reward_metrics": list(self.required_reward_metrics),
            "reward_policy_id": self.reward_policy_id,
            "workflow_qualification": self.workflow_qualification,
            "formal_delivery_qualification": self.formal_delivery_qualification,
        }


@dataclass(frozen=True, slots=True)
class MagpieCompatibilityLedger:
    magpie_commit: str
    benchmark_tree: str
    corpus_manifest_sha256: str
    entries: tuple[MagpieCompatibilityEntry, ...]
    ledger_sha256: str

    def payload(self) -> dict[str, Any]:
        compatible = sum(
            item.compatibility_status == "config_compatible" for item in self.entries
        )
        return {
            "schema": SCHEMA,
            "magpie_commit": self.magpie_commit,
            "benchmark_tree": self.benchmark_tree,
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "entries": [item.to_dict() for item in self.entries],
            "summary": {
                "config_count": len(self.entries),
                "config_compatible": compatible,
                "capability_upgrade_required": len(self.entries) - compatible,
                "workflow_qualified": 0,
                "formal_delivery_qualified": 0,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "ledger_sha256": self.ledger_sha256}


def build_magpie_compatibility_ledger(
    *,
    magpie_commit: str,
    benchmark_tree: str,
    corpus_manifest_sha256: str,
    entries: Sequence[MagpieCompatibilityEntry],
) -> MagpieCompatibilityLedger:
    """Build a sorted self-digested ledger from independently inspected entries."""

    ordered = tuple(sorted(entries, key=lambda item: item.path))
    placeholder = MagpieCompatibilityLedger(
        magpie_commit,
        benchmark_tree,
        corpus_manifest_sha256,
        ordered,
        "",
    )
    _validate_ledger(placeholder, require_digest=False)
    return MagpieCompatibilityLedger(
        magpie_commit,
        benchmark_tree,
        corpus_manifest_sha256,
        ordered,
        sha256_json(placeholder.payload()),
    )


def load_magpie_compatibility_ledger(path: Path) -> MagpieCompatibilityLedger:
    """Load a strict ledger and verify its self digest."""

    try:
        raw = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise BootstrapError(f"invalid Magpie compatibility ledger {path}: {error}") from error
    expected = {
        "schema", "magpie_commit", "benchmark_tree", "corpus_manifest_sha256",
        "entries", "summary", "ledger_sha256",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected or raw.get("schema") != SCHEMA:
        raise BootstrapError("Magpie compatibility ledger field set differs")
    entries = _parse_entries(raw.get("entries"))
    ledger = MagpieCompatibilityLedger(
        _digest(raw.get("magpie_commit"), 40, "magpie_commit"),
        _digest(raw.get("benchmark_tree"), 40, "benchmark_tree"),
        _digest(raw.get("corpus_manifest_sha256"), 64, "corpus_manifest_sha256"),
        entries,
        _digest(raw.get("ledger_sha256"), 64, "ledger_sha256"),
    )
    _validate_ledger(ledger, require_digest=True)
    if raw.get("summary") != ledger.payload()["summary"]:
        raise BootstrapError("Magpie compatibility summary differs")
    return ledger


def verify_magpie_compatibility_ledger(
    ledger: MagpieCompatibilityLedger,
    corpus: MagpieCorpusManifest,
) -> Mapping[str, Any]:
    """Bind every compatibility row to the exact corpus path and bytes."""

    expected = tuple((item.path, item.sha256) for item in corpus.files)
    observed = tuple((item.path, item.config_sha256) for item in ledger.entries)
    valid = (
        ledger.magpie_commit == corpus.commit
        and ledger.benchmark_tree == corpus.benchmark_tree
        and ledger.corpus_manifest_sha256 == corpus.manifest_sha256
        and observed == expected
    )
    if not valid:
        raise BootstrapError("Magpie compatibility ledger differs from frozen corpus")
    unsupported = [
        item.path
        for item in ledger.entries
        if item.compatibility_status != "config_compatible"
    ]
    if unsupported:
        raise BootstrapError(
            "Magpie corpus requires a capability upgrade: " + ", ".join(unsupported)
        )
    return ledger.to_dict()


def _parse_entries(value: object) -> tuple[MagpieCompatibilityEntry, ...]:
    if not isinstance(value, list) or not value:
        raise BootstrapError("Magpie compatibility entries are invalid")
    return tuple(_parse_entry(item) for item in value)


def _parse_entry(value: object) -> MagpieCompatibilityEntry:
    expected = set(MagpieCompatibilityEntry.__dataclass_fields__)
    if not isinstance(value, Mapping) or set(value) != expected:
        raise BootstrapError("Magpie compatibility entry field set differs")
    metrics = value.get("required_reward_metrics")
    if not isinstance(metrics, list) or any(not isinstance(item, str) for item in metrics):
        raise BootstrapError("Magpie compatibility reward metrics are invalid")
    return MagpieCompatibilityEntry(
        path=_text(value.get("path"), "path"),
        config_sha256=_digest(value.get("config_sha256"), 64, "config_sha256"),
        framework=_text(value.get("framework"), "framework"),
        run_mode=_text(value.get("run_mode"), "run_mode"),
        precision=_text(value.get("precision"), "precision"),
        lifecycle=_text(value.get("lifecycle"), "lifecycle"),
        image_status=_text(value.get("image_status"), "image_status"),
        model_identity_sha256=_digest(
            value.get("model_identity_sha256"), 64, "model_identity_sha256"
        ),
        compatibility_status=_text(
            value.get("compatibility_status"), "compatibility_status"
        ),
        required_reward_metrics=tuple(metrics),
        reward_policy_id=_text(value.get("reward_policy_id"), "reward_policy_id"),
        workflow_qualification=_text(
            value.get("workflow_qualification"), "workflow_qualification"
        ),
        formal_delivery_qualification=_text(
            value.get("formal_delivery_qualification"),
            "formal_delivery_qualification",
        ),
    )


def _validate_ledger(
    ledger: MagpieCompatibilityLedger, *, require_digest: bool
) -> None:
    paths = tuple(item.path for item in ledger.entries)
    if paths != tuple(sorted(set(paths))):
        raise BootstrapError("Magpie compatibility paths are duplicated or unsorted")
    for item in ledger.entries:
        valid = (
            item.path.startswith("examples/benchmarks/")
            and ".." not in Path(item.path).parts
            and item.lifecycle in _LIFECYCLE
            and item.image_status in _IMAGE_STATUS
            and item.compatibility_status in _COMPATIBILITY
            and item.required_reward_metrics == REQUIRED_REWARD_METRICS
            and item.reward_policy_id == REWARD_POLICY
            and item.workflow_qualification == "not_claimed"
            and item.formal_delivery_qualification == "not_claimed"
        )
        if not valid:
            raise BootstrapError("Magpie compatibility entry is invalid")
    if require_digest and ledger.ledger_sha256 != sha256_json(ledger.payload()):
        raise BootstrapError("Magpie compatibility ledger digest differs")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BootstrapError(f"Magpie compatibility {field} is invalid")
    return value.strip()


def _digest(value: object, length: int, field: str) -> str:
    text = _text(value, field)
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise BootstrapError(f"Magpie compatibility {field} is invalid")
    return text


__all__ = [
    "MagpieCompatibilityEntry",
    "MagpieCompatibilityLedger",
    "REQUIRED_REWARD_METRICS",
    "build_magpie_compatibility_ledger",
    "load_magpie_compatibility_ledger",
    "verify_magpie_compatibility_ledger",
]
