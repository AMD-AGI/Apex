"""Strict recovery and isolated source projection for chat-started campaigns."""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterator, Mapping

from apex.core import (
    ApexError,
    ContractError,
    IntegrityError,
    canonical_json_bytes,
    sha256_json,
)
from apex.evaluation import (
    EvaluationContractReceipt,
    load_evaluation_contract,
)
from apex.intake import ResolvedTaskSpec, TaskSpec
from apex.orchestration import RunController
from apex.reporting import resolve_run_source
from apex.runtime import WorkspaceGitIdentityResolver
from apex.storage import ArtifactReceipt, SnapshotStore

from .contract_recording import record_authorized_evaluation_contract
from .formal_authority import FormalEvaluationAuthorityProvider
from .run_record import KernelRunRecord
from .verification import candidate_source_digest
from .workspace import CandidateWorkspace, candidate_file_bytes


FormalBaselineLoader = Callable[[Path], object]


@dataclass(frozen=True, slots=True)
class FormalCandidateProjection:
    root: Path
    anchor: Path
    resolved: ResolvedTaskSpec
    source_digest: str
    changed_files: tuple[str, ...]
    files: Mapping[str, bytes]


@dataclass(slots=True)
class FormalKernelCampaign:
    """Recovered canonical draft plus evaluator-owned mutation helpers."""

    record: KernelRunRecord
    task: TaskSpec
    resolved: ResolvedTaskSpec
    draft_contract: EvaluationContractReceipt
    baseline: Mapping[str, ArtifactReceipt]
    harness: Mapping[str, ArtifactReceipt]
    repository_resolver: WorkspaceGitIdentityResolver

    @classmethod
    def load(
        cls,
        run_root: Path,
        *,
        workspace: Path,
        results: Path,
        repository_resolver: WorkspaceGitIdentityResolver | None = None,
    ) -> "FormalKernelCampaign":
        source = resolve_run_source(run_root)
        events = tuple(source.journal.iter_events(source.run_id, verify=True))
        provenance = _one_event(events, "provenance_observed", "kernel_campaign_draft")
        task = _load_task(source.artifacts, _one_binding(provenance, "task_input"))
        if task.workspace.resolve() != workspace.resolve(strict=True):
            raise IntegrityError("Campaign workspace changed", "campaign_workspace_mismatch")
        if task.results_dir.resolve() != results.resolve():
            raise IntegrityError("Campaign results root changed", "campaign_results_mismatch")
        resolver = repository_resolver or WorkspaceGitIdentityResolver()
        repository = resolver.inspect(workspace)
        contract_event = _one_event(events, "dependency_verified", "evaluation_contract")
        contract_receipt = _one_binding(contract_event, "evaluation_contract")
        contract_value = _canonical_mapping(source.artifacts, contract_receipt)
        contract = load_evaluation_contract(
            contract_value, repository_root=Path(repository.root)
        )
        _validate_draft(task, contract)
        baseline = _path_bindings(provenance, "baseline_source")
        harness = _path_bindings(provenance, "protected_harness")
        _validate_frozen_bindings(contract, baseline, harness)
        record = KernelRunRecord(
            source.run_id,
            source.root,
            source.artifacts,
            source.journal,
            RunController.recover(
                source.run_id,
                source.journal,
                SnapshotStore(source.root / "state.snapshot.json"),
            ),
            task.dataset_split,
            task.data_visibility,
        )
        resolved = ResolvedTaskSpec(
            task=task,
            workspace=workspace.resolve(strict=True),
            editable_paths=tuple(workspace / path for path in task.editable_files),
            baseline_file_hashes=dict(contract.draft.baseline_file_hashes),
            harness_file_hashes=dict(contract.draft.harness_file_hashes),
            harness_sha256=contract.draft.harness_sha256,
            resolution_hash=contract.draft.resolution_hash,
        )
        return cls(record, task, resolved, contract, baseline, harness, resolver)

    @property
    def authorized_contract(self) -> EvaluationContractReceipt | None:
        matches = [
            event
            for event in self.record.iter_events()
            if event.event_type == "dependency_verified"
            and event.payload.get("kind") == "evaluation_contract_authorized"
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise IntegrityError(
                "Authorized evaluation contract is ambiguous",
                "evaluation_contract_ambiguous",
            )
        receipt = _one_binding(matches[0], "evaluation_contract")
        value = _canonical_mapping(self.record.artifacts, receipt)
        current = self.repository_resolver.inspect(self.resolved.workspace)
        contract = load_evaluation_contract(value, repository_root=Path(current.root))
        if contract.draft != self.draft_contract.draft or not contract.verified:
            raise IntegrityError(
                "Authorized contract does not bind the frozen draft",
                "evaluation_authority_mismatch",
            )
        return contract

    def confirm(
        self,
        expected_draft_digest: str,
        provider: FormalEvaluationAuthorityProvider | None,
    ) -> EvaluationContractReceipt | None:
        if expected_draft_digest != self.draft_contract.draft.digest:
            raise ContractError(
                "Explicit confirmation names another evaluation draft",
                "evaluation_authority_mismatch",
            )
        existing = self.authorized_contract
        if existing is not None:
            return existing
        if provider is None:
            return None
        authority = provider.consume(
            run_id=self.record.run_id,
            draft=self.draft_contract.draft,
        )
        if authority is None:
            return None
        if authority.draft_digest != self.draft_contract.draft.digest:
            raise ContractError(
                "Trusted authority receipt binds another evaluation draft",
                "evaluation_authority_mismatch",
            )
        contract = EvaluationContractReceipt(
            self.draft_contract.draft, authority, "verified", None
        )
        record_authorized_evaluation_contract(
            artifacts=self.record.artifacts,
            controller=self.record.controller,
            contract=contract,
        )
        return contract

    def validate_repository(self) -> None:
        frozen = self.draft_contract.draft.repository
        current = self.repository_resolver.inspect(self.resolved.workspace)
        if not frozen.resolved or frozen.dirty_paths:
            raise ContractError(
                "Formal execution requires a clean resolved draft repository",
                "campaign_baseline_not_clean",
            )
        if (
            not current.resolved
            or current.remote != frozen.remote
            or current.commit != frozen.commit
            or current.tree != frozen.tree
        ):
            raise IntegrityError(
                "Repository identity changed after campaign start",
                "campaign_repository_drift",
            )
        observed = {_modified_path(item) for item in current.dirty_paths}
        if observed:
            raise IntegrityError(
                "The original repository must remain byte-clean during formal execution",
                "undeclared_workspace_edit",
                {"dirty_paths": list(current.dirty_paths)},
            )

    def bind_release_candidate_baseline(
        self,
        receipt: object | None,
        *,
        reason_code: str | None,
    ) -> None:
        if _baseline_event(self.record, required=False) is not None:
            raise IntegrityError(
                "Formal campaign baseline was already bound",
                "campaign_baseline_ambiguous",
            )
        artifact = None
        digest = None
        if receipt is not None:
            document = _baseline_document(receipt)
            artifact = self.record.artifacts.put_bytes(
                canonical_json_bytes(document), media_type="application/json"
            )
            digest = _baseline_digest(receipt)
            if document.get("receipt_sha256") != digest:
                raise IntegrityError(
                    "Release candidate receipt digest is incoherent",
                    "release_identity_invalid",
                )
        verified = receipt is not None and reason_code is None
        payload: dict[str, object] = {
            "kind": "formal_release_candidate_baseline",
            "status": "verified" if verified else "unverified",
            "reason_code": reason_code,
            "release_candidate_receipt_sha256": digest,
            "artifacts": [],
        }
        if artifact is not None:
            payload["artifacts"] = [
                {"role": "release_candidate_baseline", "receipt": artifact.to_dict()}
            ]
        self.record.controller.record_domain_event(
            "dependency_verified",
            payload,
            idempotency_key="formal.release_candidate_baseline",
        )

    def revalidate_release_candidate_baseline(
        self, loader: FormalBaselineLoader | None
    ) -> str | None:
        event = _baseline_event(self.record, required=False)
        if event is None:
            return "campaign_baseline_receipt_required"
        if event.payload.get("status") != "verified":
            return str(
                event.payload.get("reason_code")
                or "campaign_baseline_receipt_required"
            )
        if loader is None:
            return "campaign_baseline_verifier_unavailable"
        try:
            receipt = _one_binding(event, "release_candidate_baseline")
            self.record.artifacts.verify(receipt)
            path = self.record.artifacts.root / receipt.relative_path
            rebuilt = loader(path)
            observed = _baseline_digest(rebuilt)
        except ApexError as error:
            return error.reason_code
        expected = event.payload.get("release_candidate_receipt_sha256")
        if observed != expected:
            return "release_identity_invalid"
        self.record.controller.record_domain_event(
            "dependency_verified",
            {
                "kind": "formal_release_candidate_baseline_revalidated",
                "status": "verified",
                "release_candidate_receipt_sha256": expected,
            },
            idempotency_key="formal.release_candidate_baseline.revalidated",
        )
        return None

    def ensure_candidate_projection(self) -> Path:
        """Create or reopen the agent-editable copy without mutating the source repo."""

        self.validate_repository()
        projection = self.record.root / "formal-work" / "candidate-projection"
        anchor = projection / "anchor"
        editable = projection / "editable"
        if projection.exists():
            CandidateWorkspace.resume(
                self.resolved, root=editable, anchor=anchor
            )
            return editable
        projection.parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".projection-", dir=projection.parent
        ) as temporary:
            staging = Path(temporary)
            staged_anchor = staging / "anchor"
            CandidateWorkspace.create(
                self.resolved, destination=staged_anchor
            )
            _overwrite(staged_anchor, self.baseline, self.record)
            _overwrite(staged_anchor, self.harness, self.record)
            CandidateWorkspace.create(
                self.resolved,
                destination=staging / "editable",
                anchor=staged_anchor,
            )
            staging.replace(projection)
        return editable

    def capture_candidate(self) -> FormalCandidateProjection:
        """Freeze the persistent agent projection into an evaluator-only copy."""

        editable = self.ensure_candidate_projection()
        anchor = editable.parent / "anchor"
        workspace = CandidateWorkspace.resume(
            self.resolved, root=editable, anchor=anchor
        )
        destination = _new_evaluator_projection(self.record.root, "compile")
        frozen = workspace.freeze(destination=destination)
        return self._projection(frozen.root, anchor, frozen.changed_files)

    def candidate_files(self, attempt_id: str) -> Mapping[str, bytes]:
        event = _attempt_event(self.record, attempt_id, "candidate_frozen")
        bindings = _path_bindings(event, "candidate")
        if set(bindings) != set(self.task.editable_files):
            raise IntegrityError(
                "Frozen candidate source set is incomplete",
                "candidate_source_integrity_failed",
            )
        return {
            path: self.record.artifacts.read_bytes(receipt)
            for path, receipt in bindings.items()
        }

    @contextmanager
    def project(
        self, files: Mapping[str, bytes], *, phase: str = "replay"
    ) -> Iterator[FormalCandidateProjection]:
        self.validate_repository()
        if set(files) != set(self.task.editable_files):
            raise IntegrityError(
                "Candidate files differ from the frozen editable scope",
                "candidate_source_integrity_failed",
            )
        editable = self.ensure_candidate_projection()
        anchor = editable.parent / "anchor"
        parent = _new_evaluator_projection(self.record.root, f"{phase}-staging")
        working = CandidateWorkspace.create(
            self.resolved, destination=parent / "working", anchor=anchor
        )
        try:
            for relative, content in files.items():
                (working.root / relative).write_bytes(content)
            frozen = working.freeze(destination=parent / "candidate")
            yield self._projection(frozen.root, anchor, frozen.changed_files)
        except Exception:
            raise

    def _projection(
        self, root: Path, anchor: Path, changed_files: tuple[str, ...]
    ) -> FormalCandidateProjection:
        digest = candidate_source_digest(root, self.task.editable_files)
        files = candidate_file_bytes(root, self.task.editable_files)
        anchored_task = replace(self.task, workspace=anchor)
        anchored = replace(
            self.resolved,
            task=anchored_task,
            workspace=anchor,
            editable_paths=tuple(anchor / path for path in self.task.editable_files),
        )
        return FormalCandidateProjection(
            root, anchor, anchored, digest, changed_files, files
        )


def _load_task(artifacts, receipt: ArtifactReceipt) -> TaskSpec:
    value = _canonical_mapping(artifacts, receipt)
    if value.get("template_authority") is not None:
        raise ContractError(
            "Chat-started formal campaigns cannot import template authority",
            "template_authority_internal_only",
        )
    return TaskSpec.from_mapping(value)


def _validate_draft(task: TaskSpec, contract: EvaluationContractReceipt) -> None:
    if contract.verified or contract.draft.task_digest != sha256_json(task.to_dict()):
        raise IntegrityError(
            "Campaign draft task and evaluation contract differ",
            "evaluation_contract_task_mismatch",
        )
    if set(dict(contract.draft.baseline_file_hashes)) != set(task.editable_files):
        raise IntegrityError(
            "Evaluation contract editable scope differs from the task",
            "evaluation_contract_task_mismatch",
        )


def _validate_frozen_bindings(contract, baseline, harness) -> None:
    expected_baseline = dict(contract.draft.baseline_file_hashes)
    expected_harness = dict(contract.draft.harness_file_hashes)
    observed_baseline = {path: receipt.digest for path, receipt in baseline.items()}
    observed_harness = {path: receipt.digest for path, receipt in harness.items()}
    if observed_baseline != expected_baseline or observed_harness != expected_harness:
        raise IntegrityError(
            "Campaign frozen bytes do not match the evaluation draft",
            "campaign_frozen_input_mismatch",
        )


def _one_event(events, event_type: str, kind: str):
    matches = [
        event
        for event in events
        if event.event_type == event_type and event.payload.get("kind") == kind
    ]
    if len(matches) != 1:
        raise IntegrityError("Campaign evidence is missing or ambiguous", "campaign_evidence_ambiguous")
    return matches[0]


def _attempt_event(record, attempt_id: str, event_type: str):
    matches = [
        event
        for event in record.iter_events()
        if event.event_type == event_type
        and event.payload.get("attempt_id") == attempt_id
    ]
    if len(matches) != 1:
        raise IntegrityError("Attempt evidence is missing or ambiguous", "attempt_evidence_ambiguous")
    return matches[0]


def _baseline_event(record, *, required: bool):
    matches = [
        event
        for event in record.iter_events()
        if event.event_type == "dependency_verified"
        and event.payload.get("kind") == "formal_release_candidate_baseline"
    ]
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise IntegrityError(
            "Formal campaign baseline is missing or ambiguous",
            "campaign_baseline_ambiguous",
        )
    return matches[0]


def _one_binding(event, role: str) -> ArtifactReceipt:
    matches = [
        binding
        for binding in event.payload.get("artifacts", ())
        if isinstance(binding, Mapping) and binding.get("role") == role
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("receipt"), dict):
        raise IntegrityError("Artifact binding is missing or ambiguous", "campaign_evidence_ambiguous")
    return ArtifactReceipt.from_dict(matches[0]["receipt"])


def _path_bindings(event, role: str) -> dict[str, ArtifactReceipt]:
    result: dict[str, ArtifactReceipt] = {}
    for binding in event.payload.get("artifacts", ()):
        if not isinstance(binding, Mapping) or binding.get("role") != role:
            continue
        path, receipt = binding.get("path"), binding.get("receipt")
        if not isinstance(path, str) or path in result or not isinstance(receipt, dict):
            raise IntegrityError("Path artifact binding is invalid", "campaign_evidence_ambiguous")
        result[path] = ArtifactReceipt.from_dict(receipt)
    return result


def _canonical_mapping(artifacts, receipt: ArtifactReceipt) -> dict[str, object]:
    content = artifacts.read_bytes(receipt)
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise IntegrityError("Campaign JSON artifact is invalid", "campaign_evidence_invalid") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != content:
        raise IntegrityError("Campaign JSON artifact is not canonical", "campaign_evidence_invalid")
    return value


def _modified_path(entry: str) -> str | None:
    if len(entry) < 4 or set(entry[:2]) - {" ", "M"} or "M" not in entry[:2]:
        return None
    return entry[3:]


def _baseline_document(receipt: object) -> dict[str, object]:
    converter = getattr(receipt, "to_dict", None)
    if not callable(converter):
        raise ContractError(
            "Trusted baseline loader returned no typed receipt",
            "release_identity_invalid",
        )
    value = converter()
    if not isinstance(value, Mapping):
        raise ContractError(
            "Trusted baseline loader returned an invalid receipt",
            "release_identity_invalid",
        )
    return dict(value)


def _baseline_digest(receipt: object) -> str:
    value = getattr(receipt, "receipt_sha256", None)
    if not isinstance(value, str) or len(value) != 64:
        raise ContractError(
            "Trusted baseline receipt has no digest",
            "release_identity_invalid",
        )
    return value


def _new_evaluator_projection(root: Path, phase: str) -> Path:
    parent = root / "formal-work" / "evaluator-projections"
    parent.mkdir(parents=True, exist_ok=True)
    container = Path(tempfile.mkdtemp(prefix=f"{phase}-", dir=parent))
    return container / "projection"


def _overwrite(
    root: Path,
    bindings: Mapping[str, ArtifactReceipt],
    record: KernelRunRecord,
) -> None:
    for relative, receipt in bindings.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(record.artifacts.read_bytes(receipt))


__all__ = [
    "FormalBaselineLoader",
    "FormalCandidateProjection",
    "FormalKernelCampaign",
]
