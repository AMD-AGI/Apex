"""Delivery-owned portable bundle evidence for standalone kernel runs."""

from __future__ import annotations

from apex.delivery import (
    KernelBundle,
    capture_portable_bundle,
    kernel_reproduction_declaration,
)

from .attempts import RunSession


def record_kernel_bundle(
    session: RunSession, *, attempt_id: str, bundle: KernelBundle
) -> None:
    evidence = capture_portable_bundle(
        session.record.artifacts,
        bundle.path,
        bundle_kind="kernel",
        expected_digest=bundle.digest,
    )
    session.record.controller.record_domain_event(
        "delivery_result",
        {
            **session.record.attempt_payload(attempt_id),
            "kind": "kernel_winner_bundle",
            "verified": True,
            "bundle_kind": evidence.bundle_kind,
            "bundle_digest": evidence.bundle_digest,
            "replication": kernel_reproduction_declaration(
                session.evaluation_contract, bundle, evidence
            ),
            "artifacts": list(evidence.artifact_bindings()),
        },
        idempotency_key=f"attempt.{attempt_id}.delivery_bundle",
    )


__all__ = ["record_kernel_bundle"]
