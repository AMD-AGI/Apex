"""Persist exact derived benchmark configs for one candidate deployment."""

from __future__ import annotations

from pathlib import Path

from apex.core import IntegrityError
from apex.storage import ArtifactStore

from .services import CandidateDeployment


def persist_deployment_configs(
    store: ArtifactStore,
    run_root: Path,
    result: CandidateDeployment,
) -> tuple[dict[str, object], ...]:
    if not result.deployed:
        return ()
    digests = result.config_sha256
    if digests is None:
        raise IntegrityError(
            "Successful deployment lacks config digests",
            "deployment_config_receipt_missing",
        )
    configured = (
        ("delivery_measurement_config", result.measurement_config, digests.measurement),
        ("delivery_diagnostic_config", result.diagnostic_config, digests.diagnostic),
        ("delivery_replay_config", result.replay_config, digests.replay),
    )
    bindings: list[dict[str, object]] = []
    root = run_root.resolve()
    for role, path, expected in configured:
        _validate_path(path, root)
        receipt = store.put_file(path, media_type="application/yaml")
        if receipt.digest != expected:
            raise IntegrityError(
                "Deployment config changed before it was recorded",
                "deployment_config_digest_mismatch",
            )
        bindings.append({"role": role, "receipt": receipt.to_dict()})
    return tuple(bindings)


def _validate_path(path: Path, root: Path) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise IntegrityError(
            "Deployment config is missing or unsafe",
            "invalid_deployment_config",
        )
    try:
        path.resolve().relative_to(root)
    except ValueError as error:
        raise IntegrityError(
            "Deployment config is outside the immutable run root",
            "invalid_deployment_config",
        ) from error


__all__ = ["persist_deployment_configs"]
