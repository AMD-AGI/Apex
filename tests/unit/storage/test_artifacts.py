from __future__ import annotations

from dataclasses import replace

import pytest

from apex.core import IntegrityError
from apex.storage import ArtifactStore


def test_cas_round_trip_is_idempotent_and_verified(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    receipt = store.put_bytes(b"optimized kernel", media_type="text/x-python")

    assert store.put_bytes(b"optimized kernel", media_type="text/x-python") == receipt
    assert store.read_bytes(receipt) == b"optimized kernel"
    assert receipt.relative_path.endswith(receipt.digest)


def test_receipt_read_detects_tampered_content(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    receipt = store.put_bytes(b"trusted")
    (store.root / receipt.relative_path).write_bytes(b"evil!!!")

    with pytest.raises(IntegrityError) as failure:
        store.read_bytes(receipt)
    assert failure.value.reason_code == "artifact_digest_mismatch"


def test_receipt_cannot_redirect_outside_cas_layout(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    receipt = store.put_bytes(b"trusted")

    with pytest.raises(IntegrityError) as failure:
        store.read_bytes(replace(receipt, relative_path="../elsewhere"))
    assert failure.value.reason_code == "artifact_receipt_path_mismatch"


def test_fault_before_atomic_publish_leaves_no_artifact_or_temp(tmp_path) -> None:
    def fault(stage: str) -> None:
        if stage == "before_replace":
            raise RuntimeError("simulated crash")

    store = ArtifactStore(tmp_path / "artifacts", fault_hook=fault)
    with pytest.raises(RuntimeError, match="simulated crash"):
        store.put_bytes(b"never published")

    assert list((tmp_path / "artifacts").rglob("*artifact*")) == []
    assert [path for path in (tmp_path / "artifacts").rglob("*") if path.is_file()] == []


def test_put_file_streams_into_same_content_address(tmp_path) -> None:
    source = tmp_path / "kernel.hip"
    source.write_bytes(b"extern C kernel")
    store = ArtifactStore(tmp_path / "artifacts")

    from_file = store.put_file(source, media_type="text/x-c++")
    from_bytes = store.put_bytes(source.read_bytes(), media_type="text/x-c++")

    assert from_file == from_bytes
