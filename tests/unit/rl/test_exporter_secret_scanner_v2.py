from __future__ import annotations

import json
from pathlib import Path

import pytest

from apex.core import IntegrityError
from apex.rl import DatasetExporter, EpisodeGraphMaterializer

from .conftest import append_event, artifact_binding


_SYNTHETIC_CREDENTIAL = "test-only-credential-1234"


def _graph(canonical_run):
    return EpisodeGraphMaterializer(
        canonical_run["journal"], canonical_run["artifacts"]
    ).materialize(canonical_run["run_id"])


def _attach_artifact(canonical_run, content: bytes, media_type: str):
    receipt = canonical_run["artifacts"].put_bytes(content, media_type=media_type)
    append_event(
        canonical_run["journal"],
        canonical_run["run_id"],
        "agent_message",
        {
            "attempt_id": "attempt-1",
            "artifacts": [artifact_binding("agent_message", receipt)],
        },
        "attempt-1-secret-scanner-v2",
    )
    return receipt


def _export(canonical_run, output_dir: Path):
    return DatasetExporter(canonical_run["artifacts"]).export(
        _graph(canonical_run), output_dir
    )


def _assert_secret_rejected(canonical_run, tmp_path: Path) -> None:
    output_dir = tmp_path / "rejected"
    with pytest.raises(IntegrityError) as error:
        _export(canonical_run, output_dir)
    assert error.value.reason_code == "dataset_secret_detected"
    assert not output_dir.exists()


def _large_minified_output(*, strong_tail: bool = False) -> bytes:
    fragment = (
        "const x={password:callback,authorization=handler,secret:resolver};"
        'const y={"password":"callback","client_secret":"resolver"};'
        "x.password=nextHandler;"
    )
    output = "vendored/static/bundle.js:2:" + fragment * 16_000
    output += "\n... 400000 characters truncated ...\n" + fragment * 64
    if strong_tail:
        output += " sk-" + "x" * 20
    payload = json.dumps(
        {"type": "tool_result", "item": {"aggregated_output": output}},
        separators=(",", ":"),
    )
    return (payload + "\n").encode()


def test_minified_source_false_positive_exports_exact_observation(
    canonical_run, tmp_path: Path
) -> None:
    content = _large_minified_output()
    assert len(content) > 1024 * 1024
    receipt = _attach_artifact(canonical_run, content, "application/x-ndjson")

    result = _export(canonical_run, tmp_path / "export")

    assert result.record_count == 2
    document = json.loads((tmp_path / "export" / "dataset.json").read_text())
    record = next(
        item for item in document["records"] if item["attempt_id"] == "attempt-1"
    )
    assert record["exporter_version"] == "apex_rl_export_v3"
    artifact = next(
        item
        for item in record["artifacts_by_role"]["agent_message"]
        if item["receipt"]["digest"] == receipt.digest
    )
    assert artifact["encoding"] == "utf-8"
    assert artifact["content"].encode() == content


def test_strong_token_near_tail_of_large_minified_output_fails_before_write(
    canonical_run, tmp_path: Path
) -> None:
    content = _large_minified_output(strong_tail=True)
    assert len(content) > 1024 * 1024
    _attach_artifact(canonical_run, content, "application/x-ndjson")

    _assert_secret_rejected(canonical_run, tmp_path)


@pytest.mark.parametrize(
    ("content", "media_type"),
    [
        pytest.param(
            f"api_key={_SYNTHETIC_CREDENTIAL}".encode(),
            "text/plain",
            id="line-assignment",
        ),
        pytest.param(
            f"  DB_PASSWORD: {_SYNTHETIC_CREDENTIAL}".encode(),
            "text/plain",
            id="indented-prefixed-assignment",
        ),
        pytest.param(
            f"export client_secret={_SYNTHETIC_CREDENTIAL}".encode(),
            "text/plain",
            id="export-assignment",
        ),
        pytest.param(
            json.dumps(
                {"output": json.dumps({"password": _SYNTHETIC_CREDENTIAL})}
            ).encode(),
            "application/json",
            id="quoted-field-in-observation",
        ),
        pytest.param(
            f"'password': '{_SYNTHETIC_CREDENTIAL}'".encode(),
            "text/plain",
            id="quoted-yaml-field",
        ),
        pytest.param(
            f"--password {_SYNTHETIC_CREDENTIAL}".encode(),
            "text/plain",
            id="space-option",
        ),
        pytest.param(
            f"--api-key={_SYNTHETIC_CREDENTIAL}".encode(),
            "text/plain",
            id="equals-option",
        ),
        pytest.param(
            json.dumps({"nested": {"password": _SYNTHETIC_CREDENTIAL}}).encode(),
            "application/json",
            id="structured-key",
        ),
        pytest.param(
            json.dumps(
                {"argv": ["tool", "--api-key", _SYNTHETIC_CREDENTIAL]}
            ).encode(),
            "application/json",
            id="split-argv-option",
        ),
        pytest.param(
            (json.dumps({"type": "benign"}) + "\n"
             f"api_key={_SYNTHETIC_CREDENTIAL}\n").encode(),
            "application/x-ndjson",
            id="malformed-ndjson-raw-fallback",
        ),
        pytest.param(
            f"api_key={_SYNTHETIC_CREDENTIAL}\n{{malformed".encode(),
            "application/json",
            id="malformed-json-raw-fallback",
        ),
    ],
)
def test_synthetic_secret_forms_fail_before_write(
    canonical_run,
    tmp_path: Path,
    content: bytes,
    media_type: str,
) -> None:
    _attach_artifact(canonical_run, content, media_type)

    _assert_secret_rejected(canonical_run, tmp_path)


def test_exact_empty_and_redacted_sentinels_are_preserved(
    canonical_run, tmp_path: Path
) -> None:
    document = {
        "api_key": "eMpTy",
        "password": "[REDACTED]",
        "client_secret": "'EMPTY'",
        "access_token": "",
        "observations": [
            "api_key=EMPTY",
            "password='[REDACTED]'",
            '{"password":"EMPTY"}',
            "--api-key EMPTY",
        ],
        "argv": ["tool", "--api-key", '"EMPTY"', "--password", "'[REDACTED]'"],
    }
    content = json.dumps(document, separators=(",", ":")).encode()
    receipt = _attach_artifact(canonical_run, content, "application/json")

    _export(canonical_run, tmp_path / "export")

    exported = json.loads((tmp_path / "export" / "dataset.json").read_text())
    record = next(
        item for item in exported["records"] if item["attempt_id"] == "attempt-1"
    )
    artifact = next(
        item
        for item in record["artifacts_by_role"]["agent_message"]
        if item["receipt"]["digest"] == receipt.digest
    )
    assert artifact["content"].encode() == content


@pytest.mark.parametrize(
    ("content", "media_type"),
    [
        (b"api_key=EMPTYsuffix", "text/plain"),
        (b"password=[REDACTED]suffix", "text/plain"),
        (b'{"api_key":"EMPTYsuffix"}', "application/json"),
        (b'{"password":"[REDACTED]suffix"}', "application/json"),
    ],
)
def test_sentinel_prefix_is_not_an_allowlist(
    canonical_run,
    tmp_path: Path,
    content: bytes,
    media_type: str,
) -> None:
    _attach_artifact(canonical_run, content, media_type)

    _assert_secret_rejected(canonical_run, tmp_path)
