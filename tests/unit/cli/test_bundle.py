from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apex.cli import app
from apex.core import TaskStatus


class RecordingVerifier:
    def __init__(self, status: TaskStatus = TaskStatus.SUCCEEDED) -> None:
        self.status = status
        self.calls: list[dict[str, object]] = []

    def verify(self, **kwargs):
        self.calls.append(kwargs)
        verified = self.status is TaskStatus.SUCCEEDED
        result = SimpleNamespace(
            status=self.status,
            verified=verified,
            bundle_digest="b" * 64,
            to_dict=lambda: {
                "schema_version": 1,
                "bundle_digest": "b" * 64,
                "verified": verified,
                "status": self.status.value,
                "validation_level": (
                    "source_rebuild_verified" if verified else "none"
                ),
                "reason_code": (
                    "source_rebuild_and_second_clean_replay_verified"
                    if verified
                    else "second_clean_replay_failed"
                ),
            },
        )
        results = kwargs["results_dir"]
        final = (
            SimpleNamespace(path=results / "bundle", digest="c" * 64)
            if verified
            else None
        )
        return SimpleNamespace(
            result=result,
            result_path=results / "verification.result.json",
            verified_bundle=final,
        )


def test_bundle_verify_dispatches_kernel_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "kernel-bundle"
    root.mkdir()
    monkeypatch.setattr(app, "detect_bundle_kind", lambda path: "kernel")
    monkeypatch.setattr(
        app,
        "load_and_verify_kernel_bundle",
        lambda path, expected_digest=None: SimpleNamespace(
            task_id="rms-norm",
            path=root,
            digest="a" * 64,
            changed_files=("kernel.py",),
        ),
    )

    assert app.main(["bundle", "verify", "--bundle", str(root), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["bundle_kind"] == "kernel"
    assert result["changed_files"] == ["kernel.py"]


def test_bundle_verify_dispatches_e2e_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "e2e-bundle"
    root.mkdir()
    results = tmp_path / "verification"
    verifier = RecordingVerifier()
    monkeypatch.setattr(app, "detect_bundle_kind", lambda path: "e2e")
    monkeypatch.setattr(
        app,
        "build_application",
        lambda **kwargs: SimpleNamespace(e2e_bundle_verifier=verifier),
    )

    assert app.main(
        [
            "bundle",
            "verify",
            "--bundle",
            str(root),
            "--results",
            str(results),
            "--json",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["bundle_kind"] == "e2e"
    assert result["status"] == "succeeded"
    assert result["verified"] is True
    assert result["verified_bundle_path"] == str(results / "bundle")
    assert verifier.calls == [
        {
            "bundle_dir": root,
            "results_dir": results,
            "expected_digest": None,
        }
    ]


def test_e2e_bundle_verify_maps_typed_failure_to_nonzero_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "e2e-bundle"
    root.mkdir()
    results = tmp_path / "verification"
    verifier = RecordingVerifier(TaskStatus.VERIFICATION_FAILED)
    monkeypatch.setattr(app, "detect_bundle_kind", lambda path: "e2e")
    monkeypatch.setattr(
        app,
        "build_application",
        lambda **kwargs: SimpleNamespace(e2e_bundle_verifier=verifier),
    )

    exit_code = app.main(
        [
            "bundle",
            "verify",
            "--bundle",
            str(root),
            "--results",
            str(results),
            "--json",
        ]
    )

    assert exit_code == 3
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "verification_failed"
    assert result["verified"] is False
    assert result["verified_bundle_path"] is None


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        ((), "e2e_verification_results_required"),
        (("--results", "relative/results"), "invalid_bundle_path"),
    ],
)
def test_e2e_bundle_verify_requires_explicit_absolute_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: tuple[str, ...],
    reason: str,
) -> None:
    root = tmp_path / "e2e-bundle"
    root.mkdir()
    monkeypatch.setattr(app, "detect_bundle_kind", lambda path: "e2e")
    monkeypatch.setattr(
        app,
        "build_application",
        lambda **kwargs: pytest.fail("composition must follow path validation"),
    )

    exit_code = app.main(
        ["bundle", "verify", "--bundle", str(root), *arguments, "--json"]
    )

    assert exit_code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["reason_code"] == reason
