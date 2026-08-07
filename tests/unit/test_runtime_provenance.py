from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from apex.core import IntegrityError
from apex.execution import ProcessResult
from apex.runtime import ProvenanceResolver


class ImageOnlySupervisor:
    def __init__(self, image_id: str | None = "sha256:" + "a" * 64) -> None:
        self.image_id = image_id
        self.environments: list[dict[str, str]] = []

    def run(self, argv, *, cwd, environment, timeout_seconds, stdin_text=None):
        self.environments.append(dict(environment))
        if tuple(argv[:3]) == ("docker", "image", "inspect") and self.image_id:
            stdout = json.dumps(
                [
                    {
                        "Id": self.image_id,
                        "RepoDigests": ["example@sha256:" + "b" * 64],
                        "Config": {"Labels": {"version": "1"}},
                    }
                ]
            )
            code = 0
        else:
            stdout = ""
            code = 1
        return ProcessResult(tuple(argv), code, False, stdout, "", False, False, 0.01)


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        """benchmark:
  framework: vllm
  model: Qwen/example
  envs:
    VLLM_ROCM_USE_AITER: 1
  docker_image: example:v1
""",
        encoding="utf-8",
    )
    return path


def test_tag_only_input_is_observed_but_remains_partial(tmp_path: Path) -> None:
    provenance = ProvenanceResolver(ImageOnlySupervisor()).resolve(
        _config(tmp_path), gpu_arch="gfx950"
    )
    assert provenance.container.image_id == "sha256:" + "a" * 64
    assert provenance.active_components == ("vllm", "aiter")
    assert provenance.status == "partial"
    assert "model_revision" in provenance.missing_evidence
    assert "source_lock:vllm" in provenance.missing_evidence
    assert not provenance.source_delivery_ready


def test_missing_local_image_is_unresolved_not_an_intake_crash(tmp_path: Path) -> None:
    provenance = ProvenanceResolver(ImageOnlySupervisor(None)).resolve(
        _config(tmp_path), gpu_arch="gfx950"
    )
    assert provenance.status == "unresolved"
    assert "image_digest" in provenance.missing_evidence


def test_provenance_docker_environment_has_explicit_connection_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/user/1000/docker.sock")
    monkeypatch.setenv("DOCKER_CONTEXT", "test-context")
    monkeypatch.setenv("DOCKER_CONFIG", "/home/test/.docker")
    monkeypatch.setenv("DOCKER_AUTH_CONFIG", "registry-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "agent-secret")
    monkeypatch.setenv("BASH_ENV", "/tmp/injected-startup")
    supervisor = ImageOnlySupervisor()

    ProvenanceResolver(supervisor).resolve(_config(tmp_path), gpu_arch="gfx950")

    environment = supervisor.environments[0]
    assert environment["DOCKER_HOST"] == "unix:///run/user/1000/docker.sock"
    assert environment["DOCKER_CONTEXT"] == "test-context"
    assert environment["DOCKER_CONFIG"] == "/home/test/.docker"
    assert "DOCKER_AUTH_CONFIG" not in environment
    assert "OPENAI_API_KEY" not in environment
    assert "BASH_ENV" not in environment
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_CONFIG_SYSTEM"] == "/dev/null"
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_asserted_source_commit_mismatch_fails_closed(tmp_path: Path) -> None:
    repository = tmp_path / "source"
    repository.mkdir()
    subprocess.run(("git", "init", "-q", str(repository)), check=True)
    subprocess.run(("git", "-C", str(repository), "config", "user.email", "test@example.com"), check=True)
    subprocess.run(("git", "-C", str(repository), "config", "user.name", "Test"), check=True)
    subprocess.run(("git", "-C", str(repository), "remote", "add", "origin", "https://example.com/vllm.git"), check=True)
    (repository / "kernel.py").write_text("pass\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(repository), "add", "kernel.py"), check=True)
    subprocess.run(("git", "-C", str(repository), "commit", "-q", "-m", "baseline"), check=True)

    with pytest.raises(IntegrityError) as failure:
        ProvenanceResolver().resolve(
            _config(tmp_path),
            gpu_arch="gfx950",
            hints={
                "model_revision": "revision-1",
                "source_repositories": [
                    {"name": "vllm", "path": str(repository), "commit": "0" * 40}
                ],
            },
        )
    assert failure.value.reason_code == "repository_commit_mismatch"
