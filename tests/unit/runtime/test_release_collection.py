"""Fresh local release-evidence collection tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from apex.core import ContractError
from apex.execution import ProcessResult
from apex.runtime import LocalReleaseEvidenceCollector, ReleaseEvidence
from apex.runtime import (
    MagpieConfigResolutionEntryEvidence,
    build_magpie_config_resolution_evidence,
)
from apex.runtime.release_commands import (
    CPU_GATE_COMPILEALL_ARGV,
    CPU_GATE_PYTEST_ARGV,
    CPU_GATE_SCAN_ARGV,
)


_GIT = "a" * 40
_SHA = "b" * 64


class _Supervisor:
    def __init__(self, root: Path, *, pytest_exit: int = 0) -> None:
        self.root = root
        self.pytest_exit = pytest_exit
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv, **kwargs) -> ProcessResult:
        command = tuple(argv)
        self.calls.append(command)
        exit_code = 0
        stdout = ""
        if command == (".venv/bin/apex", "--help"):
            stdout = "usage: apex [-h]\n"
        elif command[:4] == (".venv/bin/python", "-s", "-c", command[3]):
            stdout = json.dumps({
                "entrypoints": ["apex.cli:main"],
                "origin": str(self.root / "src" / "apex" / "__init__.py"),
                "version": "0.1.0",
            })
        elif command == CPU_GATE_PYTEST_ARGV:
            exit_code = self.pytest_exit
            stdout = (
                "1 failed, 1066 passed in 9.00s\n"
                if exit_code
                else "1067 passed in 9.00s\n"
            )
        elif command == CPU_GATE_SCAN_ARGV:
            exit_code = 1
        return ProcessResult(
            command, exit_code, False, stdout, "", False, False, 0.1
        )


def _static() -> dict:
    return {
        "apex_checkout": {"tree": _GIT, "clean": True},
        "project": {
            "name": "amd-apex-optimizer",
            "version": "0.1.0",
            "entrypoint": "apex.cli:main",
            "import_file_sha256": "c" * 64,
        },
        "local_cli": {"status": "observed", "executable_sha256": "d" * 64},
        "locks": {
            "dependencies": "e" * 64,
            "e2e_sources": "f" * 64,
            "lm_eval_runtime": "1" * 64,
            "evaluator_policy": "0" * 64,
            "agent_templates": "2" * 64,
        },
        "magpie": {
            "commit": "5" * 40,
            "corpus_manifest_sha256": "3" * 64,
            "compatibility_ledger_sha256": "4" * 64,
            "config_count": 1,
            "configs": [
                {
                    "path": "examples/benchmarks/a.yaml",
                    "sha256": "9" * 64,
                }
            ],
        },
    }


def _receipt():
    dependency = {
        "repository": "https://example.invalid/magpie.git",
        "commit": "5" * 40,
        "tree": "6" * 40,
        "dirty": False,
    }
    source = {
        "repository": "https://example.invalid/vllm.git",
        "commit": "7" * 40,
        "tree": "8" * 40,
        "dirty": False,
    }
    return SimpleNamespace(
        lock_sha256="e" * 64,
        raw={
            "dependencies": {"magpie": dependency},
            "e2e_source_locks": {"sources": {"vllm": source}},
        },
        lm_eval_runtime=SimpleNamespace(
            lock_sha256="1" * 64, runtime_sha256=_SHA
        ),
        source_locks=SimpleNamespace(lock_sha256="f" * 64),
        evaluator_policy=SimpleNamespace(lock_sha256="0" * 64),
    )


def _resolver_evidence(root, static, receipt):
    entry = MagpieConfigResolutionEntryEvidence(
        "examples/benchmarks/a.yaml",
        "9" * 64,
        "a" * 64,
        "b" * 64,
        "config_compatible",
        "docker",
    )
    return build_magpie_config_resolution_evidence(
        magpie_commit="5" * 40,
        corpus_manifest_sha256="3" * 64,
        plan_schema="apex.magpie-main-resolved-plan/v1",
        capability_schema="apex.magpie-main-capability-receipt/v1",
        result_schema="apex.magpie-main-result-contract/v1",
        entries=(entry,),
    )


def test_collector_runs_fixed_argv_and_emits_only_local_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path.resolve()
    static = _static()
    monkeypatch.setattr(
        "apex.runtime.release_collection._static_identity",
        lambda value: copy.deepcopy(static),
    )
    verifier_calls = []

    def verifier(*, apex_root):
        verifier_calls.append(apex_root)
        return _receipt()

    supervisor = _Supervisor(root)
    evidence = LocalReleaseEvidenceCollector(
        supervisor=supervisor,
        dependency_verifier=verifier,
        config_resolution_collector=_resolver_evidence,
    ).collect(root)

    assert evidence.apex_baseline is None
    assert evidence.magpie_baseline is None
    assert evidence.cpu_gate.passed_count == 1067
    assert evidence.cpu_gate.forbidden_scan_exit_code == 1
    assert evidence.cpu_gate.forbidden_scan_clean is True
    assert evidence.cli_identity.executable_sha256 == "d" * 64
    assert len(evidence.magpie_config_resolution.entries) == 1
    assert [item.name for item in evidence.dependencies.components] == ["magpie", "vllm"]
    assert ReleaseEvidence.from_dict(evidence.to_dict()) == evidence
    assert verifier_calls == [root]
    assert supervisor.calls[-3:] == [
        CPU_GATE_PYTEST_ARGV,
        CPU_GATE_COMPILEALL_ARGV,
        CPU_GATE_SCAN_ARGV,
    ]


def test_failed_pytest_is_factual_failed_evidence(tmp_path: Path, monkeypatch) -> None:
    static = _static()
    monkeypatch.setattr(
        "apex.runtime.release_collection._static_identity",
        lambda value: copy.deepcopy(static),
    )
    evidence = LocalReleaseEvidenceCollector(
        supervisor=_Supervisor(tmp_path.resolve(), pytest_exit=1),
        dependency_verifier=lambda **kwargs: _receipt(),
        config_resolution_collector=_resolver_evidence,
    ).collect(tmp_path)

    assert evidence.cpu_gate.pytest_exit_code == 1
    assert evidence.cpu_gate.passed_count == 1066
    assert evidence.cpu_gate.failed_count == 1


def test_dirty_or_changing_source_never_produces_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    dirty = _static()
    dirty["apex_checkout"]["clean"] = False
    monkeypatch.setattr(
        "apex.runtime.release_collection._static_identity",
        lambda value: copy.deepcopy(dirty),
    )
    with pytest.raises(ContractError, match="clean Apex checkout"):
        LocalReleaseEvidenceCollector(
            supervisor=_Supervisor(tmp_path),
            dependency_verifier=lambda **kwargs: pytest.fail("must not verify"),
        ).collect(tmp_path)

    before = _static()
    after = copy.deepcopy(before)
    after["project"]["version"] = "0.1.1"
    identities = iter((before, after))
    monkeypatch.setattr(
        "apex.runtime.release_collection._static_identity",
        lambda value: copy.deepcopy(next(identities)),
    )
    with pytest.raises(ContractError, match="changed during"):
        LocalReleaseEvidenceCollector(
            supervisor=_Supervisor(tmp_path),
            dependency_verifier=lambda **kwargs: _receipt(),
            config_resolution_collector=_resolver_evidence,
        ).collect(tmp_path)
