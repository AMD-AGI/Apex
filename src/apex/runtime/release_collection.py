"""Fresh local dependency, CPU-gate, and installed-CLI release evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from apex.core import ContractError
from apex.execution import ProcessResult, SubprocessSupervisor, build_subprocess_environment

from .receipt import DependencyReceipt, verify_runtime_dependencies
from .dependencies import load_lock
from .magpie_corpus import load_magpie_corpus_manifest
from .magpie_config import (
    CAPABILITY_SCHEMA,
    PLAN_SCHEMA,
    RESULT_SCHEMA,
    MagpieMainConfigAdapter,
)
from .release_candidate import (
    KERNEL_SHOWCASES,
    REQUIRED_IMAGES,
    REQUIRED_QUALIFICATIONS,
    REQUIRED_SHOWCASES,
)
from .release_commands import (
    CPU_GATE_COMPILEALL_ARGV,
    CPU_GATE_PYTEST_ARGV,
    CPU_GATE_SCAN_ARGV,
)
from .release_evidence import (
    CliIdentityEvidence,
    CpuGateEvidence,
    DependencyVerificationEvidence,
    MagpieConfigResolutionEntryEvidence,
    MagpieConfigResolutionEvidence,
    ReleaseEvidence,
    VerifiedComponentEvidence,
    build_magpie_config_resolution_evidence,
)
from .release_static import collect_release_static_identity


_PYTEST_SUMMARY = re.compile(r"\b(?P<count>\d+)\s+(?P<kind>passed|failed|errors?)\b")
_CLI_PROBE_CODE = """
import importlib.metadata
import importlib.util
import json
import sys

distribution, module = sys.argv[1:3]
spec = importlib.util.find_spec(module)
dist = importlib.metadata.distribution(distribution)
entrypoints = sorted(
    item.value
    for item in dist.entry_points
    if item.group == "console_scripts" and item.name == "apex"
)
print(json.dumps({
    "entrypoints": entrypoints,
    "origin": spec.origin if spec is not None else None,
    "version": dist.version,
}, sort_keys=True))
""".strip()


class LocalReleaseEvidenceCollector:
    """Run only fixed local gates and refuse evidence for a mutable source tree."""

    def __init__(
        self,
        *,
        supervisor: SubprocessSupervisor | None = None,
        dependency_verifier: Callable[..., DependencyReceipt] | None = None,
        config_resolution_collector: Callable[
            [Path, Mapping[str, Any], DependencyReceipt], MagpieConfigResolutionEvidence
        ]
        | None = None,
    ) -> None:
        self._supervisor = supervisor or SubprocessSupervisor(max_output_bytes=32 * 1024 * 1024)
        self._dependency_verifier = dependency_verifier or verify_runtime_dependencies
        self._config_resolution_collector = (
            config_resolution_collector or _magpie_config_resolution_evidence
        )
        self._environment = build_subprocess_environment({})

    def collect(self, apex_root: Path) -> ReleaseEvidence:
        root = _root(apex_root)
        before = _static_identity(root)
        _require_clean(before)
        receipt = self._dependency_verifier(apex_root=root)
        dependency = _dependency_evidence(before, receipt)
        magpie_resolution = self._config_resolution_collector(root, before, receipt)
        cli = self._cli_identity(root, before)
        cpu_gate = self._cpu_gate(root, before)
        after = _static_identity(root)
        if before != after or not after["apex_checkout"]["clean"]:
            raise ContractError(
                "Release source identity changed during local evidence collection",
                "release_source_changed_during_gate",
            )
        return ReleaseEvidence(
            dependencies=dependency,
            magpie_config_resolution=magpie_resolution,
            cpu_gate=cpu_gate,
            cli_identity=cli,
        )

    def _cli_identity(
        self, root: Path, static: Mapping[str, Any]
    ) -> CliIdentityEvidence:
        project = _mapping(static.get("project"), "project identity")
        installed = _mapping(static.get("local_cli"), "installed CLI identity")
        if installed.get("status") != "observed":
            raise ContractError(
                "Installed Apex CLI identity is unavailable",
                "release_cli_identity_unavailable",
            )
        help_result = self._run((".venv/bin/apex", "--help"), root, 30)
        if help_result.exit_code != 0 or not help_result.stdout.startswith("usage: apex"):
            raise ContractError(
                "Installed Apex CLI help probe failed", "release_cli_probe_failed"
            )
        probe = self._run(
            (
                ".venv/bin/python", "-s", "-c", _CLI_PROBE_CODE,
                str(project["name"]), "apex",
            ),
            root,
            30,
        )
        observed = _json_mapping(probe.stdout, "installed CLI probe")
        expected_origin = root / "src" / "apex" / "__init__.py"
        if (
            probe.exit_code != 0
            or observed != {
                "entrypoints": [project["entrypoint"]],
                "origin": str(expected_origin),
                "version": project["version"],
            }
        ):
            raise ContractError(
                "Installed Apex CLI import identity differs", "release_cli_probe_failed"
            )
        return CliIdentityEvidence(
            apex_tree=str(static["apex_checkout"]["tree"]),
            project_version=str(project["version"]),
            entrypoint=str(project["entrypoint"]),
            import_module="apex",
            executable_sha256=str(installed["executable_sha256"]),
            import_file_sha256=str(project["import_file_sha256"]),
        )

    def _cpu_gate(
        self, root: Path, static: Mapping[str, Any]
    ) -> CpuGateEvidence:
        pytest_result = self._run(CPU_GATE_PYTEST_ARGV, root, 1_200)
        compile_result = self._run(CPU_GATE_COMPILEALL_ARGV, root, 300)
        scan_result = self._run(CPU_GATE_SCAN_ARGV, root, 60)
        if scan_result.exit_code not in {0, 1}:
            raise ContractError(
                "Forbidden-source scan did not complete", "release_cpu_gate_unresolved"
            )
        passed, failed = _pytest_counts(pytest_result)
        locks = _mapping(static.get("locks"), "release locks")
        magpie = _mapping(static.get("magpie"), "Magpie identity")
        return CpuGateEvidence(
            apex_tree=str(static["apex_checkout"]["tree"]),
            dependencies_lock_sha256=str(locks["dependencies"]),
            e2e_source_lock_sha256=str(locks["e2e_sources"]),
            corpus_manifest_sha256=str(magpie["corpus_manifest_sha256"]),
            compatibility_ledger_sha256=str(magpie["compatibility_ledger_sha256"]),
            pytest_argv=pytest_result.argv,
            pytest_exit_code=int(pytest_result.exit_code),
            passed_count=passed,
            failed_count=failed,
            compileall_argv=compile_result.argv,
            compileall_exit_code=int(compile_result.exit_code),
            forbidden_scan_argv=scan_result.argv,
            forbidden_scan_exit_code=int(scan_result.exit_code),
            forbidden_scan_clean=scan_result.exit_code == 1,
        )

    def _run(
        self, argv: tuple[str, ...], root: Path, timeout_seconds: int
    ) -> ProcessResult:
        try:
            result = self._supervisor.run(
                argv,
                cwd=root,
                environment=self._environment,
                timeout_seconds=timeout_seconds,
            )
        except OSError as error:
            raise ContractError(
                f"Release gate command could not start: {argv[0]}",
                "release_gate_command_unavailable",
            ) from error
        if (
            result.exit_code is None
            or result.timed_out
            or result.stdout_truncated
            or result.stderr_truncated
            or not result.cleanup_succeeded
        ):
            raise ContractError(
                f"Release gate command did not complete cleanly: {argv[0]}",
                "release_gate_command_unresolved",
            )
        return result


def collect_local_release_evidence(apex_root: Path) -> ReleaseEvidence:
    """Collect the local baseline subset without fetching or running GPU work."""

    return LocalReleaseEvidenceCollector().collect(apex_root)


def _dependency_evidence(
    static: Mapping[str, Any], receipt: DependencyReceipt
) -> DependencyVerificationEvidence:
    locks = _mapping(static.get("locks"), "release locks")
    runtime = receipt.lm_eval_runtime
    sources = receipt.source_locks
    evaluator = receipt.evaluator_policy
    if (
        runtime is None
        or sources is None
        or evaluator is None
        or receipt.lock_sha256 != locks["dependencies"]
        or sources.lock_sha256 != locks["e2e_sources"]
        or runtime.lock_sha256 != locks["lm_eval_runtime"]
        or evaluator.lock_sha256 != locks["evaluator_policy"]
    ):
        raise ContractError(
            "Runtime dependency receipt differs from release locks",
            "release_dependency_identity_mismatch",
        )
    raw_dependencies = _mapping(receipt.raw.get("dependencies"), "dependencies")
    raw_sources = _mapping(
        _mapping(receipt.raw.get("e2e_source_locks"), "source locks").get("sources"),
        "source components",
    )
    components = tuple(sorted(
        (
            _verified_component(name, item)
            for name, item in (*raw_dependencies.items(), *raw_sources.items())
        ),
        key=lambda item: item.name,
    ))
    return DependencyVerificationEvidence(
        apex_tree=str(static["apex_checkout"]["tree"]),
        dependencies_lock_sha256=str(locks["dependencies"]),
        e2e_source_lock_sha256=str(locks["e2e_sources"]),
        lm_eval_runtime_lock_sha256=str(locks["lm_eval_runtime"]),
        evaluator_policy_lock_sha256=str(locks["evaluator_policy"]),
        agent_templates_lock_sha256=str(locks["agent_templates"]),
        lm_eval_runtime_sha256=runtime.runtime_sha256,
        all_imports_exact=True,
        components=components,
    )


def _magpie_config_resolution_evidence(
    root: Path,
    static: Mapping[str, Any],
    receipt: DependencyReceipt,
) -> MagpieConfigResolutionEvidence:
    lock = load_lock(root / "scripts" / "dependencies.lock.json")
    corpus = load_magpie_corpus_manifest(lock.magpie_corpus_manifest)
    adapter = MagpieMainConfigAdapter(receipt)
    entries: list[MagpieConfigResolutionEntryEvidence] = []
    expected_static = _mapping(static.get("magpie"), "Magpie identity")
    if expected_static.get("corpus_manifest_sha256") != corpus.manifest_sha256:
        raise ContractError(
            "Magpie config corpus differs from release identity",
            "release_magpie_config_resolution_identity_mismatch",
        )
    for item in corpus.files:
        resolved = adapter.resolve(receipt.root("magpie") / item.path)
        if resolved.config_sha256 != item.sha256:
            raise ContractError(
                f"Magpie config resolution digest differs: {item.path}",
                "release_magpie_config_resolution_identity_mismatch",
            )
        entries.append(
            MagpieConfigResolutionEntryEvidence(
                item.path,
                item.sha256,
                str(resolved.plan["plan_sha256"]),
                str(resolved.capability_receipt["receipt_sha256"]),
                resolved.status,
                str(resolved.capability_receipt["run_mode"]),
                str(resolved.plan["lifecycle"]),
            )
        )
    return build_magpie_config_resolution_evidence(
        magpie_commit=str(receipt.commits.get("magpie", "")),
        corpus_manifest_sha256=corpus.manifest_sha256,
        plan_schema=PLAN_SCHEMA,
        capability_schema=CAPABILITY_SCHEMA,
        result_schema=RESULT_SCHEMA,
        entries=entries,
    )


def _verified_component(name: object, value: object) -> VerifiedComponentEvidence:
    item = _mapping(value, f"dependency component {name}")
    try:
        return VerifiedComponentEvidence(
            name=str(name),
            repository=str(item["repository"]),
            commit=str(item["commit"]),
            tree=str(item["tree"]),
            clean=item.get("dirty") is False,
        )
    except KeyError as error:
        raise ContractError(
            f"Dependency component is incomplete: {name}",
            "release_dependency_identity_mismatch",
        ) from error


def _pytest_counts(result: ProcessResult) -> tuple[int, int]:
    for line in reversed(result.stdout.splitlines()):
        matches = tuple(_PYTEST_SUMMARY.finditer(line))
        if not matches:
            continue
        counts = {"passed": 0, "failed": 0, "error": 0, "errors": 0}
        for match in matches:
            counts[match.group("kind")] += int(match.group("count"))
        return counts["passed"], counts["failed"] + counts["error"] + counts["errors"]
    if result.exit_code == 0:
        raise ContractError(
            "Successful pytest gate lacks a parseable summary",
            "release_cpu_gate_unresolved",
        )
    return 0, 0


def _static_identity(root: Path) -> dict[str, Any]:
    return collect_release_static_identity(
        root,
        kernel_showcases=KERNEL_SHOWCASES,
        required_showcases=REQUIRED_SHOWCASES,
        required_images=REQUIRED_IMAGES,
        required_qualifications=REQUIRED_QUALIFICATIONS,
    )


def _require_clean(static: Mapping[str, Any]) -> None:
    checkout = _mapping(static.get("apex_checkout"), "Apex checkout")
    if checkout.get("clean") is not True:
        raise ContractError(
            "Local release evidence requires a clean Apex checkout",
            "release_source_dirty",
        )


def _root(value: Path) -> Path:
    selected = value.expanduser()
    if selected.is_symlink():
        raise ContractError("Apex root cannot be a symlink", "unsafe_release_root")
    try:
        root = selected.resolve(strict=True)
    except OSError as error:
        raise ContractError("Apex root does not exist", "release_root_missing") from error
    if not root.is_dir():
        raise ContractError("Apex root is not a directory", "invalid_release_root")
    return root


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field} is invalid", "release_evidence_collection_failed")
    return value


def _json_mapping(value: str, field: str) -> Mapping[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise ContractError(
            f"{field} is not JSON", "release_evidence_collection_failed"
        ) from error
    return _mapping(result, field)


__all__ = ["LocalReleaseEvidenceCollector", "collect_local_release_evidence"]
