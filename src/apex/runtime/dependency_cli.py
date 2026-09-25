"""Command-line composition for pinned runtime dependency management."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dependencies import (
    DependencyBootstrapper,
    PythonEnvironment,
    load_lock,
)
from .evaluator_lock import load_evaluator_policy_lock
from .lm_eval_lock import load_lm_eval_runtime_lock
from .lm_eval_prepare import LmEvalRuntimePreparer
from .lm_eval_runtime import default_lm_eval_runtime_root, verify_lm_eval_runtime
from .repositories import BootstrapError, RepositoryResolver
from .source_locks import (
    SourceLockManager,
    default_source_checkout_root,
    default_source_lock_path,
    load_source_lock,
)


def build_parser(apex_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install or verify Apex's pinned repositories and deterministic "
            "E2E source checkouts and lm-eval runtime."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("install", "verify", "prepare-runtime", "verify-runtime"),
        default="install",
    )
    parser.add_argument(
        "--lock", type=Path, default=apex_root / "scripts" / "dependencies.lock.json"
    )
    parser.add_argument("--sibling-root", type=Path, default=apex_root.parent)
    parser.add_argument(
        "--checkout-root",
        type=Path,
        default=apex_root / ".cache" / "apex-dependencies",
    )
    parser.add_argument("--venv", type=Path, default=apex_root / ".venv")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--magpie-root", type=Path)
    parser.add_argument("--tracelens-root", type=Path)
    parser.add_argument("--inferencex-root", type=Path)
    parser.add_argument(
        "--e2e-source-lock", type=Path, default=default_source_lock_path(apex_root)
    )
    parser.add_argument(
        "--source-lock-root", type=Path, default=default_source_checkout_root()
    )
    parser.add_argument("--vllm-source-root", type=Path)
    parser.add_argument("--aiter-source-root", type=Path)
    parser.add_argument(
        "--lm-eval-lock",
        type=Path,
        default=apex_root / "scripts" / "lm_eval_runtime.lock.json",
    )
    parser.add_argument(
        "--evaluator-policy-lock",
        type=Path,
        default=apex_root / "scripts" / "evaluator_policy.lock.json",
    )
    parser.add_argument("--lm-eval-runtime", type=Path)
    parser.add_argument("--artifact-cache", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def emit_result(result: Mapping[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"Apex dependencies: {result['status']}")
    print(f"  lock:   {result['lock']} ({result['lock_sha256'][:12]})")
    print(f"  python: {result['python']} ({result['venv_action']})")
    for value in result["dependencies"].values():
        print(
            f"  {value['name']}: {value['action']} at {value['root']} "
            f"({value['commit'][:12]}, {value['resolution']})"
        )
    corpus = result.get("magpie_corpus")
    if isinstance(corpus, Mapping):
        print(
            f"  Magpie corpus: {corpus['summary']['config_count']} configs "
            f"(tree {str(corpus['benchmark_tree'])[:12]}, "
            f"manifest {str(corpus['manifest_sha256'])[:12]})"
        )
    source_locks = result.get("e2e_source_locks")
    if isinstance(source_locks, Mapping):
        for value in source_locks["sources"].values():
            print(
                f"  {value['name']} source: {value.get('action', 'verified')} "
                f"at {value['root']} ({value['commit'][:12]}, "
                f"tree {value['tree'][:12]}, {value['resolution']})"
            )
    runtime = result.get("lm_eval_runtime")
    if isinstance(runtime, Mapping):
        print(
            f"  lm-eval runtime: {runtime['status']} at {runtime['path']} "
            f"({str(runtime['sha256'])[:12]})"
        )
    evaluator = result.get("evaluator_policy")
    if isinstance(evaluator, Mapping):
        print(
            f"  evaluator policy: {evaluator['policy_id']} "
            f"({str(evaluator['lock_sha256'])[:12]})"
        )


def _explicit_roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        key: value
        for key, value in (
            ("magpie", args.magpie_root),
            ("tracelens", args.tracelens_root),
            ("inferencex", args.inferencex_root),
        )
        if value is not None
    }


def _runtime_root(args: argparse.Namespace, apex_root: Path, lock: Any) -> Path:
    explicit = args.lm_eval_runtime or os.environ.get("APEX_LM_EVAL_RUNTIME")
    if explicit:
        return Path(explicit).expanduser().resolve()
    return default_lm_eval_runtime_root(apex_root, lock)


def _source_roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        key: value
        for key, value in (
            ("vllm", args.vllm_source_root),
            ("aiter", args.aiter_source_root),
        )
        if value is not None
    }


def _source_manager(args: argparse.Namespace) -> SourceLockManager:
    lock = load_source_lock(args.e2e_source_lock.expanduser().resolve())
    return SourceLockManager(
        lock,
        sibling_root=args.sibling_root,
        checkout_root=args.source_lock_root,
        explicit_roots=_source_roots(args),
        offline=args.offline,
    )


def _attach_source_locks(
    result: dict[str, Any], manager: SourceLockManager, *, action: str
) -> None:
    if action == "plan":
        result["e2e_source_locks"] = manager.plan()
        return
    receipt = manager.materialize() if action == "materialize" else manager.verify()
    result["e2e_source_locks"] = receipt.to_dict()


def _runtime_result(receipt: Any, status: str) -> dict[str, Any]:
    result = receipt.to_dict()
    result["status"] = status
    return result


def _attach_evaluator_policy(
    result: dict[str, Any], lock_path: Path
) -> None:
    dependencies = result.get("dependencies")
    if not isinstance(dependencies, Mapping):
        raise BootstrapError("dependency result lacks InferenceX")
    inferencex = dependencies.get("inferencex")
    if not isinstance(inferencex, Mapping):
        raise BootstrapError("dependency result lacks InferenceX")
    receipt = load_evaluator_policy_lock(
        lock_path.expanduser().resolve(),
        inferencex_root=Path(str(inferencex["root"])),
    )
    result["evaluator_policy"] = receipt.to_dict()


def _run(args: argparse.Namespace, apex_root: Path) -> Mapping[str, Any]:
    lock = load_lock(args.lock.expanduser().resolve())
    resolver = RepositoryResolver(
        sibling_root=args.sibling_root.expanduser(),
        checkout_root=args.checkout_root.expanduser(),
        explicit_roots=_explicit_roots(args),
        offline=args.offline,
        dry_run=args.dry_run,
    )
    environment = PythonEnvironment(args.venv, args.python, args.offline)
    bootstrapper = DependencyBootstrapper(lock, resolver, environment)
    sources = _source_manager(args)
    if args.command == "install":
        result = bootstrapper.install(dry_run=args.dry_run)
        _attach_source_locks(
            result, sources, action="plan" if args.dry_run else "materialize"
        )
        if not args.dry_run:
            _attach_evaluator_policy(result, args.evaluator_policy_lock)
        return result
    result = bootstrapper.verify()
    _attach_source_locks(
        result,
        sources,
        action="materialize" if args.command == "prepare-runtime" else "verify",
    )
    _attach_evaluator_policy(result, args.evaluator_policy_lock)
    runtime_lock = load_lm_eval_runtime_lock(args.lm_eval_lock.expanduser().resolve())
    runtime_root = _runtime_root(args, apex_root, runtime_lock)
    if args.command == "verify":
        if runtime_root.exists():
            receipt = verify_lm_eval_runtime(runtime_root, runtime_lock)
            result["lm_eval_runtime"] = _runtime_result(receipt, "verified")
        return result
    if args.command == "verify-runtime":
        receipt = verify_lm_eval_runtime(runtime_root, runtime_lock)
        result["lm_eval_runtime"] = _runtime_result(receipt, "verified")
        return result
    inferencex = Path(str(result["dependencies"]["inferencex"]["root"]))
    receipt = LmEvalRuntimePreparer(
        runtime_lock,
        apex_root=apex_root,
        inferencex_root=inferencex,
        runtime_root=runtime_root,
        artifact_cache=args.artifact_cache,
        offline=args.offline,
    ).prepare()
    result["lm_eval_runtime"] = _runtime_result(receipt, "prepared")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dependency bootstrap CLI."""

    apex_root = Path(__file__).resolve().parents[3]
    parser = build_parser(apex_root)
    args = parser.parse_args(argv)
    if args.command != "install" and args.dry_run:
        parser.error("--dry-run is only valid with install")
    try:
        emit_result(_run(args, apex_root), json_output=args.json)
        return 0
    except BootstrapError as error:
        payload = {
            "schema": "apex.dependencies.receipt/v1",
            "status": "error",
            "error": str(error),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"dependency bootstrap failed: {error}", file=sys.stderr)
        return 2


__all__ = ["build_parser", "emit_result", "main"]
