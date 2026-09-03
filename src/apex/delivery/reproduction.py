"""Task-kind-specific reproduction declarations bound to verified bundles."""

from __future__ import annotations

from typing import Any

from apex.core import sha256_file, sha256_json
from apex.evaluation import EvaluationContractReceipt

from .e2e_bundle_common import E2EPatchBundle
from .kernel_bundle import KernelBundle
from .portable_bundle import PortableBundleEvidence


def kernel_reproduction_declaration(
    contract: EvaluationContractReceipt,
    bundle: KernelBundle,
    portable: PortableBundleEvidence,
) -> dict[str, Any]:
    """Describe only exact standalone inputs already frozen by the evaluator."""

    repository = contract.draft.repository
    sources = []
    if repository.resolved:
        sources.append(
            {
                "name": "workspace",
                "repository": repository.remote,
                "commit": repository.commit,
                "tree": repository.tree,
            }
        )
    dependencies = [
        {"name": "evaluation_contract", "digest": contract.digest}
    ]
    if contract.authority is not None:
        dependencies.append(
            {"name": "evaluation_authority", "digest": contract.authority.digest}
        )
    commands = [
        {
            "name": "verify_bundle",
            "argv": [
                "apex", "bundle", "verify", "--bundle", "./bundle",
                "--digest", bundle.digest, "--json",
            ],
            "cwd": ".",
            "env": {},
        },
        {
            "name": "apply_bundle",
            "argv": [
                "apex", "bundle", "apply", "--bundle", "./bundle",
                "--workspace", "./workspace", "--digest", bundle.digest, "--json",
            ],
            "cwd": ".",
            "env": {},
        },
        *(
            {
                "name": name,
                "argv": list(command["argv"]),
                "cwd": command["cwd"],
                "env": dict(command["env"]),
            }
            for name, command in sorted(contract.draft.commands.items())
            if name in {"compile", "correctness", "performance"}
        ),
    ]
    authority = contract.draft.source_scope.get("template_authority")
    parent_image = (
        authority.get("runtime_image_id")
        if isinstance(authority, dict)
        else None
    )
    return {
        "schema": "apex.replication-declaration/v1",
        "task_kind": "single_kernel",
        "dependency_receipts": dependencies,
        "source_commits": sources,
        "parent_image_digest": parent_image,
        "derived_image_digest": None,
        "commands": commands,
        "benchmark_config_receipts": [],
        "bundle_receipt": {
            "kind": "kernel",
            "digest": bundle.digest,
            "evidence_receipt": portable.evidence_receipt.digest,
            "verification_receipt": portable.verification_receipt.digest,
        },
    }


def e2e_reproduction_declaration(
    bundle: E2EPatchBundle,
    portable: PortableBundleEvidence,
) -> dict[str, Any]:
    """Project final bundle locks into an exact E2E replay declaration."""

    verification_argv = [
        "apex", "bundle", "verify", "--bundle", "./bundle",
        "--results", "./verification-results", "--digest", bundle.digest, "--json",
    ]
    commands = [
        {"name": "verify_bundle", "argv": verification_argv, "cwd": ".", "env": {}},
        *(
            {
                "name": "build_image" if index == 0 else f"build_image_{index:04d}",
                "argv": list(step.argv),
                "cwd": step.cwd,
                "env": dict(step.environment),
            }
            for index, step in enumerate(bundle.recipe.steps)
        ),
        {"name": "clean_replay", "argv": verification_argv, "cwd": ".", "env": {}},
    ]
    return {
        "schema": "apex.replication-declaration/v1",
        "task_kind": "e2e_kernel_only",
        "dependency_receipts": [
            {"name": "build_recipe", "digest": bundle.recipe.computed_sha256},
            {
                "name": "delivery_provenance",
                "digest": sha256_json(bundle.provenance.to_dict()),
            },
        ],
        "source_commits": [
            {
                "name": item.repository_id,
                "repository": item.url,
                "commit": item.base_commit,
                "tree": item.base_tree,
            }
            for item in bundle.repositories
        ],
        "parent_image_digest": bundle.recipe.parent_image_digest,
        "derived_image_digest": bundle.derived_image.image_digest,
        "commands": commands,
        "benchmark_config_receipts": [
            {"name": name, "digest": sha256_file(path)}
            for name, path in sorted(bundle.config_paths.items())
        ],
        "bundle_receipt": {
            "kind": "e2e",
            "digest": bundle.digest,
            "evidence_receipt": portable.evidence_receipt.digest,
            "verification_receipt": portable.verification_receipt.digest,
        },
    }


__all__ = ["e2e_reproduction_declaration", "kernel_reproduction_declaration"]
