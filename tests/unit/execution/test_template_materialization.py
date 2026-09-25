from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from apex.core import AgentBackendName, ContractError, IntegrityError
from apex.execution import (
    KernelTemplateMaterializer,
    TemplateImageSourceReceipt,
    template_source_tree_sha256,
)
from apex.execution.kernel_measurement import (
    STRUCTURED_KERNEL_MEASUREMENT_ADAPTER_ID,
    STRUCTURED_KERNEL_MEASUREMENT_METHOD_SHA256,
)
from apex.intake import ReviewedKernelTemplate, TaskSpec
from apex.intake.template import TemplateEvaluator, TemplateRuntime, TemplateSource
from apex.runtime import WorkspaceGitIdentityResolver


class FakeImageRuntime:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.calls = 0

    def materialize(
        self,
        *,
        immutable_locator: str,
        expected_image_id: str,
        container_root: str,
        destination: Path,
    ) -> TemplateImageSourceReceipt:
        self.calls += 1
        for relative, content in self.files.items():
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return TemplateImageSourceReceipt(
            immutable_locator, expected_image_id, container_root
        )


def _source_digest(tmp_path: Path, files: dict[str, bytes]) -> str:
    root = tmp_path / "expected-source"
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    digest = template_source_tree_sha256(root)
    shutil.rmtree(root)
    return digest


def _reviewed_template(tmp_path: Path, files: dict[str, bytes]) -> ReviewedKernelTemplate:
    root = tmp_path / "template"
    harness = root / "template" / "evaluator" / "harness.py"
    harness.parent.mkdir(parents=True)
    harness.write_text("print('not executed in this test')\n", encoding="utf-8")
    return ReviewedKernelTemplate(
        root=root,
        template_id="reviewed-ck-template",
        showcase_id="reviewed-ck-showcase",
        status="reviewed",
        manifest_sha256="1" * 64,
        upstream={},
        runtime=TemplateRuntime(
            "MI355X",
            "gfx950",
            "example.invalid/image:mutable",
            "example.invalid/image@sha256:" + "2" * 64,
            "sha256:" + "3" * 64,
        ),
        source=TemplateSource(
            "/opt/aiter_meta",
            "aiter_meta",
            "hip",
            ("csrc/kernel.cu",),
            ("kernel",),
            _source_digest(tmp_path, files),
        ),
        evaluator=TemplateEvaluator(
            STRUCTURED_KERNEL_MEASUREMENT_ADAPTER_ID,
            STRUCTURED_KERNEL_MEASUREMENT_METHOD_SHA256,
            ("evaluator/harness.py",),
            ("python3", "evaluator/harness.py", "compile"),
            ("python3", "evaluator/harness.py", "correctness"),
            ("python3", "evaluator/harness.py", "performance"),
            ("python3", "evaluator/harness.py", "measurement"),
            "reviewed-ck-recipe",
            "4" * 64,
        ),
        blockers=(),
        snapshot_files=(),
    )


def test_reviewed_hip_template_materializes_one_authority_bound_task(
    tmp_path: Path,
) -> None:
    files = {"csrc/kernel.cu": b"extern \"C\" __global__ void kernel() {}\n"}
    template = _reviewed_template(tmp_path, files)
    runtime = FakeImageRuntime(files)
    results = (tmp_path / "results").resolve()

    materialized = KernelTemplateMaterializer(runtime).materialize(
        template,
        results_dir=results,
        instructions="Optimize the reviewed kernel",
        backend=AgentBackendName.CODEX,
    )

    task = materialized.task
    assert runtime.calls == 1
    assert task.language == "hip"
    assert task.mode == "template_bound_image_kernel"
    assert task.recipe is not None and task.recipe.kind == "fixed_hip"
    assert task.template_authority is not None
    assert task.template_authority.materialization_receipt_sha256 == materialized.receipt.digest
    assert task.workspace.joinpath("csrc/kernel.cu").read_bytes() == files["csrc/kernel.cu"]
    assert materialized.receipt_path == results / "template_materialization.json"
    assert json.loads(materialized.receipt_path.read_text())["image_id"] == "sha256:" + "3" * 64
    repository = WorkspaceGitIdentityResolver().inspect(task.workspace)
    assert repository.resolved and not repository.dirty_paths


def test_serialized_template_authority_cannot_be_replayed_as_user_task(
    tmp_path: Path,
) -> None:
    files = {"csrc/kernel.cu": b"kernel\n"}
    materialized = KernelTemplateMaterializer(FakeImageRuntime(files)).materialize(
        _reviewed_template(tmp_path, files),
        results_dir=(tmp_path / "results").resolve(),
        instructions="Optimize",
        backend=AgentBackendName.CODEX,
    )

    with pytest.raises(ContractError) as raised:
        TaskSpec.from_mapping(materialized.task.to_dict())

    assert raised.value.reason_code == "template_authority_internal_only"


def test_source_tree_mismatch_cleans_partial_materialization(tmp_path: Path) -> None:
    expected = {"csrc/kernel.cu": b"expected\n"}
    observed = {"csrc/kernel.cu": b"different\n"}
    results = (tmp_path / "results").resolve()

    with pytest.raises(IntegrityError) as raised:
        KernelTemplateMaterializer(FakeImageRuntime(observed)).materialize(
            _reviewed_template(tmp_path, expected),
            results_dir=results,
            instructions="Optimize",
            backend=AgentBackendName.CODEX,
        )

    assert raised.value.reason_code == "template_source_tree_mismatch"
    assert not (results / "template_workspace").exists()
    assert not (results / "template_materialization.json").exists()


def test_pending_template_never_calls_container_runtime(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    template_path = (
        root / "examples" / "optimization_showcases" / "kernel_ck_moe_2stage"
    )
    from apex.intake import load_kernel_template

    runtime = FakeImageRuntime({"unused": b"unused"})
    with pytest.raises(ContractError) as raised:
        KernelTemplateMaterializer(runtime).materialize(
            load_kernel_template(template_path),
            results_dir=(tmp_path / "results").resolve(),
            instructions="Optimize",
            backend=AgentBackendName.CODEX,
        )

    assert raised.value.reason_code == "template_not_materializable"
    assert runtime.calls == 0
    assert not (tmp_path / "results").exists()


def test_template_source_tree_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")
    (root / "linked").symlink_to(outside)

    with pytest.raises(IntegrityError) as raised:
        template_source_tree_sha256(root)

    assert raised.value.reason_code == "unsafe_template_source"


def test_bootstrap_composes_materializer_without_e2e_dependency_verification() -> None:
    from apex.bootstrap import build_application

    application = build_application(
        include_kernel=False,
        include_kernel_templates=True,
        knowledge_enabled=False,
    )

    assert application.kernel_optimizer is None
    assert isinstance(application.kernel_template_materializer, KernelTemplateMaterializer)
