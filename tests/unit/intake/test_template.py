from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from apex.core import ContractError, sha256_json
from apex.intake import load_kernel_template


ROOT = Path(__file__).resolve().parents[3]
SHOWCASES = ROOT / "examples" / "optimization_showcases"
TEMPLATES = (
    "kernel_triton_paged_attention_2d",
    "kernel_ck_moe_2stage",
    "kernel_cktile_moe_2stage",
)


def _copy(tmp_path: Path, name: str = TEMPLATES[0]) -> Path:
    destination = tmp_path / name
    shutil.copytree(SHOWCASES / name, destination)
    return destination


def _rewrite_manifest(root: Path, update) -> None:
    path = root / "template" / "template_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    update(value)
    unsigned = dict(value)
    unsigned.pop("manifest_sha256")
    value["manifest_sha256"] = sha256_json(unsigned)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_checked_in_templates_are_attributed_but_pending() -> None:
    loaded = tuple(load_kernel_template(SHOWCASES / name) for name in TEMPLATES)

    assert {item.template_id for item in loaded} == {
        "kernel-triton-paged-attention-2d",
        "kernel-ck-moe-2stage",
        "kernel-cktile-moe-2stage",
    }
    assert all(item.status == "pending" and not item.materializable for item in loaded)
    assert all(item.upstream["commit"] == "1292b4531fad8bed02c0ecc292704c44cb63c49a" for item in loaded)
    assert {item.source.language for item in loaded} == {"triton", "hip"}


def test_pending_template_reports_every_blocker() -> None:
    template = load_kernel_template(SHOWCASES / TEMPLATES[1])

    with pytest.raises(ContractError) as raised:
        template.require_materializable()

    assert raised.value.reason_code == "template_not_materializable"
    assert raised.value.details == {
        "template_id": "kernel-ck-moe-2stage",
        "blockers": [
            "immutable_image_digest_missing",
            "in_image_source_digest_missing",
            "apex_evaluator_missing",
        ],
    }


def test_manifest_byte_tamper_is_rejected(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    path = root / "template" / "template_manifest.json"
    path.write_text(path.read_text().replace("MI355X", "MI300X"), encoding="utf-8")

    with pytest.raises(ContractError) as raised:
        load_kernel_template(root)

    assert raised.value.reason_code == "template_manifest_mismatch"


def test_attributed_snapshot_tamper_is_rejected(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    path = root / "template" / "upstream" / "config.yaml"
    path.write_text(path.read_text() + "# changed\n", encoding="utf-8")

    with pytest.raises(ContractError) as raised:
        load_kernel_template(root)

    assert raised.value.reason_code == "template_snapshot_mismatch"


def test_unknown_manifest_field_is_rejected_even_with_new_digest(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    _rewrite_manifest(root, lambda value: value.update({"surprise": True}))

    with pytest.raises(ContractError) as raised:
        load_kernel_template(root)

    assert raised.value.reason_code == "invalid_template_manifest"


def test_recomputed_structurally_valid_manifest_is_not_registry_authority(
    tmp_path: Path,
) -> None:
    root = _copy(tmp_path)
    _rewrite_manifest(
        root,
        lambda value: value.update(
            {"template_id": "caller-recomputed-template"}
        ),
    )

    with pytest.raises(ContractError) as raised:
        load_kernel_template(root)

    assert raised.value.reason_code == "template_not_registered"


def test_pending_status_cannot_hide_undeclared_blockers(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    _rewrite_manifest(root, lambda value: value.update({"blockers": []}))

    with pytest.raises(ContractError) as raised:
        load_kernel_template(root)

    assert raised.value.reason_code == "invalid_template_manifest"


def test_reviewed_status_requires_runtime_source_and_evaluator_proof(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    _rewrite_manifest(
        root,
        lambda value: value.update({"status": "reviewed", "blockers": []}),
    )

    with pytest.raises(ContractError) as raised:
        load_kernel_template(root)

    assert raised.value.reason_code == "invalid_template_manifest"


def test_template_root_symlink_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "template-link"
    link.symlink_to(SHOWCASES / TEMPLATES[0], target_is_directory=True)

    with pytest.raises(ContractError) as raised:
        load_kernel_template(link)

    assert raised.value.reason_code == "unsafe_template_path"
