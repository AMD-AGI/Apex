"""Internal authority receipt for reviewed template materialization."""

from __future__ import annotations

from dataclasses import dataclass

from apex.core import ContractError, validate_identifier


@dataclass(frozen=True, slots=True)
class TemplateTaskAuthority:
    """Prove one task came from exact reviewed template and materialized bytes."""

    template_id: str
    showcase_id: str
    manifest_sha256: str
    runtime_image_locator: str
    runtime_image_id: str
    source_tree_sha256: str
    evaluator_recipe_sha256: str
    materialization_receipt_sha256: str

    def __post_init__(self) -> None:
        validate_identifier(self.template_id, field_name="template ID")
        validate_identifier(self.showcase_id, field_name="showcase ID")
        digests = (
            self.manifest_sha256,
            self.source_tree_sha256,
            self.evaluator_recipe_sha256,
            self.materialization_receipt_sha256,
        )
        if any(
            len(item) != 64
            or any(char not in "0123456789abcdef" for char in item)
            for item in digests
        ):
            raise ContractError(
                "Template authority digest is invalid", "invalid_template_authority"
            )
        if not self.runtime_image_locator or not self.runtime_image_id.startswith("sha256:"):
            raise ContractError(
                "Template runtime authority is invalid", "invalid_template_authority"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "schema": "apex.template-task-authority/v1",
            "template_id": self.template_id,
            "showcase_id": self.showcase_id,
            "manifest_sha256": self.manifest_sha256,
            "runtime_image_locator": self.runtime_image_locator,
            "runtime_image_id": self.runtime_image_id,
            "source_tree_sha256": self.source_tree_sha256,
            "evaluator_recipe_sha256": self.evaluator_recipe_sha256,
            "materialization_receipt_sha256": self.materialization_receipt_sha256,
        }


__all__ = ["TemplateTaskAuthority"]
