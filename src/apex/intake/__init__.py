"""Caller-neutral task and workload intake contracts."""

from .e2e_spec import E2EOptimizeSpec, MetricGoal, RegressionGates
from .resolver import NaturalLanguageTaskResolver, TaskResolver
from .task_intent import NaturalLanguageRequest
from .template import ReviewedKernelTemplate, load_kernel_template
from .template_authority import TemplateTaskAuthority
from .task_spec import (
    AgentOptions,
    CommandSpec,
    DeliverySpec,
    KernelMeasurementSpec,
    ResolvedTaskSpec,
    TaskBudget,
    TaskRecipe,
    TaskScope,
    TaskSpec,
)

__all__ = [
    "AgentOptions",
    "CommandSpec",
    "DeliverySpec",
    "E2EOptimizeSpec",
    "MetricGoal",
    "KernelMeasurementSpec",
    "NaturalLanguageRequest",
    "NaturalLanguageTaskResolver",
    "RegressionGates",
    "ReviewedKernelTemplate",
    "ResolvedTaskSpec",
    "TaskRecipe",
    "TaskBudget",
    "TaskResolver",
    "TaskScope",
    "TaskSpec",
    "TemplateTaskAuthority",
    "load_kernel_template",
]
