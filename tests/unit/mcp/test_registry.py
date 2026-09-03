from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anyio
import pytest
from mcp import types

import apex.bootstrap as bootstrap_module
from apex.core import ContractError
from apex.knowledge import KnowledgeRetriever, load_knowledge_catalog
from apex.mcp import (
    CapabilityGrantGate,
    CapabilityRegistry,
    KernelDraftSessionGrantAuthority,
    KnowledgeExplainHandler,
    KnowledgeSearchHandler,
    build_low_level_server,
    knowledge_search_descriptor,
    knowledge_explain_descriptor,
    planned_capability_descriptors,
)
from apex.ports import (
    CapabilityAuthority,
    CapabilityDescriptor,
    CapabilityGpuRequirement,
    CapabilityKind,
    CapabilityRequest,
    CapabilityResult,
    CapabilityRewardRole,
    CapabilitySideEffect,
)


ROOT = Path(__file__).resolve().parents[3]


@dataclass
class _Handler:
    result: CapabilityResult

    def invoke(self, request: CapabilityRequest) -> CapabilityResult:
        del request
        return self.result


def _descriptor(
    *, authority: CapabilityAuthority = CapabilityAuthority.NONE
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        "test.read",
        "Test read",
        "Read a deterministic test value.",
        CapabilityKind.TOOL,
        {"type": "object", "additionalProperties": False},
        {"type": "object"},
        (CapabilitySideEffect.NONE,),
        authority,
        CapabilityGpuRequirement.NONE,
        1,
        ("test",),
        CapabilityRewardRole.INELIGIBLE,
    )


def test_registry_enforces_identity_authority_and_unique_registration() -> None:
    registry = CapabilityRegistry()
    descriptor = _descriptor(authority=CapabilityAuthority.WORKSPACE_USER)
    registry.register(
        descriptor,
        _Handler(CapabilityResult("test.read", {"value": 1})),
    )

    with pytest.raises(ContractError, match="authority"):
        registry.invoke(CapabilityRequest("test.read", {}))
    result = registry.invoke(
        CapabilityRequest(
            "test.read",
            {},
            frozenset({CapabilityAuthority.WORKSPACE_USER}),
        )
    )
    assert result.content == {"value": 1}
    with pytest.raises(ContractError) as duplicate:
        registry.register(descriptor, _Handler(result))
    assert duplicate.value.reason_code == "duplicate_capability"


def test_unavailable_capability_is_visible_but_not_callable() -> None:
    registry = CapabilityRegistry()
    registry.register(_descriptor(), None, unavailable_reason="dependency_missing")

    assert registry.inventory()[0].unavailable_reason == "dependency_missing"
    with pytest.raises(ContractError) as error:
        registry.invoke(CapabilityRequest("test.read", {}))
    assert error.value.reason_code == "dependency_missing"


def test_knowledge_search_is_bounded_attributed_and_reward_ineligible() -> None:
    catalog = load_knowledge_catalog(
        ROOT / "src" / "apex" / "knowledge" / "data" / "cards.json"
    )
    registry = CapabilityRegistry()
    registry.register(
        knowledge_search_descriptor(),
        KnowledgeSearchHandler(KnowledgeRetriever(catalog.cards)),
    )

    result = registry.invoke(
        CapabilityRequest(
            "knowledge.search",
            {
                "gpu_arch": "gfx950",
                "language": "triton",
                "operator": "rmsnorm",
                "independent_hypothesis": "RMSNorm is limited by memory traffic",
                "limit": 2,
            },
        )
    )

    assert result.reward_eligible is False
    assert len(result.content["cards"]) == 2
    assert result.content["advisory_only"] is True
    for card in result.content["cards"]:
        assert len(card["source"]["content_sha256"]) == 64
        assert len(card["source"]["git_sha"]) == 40
    with pytest.raises(ContractError) as invalid:
        registry.invoke(
            CapabilityRequest(
                "knowledge.search",
                {"gpu_arch": "gfx950", "language": "triton"},
            )
        )
    assert invalid.value.reason_code == "invalid_capability_arguments"


def test_knowledge_explain_returns_only_the_exact_attributed_card() -> None:
    catalog = load_knowledge_catalog(
        ROOT / "src" / "apex" / "knowledge" / "data" / "cards.json"
    )
    registry = CapabilityRegistry()
    registry.register(
        knowledge_explain_descriptor(),
        KnowledgeExplainHandler(KnowledgeRetriever(catalog.cards)),
    )
    selected = catalog.cards[0]

    result = registry.invoke(
        CapabilityRequest("knowledge.explain", {"card_id": selected.card_id})
    )
    missing = registry.invoke(
        CapabilityRequest("knowledge.explain", {"card_id": "card-does-not-exist"})
    )

    assert result.content["card"]["card_id"] == selected.card_id
    assert result.content["card"]["source"]["content_sha256"] == (
        selected.source.content_sha256
    )
    assert result.content["advisory_only"] is True
    assert missing.content["card"] is None
    assert missing.content["unavailable_reason"] == "knowledge_card_unavailable"


def test_mcp_tool_schema_is_the_registry_descriptor() -> None:
    registry = CapabilityRegistry()
    descriptor = knowledge_search_descriptor()
    registry.register(
        descriptor,
        KnowledgeSearchHandler(KnowledgeRetriever((), enabled=False)),
    )
    server = build_low_level_server(registry)

    async def project():
        request = types.ListToolsRequest(method="tools/list")
        return await server.request_handlers[types.ListToolsRequest](request)

    projected = anyio.run(project).root.tools
    assert len(projected) == 1
    assert projected[0].name == descriptor.capability_id
    assert projected[0].inputSchema == descriptor.input_schema
    assert projected[0].outputSchema == descriptor.output_schema


def test_mcp_active_tool_without_grant_never_reaches_handler() -> None:
    class CountingHandler:
        calls = 0

        def invoke(self, request: CapabilityRequest) -> CapabilityResult:
            self.calls += 1
            return CapabilityResult(request.capability_id, {"ok": True})

    descriptor = _descriptor(authority=CapabilityAuthority.WORKSPACE_USER)
    handler = CountingHandler()
    registry = CapabilityRegistry()
    registry.register(descriptor, handler)
    server = build_low_level_server(registry)

    async def invoke():
        request = types.CallToolRequest(
            params=types.CallToolRequestParams(
                name=descriptor.capability_id,
                arguments={},
            )
        )
        return await server.request_handlers[types.CallToolRequest](request)

    async def project():
        request = types.ListToolsRequest(method="tools/list")
        return await server.request_handlers[types.ListToolsRequest](request)

    response = anyio.run(invoke).root
    assert anyio.run(project).root.tools == []
    assert response.isError is True
    assert response.content[0].text == (
        "Capability invocation requires explicit caller approval"
    )
    assert handler.calls == 0


def test_kernel_session_grant_exposes_only_unverified_campaign_start() -> None:
    registry = CapabilityRegistry()
    for descriptor in planned_capability_descriptors():
        if descriptor.kind is CapabilityKind.SKILL:
            continue
        registry.register(
            descriptor,
            _Handler(CapabilityResult(descriptor.capability_id, {})),
        )
    server = build_low_level_server(
        registry,
        grant_authority=KernelDraftSessionGrantAuthority(),
        session_id="kernel-session-1",
    )

    async def project():
        request = types.ListToolsRequest(method="tools/list")
        return await server.request_handlers[types.ListToolsRequest](request)

    names = {tool.name for tool in anyio.run(project).root.tools}
    assert names == {"campaign.start"}


def test_kernel_session_grant_can_create_only_one_draft() -> None:
    authority = KernelDraftSessionGrantAuthority()
    descriptor = next(
        item
        for item in planned_capability_descriptors()
        if item.capability_id == "campaign.start"
    )
    gate = CapabilityGrantGate(authority, session_id="kernel-session-once")

    first = gate.authorize(descriptor, {"task": {"task_id": "one"}})

    assert first is not None
    with pytest.raises(ContractError) as replay:
        gate.authorize(descriptor, {"task": {"task_id": "two"}})
    assert replay.value.reason_code == "capability_grant_replayed"


def test_campaign_start_schema_declares_the_chat_to_formal_task_shape() -> None:
    descriptor = next(
        item
        for item in planned_capability_descriptors()
        if item.capability_id == "campaign.start"
    )

    task = descriptor.input_schema["properties"]["task"]
    assert task["additionalProperties"] is False
    assert task["properties"]["language"]["enum"] == ["python", "triton"]
    assert "workspace" not in task["properties"]
    commands = task["properties"]["commands"]
    assert commands["required"] == ["compile", "correctness", "performance"]
    assert commands["properties"]["compile"]["properties"]["argv"]["type"] == "array"
    measurement = task["properties"]["measurement"]
    assert measurement["properties"]["runner"]["properties"]["argv"]["type"] == "array"
    assert measurement["properties"]["schema"]["const"] == (
        "apex.kernel-measurement/v1"
    )


def test_presented_skill_is_available_but_not_an_mcp_tool() -> None:
    registry = CapabilityRegistry()
    descriptor = next(
        item
        for item in planned_capability_descriptors()
        if item.capability_id == "amd-kernel-optimization"
    )
    registry.register_presentation(descriptor)
    server = build_low_level_server(registry)

    async def project():
        request = types.ListToolsRequest(method="tools/list")
        return await server.request_handlers[types.ListToolsRequest](request)

    assert registry.inventory()[0].available is True
    assert anyio.run(project).root.tools == []
    with pytest.raises(ContractError) as error:
        registry.invoke(CapabilityRequest(descriptor.capability_id, {}))
    assert error.value.reason_code == "capability_not_callable"


def test_application_inventory_does_not_verify_workload_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def forbidden_probe():
        nonlocal calls
        calls += 1
        raise AssertionError("dependency verification must remain lazy")

    monkeypatch.setattr(
        bootstrap_module, "verify_runtime_dependencies", forbidden_probe
    )
    application = bootstrap_module.build_application(
        include_kernel=False,
        include_capabilities=True,
        knowledge_enabled=False,
        capability_workspace=tmp_path,
        capability_results=tmp_path / "results",
    )

    assert calls == 0
    assert application.capabilities is not None
    inventory = {
        item.descriptor.capability_id: item
        for item in application.capabilities.inventory()
    }
    assert inventory["workload.inspect"].available is True
    assert inventory["trace.compare"].available is True
    assert inventory["amd-hip-kernel-optimization"].available is True
    assert inventory["amd-kernel-debugging"].available is True
    assert inventory["amd-kernel-optimization"].available is True
    assert inventory["benchmark.run"].available is True
    assert inventory["profile.capture"].available is True
    assert inventory["bundle.verify"].available is True
    assert inventory["campaign.status"].available is True
    assert inventory["campaign.checkpoint"].available is True
    assert inventory["campaign.start"].available is True
    assert inventory["campaign.stop"].available is True
    assert inventory["campaign.resume"].available is True
    assert inventory["bundle.verify"].descriptor.side_effects == (
        CapabilitySideEffect.READ_WORKSPACE,
        CapabilitySideEffect.READ_RESULTS,
    )
    assert inventory["campaign.status"].descriptor.side_effects == (
        CapabilitySideEffect.READ_RESULTS,
    )
    assert inventory["campaign.checkpoint"].descriptor.side_effects == (
        CapabilitySideEffect.WRITE_RESULTS,
    )
    assert inventory["campaign.start"].descriptor.side_effects == (
        CapabilitySideEffect.READ_WORKSPACE,
        CapabilitySideEffect.WRITE_RESULTS,
    )
    assert inventory["campaign.stop"].descriptor.side_effects == (
        CapabilitySideEffect.READ_RESULTS,
        CapabilitySideEffect.WRITE_RESULTS,
    )
    assert inventory["campaign.resume"].descriptor.side_effects == (
        CapabilitySideEffect.READ_RESULTS,
        CapabilitySideEffect.WRITE_RESULTS,
        CapabilitySideEffect.RUN_PROCESS,
    )
    assert inventory["kernel.grade"].descriptor.reward_role is (
        CapabilityRewardRole.EVALUATOR_OWNED
    )
    assert "kernel.sanitize" not in inventory
    assert calls == 0


def test_unscoped_implemented_capabilities_report_scope_gap() -> None:
    application = bootstrap_module.build_application(
        include_kernel=False,
        include_capabilities=True,
        knowledge_enabled=False,
    )

    assert application.capabilities is not None
    inventory = {
        item.descriptor.capability_id: item
        for item in application.capabilities.inventory()
    }
    assert inventory["bundle.verify"].unavailable_reason == "capability_scope_missing"
    assert inventory["campaign.status"].unavailable_reason == "capability_scope_missing"
    assert inventory["campaign.checkpoint"].unavailable_reason == "capability_scope_missing"
    assert inventory["campaign.start"].unavailable_reason == "capability_scope_missing"
    assert inventory["campaign.stop"].unavailable_reason == "capability_scope_missing"
    assert inventory["campaign.resume"].unavailable_reason == "capability_scope_missing"
    assert inventory["benchmark.run"].unavailable_reason == "capability_scope_missing"
    assert inventory["profile.capture"].unavailable_reason == "capability_scope_missing"
    assert inventory["trace.compare"].unavailable_reason == "capability_scope_missing"


def test_planned_campaign_mutations_have_specific_unavailable_reasons() -> None:
    application = bootstrap_module.build_application(
        include_kernel=False,
        include_capabilities=True,
        knowledge_enabled=False,
    )

    assert application.capabilities is not None
    inventory = {
        item.descriptor.capability_id: item
        for item in application.capabilities.inventory()
    }
    assert inventory["campaign.start"].unavailable_reason == "capability_scope_missing"
    assert inventory["campaign.stop"].unavailable_reason == "capability_scope_missing"
    assert inventory["campaign.checkpoint"].unavailable_reason == "capability_scope_missing"
    assert inventory["campaign.resume"].unavailable_reason == "capability_scope_missing"


def test_planned_catalog_is_typed_unique_and_excludes_sanitizer_runtime() -> None:
    descriptors = planned_capability_descriptors()
    identities = [item.capability_id for item in descriptors]

    assert len(identities) == len(set(identities))
    assert {
        "amd-hip-kernel-optimization",
        "amd-kernel-optimization",
        "benchmark.run",
        "profile.capture",
        "kernel.compile",
        "kernel.correctness",
        "kernel.measure",
        "kernel.grade",
        "campaign.start",
        "campaign.stop",
        "campaign.resume",
        "bundle.build",
        "bundle.verify",
    }.issubset(identities)
    assert "kernel.sanitize" not in identities
    assert all(len(item.digest) == 64 for item in descriptors)
