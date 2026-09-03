"""Local stdio MCP projection of the canonical capability registry."""

from __future__ import annotations

from typing import Any, Mapping

from apex.ports import (
    CapabilityAuthority,
    CapabilityGrantAuthority,
    CapabilityKind,
    CapabilityRequest,
)

from .grants import CapabilityGrantGate
from .registry import CapabilityRegistry


def run_stdio_server(
    registry: CapabilityRegistry,
    *,
    grant_authority: CapabilityGrantAuthority | None = None,
    session_id: str | None = None,
) -> None:
    """Serve only currently available registry entries over MCP stdio."""

    import anyio

    anyio.run(_serve, registry, grant_authority, session_id)


def build_low_level_server(
    registry: CapabilityRegistry,
    *,
    grant_authority: CapabilityGrantAuthority | None = None,
    session_id: str | None = None,
):
    """Build without opening stdio so schema projection can be contract-tested."""

    from mcp import types
    from mcp.server.lowlevel import Server

    server = Server("apex-capabilities")
    grants = CapabilityGrantGate(grant_authority, session_id=session_id)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=item.descriptor.capability_id,
                title=item.descriptor.title,
                description=item.descriptor.summary,
                inputSchema=dict(item.descriptor.input_schema),
                outputSchema=dict(item.descriptor.output_schema),
            )
            for item in registry.inventory()
            if (
                item.available
                and item.descriptor.kind is not CapabilityKind.SKILL
                and grants.available(item.descriptor)
            )
        ]

    @server.call_tool(validate_input=True)
    async def call_tool(name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        import anyio

        descriptor = registry.validate_arguments(name, arguments)
        grant = grants.authorize(descriptor, arguments)
        authorities = (
            frozenset()
            if grant is None or grant.authority is CapabilityAuthority.NONE
            else frozenset({grant.authority})
        )
        request = CapabilityRequest(
            capability_id=name,
            arguments=arguments,
            authorities=authorities,
            grant=grant,
        )
        timeout = grant.timeout_seconds if grant is not None else descriptor.timeout_seconds
        with anyio.fail_after(timeout):
            result = await anyio.to_thread.run_sync(registry.invoke, request)
        return dict(result.content)

    return server


async def _serve(registry, grant_authority, session_id) -> None:
    from mcp.server.stdio import stdio_server

    server = build_low_level_server(
        registry,
        grant_authority=grant_authority,
        session_id=session_id,
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


__all__ = ["build_low_level_server", "run_stdio_server"]
