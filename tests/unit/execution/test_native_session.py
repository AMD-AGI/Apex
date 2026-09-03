from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from apex.core import AgentBackendName, ConfigurationError
from apex.execution import (
    NativeCodingSessionLauncher,
    NativeSessionInvocation,
    SubprocessSessionRunner,
    default_capability_results,
    load_kernel_skill_package,
)
from apex.ports import (
    CodingSessionOutput,
    CodingSessionRequest,
    KernelEnhancement,
)


class _Runner:
    def __init__(self) -> None:
        self.invocation: NativeSessionInvocation | None = None

    def run(self, invocation: NativeSessionInvocation) -> int:
        self.invocation = invocation
        return 17


def _launcher(runner: _Runner) -> NativeCodingSessionLauncher:
    return NativeCodingSessionLauncher(
        mcp_command=("/usr/bin/python3", "-m", "apex.cli", "mcp-server"),
        skill_package=load_kernel_skill_package(),
        runner=runner,
        executable_resolver=lambda name: f"/tools/{name}",
    )


def _request(tmp_path: Path, prompt: str, **values) -> CodingSessionRequest:
    return CodingSessionRequest(
        workspace=tmp_path.resolve(),
        prompt=prompt,
        **values,
    )


def test_non_kernel_session_preserves_native_codex_behavior(tmp_path: Path) -> None:
    runner = _Runner()

    exit_code = _launcher(runner).launch(
        _request(tmp_path, "Refactor the HTTP request parser")
    )

    assert exit_code == 17
    assert runner.invocation is not None
    argv = runner.invocation.argv
    assert argv[:3] == ("/tools/codex", "-C", str(tmp_path.resolve()))
    assert "exec" not in argv
    assert "--sandbox" not in argv
    assert "--config" not in argv
    assert runner.invocation.kernel_capabilities_enabled is False
    assert runner.invocation.kernel_skill_ids == ()
    assert runner.invocation.kernel_skill_digest is None


def test_kernel_hint_lazily_mounts_read_only_mcp_for_codex(tmp_path: Path) -> None:
    runner = _Runner()

    _launcher(runner).launch(
        _request(
            tmp_path,
            "Explain this Triton kernel on gfx950",
            output=CodingSessionOutput.JSONL,
        )
    )

    assert runner.invocation is not None
    argv = runner.invocation.argv
    assert argv[1] == "exec"
    assert "--json" in argv
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv
    mcp_config = next(value for value in argv if value.startswith("mcp_servers.apex="))
    assert 'enabled_tools=["campaign.start","knowledge.search","knowledge.explain"]' in mcp_config
    assert 'default_tools_approval_mode="prompt"' in mcp_config
    assert 'tools={"campaign.start"={approval_mode="approve"}}' in mcp_config
    skill_config = next(value for value in argv if value.startswith("skills.config="))
    assert "amd-hip-kernel-optimization/SKILL.md" in skill_config
    assert "amd-kernel-debugging/SKILL.md" in skill_config
    assert "amd-kernel-optimization/SKILL.md" in skill_config
    assert any("--workspace" in value for value in argv)
    assert any(str(default_capability_results(tmp_path)) in value for value in argv)
    assert any("--session-kernel-draft-grants" in value for value in argv)
    assert runner.invocation.kernel_capabilities_enabled is True
    assert runner.invocation.kernel_skill_ids == (
        "amd-hip-kernel-optimization",
        "amd-kernel-debugging",
        "amd-kernel-optimization",
    )
    assert len(runner.invocation.kernel_skill_digest or "") == 64
    assert runner.invocation.prompt_transport == "stdin"
    assert runner.invocation.stdin_text is not None
    assert "use the packaged amd-kernel-optimization skill" in runner.invocation.stdin_text
    assert runner.invocation.stdin_text.endswith(
        "User request:\nExplain this Triton kernel on gfx950"
    )
    assert runner.invocation.stdin_text not in runner.invocation.argv


def test_kernel_session_passes_caller_selected_capability_results(tmp_path: Path) -> None:
    results = tmp_path / "external-results"
    invocation = _launcher(_Runner()).prepare(
        _request(
            tmp_path,
            "Analyze this GPU trace",
            results_dir=results,
            enhancement=KernelEnhancement.KERNEL,
        )
    )

    assert any(str(results) in value for value in invocation.argv)


def test_plain_mode_disables_kernel_augmentation(tmp_path: Path) -> None:
    invocation = _launcher(_Runner()).prepare(
        _request(
            tmp_path,
            "Optimize this AMD kernel",
            enhancement=KernelEnhancement.PLAIN,
        )
    )

    assert invocation.kernel_capabilities_enabled is False
    assert not any("mcp_servers.apex" in value for value in invocation.argv)
    assert invocation.argv[-1] == "Optimize this AMD kernel"


def test_claude_kernel_session_uses_native_permissions_and_explicit_mcp(
    tmp_path: Path,
) -> None:
    invocation = _launcher(_Runner()).prepare(
        _request(
            tmp_path,
            "Profile the ROCm kernel",
            backend=AgentBackendName.CLAUDE,
            model="sonnet",
            effort="high",
        )
    )

    assert "--mcp-config" in invocation.argv
    assert "--session-kernel-draft-grants" in invocation.argv[
        invocation.argv.index("--mcp-config") + 1
    ]
    assert "--plugin-dir" in invocation.argv
    allowed_tools = invocation.argv.index("--allowedTools") + 1
    assert invocation.argv[allowed_tools] == (
        "mcp__apex__campaign_start,mcp__apex__knowledge_search,"
        "mcp__apex__knowledge_explain"
    )
    assert allowed_tools > invocation.argv.index("--mcp-config")
    assert "--bare" not in invocation.argv
    assert "--safe-mode" not in invocation.argv
    assert "--permission-mode" not in invocation.argv
    assert invocation.kernel_capabilities_enabled is True


def test_cursor_mounts_skills_when_mcp_bridge_is_unavailable(tmp_path: Path) -> None:
    invocation = _launcher(_Runner()).prepare(
        _request(
            tmp_path,
            "Optimize this kernel",
            backend=AgentBackendName.CURSOR,
            enhancement=KernelEnhancement.KERNEL,
        )
    )

    assert invocation.kernel_capabilities_enabled is True
    assert "--workspace" in invocation.argv
    assert "--plugin-dir" in invocation.argv
    assert invocation.capability_notices == (
        "kernel_mcp_bridge_unavailable: Cursor session includes Apex skills "
        "without Apex MCP tools",
    )
    with pytest.raises(ConfigurationError) as error:
        _launcher(_Runner()).prepare(
            _request(
                tmp_path,
                "Explain code",
                backend=AgentBackendName.CURSOR,
                effort="high",
            )
        )
    assert error.value.reason_code == "agent_effort_unsupported"


@pytest.mark.parametrize(
    "backend", (AgentBackendName.CLAUDE, AgentBackendName.CURSOR)
)
def test_headless_native_prompt_is_not_exposed_in_process_argv(
    tmp_path: Path, backend: AgentBackendName
) -> None:
    prompt = "private optimizer instruction test-only-secret-value"

    invocation = _launcher(_Runner()).prepare(
        _request(
            tmp_path,
            prompt,
            backend=backend,
            output=CodingSessionOutput.TEXT,
        )
    )

    assert prompt not in invocation.argv
    assert invocation.stdin_text == prompt
    assert invocation.prompt_transport == "stdin"


def test_interactive_prompt_receipt_matches_argv_transport(tmp_path: Path) -> None:
    prompt = "Review the parser"

    invocation = _launcher(_Runner()).prepare(_request(tmp_path, prompt))

    assert invocation.argv[-1] == prompt
    assert invocation.stdin_text is None
    assert invocation.prompt_transport == "argv"


def test_kernel_interactive_prompt_activates_skill_and_preserves_request(
    tmp_path: Path,
) -> None:
    prompt = "Make vector_add faster"

    invocation = _launcher(_Runner()).prepare(
        _request(tmp_path, prompt, enhancement=KernelEnhancement.KERNEL)
    )

    rendered = invocation.argv[-1]
    assert "amd-kernel-optimization skill" in rendered
    assert "campaign.start" in rendered
    assert rendered.endswith(f"User request:\n{prompt}")


def test_subprocess_session_runner_writes_prompt_only_to_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorded: dict[str, object] = {}

    def fake_run(argv, **values):
        recorded["argv"] = tuple(argv)
        recorded.update(values)
        return SimpleNamespace(returncode=23)

    monkeypatch.setattr("apex.execution.native_session.subprocess.run", fake_run)
    invocation = _launcher(_Runner()).prepare(
        _request(
            tmp_path,
            "do not expose this prompt",
            backend=AgentBackendName.CLAUDE,
            output=CodingSessionOutput.TEXT,
        )
    )

    assert SubprocessSessionRunner().run(invocation) == 23
    assert recorded["input"] == "do not expose this prompt"
    assert recorded["text"] is True
    assert recorded["check"] is False
    assert recorded["input"] not in recorded["argv"]


@pytest.mark.parametrize(
    ("backend", "allowed"),
    (
        (AgentBackendName.CODEX, "OPENAI_API_KEY"),
        (AgentBackendName.CLAUDE, "ANTHROPIC_API_KEY"),
        (AgentBackendName.CURSOR, "CURSOR_API_KEY"),
    ),
)
def test_native_backend_environment_contains_only_its_credential(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: AgentBackendName,
    allowed: str,
) -> None:
    credentials = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CURSOR_API_KEY")
    for key in credentials:
        monkeypatch.setenv(key, f"{key}-test-value")

    invocation = _launcher(_Runner()).prepare(
        _request(tmp_path, "Review code", backend=backend)
    )

    assert invocation.environment[allowed] == f"{allowed}-test-value"
    assert all(
        key not in invocation.environment for key in credentials if key != allowed
    )
