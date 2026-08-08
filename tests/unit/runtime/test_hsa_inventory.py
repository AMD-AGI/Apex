from __future__ import annotations

from pathlib import Path

import pytest

from apex.core import ContractError
from apex.execution import ProcessResult
from apex.runtime import CleanHsaInventoryProvider


_OUTPUT = (
    '{"devices":[{"bdf_id":100,"domain":0,"generic_node_id":2,'
    '"hsa_gpu_index":0,"node_id":2,"unique_id":"GPU-0000000000000001"}],'
    '"schema_version":1}\n'
)


class _Supervisor:
    def __init__(self, stdout: str = _OUTPUT, *, exit_code: int = 0) -> None:
        self.stdout = stdout
        self.exit_code = exit_code
        self.environment: dict[str, str] = {}
        self.mutate: Path | None = None

    def run(self, argv, *, cwd, environment, timeout_seconds, **_kwargs):
        self.environment = dict(environment)
        if self.mutate is not None:
            self.mutate.write_text("changed", encoding="utf-8")
        return ProcessResult(
            argv=tuple(argv),
            exit_code=self.exit_code,
            timed_out=False,
            stdout=self.stdout,
            stderr="failure" if self.exit_code else "",
            stdout_truncated=False,
            stderr_truncated=False,
            duration_seconds=0.01,
        )


def _provider(tmp_path: Path, supervisor: _Supervisor) -> CleanHsaInventoryProvider:
    helper = tmp_path / "helper.py"
    library = tmp_path / "libhsa.so"
    helper.write_text("helper", encoding="utf-8")
    library.write_bytes(b"library")
    return CleanHsaInventoryProvider(
        helper_path=helper,
        library_path=library,
        python_path=Path("/usr/bin/python3"),
        supervisor=supervisor,
    )


def test_clean_helper_does_not_inherit_visibility_and_binds_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in (
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.setenv(name, "0")
    supervisor = _Supervisor()

    evidence = _provider(tmp_path, supervisor).collect()

    assert evidence.devices[0].node_id == 2
    assert evidence.devices[0].unique_id == "GPU-0000000000000001"
    assert len(evidence.helper_sha256) == 64
    assert not set(supervisor.environment).intersection(
        {
            "ROCR_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
        }
    )


@pytest.mark.parametrize(("stdout", "exit_code"), [("{}", 0), ("", 1)])
def test_helper_failure_or_malformed_output_fails_closed(
    tmp_path: Path, stdout: str, exit_code: int
) -> None:
    with pytest.raises(ContractError) as raised:
        _provider(tmp_path, _Supervisor(stdout, exit_code=exit_code)).collect()

    assert raised.value.reason_code == "gpu_hsa_helper_failed"


def test_helper_bytes_changing_during_enumeration_fail_closed(tmp_path: Path) -> None:
    supervisor = _Supervisor()
    provider = _provider(tmp_path, supervisor)
    supervisor.mutate = tmp_path / "helper.py"

    with pytest.raises(ContractError) as raised:
        provider.collect()

    assert raised.value.reason_code == "gpu_hsa_helper_changed"
