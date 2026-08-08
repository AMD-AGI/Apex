from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from apex.core import ContractError
from apex.runtime import (
    GpuDeviceIdentity,
    GpuOwnershipReceipt,
    GpuProcessIdentity,
    GpuSelectorRequest,
    HsaGpuIdentity,
    HsaInventoryEvidence,
    LocalGpuLeaseManager,
    RsmiDeviceIdentity,
    collect_gpu_ownership,
)


_UUID_ONE = "GPU-0000000000000001"
_UUID_TWO = "GPU-0000000000000002"


class _FakeApi:
    def __init__(
        self,
        snapshots: list[dict[int, tuple[int, ...]]] | None = None,
        *,
        init_status: int = 0,
        shutdown_status: int = 0,
        device_status: int = 0,
    ) -> None:
        self.snapshots = snapshots or [{}, {}]
        self.init_status = init_status
        self.shutdown_status = shutdown_status
        self.device_status = device_status
        self._snapshot: dict[int, tuple[int, ...]] = {}
        self._next = 0

    def init(self) -> int:
        return self.init_status

    def shutdown(self) -> int:
        return self.shutdown_status

    def device_count(self) -> tuple[int, int]:
        return self.device_status, 2

    def device_identity(self, index: int) -> tuple[int, int, int, int, int]:
        # RSMI order is deliberately different from clean HSA order.
        identities = (
            (0, 2, 3, 200, 129),
            (0, 1, 2, 100, 128),
        )
        return identities[index]

    def process_pids(self) -> tuple[int, tuple[int, ...]]:
        self._snapshot = self.snapshots[min(self._next, len(self.snapshots) - 1)]
        self._next += 1
        return 0, tuple(self._snapshot)

    def process_devices(self, pid: int) -> tuple[int, tuple[int, ...]]:
        return 0, self._snapshot[pid]


def _hsa_inventory() -> HsaInventoryEvidence:
    return HsaInventoryEvidence(
        schema_version=1,
        policy_id="clean_unfiltered_hsa_gpu_inventory_v1",
        helper_path="/trusted/hsa-helper.py",
        helper_sha256="b" * 64,
        library_path="/opt/rocm/lib/libhsa-runtime64.so.1",
        library_sha256="c" * 64,
        devices=(
            HsaGpuIdentity(0, 2, 2, 100, 0, _UUID_ONE),
            HsaGpuIdentity(1, 3, 3, 200, 0, _UUID_TWO),
        ),
    )


def _single_hsa_inventory() -> HsaInventoryEvidence:
    inventory = _hsa_inventory()
    return HsaInventoryEvidence(
        schema_version=inventory.schema_version,
        policy_id=inventory.policy_id,
        helper_path=inventory.helper_path,
        helper_sha256=inventory.helper_sha256,
        library_path=inventory.library_path,
        library_sha256=inventory.library_sha256,
        devices=(inventory.devices[0],),
    )


class _SingleVisibleApi(_FakeApi):
    def device_count(self) -> tuple[int, int]:
        return self.device_status, 1

    def device_identity(self, index: int) -> tuple[int, int, int, int, int]:
        assert index == 0
        return 0, 1, 2, 100, 128


class _HostGlobalApi(_FakeApi):
    def device_count(self) -> tuple[int, int]:
        return self.device_status, 8

    def device_identity(self, index: int) -> tuple[int, int, int, int, int]:
        if index == 3:
            return 0, 1, 2, 100, 128
        return 0, 100 + index, 10 + index, 1000 + index, 200 + index


class _NoMatchingRsmiApi(_FakeApi):
    def device_count(self) -> tuple[int, int]:
        return self.device_status, 1

    def device_identity(self, index: int) -> tuple[int, int, int, int, int]:
        assert index == 0
        return 0, 9, 9, 900, 190


class _DuplicateMatchingRsmiApi(_FakeApi):
    def device_identity(self, index: int) -> tuple[int, int, int, int, int]:
        return 0, 1, 2, 100, 128


def _topology(root: Path) -> Path:
    for node, unique_id, bdf, render in ((2, 1, 100, 128), (3, 2, 200, 129)):
        path = root / str(node)
        path.mkdir(parents=True)
        (path / "properties").write_text(
            "\n".join(
                (
                    "cpu_cores_count 0",
                    "simd_count 1",
                    f"location_id {bdf}",
                    "domain 0",
                    f"drm_render_minor {render}",
                    f"unique_id {unique_id}",
                )
            )
            + "\n",
            encoding="ascii",
        )
    return root


def _add_host_only_topology_nodes(root: Path) -> None:
    for node in range(4, 10):
        path = root / str(node)
        path.mkdir(parents=True)
        (path / "properties").write_text(
            "\n".join(
                (
                    "cpu_cores_count 0",
                    "simd_count 1",
                    f"location_id {node * 100}",
                    "domain 0",
                    f"drm_render_minor {128 + node}",
                    f"unique_id {node}",
                )
            )
            + "\n",
            encoding="ascii",
        )


def _proc_process(root: Path, pid: int, *, start: int = 777) -> None:
    process = root / str(pid)
    process.mkdir(parents=True)
    fields = ["S", *("0" for _ in range(18)), str(start)]
    (process / "stat").write_text(
        f"{pid} (gpu worker) " + " ".join(fields) + "\n", encoding="utf-8"
    )
    (process / "cmdline").write_bytes(b"worker\x00--token=redacted-by-digest\x00")


def _collect(
    api: _FakeApi,
    root: Path,
    *,
    selector: str = "amd-gpu-set=0",
    selector_inputs: GpuSelectorRequest | None = None,
    allowed: tuple[int, ...] = (),
) -> GpuOwnershipReceipt:
    return collect_gpu_ownership(
        api,
        selector_scope=selector,
        selector_inputs=selector_inputs,
        allowed_pids=allowed,
        library_path=Path("/opt/rocm/lib/librocm_smi64.so.7"),
        library_sha256="a" * 64,
        topology_root=_topology(root / "kfd"),
        hsa_inventory=_hsa_inventory(),
        proc_root=root / "proc",
        observed_unix_ns=123,
    )


def test_numeric_selector_uses_hsa_ordinal_not_rsmi_index(tmp_path: Path) -> None:
    receipt = _collect(_FakeApi(), tmp_path)

    assert receipt.selected_devices == (
        GpuDeviceIdentity(0, 2, 1, _UUID_ONE, "/dev/dri/renderD128"),
    )
    assert receipt.device_inventory[1].rsmi_index == 0
    assert receipt.execution_scope == f"amd-gpu-set={_UUID_ONE}"
    assert receipt.physical_scope == f"amd-gpu-unique-id-set={_UUID_ONE}"
    assert receipt.foreign_owners == ()
    assert len(receipt.digest) == 64


def test_uuid_request_and_magpie_ambient_resolve_to_same_gpu(tmp_path: Path) -> None:
    inputs = GpuSelectorRequest(
        requested=(_UUID_ONE,),
        rocr_visible_devices=("0",),
        hip_visible_devices=("0",),
    )
    receipt = _collect(
        _FakeApi(),
        tmp_path,
        selector=f"amd-gpu-set={_UUID_ONE}",
        selector_inputs=inputs,
    )

    assert receipt.selected_devices[0].rsmi_index == 1
    assert receipt.selector_inputs == inputs


def test_aka_single_gpu_aliases_resolve_to_one_physical_device(
    tmp_path: Path,
) -> None:
    inputs = GpuSelectorRequest(
        rocr_visible_devices=("0",),
        hip_visible_devices=("0",),
        cuda_visible_devices=("0",),
        gpu_device_ordinal=("0",),
    )
    receipt = _collect(
        _FakeApi(),
        tmp_path,
        selector="all-visible-amd-gpus",
        selector_inputs=inputs,
    )

    assert receipt.selected_devices == (
        GpuDeviceIdentity(0, 2, 1, _UUID_ONE, "/dev/dri/renderD128"),
    )


def test_rocr_order_then_hip_ordinals_are_composed(tmp_path: Path) -> None:
    inputs = GpuSelectorRequest(
        requested=("1", "0"),
        rocr_visible_devices=("1", "0"),
        hip_visible_devices=("0", "1"),
    )
    receipt = _collect(
        _FakeApi(),
        tmp_path,
        selector="amd-gpu-set=1,0",
        selector_inputs=inputs,
    )

    assert tuple(item.unique_id for item in receipt.selected_devices) == (
        _UUID_TWO,
        _UUID_ONE,
    )


def test_conflicting_ambient_selector_fails_closed(tmp_path: Path) -> None:
    inputs = GpuSelectorRequest(
        requested=("0",),
        rocr_visible_devices=("0", "1"),
        hip_visible_devices=("1",),
    )
    with pytest.raises(ContractError) as raised:
        _collect(
            _FakeApi(),
            tmp_path,
            selector="amd-gpu-set=0",
            selector_inputs=inputs,
        )

    assert raised.value.reason_code == "gpu_visibility_mismatch"


def test_kfd_owner_intersection_uses_rsmi_index(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    _proc_process(proc, 4242)
    receipt = _collect(_FakeApi([{4242: (1,)}, {4242: (1,)}]), tmp_path)

    assert receipt.foreign_owners[0].pid == 4242
    assert receipt.foreign_owners[0].rsmi_device_indices == (1,)


def test_current_runner_may_be_recorded_as_an_allowed_owner(tmp_path: Path) -> None:
    _proc_process(tmp_path / "proc", 4242)
    receipt = _collect(
        _FakeApi([{4242: (1,)}, {4242: (1,)}]), tmp_path, allowed=(4242,)
    )

    assert receipt.foreign_owners == ()
    assert receipt.allowed_owners[0].pid == 4242


def test_ownership_change_during_query_fails_closed(tmp_path: Path) -> None:
    _proc_process(tmp_path / "proc", 4242)
    with pytest.raises(ContractError) as raised:
        _collect(_FakeApi([{}, {4242: (1,)}]), tmp_path)

    assert raised.value.reason_code == "gpu_ownership_race"


def test_process_map_rejects_rsmi_index_outside_monitor_inventory(
    tmp_path: Path,
) -> None:
    with pytest.raises(ContractError) as raised:
        _collect(_FakeApi([{4242: (99,)}]), tmp_path)

    assert raised.value.reason_code == "gpu_ownership_query_failed"


@pytest.mark.parametrize(
    ("api", "reason"),
    [
        (_FakeApi(init_status=4), "gpu_ownership_query_failed"),
        (_FakeApi(device_status=4), "gpu_physical_mapping_unresolved"),
        (_FakeApi(shutdown_status=4), "gpu_ownership_query_failed"),
    ],
)
def test_every_api_failure_is_fail_closed(
    tmp_path: Path, api: _FakeApi, reason: str
) -> None:
    with pytest.raises(ContractError) as raised:
        _collect(api, tmp_path)

    assert raised.value.reason_code == reason


def test_kfd_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    topology = _topology(tmp_path / "kfd")
    (topology / "2" / "properties").write_text(
        "cpu_cores_count 0\nsimd_count 1\nlocation_id 999\n"
        "domain 0\ndrm_render_minor 128\nunique_id 1\n",
        encoding="ascii",
    )
    with pytest.raises(ContractError) as raised:
        collect_gpu_ownership(
            _FakeApi(),
            selector_scope="amd-gpu-set=0",
            allowed_pids=(),
            library_path=Path("/opt/rocm/lib/librocm_smi64.so.7"),
            library_sha256="a" * 64,
            topology_root=topology,
            hsa_inventory=_hsa_inventory(),
            proc_root=tmp_path / "proc",
        )

    assert raised.value.reason_code == "gpu_physical_mapping_unresolved"


def test_receipt_rejects_duplicate_monitor_identity(tmp_path: Path) -> None:
    receipt = _collect(_FakeApi(), tmp_path)
    first = receipt.rsmi_monitor_inventory[0]
    duplicate = RsmiDeviceIdentity(
        1, first.node_id, first.pci_id, first.unique_id, first.render_minor
    )

    with pytest.raises(ContractError) as raised:
        replace(receipt, rsmi_monitor_inventory=(first, duplicate))

    assert raised.value.reason_code == "invalid_gpu_ownership_receipt"


def test_receipt_rejects_duplicate_joined_rsmi_index(tmp_path: Path) -> None:
    receipt = _collect(_FakeApi(), tmp_path)
    duplicated = replace(receipt.device_inventory[1], rsmi_index=1)

    with pytest.raises(ContractError) as raised:
        replace(
            receipt,
            device_inventory=(receipt.device_inventory[0], duplicated),
        )

    assert raised.value.reason_code == "invalid_gpu_ownership_receipt"


def test_receipt_rejects_owner_outside_monitor_inventory(tmp_path: Path) -> None:
    receipt = _collect(_FakeApi(), tmp_path)
    owner = GpuProcessIdentity(4242, 1000, 777, "b" * 64, (99,))

    with pytest.raises(ContractError) as raised:
        replace(receipt, allowed_owners=(owner,))

    assert raised.value.reason_code == "invalid_gpu_ownership_receipt"


def test_host_global_kfd_nodes_are_ignored_for_single_gpu_container(
    tmp_path: Path,
) -> None:
    topology = _topology(tmp_path / "kfd")
    _add_host_only_topology_nodes(topology)

    receipt = collect_gpu_ownership(
        _HostGlobalApi(),
        selector_scope="amd-gpu-set=0",
        allowed_pids=(),
        library_path=Path("/opt/rocm/lib/librocm_smi64.so.7"),
        library_sha256="a" * 64,
        topology_root=topology,
        hsa_inventory=_single_hsa_inventory(),
        proc_root=tmp_path / "proc",
    )

    assert receipt.selected_devices == (
        GpuDeviceIdentity(0, 2, 3, _UUID_ONE, "/dev/dri/renderD128"),
    )
    assert len(tuple(topology.iterdir())) == 8
    assert len(receipt.rsmi_monitor_inventory) == 8


@pytest.mark.parametrize("failure", ["missing", "tampered"])
def test_selected_kfd_node_must_exist_and_match(
    tmp_path: Path, failure: str
) -> None:
    topology = _topology(tmp_path / "kfd")
    selected = topology / "2" / "properties"
    if failure == "missing":
        selected.unlink()
    else:
        selected.write_text(
            "cpu_cores_count 0\nsimd_count 1\nlocation_id 999\n"
            "domain 0\ndrm_render_minor 128\nunique_id 1\n",
            encoding="ascii",
        )

    with pytest.raises(ContractError) as raised:
        collect_gpu_ownership(
            _SingleVisibleApi(),
            selector_scope="amd-gpu-set=0",
            allowed_pids=(),
            library_path=Path("/opt/rocm/lib/librocm_smi64.so.7"),
            library_sha256="a" * 64,
            topology_root=topology,
            hsa_inventory=_single_hsa_inventory(),
            proc_root=tmp_path / "proc",
        )

    assert raised.value.reason_code == "gpu_physical_mapping_unresolved"


@pytest.mark.parametrize(
    "api", [_NoMatchingRsmiApi(), _DuplicateMatchingRsmiApi()]
)
def test_hsa_device_requires_exactly_one_rsmi_peer(
    tmp_path: Path, api: _FakeApi
) -> None:
    topology = _topology(tmp_path / "kfd")

    with pytest.raises(ContractError) as raised:
        collect_gpu_ownership(
            api,
            selector_scope="amd-gpu-set=0",
            allowed_pids=(),
            library_path=Path("/opt/rocm/lib/librocm_smi64.so.7"),
            library_sha256="a" * 64,
            topology_root=topology,
            hsa_inventory=_single_hsa_inventory(),
            proc_root=tmp_path / "proc",
        )

    assert raised.value.reason_code == "gpu_physical_mapping_unresolved"


class _ForeignInspector:
    def inspect(
        self, selector_scope: str, *, allowed_pids: tuple[int, ...] = ()
    ) -> GpuOwnershipReceipt:
        receipt = _collect(_FakeApi(), Path(self.root))
        owner = GpuProcessIdentity(4242, 1000, 777, "b" * 64, (1,))
        return GpuOwnershipReceipt(
            schema_version=2,
            policy_id=receipt.policy_id,
            selector_inputs=receipt.selector_inputs,
            observed_unix_ns=receipt.observed_unix_ns,
            library_path=receipt.library_path,
            library_sha256=receipt.library_sha256,
            topology_root=receipt.topology_root,
            hsa_inventory=receipt.hsa_inventory,
            rsmi_monitor_inventory=receipt.rsmi_monitor_inventory,
            device_inventory=receipt.device_inventory,
            selected_devices=receipt.selected_devices,
            allowed_owners=(),
            foreign_owners=(owner,),
        )

    def __init__(self, root: Path) -> None:
        self.root = root


def test_lease_refuses_foreign_owner_without_terminating_it(tmp_path: Path) -> None:
    manager = LocalGpuLeaseManager(
        lock_root=tmp_path / "leases",
        ownership_inspector=_ForeignInspector(tmp_path / "inspection"),
    )

    with pytest.raises(ContractError) as raised:
        manager.acquire("run", requested_devices="0")

    assert raised.value.reason_code == "gpu_foreign_owner"
    assert raised.value.details["ownership_receipt"]["foreign_owners"][0]["pid"] == 4242
    assert not (tmp_path / "leases").exists()
