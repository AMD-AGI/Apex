from __future__ import annotations

from copy import deepcopy

import pytest

from apex.core import ContractError
from apex.evaluation import E2EAcceptancePolicy, E2ERewardContract
from apex.intake import RegressionGates


def test_e2e_reward_contract_round_trips_without_runtime_paths() -> None:
    contract = E2ERewardContract(
        "run-contract",
        "a" * 64,
        E2EAcceptancePolicy(RegressionGates(0.0, 5.0, 2.0)),
    )
    document = contract.to_dict()

    assert E2ERewardContract.from_mapping(document) == contract
    assert document["objective_policy_hash"] == contract.objective_policy_hash
    assert not {"config_path", "results_dir", "workspace"} & set(document)
    assert "/home/" not in str(document) and "/tmp/" not in str(document)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("primary_metric",), "output_throughput"),
        (("measurement_protocol_hash",), "bad"),
        (("acceptance_policy", "min_paired_windows"), 2),
        (("acceptance_policy", "policy_id"), "dynamic_policy"),
    ],
)
def test_e2e_reward_contract_rejects_drift(
    path: tuple[str, ...], value: object
) -> None:
    document = E2ERewardContract(
        "run-contract", "a" * 64, E2EAcceptancePolicy()
    ).to_dict()
    changed = deepcopy(document)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ContractError) as raised:
        E2ERewardContract.from_mapping(changed)
    assert raised.value.reason_code == "invalid_e2e_reward_contract"
