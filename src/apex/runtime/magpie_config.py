"""Apex-owned workload projection over the published Magpie main config model."""

from __future__ import annotations

import copy
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from yaml.events import AliasEvent, CollectionEndEvent, CollectionStartEvent

from apex.core import ConfigurationError, sha256_bytes, sha256_file, sha256_json

from .magpie_main import MagpieMainPublicApi
from .magpie_fields import nested_unknown_fields
from .magpie_result_contract import (
    EXECUTION_ATTESTATION_SCHEMA,
    RESULT_SCHEMA,
    build_magpie_result_contract,
    validate_magpie_result_contract,
)
from .receipt import DependencyReceipt
from .evaluator_lock import EvaluatorPolicyLock


PLAN_SCHEMA = "apex.magpie-main-resolved-plan/v1"
CAPABILITY_SCHEMA = "apex.magpie-main-capability-receipt/v1"
REWARD_POLICY = "e2e_throughput_qos_v1"
REQUIRED_REWARD_METRICS = (
    "total_token_throughput", "ttft_p99", "tpot_p99", "quality"
)
RESOLUTION_METHOD = "apex_magpie_main_public_config_projection_v1"
_SERVING = frozenset({"atom", "sglang", "vllm"})
_COMMIT = re.compile(r"[0-9a-f]{40}")
_PUBLIC_FIELDS = frozenset(
    {
        "framework", "model", "precision", "run_mode", "envs", "profiler",
        "docker_image", "gpu_arch", "timeout_seconds", "inferencex_path",
        "inferencemax_path", "hf_cache_path", "gap_analysis", "runner_type",
        "benchmark_script", "gpu_selection", "ray_config", "server_lifecycle",
    }
)
_PLAN_FIELDS = {
    "schema", "raw_config_sha256", "effective_config_sha256",
    "scoring_config_sha256", "phase_views_sha256", "identity", "lifecycle",
    "phase_views", "diagnostics", "objective", "requirements",
    "source_runtime", "expected_result", "extensions", "redactions",
    "plan_sha256",
}
_CAPABILITY_FIELDS = {
    "schema", "plan_sha256", "status", "framework", "run_mode", "lifecycle",
    "optimization_applicable", "reward_contract", "capabilities", "blockers",
    "receipt_sha256",
}


@dataclass(frozen=True, slots=True)
class MagpieConfigContract:
    """Content-bound Apex plan derived from one exact published Magpie config."""

    config_path: Path
    config_sha256: str
    magpie_commit: str
    dependency_lock_sha256: str
    resolution_method_sha256: str
    plan: Mapping[str, Any]
    capability_receipt: Mapping[str, Any]

    @property
    def status(self) -> str:
        return str(self.capability_receipt["status"])

    @property
    def scoring_config(self) -> Mapping[str, Any]:
        return self.plan["phase_views"]["scoring_measurement"]

    @property
    def requested_components(self) -> tuple[str, ...]:
        return tuple(self.plan["source_runtime"]["requested_components"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "apex.magpie-config-contract/v1",
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "magpie_commit": self.magpie_commit,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "resolution_method_sha256": self.resolution_method_sha256,
            "plan": dict(self.plan),
            "capability_receipt": dict(self.capability_receipt),
        }


class MagpieMainConfigAdapter:
    """Build an Apex contract using only published Magpie main public APIs."""

    def __init__(
        self,
        dependency_receipt: DependencyReceipt,
        public_api: MagpieMainPublicApi | None = None,
    ) -> None:
        self._receipt = dependency_receipt
        self._root = dependency_receipt.root("magpie").resolve()
        self._public_api = public_api
        self._evaluator_policy = dependency_receipt.evaluator_policy

    def resolve(self, config_path: Path) -> MagpieConfigContract:
        selected = _regular_config(config_path)
        before, document = _strict_document(selected)
        raw = document["benchmark"]
        public_api = self._public_api or MagpieMainPublicApi(self._root)
        effective = public_api.load_and_normalize(selected, raw)
        after = selected.read_bytes()
        if after != before or sha256_bytes(after) != sha256_bytes(before):
            raise ConfigurationError(
                "Magpie config bytes changed during resolution",
                "benchmark_config_changed",
            )
        plan, capability = _build_documents(
            selected, document, effective, self._evaluator_policy
        )
        validate_apex_magpie_config_documents(selected, plan, capability)
        method = sha256_json(
            {
                "method": RESOLUTION_METHOD,
                "magpie_commit": self._receipt.commits.get("magpie", ""),
                "public_apis": [
                    "Magpie.main.load_benchmark_config",
                    "Magpie.modes.benchmark.config.BenchmarkConfig.from_dict",
                    "Magpie.modes.benchmark.config.BenchmarkConfig.to_dict",
                ],
            }
        )
        return MagpieConfigContract(
            selected,
            str(plan["raw_config_sha256"]),
            str(self._receipt.commits.get("magpie", "")),
            self._receipt.lock_sha256,
            method,
            plan,
            capability,
        )


def _build_documents(
    path: Path,
    document: Mapping[str, Any],
    effective: Mapping[str, Any],
    evaluator_policy: EvaluatorPolicyLock | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = dict(document["benchmark"])
    unknown = sorted(set(raw) - _PUBLIC_FIELDS)
    nested_unknown = nested_unknown_fields(raw)
    top_unknown = sorted(set(document) - {"benchmark"})
    blockers = [*(f"unrecognized_benchmark_field:{key}" for key in unknown)]
    blockers.extend(f"unrecognized_top_level_field:{key}" for key in top_unknown)
    blockers.extend(f"unrecognized_nested_field:{key}" for key in nested_unknown)
    if "sweep_matrix" in raw:
        blockers.append("sweep_matrix_requires_capability_upgrade")
    scoring = copy.deepcopy(dict(effective))
    blockers.extend(_apply_apex_scoring_policy(scoring, evaluator_policy))
    _disable_instrumentation(scoring)
    scoring["run_kind"] = "measurement"
    phase_full = {
        "requested": copy.deepcopy(dict(effective)),
        "scoring_measurement": scoring,
    }
    extensions_full = {
        "owner": "apex",
        "resolution_method": RESOLUTION_METHOD,
        "unrecognized_benchmark_fields": {key: raw[key] for key in unknown},
        "unrecognized_nested_fields": nested_unknown,
        "unrecognized_top_level_fields": {
            key: document[key] for key in top_unknown
        },
    }
    phase_views, phase_paths = _redact(phase_full, ("phase_views",))
    extensions, extension_paths = _redact(extensions_full, ("extensions",))
    identity = _identity(effective)
    lifecycle = _lifecycle(effective.get("server_lifecycle"))
    requested = _requested_components(effective)
    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "raw_config_sha256": sha256_file(path),
        "effective_config_sha256": sha256_json(effective),
        "scoring_config_sha256": sha256_json(scoring),
        "phase_views_sha256": sha256_json(phase_views),
        "identity": identity,
        "lifecycle": lifecycle,
        "phase_views": phase_views,
        "diagnostics": {"owner": "apex", "reward_eligible": False},
        "objective": {
            "owner": "apex", "primary": "total_token_throughput",
            "hard_gates": ["quality", "ttft_p99", "tpot_p99"],
        },
        "requirements": {"gpu_count": _gpu_count(effective)},
        "source_runtime": {
            "requested_image": effective.get("docker_image"),
            "benchmark_script": effective.get("benchmark_script"),
            "requested_components": requested,
            "potential_components": list(dict.fromkeys([*requested, "aiter", "triton", "flydsl"])),
            "observation": "required_at_runtime",
            "build_engagement_hooks": "adapter_required",
        },
        "expected_result": build_magpie_result_contract(),
        "extensions": extensions,
        "redactions": {
            "policy_id": "secret_name_markers_v2",
            "paths": sorted([*phase_paths, *extension_paths]),
        },
    }
    plan["plan_sha256"] = sha256_json(plan)
    return plan, _capability_document(plan, blockers)


def _capability_document(
    plan: Mapping[str, Any], blockers: list[str]
) -> dict[str, Any]:
    identity = plan["identity"]
    lifecycle = str(plan["lifecycle"])
    status = "config_compatible" if not blockers else "capability_upgrade_required"
    capability: dict[str, Any] = {
        "schema": CAPABILITY_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "status": status,
        "framework": identity["framework"],
        "run_mode": identity["run_mode"],
        "lifecycle": lifecycle,
        "optimization_applicable": lifecycle != "cleanup",
        "reward_contract": {
            "owner": "apex",
            "policy_id": REWARD_POLICY,
            "measurement_view_sha256": plan["scoring_config_sha256"],
            "required_metrics": list(REQUIRED_REWARD_METRICS),
        },
        "capabilities": _capabilities(identity),
        "blockers": sorted(set(blockers)),
    }
    capability["receipt_sha256"] = sha256_json(capability)
    return capability


def _apply_apex_scoring_policy(
    scoring: dict[str, Any], evaluator_policy: EvaluatorPolicyLock | None
) -> list[str]:
    if str(scoring.get("framework", "")).lower() not in _SERVING:
        return []
    if evaluator_policy is None:
        raise ConfigurationError(
            "Serving config lacks an evaluator policy lock",
            "evaluator_policy_lock_missing",
        )
    envs = scoring.setdefault("envs", {})
    if not isinstance(envs, dict):
        raise ConfigurationError("benchmark.envs must be a mapping", "invalid_benchmark_config")
    blockers: list[str] = []
    if "RUN_EVAL" in envs and not _enabled(envs["RUN_EVAL"]):
        blockers.append("quality_contract_disabled")
    else:
        envs["RUN_EVAL"] = "true"
    max_length = _positive_int(envs.get("MAX_MODEL_LEN")) or 2248
    policy = {
        **evaluator_policy.env(),
        "MAGPIE_EVAL_BATCH_SIZE": "auto",
        "MAGPIE_EVAL_MAX_LENGTH": str(max_length),
        "MAGPIE_EVAL_MAX_GEN_TOKENS": str(min(480, max_length - 1)),
    }
    for key, expected in policy.items():
        observed = envs.get(key)
        if observed is not None and str(observed) != expected:
            blockers.append(f"apex_evaluator_policy_override:{key}")
        envs[key] = expected
    return blockers


def _disable_instrumentation(scoring: dict[str, Any]) -> None:
    profiler = scoring.setdefault("profiler", {})
    gap = scoring.setdefault("gap_analysis", {})
    if not isinstance(profiler, dict) or not isinstance(gap, dict):
        raise ConfigurationError("Magpie instrumentation config is invalid", "invalid_benchmark_config")
    for value in profiler.values():
        if isinstance(value, dict):
            value["enabled"] = False
    gap["enabled"] = False


def _identity(effective: Mapping[str, Any]) -> dict[str, str]:
    framework = str(effective.get("framework", "")).lower()
    model = str(effective.get("model", ""))
    return {
        "framework": framework,
        "model": model,
        "model_sha256": sha256_json({"model": model}),
        "precision": str(effective.get("precision", "fp8")),
        "run_mode": str(effective.get("run_mode", "docker")).lower(),
        "workload_kind": "serving" if framework in _SERVING else "scriptable",
    }


def _lifecycle(value: object) -> str:
    if not isinstance(value, Mapping) or not _enabled(value.get("enabled")):
        return "one_shot"
    return "cleanup" if _enabled(value.get("cleanup")) else "reuse"


def _requested_components(effective: Mapping[str, Any]) -> list[str]:
    framework = str(effective.get("framework", ""))
    result = [framework]
    envs = effective.get("envs")
    if isinstance(envs, Mapping) and _enabled(envs.get("VLLM_ROCM_USE_AITER")):
        result.append("aiter")
    extra = envs.get("EXTRA_VLLM_ARGS") if isinstance(envs, Mapping) else None
    if isinstance(extra, str) and "--moe-backend flydsl" in extra:
        result.append("flydsl")
    return list(dict.fromkeys(result))


def _gpu_count(effective: Mapping[str, Any]) -> int:
    envs = effective.get("envs")
    value = envs.get("TP") if isinstance(envs, Mapping) else None
    return _positive_int(value) or 1


def _capabilities(identity: Mapping[str, str]) -> dict[str, str]:
    if identity["workload_kind"] == "scriptable":
        quality = "framework_gate_required"
    elif identity["run_mode"] == "ray":
        quality = "shared_runtime_required"
    else:
        quality = "runtime_required"
    return {
        "benchmark_execution": "published_magpie_main",
        "quality_evaluation": quality,
        "source_observation": "runtime_required",
        "source_build_engagement": "adapter_required",
    }


def validate_apex_magpie_config_documents(
    config_path: Path, plan_value: object, capability_value: object
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(plan_value, Mapping) or not isinstance(capability_value, Mapping):
        raise _invalid("Apex Magpie config contract did not contain two objects")
    plan, capability = dict(plan_value), dict(capability_value)
    if set(plan) != _PLAN_FIELDS or plan.get("schema") != PLAN_SCHEMA:
        raise ConfigurationError("Unsupported Apex Magpie plan schema", "unsupported_magpie_config_schema")
    if set(capability) != _CAPABILITY_FIELDS or capability.get("schema") != CAPABILITY_SCHEMA:
        raise ConfigurationError("Unsupported Apex Magpie capability schema", "unsupported_magpie_config_schema")
    _validate_plan(config_path, plan)
    _validate_capability(plan, capability)
    return plan, capability


def _validate_plan(config_path: Path, plan: Mapping[str, Any]) -> None:
    _, raw_document = _strict_document(_regular_config(config_path))
    phases = plan.get("phase_views")
    identity = plan.get("identity")
    valid = (
        all(_sha256(plan.get(key)) for key in (
            "raw_config_sha256", "effective_config_sha256",
            "scoring_config_sha256", "phase_views_sha256", "plan_sha256"
        ))
        and plan.get("raw_config_sha256") == sha256_file(config_path)
        and isinstance(phases, Mapping)
        and set(phases) == {"requested", "scoring_measurement"}
        and plan.get("phase_views_sha256") == sha256_json(phases)
        and isinstance(identity, Mapping)
        and set(identity) == {"framework", "model", "model_sha256", "precision", "run_mode", "workload_kind"}
        and all(isinstance(identity.get(key), str) and identity.get(key) for key in ("framework", "precision", "run_mode", "workload_kind"))
        and isinstance(plan.get("extensions"), Mapping)
        and isinstance(plan.get("redactions"), Mapping)
    )
    if not valid:
        raise _invalid("Apex Magpie plan fields are invalid")
    raw = raw_document["benchmark"]
    effective = _restore_redactions(phases["requested"], raw)
    scoring = _restore_redactions(phases["scoring_measurement"], raw)
    if sha256_json(effective) != plan["effective_config_sha256"] or sha256_json(scoring) != plan["scoring_config_sha256"]:
        raise _invalid("Apex Magpie effective configuration digest differs")
    validate_magpie_result_contract(plan.get("expected_result"))
    _validate_source_runtime(plan.get("source_runtime"), identity)
    if identity["workload_kind"] == "serving":
        _validate_scoring_quality(scoring)
    _validate_redactions(phases, plan["extensions"], plan["redactions"])
    if sha256_json({key: value for key, value in plan.items() if key != "plan_sha256"}) != plan["plan_sha256"]:
        raise _invalid("Apex Magpie plan digest differs")


def _validate_source_runtime(value: object, identity: Mapping[str, Any]) -> None:
    fields = {"requested_image", "benchmark_script", "requested_components", "potential_components", "observation", "build_engagement_hooks"}
    valid = isinstance(value, Mapping) and set(value) == fields
    if valid:
        requested, potential = value.get("requested_components"), value.get("potential_components")
        valid = _string_list(requested) and _string_list(potential) and requested[0] == identity.get("framework") and set(requested).issubset(potential) and value.get("observation") == "required_at_runtime" and value.get("build_engagement_hooks") == "adapter_required"
    if not valid:
        raise _invalid("Apex Magpie source-runtime contract is invalid")


def _validate_scoring_quality(value: Mapping[str, Any]) -> None:
    envs = value.get("envs")
    required = (
        "MAGPIE_EVAL_TASKS", "MAGPIE_EVAL_BATCH_SIZE",
        "MAGPIE_EVAL_POLICY_ID", "MAGPIE_EVAL_PRIMARY_METRIC",
        "MAGPIE_EVAL_TASK_DEFINITION_PATH",
        "MAGPIE_EVAL_TASK_DEFINITION_SHA256",
        "MAGPIE_EVAL_DATASET_PATH", "MAGPIE_EVAL_DATASET_NAME",
        "MAGPIE_EVAL_DATASET_REVISION", "EVAL_TASKS_DIR",
    )
    valid = isinstance(envs, Mapping) and all(isinstance(envs.get(key), str) and str(envs[key]).strip() for key in required)
    max_length = _positive_int(envs.get("MAGPIE_EVAL_MAX_LENGTH")) if isinstance(envs, Mapping) else None
    max_tokens = _positive_int(envs.get("MAGPIE_EVAL_MAX_GEN_TOKENS")) if isinstance(envs, Mapping) else None
    if (
        not valid
        or envs.get("EVAL_TASKS_DIR") != envs.get("MAGPIE_EVAL_TASK_DEFINITION_PATH")
        or not _sha256(envs.get("MAGPIE_EVAL_TASK_DEFINITION_SHA256"))
        or not _COMMIT.fullmatch(str(envs.get("MAGPIE_EVAL_DATASET_REVISION")))
        or max_length is None
        or max_tokens is None
        or max_tokens >= max_length
    ):
        raise _invalid("Apex scoring quality contract is incomplete")


def _validate_capability(plan: Mapping[str, Any], capability: Mapping[str, Any]) -> None:
    identity = plan["identity"]
    blockers, reward = capability.get("blockers"), capability.get("reward_contract")
    status = capability.get("status")
    valid = (
        capability.get("plan_sha256") == plan.get("plan_sha256")
        and status in {"config_compatible", "capability_upgrade_required"}
        and isinstance(blockers, list) and blockers == sorted(set(blockers))
        and all(isinstance(item, str) and item for item in blockers)
        and (status == "config_compatible") == (not blockers)
        and capability.get("framework") == identity.get("framework")
        and capability.get("run_mode") == identity.get("run_mode")
        and capability.get("lifecycle") == plan.get("lifecycle")
        and capability.get("optimization_applicable") == (plan.get("lifecycle") != "cleanup")
        and isinstance(reward, Mapping)
        and set(reward) == {"owner", "policy_id", "measurement_view_sha256", "required_metrics"}
        and reward.get("owner") == "apex" and reward.get("policy_id") == REWARD_POLICY
        and reward.get("measurement_view_sha256") == plan.get("scoring_config_sha256")
        and tuple(reward.get("required_metrics", ())) == REQUIRED_REWARD_METRICS
        and capability.get("capabilities") == _capabilities(identity)
        and _sha256(capability.get("receipt_sha256"))
    )
    if not valid:
        raise _invalid("Apex Magpie capability receipt conflicts with its plan")
    if sha256_json({key: value for key, value in capability.items() if key != "receipt_sha256"}) != capability["receipt_sha256"]:
        raise _invalid("Apex Magpie capability receipt digest differs")


def _strict_document(path: Path) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()
    try:
        depth, events = 0, 0
        for event in yaml.parse(payload):
            events += 1
            if events > 10000 or isinstance(event, AliasEvent):
                raise ValueError("aliases or excessive YAML events are forbidden")
            if isinstance(event, CollectionStartEvent):
                depth += 1
                if depth > 32:
                    raise ValueError("YAML nesting is excessive")
            elif isinstance(event, CollectionEndEvent):
                depth -= 1
        loaded = yaml.load(payload, Loader=_UniqueSafeLoader)
    except (ValueError, yaml.YAMLError, RecursionError) as error:
        raise ConfigurationError(f"Invalid benchmark YAML: {error}", "invalid_benchmark_config") from error
    if not isinstance(loaded, dict) or not isinstance(loaded.get("benchmark"), dict):
        raise ConfigurationError("Benchmark config must contain a benchmark mapping", "invalid_benchmark_config")
    _require_string_keys(loaded)
    return payload, loaded


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def _require_string_keys(value: object) -> None:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ConfigurationError("Benchmark YAML keys must be strings", "invalid_benchmark_config")
        for child in value.values():
            _require_string_keys(child)
    elif isinstance(value, list):
        for child in value:
            _require_string_keys(child)


def _redact(value: object, path: tuple[str, ...]) -> tuple[Any, list[str]]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            selected = (*path, str(key))
            if _is_secret_name(key):
                result[str(key)] = "<redacted>"
                paths.append(".".join(selected))
            else:
                result[str(key)], nested = _redact(child, selected)
                paths.extend(nested)
        return result, paths
    if isinstance(value, list):
        result = []
        for index, child in enumerate(value):
            redacted, nested = _redact(child, (*path, str(index)))
            result.append(redacted)
            paths.extend(nested)
        return result, paths
    return copy.deepcopy(value), paths


def _restore_redactions(value: object, raw: object) -> Any:
    if value == "<redacted>":
        if raw is None or raw == "<redacted>":
            raise _invalid("Apex redacted config value cannot be reconstructed")
        return copy.deepcopy(raw)
    if isinstance(value, Mapping):
        raw_map = raw if isinstance(raw, Mapping) else {}
        return {str(key): _restore_redactions(child, raw_map.get(key)) for key, child in value.items()}
    if isinstance(value, list):
        raw_list = raw if isinstance(raw, list) else []
        return [_restore_redactions(child, raw_list[index] if index < len(raw_list) else None) for index, child in enumerate(value)]
    return copy.deepcopy(value)


def _validate_redactions(phases: Mapping[str, Any], extensions: Mapping[str, Any], redactions: object) -> None:
    if not isinstance(redactions, Mapping) or set(redactions) != {"policy_id", "paths"} or redactions.get("policy_id") != "secret_name_markers_v2":
        raise _invalid("Apex Magpie redaction policy differs")
    _, phase_paths = _redact(phases, ("phase_views",))
    _, extension_paths = _redact(extensions, ("extensions",))
    paths = redactions.get("paths")
    if paths != sorted([*phase_paths, *extension_paths]):
        raise _invalid("Apex Magpie redaction evidence differs")


def _regular_config(path: Path) -> Path:
    selected = Path(path)
    try:
        observed = selected.lstat()
    except OSError as error:
        raise ConfigurationError(f"Cannot inspect Magpie config: {error}", "invalid_benchmark_config") from error
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1 or not 0 < observed.st_size <= 1024 * 1024:
        raise ConfigurationError("Magpie config must be one bounded non-linked regular file", "invalid_benchmark_config")
    return selected.resolve()


def _enabled(value: object) -> bool:
    return value is True or value == 1 or isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    text = str(value)
    return int(text) if text.isdigit() and not text.startswith("0") else None


def _is_secret_name(value: object) -> bool:
    normalized = str(value).upper().replace("-", "_")
    parts = tuple(part for part in normalized.split("_") if part)
    return any(part in {"TOKEN", "PASSWORD", "SECRET", "AUTHORIZATION"} for part in parts) or "API_KEY" in normalized


def _string_list(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and item for item in value)


def _sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not set(value) - set("0123456789abcdef")


def _invalid(message: str) -> ConfigurationError:
    return ConfigurationError(message, "invalid_magpie_config_resolution")


__all__ = [
    "CAPABILITY_SCHEMA", "MagpieConfigContract", "MagpieMainConfigAdapter",
    "EXECUTION_ATTESTATION_SCHEMA", "PLAN_SCHEMA", "REQUIRED_REWARD_METRICS",
    "RESULT_SCHEMA",
    "validate_apex_magpie_config_documents",
]
