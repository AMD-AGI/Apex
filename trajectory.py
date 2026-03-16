# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
trajectory.py — Trajectory persistence for the RL kernel-optimization pipeline.

Captures full agent trajectories (every message, tool call, grading result) in a
structured format suitable for RL training data export.

Storage backends:
  - FileStore:    JSONL files in trajectories/ directory (dev/local)
  - CouchDBStore: Push to CouchDB (integrates with Grafana dashboards)
  - S3Store:      Upload to S3 bucket for large-scale RL training
"""

from __future__ import annotations

import json
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent


@dataclass
class TrajectoryRecord:
    """One complete agent run: task context, iterations, and final outcome."""

    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    agent_model: str = ""
    agent_version: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Task context
    prompt: str = ""
    baseline_tps: float = 0.0
    gpu_arch: str = ""
    model_id: str = ""
    kernel_type: str = ""
    framework: str = ""

    # Iterations — each dict contains: messages, tool_calls, solution_code, score, reflection
    iterations: list[dict] = field(default_factory=list)

    # Final outcome
    final_score: float = 0.0
    final_speedup: float = 0.0
    final_tps: float = 0.0
    throughput_improvement: float = 0.0

    # RL training fields
    reward: float = 0.0
    trajectory_quality: str = "unknown"  # good | mediocre | bad | unknown

    # Raw results for debugging
    baseline_benchmark: dict = field(default_factory=dict)
    final_benchmark: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def compute_reward(self) -> float:
        """Compute a normalised reward signal from grading outcomes."""
        reward = 0.0
        if self.final_score > 0:
            reward += min(self.final_score / 500.0, 1.0)
        if self.throughput_improvement > 1.0:
            reward += min(self.throughput_improvement - 1.0, 1.0)
        self.reward = round(reward, 4)

        if self.reward >= 0.8:
            self.trajectory_quality = "good"
        elif self.reward >= 0.3:
            self.trajectory_quality = "mediocre"
        else:
            self.trajectory_quality = "bad"

        return self.reward

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TrajectoryRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


@dataclass
class WorkloadTrajectoryRecord:
    """Full workload optimization trajectory: benchmark → bottleneck → optimize → re-benchmark."""

    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workload_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Agent
    agent_model: str = ""
    agent_version: str = ""

    # Benchmark config
    benchmark_config_path: str = ""
    benchmark_config: dict = field(default_factory=dict)
    framework: str = ""
    model_id: str = ""
    gpu_arch: str = ""

    # Baseline benchmark
    baseline_benchmark: dict = field(default_factory=dict)
    baseline_tps: float = 0.0

    # Bottleneck discovery
    bottleneck_kernels: list[dict] = field(default_factory=list)
    kernel_type_filter: list[str] = field(default_factory=list)
    selected_kernels: list[str] = field(default_factory=list)

    # Per-kernel optimization
    kernel_optimizations: list[dict] = field(default_factory=list)

    # Re-injection
    reinjected_kernels: list[str] = field(default_factory=list)

    # Final benchmark
    final_benchmark: dict = field(default_factory=dict)
    final_tps: float = 0.0

    # Reward
    per_kernel_scores: list[float] = field(default_factory=list)
    avg_kernel_score: float = 0.0
    normalized_kernel_score: float = 0.0
    model_reward: float = 0.0
    trajectory_quality: str = "unknown"

    # Metadata
    skip_benchmark_used: bool = False
    total_duration_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def apply_reward(self, reward_dict: dict) -> None:
        """Apply reward results from trajectory_reward()."""
        self.per_kernel_scores = reward_dict.get("per_kernel_scores", [])
        self.avg_kernel_score = reward_dict.get("avg_kernel_score", 0.0)
        self.normalized_kernel_score = reward_dict.get("normalized_kernel_score", 0.0)
        self.model_reward = reward_dict.get("model_reward", 0.0)

        if self.model_reward >= 0.8:
            self.trajectory_quality = "good"
        elif self.model_reward >= 0.3:
            self.trajectory_quality = "mediocre"
        else:
            self.trajectory_quality = "bad"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> WorkloadTrajectoryRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


class TrajectoryStore(ABC):
    """Pluggable storage backend for trajectory records."""

    @abstractmethod
    def save(self, record: TrajectoryRecord) -> str:
        """Persist a trajectory. Returns the trajectory_id."""

    @abstractmethod
    def load(self, trajectory_id: str) -> TrajectoryRecord | None:
        """Load a trajectory by ID. Returns None if not found."""

    @abstractmethod
    def list_ids(self, filters: dict | None = None) -> list[str]:
        """List trajectory IDs, optionally filtered."""

    def export_for_rl(
        self, output_path: Path, quality: str | None = None, fmt: str = "jsonl"
    ) -> int:
        """Export trajectories to a file for RL training. Returns count exported."""
        ids = self.list_ids({"trajectory_quality": quality} if quality else None)
        count = 0
        with open(output_path, "w") as f:
            for tid in ids:
                record = self.load(tid)
                if record is None:
                    continue
                if quality and record.trajectory_quality != quality:
                    continue
                if fmt == "jsonl":
                    f.write(json.dumps(record.to_dict(), default=str) + "\n")
                count += 1
        return count

    def export_for_keystone_rl(
        self,
        output_dir: Path,
        results_dirs: list[Path] | None = None,
        quality: str | None = None,
        include_sft: bool = False,
        min_score: float = 0.0,
        gpu_arch: str = "gfx950",
    ) -> dict:
        """Export trajectories to keystone-rl-training format.

        Delegates to export_rl_dataset.export() after loading all workload
        trajectories from this store.

        Returns dict with counts: tasks_exported, sft_pairs_exported.
        """
        from export_rl_dataset import export

        traj_dir = self.base_dir if hasattr(self, "base_dir") else REPO_ROOT / "trajectories"
        if results_dirs is None:
            results_dirs = [REPO_ROOT / "output"]

        return export(
            trajectories_dir=traj_dir,
            results_dirs=results_dirs,
            output_dir=output_dir,
            include_sft=include_sft,
            quality_filter=quality,
            min_score=min_score,
            gpu_arch=gpu_arch,
        )


def _record_from_dict(d: dict) -> TrajectoryRecord | WorkloadTrajectoryRecord:
    """Deserialize a dict into the appropriate record type."""
    if "workload_id" in d:
        return WorkloadTrajectoryRecord.from_dict(d)
    return TrajectoryRecord.from_dict(d)


class FileStore(TrajectoryStore):
    """Store trajectories as individual JSON files in a directory."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or REPO_ROOT / "trajectories"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, trajectory_id: str) -> Path:
        return self.base_dir / f"{trajectory_id}.json"

    def save(self, record: TrajectoryRecord) -> str:
        path = self._path(record.trajectory_id)
        with open(path, "w") as f:
            json.dump(record.to_dict(), f, indent=2, default=str)
        return record.trajectory_id

    def load(self, trajectory_id: str) -> TrajectoryRecord | WorkloadTrajectoryRecord | None:
        path = self._path(trajectory_id)
        if not path.exists():
            return None
        with open(path) as f:
            return _record_from_dict(json.load(f))

    def list_ids(self, filters: dict | None = None) -> list[str]:
        ids = []
        for p in sorted(self.base_dir.glob("*.json")):
            tid = p.stem
            if filters:
                record = self.load(tid)
                if record is None:
                    continue
                skip = False
                for k, v in filters.items():
                    if v is not None and getattr(record, k, None) != v:
                        skip = True
                        break
                if skip:
                    continue
            ids.append(tid)
        return ids


class CouchDBStore(TrajectoryStore):
    """Store trajectories in CouchDB (integrates with existing Grafana infra)."""

    def __init__(
        self,
        url: str | None = None,
        db_name: str = "keystone-trajectories",
        auth: tuple[str, str] | None = None,
    ):
        self.url = (url or os.environ.get("COUCHDB_URL", "http://localhost:5984")).rstrip("/")
        self.db_name = db_name
        self.auth = auth or (
            os.environ.get("COUCHDB_USER", "admin"),
            os.environ.get("COUCHDB_PASS", "admin"),
        )
        self._ensure_db()

    def _ensure_db(self) -> None:
        try:
            import requests
            requests.put(f"{self.url}/{self.db_name}", auth=self.auth, timeout=5)
        except Exception:
            pass

    def save(self, record: TrajectoryRecord) -> str:
        import requests
        doc = record.to_dict()
        doc["_id"] = f"trajectory:{record.trajectory_id}"
        doc["type"] = "trajectory"
        try:
            resp = requests.put(
                f"{self.url}/{self.db_name}/{doc['_id']}",
                json=doc, auth=self.auth, timeout=10,
            )
            resp.raise_for_status()
        except Exception:
            FileStore().save(record)
        return record.trajectory_id

    def load(self, trajectory_id: str) -> TrajectoryRecord | None:
        import requests
        doc_id = f"trajectory:{trajectory_id}"
        try:
            resp = requests.get(
                f"{self.url}/{self.db_name}/{doc_id}",
                auth=self.auth, timeout=10,
            )
            if resp.status_code == 200:
                return TrajectoryRecord.from_dict(resp.json())
        except Exception:
            pass
        return None

    def list_ids(self, filters: dict | None = None) -> list[str]:
        import requests
        try:
            resp = requests.get(
                f"{self.url}/{self.db_name}/_all_docs",
                params={"startkey": '"trajectory:"', "endkey": '"trajectory:\ufff0"'},
                auth=self.auth, timeout=10,
            )
            if resp.status_code == 200:
                rows = resp.json().get("rows", [])
                return [r["id"].replace("trajectory:", "") for r in rows]
        except Exception:
            pass
        return []


class S3Store(TrajectoryStore):
    """Store trajectories in S3 (for large-scale RL training data)."""

    def __init__(
        self,
        bucket: str | None = None,
        prefix: str = "trajectories/",
        region: str | None = None,
    ):
        self.bucket = bucket or os.environ.get("TRAJECTORY_S3_BUCKET", "keystone-trajectories")
        self.prefix = prefix
        self.region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self._file_fallback = FileStore()

    def _key(self, trajectory_id: str) -> str:
        return f"{self.prefix}{trajectory_id}.json"

    def _client(self) -> Any:
        import boto3
        return boto3.client("s3", region_name=self.region)

    def save(self, record: TrajectoryRecord) -> str:
        try:
            s3 = self._client()
            body = json.dumps(record.to_dict(), default=str)
            s3.put_object(Bucket=self.bucket, Key=self._key(record.trajectory_id), Body=body)
        except Exception:
            self._file_fallback.save(record)
        return record.trajectory_id

    def load(self, trajectory_id: str) -> TrajectoryRecord | None:
        try:
            s3 = self._client()
            obj = s3.get_object(Bucket=self.bucket, Key=self._key(trajectory_id))
            data = json.loads(obj["Body"].read().decode())
            return TrajectoryRecord.from_dict(data)
        except Exception:
            return None

    def list_ids(self, filters: dict | None = None) -> list[str]:
        try:
            s3 = self._client()
            resp = s3.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix)
            ids = []
            for obj in resp.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".json"):
                    tid = key[len(self.prefix):].replace(".json", "")
                    ids.append(tid)
            return ids
        except Exception:
            return []


def get_store(backend: str = "file", **kwargs: Any) -> TrajectoryStore:
    """Factory for trajectory stores."""
    if backend == "couchdb":
        return CouchDBStore(**kwargs)
    if backend == "s3":
        return S3Store(**kwargs)
    return FileStore(**kwargs)
