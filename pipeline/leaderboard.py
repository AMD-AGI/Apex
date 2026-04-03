# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
leaderboard.py — Agent performance leaderboard for the RL kernel-optimization pipeline.

Tracks agent scores over time so RL fine-tuning iterations can be compared.
Supports CouchDB (integrates with Grafana) and file-based backends.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent


@dataclass
class LeaderboardEntry:
    """One scored run on the leaderboard."""

    agent_model: str = ""
    agent_version: str = ""
    task_id: str = ""
    kernel_type: str = ""
    model_id: str = ""
    gpu_arch: str = ""

    # Scores
    kernel_score: float = 0.0
    model_score: float = 0.0
    arena_score: float = 0.0

    # Performance
    baseline_tps: float = 0.0
    optimized_tps: float = 0.0
    throughput_ratio: float = 0.0
    speedup: float = 0.0

    # Meta
    iterations_used: int = 0
    total_agent_turns: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    trajectory_id: str = ""

    def doc_id(self) -> str:
        ts = self.timestamp.replace(":", "-").replace("+", "p")
        return f"leaderboard:{self.agent_version}:{self.task_id}:{ts}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_id"] = self.doc_id()
        d["type"] = "leaderboard"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> LeaderboardEntry:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


class Leaderboard:
    """Leaderboard with pluggable backend (file or CouchDB)."""

    def __init__(self, backend: str = "file", **kwargs: Any):
        self.backend = backend
        if backend == "couchdb":
            self._store = _CouchDBLeaderboard(**kwargs)
        else:
            self._store = _FileLeaderboard(**kwargs)

    def push(self, entry: LeaderboardEntry) -> None:
        self._store.push(entry)

    def get_rankings(
        self, task_id: str | None = None, limit: int = 50
    ) -> list[LeaderboardEntry]:
        entries = self._store.all_entries()
        if task_id:
            entries = [e for e in entries if e.task_id == task_id]
        entries.sort(key=lambda e: e.arena_score, reverse=True)
        return entries[:limit]

    def get_agent_history(self, agent_version: str) -> list[LeaderboardEntry]:
        entries = self._store.all_entries()
        history = [e for e in entries if e.agent_version == agent_version]
        history.sort(key=lambda e: e.timestamp)
        return history

    def compare_versions(self, v1: str, v2: str) -> dict:
        h1 = self.get_agent_history(v1)
        h2 = self.get_agent_history(v2)

        def _avg(entries: list[LeaderboardEntry], attr: str) -> float:
            vals = [getattr(e, attr) for e in entries]
            return sum(vals) / len(vals) if vals else 0.0

        return {
            "v1": v1,
            "v2": v2,
            "v1_runs": len(h1),
            "v2_runs": len(h2),
            "v1_avg_arena": round(_avg(h1, "arena_score"), 2),
            "v2_avg_arena": round(_avg(h2, "arena_score"), 2),
            "v1_avg_speedup": round(_avg(h1, "speedup"), 4),
            "v2_avg_speedup": round(_avg(h2, "speedup"), 4),
            "v1_avg_throughput_ratio": round(_avg(h1, "throughput_ratio"), 4),
            "v2_avg_throughput_ratio": round(_avg(h2, "throughput_ratio"), 4),
            "improvement_arena": round(
                _avg(h2, "arena_score") - _avg(h1, "arena_score"), 2
            ),
        }

    def export_summary(self) -> dict:
        entries = self._store.all_entries()
        if not entries:
            return {"total_runs": 0, "agents": [], "top_scores": []}

        agents: dict[str, list[LeaderboardEntry]] = {}
        for e in entries:
            agents.setdefault(e.agent_version, []).append(e)

        agent_summaries = []
        for version, runs in sorted(agents.items()):
            avg_arena = sum(r.arena_score for r in runs) / len(runs)
            avg_sp = sum(r.speedup for r in runs) / len(runs)
            agent_summaries.append({
                "agent_version": version,
                "runs": len(runs),
                "avg_arena_score": round(avg_arena, 2),
                "avg_speedup": round(avg_sp, 4),
                "best_arena_score": round(max(r.arena_score for r in runs), 2),
            })

        top = sorted(entries, key=lambda e: e.arena_score, reverse=True)[:10]

        return {
            "total_runs": len(entries),
            "agents": agent_summaries,
            "top_scores": [e.to_dict() for e in top],
        }


class _FileLeaderboard:
    """File-based leaderboard (JSONL)."""

    def __init__(self, path: Path | None = None, **_: Any):
        self.path = path or REPO_ROOT / "leaderboard.jsonl"

    def push(self, entry: LeaderboardEntry) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(entry.to_dict(), default=str) + "\n")

    def all_entries(self) -> list[LeaderboardEntry]:
        if not self.path.exists():
            return []
        entries = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(LeaderboardEntry.from_dict(json.loads(line)))
        return entries


class _CouchDBLeaderboard:
    """CouchDB-backed leaderboard."""

    _LEGACY_DB_NAME = "keystone-leaderboard"

    def __init__(
        self,
        url: str | None = None,
        db_name: str = "apex-leaderboard",
        auth: tuple[str, str] | None = None,
        **_: Any,
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
            resp = requests.head(f"{self.url}/{self.db_name}", auth=self.auth, timeout=5)
            if resp.status_code == 404 and self.db_name != self._LEGACY_DB_NAME:
                legacy_resp = requests.head(
                    f"{self.url}/{self._LEGACY_DB_NAME}", auth=self.auth, timeout=5,
                )
                if legacy_resp.status_code == 200:
                    self.db_name = self._LEGACY_DB_NAME
                    return
            requests.put(f"{self.url}/{self.db_name}", auth=self.auth, timeout=5)
        except Exception:
            pass

    def push(self, entry: LeaderboardEntry) -> None:
        import requests
        doc = entry.to_dict()
        try:
            requests.put(
                f"{self.url}/{self.db_name}/{doc['_id']}",
                json=doc, auth=self.auth, timeout=10,
            )
        except Exception:
            _FileLeaderboard().push(entry)

    def all_entries(self) -> list[LeaderboardEntry]:
        import requests
        try:
            resp = requests.get(
                f"{self.url}/{self.db_name}/_all_docs",
                params={
                    "startkey": '"leaderboard:"',
                    "endkey": '"leaderboard:\ufff0"',
                    "include_docs": "true",
                },
                auth=self.auth, timeout=10,
            )
            if resp.status_code == 200:
                rows = resp.json().get("rows", [])
                return [LeaderboardEntry.from_dict(r["doc"]) for r in rows if "doc" in r]
        except Exception:
            pass
        return []
