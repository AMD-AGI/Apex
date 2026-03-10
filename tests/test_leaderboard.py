"""Tests for leaderboard.py — LeaderboardEntry and Leaderboard."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from leaderboard import LeaderboardEntry, Leaderboard


class TestLeaderboardEntry:
    def test_defaults(self):
        e = LeaderboardEntry()
        assert e.timestamp
        assert e.agent_model == ""
        assert e.arena_score == 0.0

    def test_doc_id(self):
        e = LeaderboardEntry(agent_version="v1.0", task_id="test_task")
        doc_id = e.doc_id()
        assert doc_id.startswith("leaderboard:v1.0:test_task:")

    def test_to_dict(self):
        e = LeaderboardEntry(
            agent_model="claude-sonnet-4-6",
            agent_version="v1.0",
            task_id="dsr1__fused_moe",
            arena_score=350.0,
            speedup=2.5,
        )
        d = e.to_dict()
        assert d["_id"].startswith("leaderboard:")
        assert d["type"] == "leaderboard"
        assert d["arena_score"] == 350.0

    def test_from_dict(self):
        d = {
            "agent_version": "v2.0",
            "task_id": "llama__gemm",
            "arena_score": 200.0,
            "_id": "ignored",
            "type": "leaderboard",
        }
        e = LeaderboardEntry.from_dict(d)
        assert e.agent_version == "v2.0"
        assert e.arena_score == 200.0


class TestFileLeaderboard:
    def test_push_and_rankings(self, tmp_path):
        lb_path = tmp_path / "lb.jsonl"
        lb = Leaderboard(backend="file", path=lb_path)

        lb.push(LeaderboardEntry(
            agent_version="v1.0", task_id="task_a", arena_score=100.0,
        ))
        lb.push(LeaderboardEntry(
            agent_version="v1.0", task_id="task_b", arena_score=300.0,
        ))
        lb.push(LeaderboardEntry(
            agent_version="v2.0", task_id="task_a", arena_score=250.0,
        ))

        rankings = lb.get_rankings()
        assert len(rankings) == 3
        assert rankings[0].arena_score == 300.0

    def test_filter_by_task(self, tmp_path):
        lb = Leaderboard(backend="file", path=tmp_path / "lb.jsonl")
        lb.push(LeaderboardEntry(task_id="task_a", arena_score=100.0))
        lb.push(LeaderboardEntry(task_id="task_b", arena_score=200.0))

        filtered = lb.get_rankings(task_id="task_a")
        assert len(filtered) == 1
        assert filtered[0].task_id == "task_a"

    def test_agent_history(self, tmp_path):
        lb = Leaderboard(backend="file", path=tmp_path / "lb.jsonl")
        lb.push(LeaderboardEntry(agent_version="v1.0", arena_score=100.0))
        lb.push(LeaderboardEntry(agent_version="v1.0", arena_score=200.0))
        lb.push(LeaderboardEntry(agent_version="v2.0", arena_score=300.0))

        h = lb.get_agent_history("v1.0")
        assert len(h) == 2
        assert all(e.agent_version == "v1.0" for e in h)

    def test_compare_versions(self, tmp_path):
        lb = Leaderboard(backend="file", path=tmp_path / "lb.jsonl")
        lb.push(LeaderboardEntry(agent_version="v1.0", arena_score=100.0, speedup=1.5))
        lb.push(LeaderboardEntry(agent_version="v1.0", arena_score=200.0, speedup=2.0))
        lb.push(LeaderboardEntry(agent_version="v2.0", arena_score=300.0, speedup=3.0))

        cmp = lb.compare_versions("v1.0", "v2.0")
        assert cmp["v1_runs"] == 2
        assert cmp["v2_runs"] == 1
        assert cmp["v2_avg_arena"] > cmp["v1_avg_arena"]
        assert cmp["improvement_arena"] > 0

    def test_export_summary(self, tmp_path):
        lb = Leaderboard(backend="file", path=tmp_path / "lb.jsonl")
        lb.push(LeaderboardEntry(agent_version="v1.0", arena_score=100.0))
        lb.push(LeaderboardEntry(agent_version="v2.0", arena_score=300.0))

        summary = lb.export_summary()
        assert summary["total_runs"] == 2
        assert len(summary["agents"]) == 2
        assert len(summary["top_scores"]) == 2

    def test_empty_leaderboard(self, tmp_path):
        lb = Leaderboard(backend="file", path=tmp_path / "lb.jsonl")
        assert lb.get_rankings() == []
        assert lb.export_summary()["total_runs"] == 0
