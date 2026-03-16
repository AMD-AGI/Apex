# Copyright (c) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Tests for trajectory.py — TrajectoryRecord and TrajectoryStore backends."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from trajectory import TrajectoryRecord, FileStore, get_store


class TestTrajectoryRecord:
    def test_defaults(self):
        r = TrajectoryRecord()
        assert r.trajectory_id
        assert r.timestamp
        assert r.final_score == 0.0
        assert r.trajectory_quality == "unknown"

    def test_to_dict_roundtrip(self):
        r = TrajectoryRecord(
            task_id="dsr1__fused_moe__sglang",
            agent_model="claude-sonnet-4-6",
            final_score=300.0,
            final_speedup=2.5,
        )
        d = r.to_dict()
        assert d["task_id"] == "dsr1__fused_moe__sglang"
        assert d["final_score"] == 300.0

        r2 = TrajectoryRecord.from_dict(d)
        assert r2.task_id == r.task_id
        assert r2.final_score == r.final_score
        assert r2.trajectory_id == r.trajectory_id

    def test_compute_reward_good(self):
        r = TrajectoryRecord(final_score=400.0, throughput_improvement=1.8)
        reward = r.compute_reward()
        assert reward > 0.8
        assert r.trajectory_quality == "good"

    def test_compute_reward_mediocre(self):
        r = TrajectoryRecord(final_score=150.0, throughput_improvement=1.1)
        reward = r.compute_reward()
        assert 0.3 <= reward < 0.8
        assert r.trajectory_quality == "mediocre"

    def test_compute_reward_bad(self):
        r = TrajectoryRecord(final_score=0.0, throughput_improvement=0.0)
        reward = r.compute_reward()
        assert reward < 0.3
        assert r.trajectory_quality == "bad"

    def test_iterations_list(self):
        r = TrajectoryRecord()
        r.iterations.append({"iteration": 1, "score": 100})
        r.iterations.append({"iteration": 2, "score": 250})
        assert len(r.iterations) == 2

    def test_from_dict_ignores_unknown_keys(self):
        d = {"task_id": "test", "unknown_field": 42}
        r = TrajectoryRecord.from_dict(d)
        assert r.task_id == "test"
        assert not hasattr(r, "unknown_field")


class TestFileStore:
    def test_save_and_load(self, tmp_path):
        store = FileStore(base_dir=tmp_path)
        r = TrajectoryRecord(task_id="test_task", final_score=250.0)
        tid = store.save(r)
        assert (tmp_path / f"{tid}.json").exists()

        loaded = store.load(tid)
        assert loaded is not None
        assert loaded.task_id == "test_task"
        assert loaded.final_score == 250.0

    def test_load_nonexistent(self, tmp_path):
        store = FileStore(base_dir=tmp_path)
        assert store.load("nonexistent") is None

    def test_list_ids(self, tmp_path):
        store = FileStore(base_dir=tmp_path)
        for i in range(3):
            store.save(TrajectoryRecord(task_id=f"task_{i}"))
        ids = store.list_ids()
        assert len(ids) == 3

    def test_list_ids_with_filter(self, tmp_path):
        store = FileStore(base_dir=tmp_path)
        r1 = TrajectoryRecord(task_id="task_a", trajectory_quality="good")
        r2 = TrajectoryRecord(task_id="task_b", trajectory_quality="bad")
        store.save(r1)
        store.save(r2)
        good = store.list_ids({"trajectory_quality": "good"})
        assert len(good) == 1

    def test_export_for_rl(self, tmp_path):
        store = FileStore(base_dir=tmp_path)
        r1 = TrajectoryRecord(task_id="a", trajectory_quality="good")
        r2 = TrajectoryRecord(task_id="b", trajectory_quality="bad")
        r3 = TrajectoryRecord(task_id="c", trajectory_quality="good")
        for r in [r1, r2, r3]:
            store.save(r)

        output = tmp_path / "export.jsonl"
        count = store.export_for_rl(output, quality="good")
        assert count == 2
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            d = json.loads(line)
            assert d["trajectory_quality"] == "good"


class TestGetStore:
    def test_file_backend(self):
        store = get_store("file")
        assert isinstance(store, FileStore)

    def test_default_backend(self):
        store = get_store()
        assert isinstance(store, FileStore)

    def test_unknown_falls_to_file(self):
        store = get_store("unknown")
        assert isinstance(store, FileStore)
