from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from apex.benchmark.magpie_launch_projection import materialize_magpie_launch_config
from apex.core import ConfigurationError, sha256_file


def _config(tmp_path: Path, inferencex: Path) -> Path:
    path = tmp_path / "canonical.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "vllm",
                    "model": "Qwen/example",
                    "inferencex_path": str(inferencex),
                    "envs": {"TP": 8, "CONC": 64},
                },
                "apex": {"benchmark_view": {"kind": "measurement"}},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_changes_only_private_inferencex_locator(tmp_path: Path) -> None:
    source = tmp_path / "InferenceX"
    projection = tmp_path / "run" / "authority" / "inferencex"
    source.mkdir()
    projection.mkdir(parents=True)
    canonical = _config(tmp_path, source)
    canonical_sha = sha256_file(canonical)
    launch = tmp_path / "run" / "authority" / "launch.yaml"

    receipt = materialize_magpie_launch_config(
        canonical,
        launch,
        canonical_sha256=canonical_sha,
        inferencex_source_root=source,
        inferencex_projection_root=projection,
        inferencex_projection_receipt_sha256="a" * 64,
    )

    assert sha256_file(canonical) == canonical_sha
    assert yaml.safe_load(launch.read_text())["benchmark"]["inferencex_path"] == str(
        projection.resolve()
    )
    assert receipt.canonical_config_sha256 == canonical_sha
    assert receipt.launch_config_sha256 == sha256_file(launch)
    assert receipt.to_dict()["allowed_change"] == "/benchmark/inferencex_path"


def test_rejects_unexpected_canonical_inferencex_locator(tmp_path: Path) -> None:
    source = tmp_path / "InferenceX"
    other = tmp_path / "other"
    projection = tmp_path / "projection"
    for path in (source, other, projection):
        path.mkdir()
    canonical = _config(tmp_path, other)

    with pytest.raises(ConfigurationError, match="differs from dependencies"):
        materialize_magpie_launch_config(
            canonical,
            tmp_path / "launch.yaml",
            canonical_sha256=sha256_file(canonical),
            inferencex_source_root=source,
            inferencex_projection_root=projection,
            inferencex_projection_receipt_sha256="a" * 64,
        )


def test_resolves_relative_inferencex_locator_from_config_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    source = project / "InferenceX"
    config_directory = project / "configs"
    projection = tmp_path / "projection"
    working_directory = tmp_path / "unrelated-cwd"
    for path in (source, config_directory, projection, working_directory):
        path.mkdir(parents=True)
    canonical = _config(config_directory, Path("../InferenceX"))
    launch = tmp_path / "launch.yaml"
    monkeypatch.chdir(working_directory)

    receipt = materialize_magpie_launch_config(
        canonical,
        launch,
        canonical_sha256=sha256_file(canonical),
        inferencex_source_root=source,
        inferencex_projection_root=projection,
        inferencex_projection_receipt_sha256="a" * 64,
    )

    assert receipt.inferencex_source_root == str(source.resolve())
    assert yaml.safe_load(launch.read_text())["benchmark"]["inferencex_path"] == str(
        projection.resolve()
    )
