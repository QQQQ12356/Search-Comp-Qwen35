import json

import pytest

from search_comp.evaluation.statistics import summarize_results
from search_comp.utils.runtime import load_yaml_config, prepare_run_artifacts


def test_yaml_overrides_support_nested_values(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "learning_rate: 5.0e-5\nbeacon:\n  beacon_ratio: 16\n",
        encoding="utf-8",
    )

    config = load_yaml_config(
        str(config_path),
        ["learning_rate=1.0e-5", "beacon.beacon_ratio=32", "use_lora=true"],
    )

    assert config["learning_rate"] == pytest.approx(1.0e-5)
    assert config["beacon"]["beacon_ratio"] == 32
    assert config["use_lora"] is True


def test_prepare_run_artifacts_records_resolved_config(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["trainer", "--config", "train.yaml"])
    run_dir = prepare_run_artifacts(
        tmp_path / "run",
        {"exp_name": "smoke"},
        "train.yaml",
        ["exp_name=smoke"],
    )

    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert config["exp_name"] == "smoke"
    assert metadata["overrides"] == ["exp_name=smoke"]
    assert metadata["command"].startswith("trainer")


def test_statistics_cover_quality_search_latency_and_compression():
    summary = summarize_results(
        [
            {
                "id": "1",
                "prediction": "Paris",
                "ground_truth": "Paris",
                "turns": 1,
                "latency_seconds": 2.0,
                "information_tokens": 64,
                "beacon_tokens": 4,
            },
            {
                "id": "2",
                "prediction": "[无作答]",
                "ground_truth": "Rome",
                "turns": 2,
                "latency_seconds": 4.0,
                "information_tokens": 32,
                "beacon_tokens": 2,
            },
        ]
    )

    assert summary["em"] == 0.5
    assert summary["search_rate"] == 1.0
    assert summary["multi_turn_samples"] == 1
    assert summary["average_turns"] == 1.5
    assert summary["latency_seconds"]["mean"] == 3.0
    assert summary["effective_information_compression_ratio"] == 16.0
    assert summary["answer_rate"] == 0.5


def test_invalid_override_is_rejected(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    with pytest.raises(ValueError, match="key=value"):
        load_yaml_config(str(config_path), ["seed"])
