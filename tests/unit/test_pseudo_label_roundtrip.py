from __future__ import annotations

import numpy as np

from src.safety_planner.interfaces import MpcRequest, PseudoLabelRecord
from src.safety_planner.safety_teacher import (
    generate_pseudo_labels,
    records_to_safety_targets,
    write_ranker_features,
)


def test_pseudo_label_roundtrip_via_dict() -> None:
    original = PseudoLabelRecord(
        scenario_id="scenario_001",
        candidate_id=3,
        controls=np.array([[0.1, 0.05], [0.2, 0.1]], dtype=np.float64),
        h_values=np.array([[1.0, 0.9, 0.8], [2.0, 1.9, 1.8]], dtype=np.float64),
        cbf_residuals=np.array([[0.0, 0.01], [0.0, 0.02]], dtype=np.float64),
        slack=np.array([[0.0, 0.0], [0.0, 0.01]], dtype=np.float64),
        feasible=True,
        solver_status=0,
        objective_total=12.3,
        correction_l2=0.15,
        solve_time_ms=42.5,
        quality_grade="A",
        solver_config_hash="abc123",
    )

    restored = PseudoLabelRecord.from_dict(original.to_dict())

    assert restored.scenario_id == original.scenario_id
    assert restored.candidate_id == original.candidate_id
    assert restored.feasible == original.feasible
    assert restored.quality_grade == original.quality_grade
    assert restored.solver_config_hash == original.solver_config_hash
    assert np.allclose(restored.controls, original.controls)
    assert np.allclose(restored.h_values, original.h_values)
    assert np.allclose(restored.cbf_residuals, original.cbf_residuals)
    assert np.isclose(restored.objective_total, original.objective_total)
    assert np.isclose(restored.correction_l2, original.correction_l2)
    assert np.isclose(restored.solve_time_ms, original.solve_time_ms)


def test_pseudo_label_quality_grade_validation() -> None:
    r = PseudoLabelRecord(quality_grade="B")
    r.validate()

    r2 = PseudoLabelRecord(quality_grade="X")
    try:
        r2.validate()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_pseudo_label_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "pseudo_label.json"
    original = PseudoLabelRecord(
        scenario_id="scenario_002",
        candidate_id=1,
        controls=np.zeros((2, 2), dtype=np.float64),
        h_values=np.ones((1, 3), dtype=np.float64),
        cbf_residuals=np.ones((1, 2), dtype=np.float64),
        slack=np.zeros((1, 2), dtype=np.float64),
        feasible=True,
        quality_grade="B",
    )
    original.save_json(path)
    restored = PseudoLabelRecord.load_json(path)

    assert restored.scenario_id == original.scenario_id
    assert restored.quality_grade == original.quality_grade
    assert np.allclose(restored.controls, original.controls)


def test_pseudo_label_generation_preserves_solver_exceptions(tmp_path) -> None:
    class FailingSolver:
        def solve(self, request):
            raise RuntimeError("synthetic failure")

    request = MpcRequest(
        schema_version="1.0",
        scenario_id="failed-scene",
        candidate_id=3,
        initial_state=np.zeros(4),
        reference_states=np.zeros((3, 4)),
        reference_timestamps=np.array([0.0, 0.2, 0.4]),
        predicted_obstacles=[],
    )
    records = generate_pseudo_labels([request], FailingSolver(), tmp_path, "commit")
    assert records[0].quality_grade == "D"
    assert records[0].solver_status == -1
    assert "synthetic failure" in records[0].failure_reason
    assert (tmp_path / "manifest.jsonl").exists()
    assert (tmp_path / "manifest.parquet").exists()
    with np.load(tmp_path / "pseudo-labels-00000.npz") as shard:
        assert shard["scenario_id"].tolist() == ["failed-scene"]
        assert shard["quality_grade"].tolist() == ["D"]
        assert "synthetic failure" in shard["failure_reason"][0]
        assert shard["controls"].shape == (1, 0)


def test_pseudo_labels_feed_safety_and_ranker_training(tmp_path) -> None:
    records = [
        PseudoLabelRecord(
            scenario_id="scene",
            candidate_id=0,
            h_values=np.array([[1.0, 0.5]]),
            slack=np.zeros((1, 1)),
            feasible=True,
            objective_total=2.0,
            correction_l2=0.1,
            quality_grade="A",
        ),
        PseudoLabelRecord(
            scenario_id="scene",
            candidate_id=1,
            h_values=np.array([[0.2, -0.1]]),
            slack=np.array([[0.1]]),
            feasible=False,
            objective_total=4.0,
            correction_l2=0.5,
            quality_grade="C",
        ),
    ]
    targets = records_to_safety_targets(records, ["scene"], num_candidates=2)
    assert targets["target_h_min"].shape == (1, 2)
    assert targets["target_feasible"].tolist() == [[1.0, 0.0]]
    metadata = write_ranker_features(records, tmp_path / "ranker.npz")
    assert metadata == {"num_rows": 2, "num_features": 5}
    with np.load(tmp_path / "ranker.npz") as table:
        assert table["features"].shape == (2, 5)
        assert table["labels"][0] > table["labels"][1]
