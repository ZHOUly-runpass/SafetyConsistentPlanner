from __future__ import annotations

import numpy as np

from src.safety_planner.interfaces import PseudoLabelRecord


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
