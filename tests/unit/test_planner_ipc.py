from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest

from src.safety_planner.planning.ipc import read_message, serve_messages, write_message


def test_message_round_trip_preserves_numpy_arrays() -> None:
    stream = BytesIO()
    payload = {
        "schema_version": "1.0",
        "request_id": "abc",
        "array": np.arange(12, dtype=np.float32).reshape(3, 4),
    }
    write_message(stream, payload)
    stream.seek(0)
    decoded = read_message(stream)
    assert decoded["request_id"] == "abc"
    np.testing.assert_array_equal(decoded["array"], payload["array"])


def test_message_rejects_truncated_body() -> None:
    stream = BytesIO(b"\x00\x00\x00\x04x")
    with pytest.raises(EOFError, match="body"):
        read_message(stream)


def test_server_returns_versioned_error_without_crashing() -> None:
    reader = BytesIO()
    write_message(reader, {"schema_version": "0.0", "request_id": "bad"})
    reader.seek(0)
    writer = BytesIO()
    serve_messages(reader, writer, lambda request: {"ok": True})
    writer.seek(0)
    response = read_message(writer)
    assert response["request_id"] == "bad"
    assert response["ok"] is False
    assert response["error_type"] == "ValueError"


def test_worker_response_contains_mpc_control_and_fallback() -> None:
    torch = pytest.importorskip("torch")
    from src.safety_planner.dcbf_mpc import NumpyVehicleDcbfMpcSolver
    from src.safety_planner.planning.worker import PlannerWorker

    class FakeModel:
        def __call__(self, batch):
            candidate = torch.zeros((1, 2, 8, 4), dtype=torch.float32)
            candidate[0, :, :, 0] = torch.linspace(0.0, 4.0, 8)
            candidate[0, :, :, 3] = 2.0
            return {
                "base_trajectory": candidate[:, 0],
                "candidate_trajectories": candidate,
                "predicted_h_min": torch.ones((1, 2)),
                "feasibility_logits": torch.full((1, 2), 4.0),
                "predicted_correction": torch.zeros((1, 2)),
                "predicted_risk": torch.zeros((1, 2)),
            }

    worker = object.__new__(PlannerWorker)
    worker.device = torch.device("cpu")
    worker.model = FakeModel()
    worker.solver = NumpyVehicleDcbfMpcSolver()
    worker.mpc_num_intervals = 15
    worker.mpc_dt = 0.2
    response = worker.handle(
        {
            "scenario_id": "scene",
            "scene": {"ego_history": np.zeros((5, 4), dtype=np.float32)},
            "initial_state": np.asarray([0.0, 0.0, 0.0, 2.0]),
            "planner_timestamps": np.arange(1, 9, dtype=np.float64) * 0.5,
            "predicted_obstacles": [],
        }
    )

    assert response["ok"] is True
    assert response["first_control"].shape == (2,)
    assert response["selected_trajectory"].shape == (16, 4)
    assert response["mpc"]["feasible"] is True
    assert response["fallback"] == "none"
