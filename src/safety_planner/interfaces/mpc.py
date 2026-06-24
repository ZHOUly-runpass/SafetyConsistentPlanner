from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .obstacle import PredictedObstacle

FloatArray = NDArray[np.float64]


@dataclass
class MpcRequest:
    schema_version: str

    scenario_id: str
    candidate_id: int

    initial_state: FloatArray
    reference_states: FloatArray
    reference_timestamps: FloatArray

    predicted_obstacles: list[PredictedObstacle]

    road_boundaries: Any | None = None

    initial_guess_states: FloatArray | None = None
    initial_guess_controls: FloatArray | None = None

    prediction_source: str = "unknown"
    solver_config_hash: str = ""

    def validate(self, num_intervals: int) -> None:
        expected_states = (num_intervals + 1, 4)
        expected_controls = (num_intervals, 2)

        if self.initial_state.shape != (4,):
            raise ValueError("initial_state must have shape [4].")

        if self.reference_states.shape != expected_states:
            raise ValueError(
                f"reference_states must have shape {expected_states}."
            )

        if self.reference_timestamps.shape != (num_intervals + 1,):
            raise ValueError("reference_timestamps has invalid shape.")
        if np.any(np.diff(self.reference_timestamps) <= 0.0):
            raise ValueError("reference_timestamps must be strictly increasing.")

        if (
            self.initial_guess_states is not None
            and self.initial_guess_states.shape != expected_states
        ):
            raise ValueError("Invalid initial_guess_states shape.")

        if (
            self.initial_guess_controls is not None
            and self.initial_guess_controls.shape != expected_controls
        ):
            raise ValueError("Invalid initial_guess_controls shape.")

        for obstacle in self.predicted_obstacles:
            obstacle.validate(num_intervals=num_intervals)


@dataclass
class MpcResult:
    schema_version: str = "1.0"

    safe_states: FloatArray | None = None
    controls: FloatArray | None = None

    h_values: FloatArray | None = None
    cbf_residuals: FloatArray | None = None
    obstacle_slack: FloatArray | None = None
    road_slack: FloatArray | None = None

    feasible: bool = False
    timed_out: bool = False
    solver_status: int = 0
    iterations: int = 0

    objective_total: float = 0.0
    objective_tracking: float = 0.0
    objective_control: float = 0.0
    objective_smoothness: float = 0.0
    objective_slack: float = 0.0

    h_min: float = 0.0
    slack_max: float = 0.0
    slack_sum: float = 0.0

    correction_l2: float = 0.0
    correction_max: float = 0.0

    solve_time_ms: float = 0.0

    first_control: FloatArray | None = None

    metadata: dict | None = None

    def validate(self, num_intervals: int, num_obstacles: int = 0) -> None:
        nm = num_intervals

        if self.safe_states is not None:
            if self.safe_states.shape != (nm + 1, 4):
                raise ValueError(f"safe_states must have shape [{nm + 1}, 4].")

        if self.controls is not None:
            if self.controls.shape != (nm, 2):
                raise ValueError(f"controls must have shape [{nm}, 2].")

        if self.h_values is not None:
            if self.h_values.shape != (num_obstacles, nm + 1):
                raise ValueError(f"h_values must have shape [{num_obstacles}, {nm + 1}].")

        if self.cbf_residuals is not None:
            if self.cbf_residuals.shape != (num_obstacles, nm):
                raise ValueError(f"cbf_residuals must have shape [{num_obstacles}, {nm}].")

        if self.obstacle_slack is not None:
            if self.obstacle_slack.shape != (num_obstacles, nm):
                raise ValueError(f"obstacle_slack must have shape [{num_obstacles}, {nm}].")

        if self.first_control is not None:
            if self.first_control.shape != (2,):
                raise ValueError("first_control must have shape [2].")
