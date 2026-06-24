from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class FallbackMode(str, Enum):
    NONE = "none"
    NEXT_CANDIDATE = "next_candidate"
    REDUCED_SPEED = "reduced_speed"
    CONSERVATIVE_REFERENCE = "conservative_reference"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class ExecutionOutput:
    schema_version: str = "1.0"

    selected_candidate_id: int = -1
    safe_trajectory: FloatArray | None = None
    first_control: FloatArray | None = None

    feasible: bool = False
    fallback_mode: FallbackMode = FallbackMode.NONE

    h_min: float = 0.0
    slack_max: float = 0.0
    solve_time_ms: float = 0.0

    diagnostics: dict | None = None

    def validate(self) -> None:
        if self.first_control is not None:
            if self.first_control.shape != (2,):
                raise ValueError("first_control must have shape [2].")
