from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass
class PseudoLabelRecord:
    schema_version: str = "1.0"

    scenario_id: str = ""
    candidate_id: int = 0

    ref_planner_states: FloatArray | None = None
    safe_planner_states: FloatArray | None = None

    ref_mpc_states: FloatArray | None = None
    safe_mpc_states: FloatArray | None = None

    controls: FloatArray | None = None
    h_values: FloatArray | None = None
    cbf_residuals: FloatArray | None = None
    slack: FloatArray | None = None

    feasible: bool = False
    solver_status: int = 0

    objective_total: float = 0.0
    objective_tracking: float = 0.0
    objective_control: float = 0.0
    objective_smoothness: float = 0.0
    objective_slack: float = 0.0

    correction_l2: float = 0.0
    correction_max: float = 0.0

    solve_time_ms: float = 0.0

    quality_grade: str = ""
    failure_reason: str = ""

    solver_config_hash: str = ""
    code_commit: str = ""
    prediction_source: str = "unknown"

    def validate(self) -> None:
        if self.controls is not None:
            if self.controls.ndim != 2 or self.controls.shape[1] != 2:
                raise ValueError("controls must have shape [Nm, 2].")

        if self.h_values is not None:
            if self.h_values.ndim != 2:
                raise ValueError("h_values must have shape [A, Nm+1].")

        if self.cbf_residuals is not None:
            if self.cbf_residuals.ndim != 2:
                raise ValueError("cbf_residuals must have shape [A, Nm].")

        if self.slack is not None:
            if self.slack.ndim != 2:
                raise ValueError("slack must have shape [A, Nm].")

        if self.quality_grade not in ("A", "B", "C", "D", ""):
            raise ValueError("quality_grade must be one of A, B, C, D.")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()
            else:
                payload[key] = value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PseudoLabelRecord":
        array_fields = {
            "ref_planner_states",
            "safe_planner_states",
            "ref_mpc_states",
            "safe_mpc_states",
            "controls",
            "h_values",
            "cbf_residuals",
            "slack",
        }
        kwargs: dict[str, Any] = {}
        for field_name in cls.__dataclass_fields__:
            if field_name not in payload:
                continue
            value = payload[field_name]
            if field_name in array_fields and value is not None:
                kwargs[field_name] = np.asarray(value, dtype=np.float64)
            else:
                kwargs[field_name] = value
        record = cls(**kwargs)
        record.validate()
        return record

    def save_json(self, path: str | Path) -> None:
        self.validate()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "PseudoLabelRecord":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)
