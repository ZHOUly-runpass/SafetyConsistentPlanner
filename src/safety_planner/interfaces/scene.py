from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass
class SceneSample:
    schema_version: str = "1.0"

    scenario_id: str = ""
    timestamp_us: int = 0

    ego_history: FloatArray | None = None
    ego_history_mask: BoolArray | None = None

    agent_history: FloatArray | None = None
    agent_history_mask: BoolArray | None = None
    agent_type: NDArray[np.int32] | None = None
    agent_track_ids: list[str] | None = None

    map_polylines: list[FloatArray] | None = None
    map_mask: list[BoolArray] | None = None
    map_type: list[int] | None = None

    route_polyline: FloatArray | None = None
    route_mask: BoolArray | None = None

    expert_future: FloatArray | None = None
    expert_future_mask: BoolArray | None = None

    metadata: dict | None = None

    def validate(self) -> None:
        if self.ego_history is not None:
            if self.ego_history.ndim != 2:
                raise ValueError("ego_history must have shape [H, D_ego].")
            if self.ego_history_mask is not None:
                if self.ego_history_mask.shape[0] != self.ego_history.shape[0]:
                    raise ValueError("ego_history_mask length mismatch.")

        if self.agent_history is not None:
            if self.agent_history.ndim != 3:
                raise ValueError("agent_history must have shape [A, H, D_agent].")
            num_agents = self.agent_history.shape[0]
            if self.agent_history_mask is not None:
                if self.agent_history_mask.shape != (num_agents, self.agent_history.shape[1]):
                    raise ValueError("agent_history_mask shape mismatch.")
            if self.agent_type is not None:
                if self.agent_type.shape[0] != num_agents:
                    raise ValueError("agent_type length mismatch.")
            if self.agent_track_ids is not None:
                if len(self.agent_track_ids) != num_agents:
                    raise ValueError("agent_track_ids length mismatch.")

        if self.expert_future is not None:
            if self.expert_future.ndim != 2:
                raise ValueError("expert_future must have shape [Np, 4].")
            if self.expert_future_mask is not None:
                if self.expert_future_mask.shape[0] != self.expert_future.shape[0]:
                    raise ValueError("expert_future_mask length mismatch.")
