from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class TensorizedScene:
    scenario_id: str
    timestamp_us: int
    ego_history: NDArray[np.float32]
    ego_history_mask: NDArray[np.bool_]
    agent_history: NDArray[np.float32]
    agent_history_mask: NDArray[np.bool_]
    agent_type: NDArray[np.int32]
    map_polylines: NDArray[np.float32]
    map_mask: NDArray[np.bool_]
    map_type: NDArray[np.int32]
    route_polyline: NDArray[np.float32]
    route_mask: NDArray[np.bool_]
    expert_future: NDArray[np.float32]
    expert_future_mask: NDArray[np.bool_]
    schema_version: str = "1.0"

    def validate(self) -> None:
        if self.ego_history.ndim != 2 or self.ego_history.shape[1] != 4:
            raise ValueError("ego_history must have shape [H, 4].")
        if self.ego_history_mask.shape != self.ego_history.shape[:1]:
            raise ValueError("ego_history_mask shape mismatch.")
        if self.agent_history.ndim != 3 or self.agent_history.shape[2] != 7:
            raise ValueError("agent_history must have shape [A, H, 7].")
        if self.agent_history_mask.shape != self.agent_history.shape[:2]:
            raise ValueError("agent_history_mask shape mismatch.")
        if self.agent_type.shape != self.agent_history.shape[:1]:
            raise ValueError("agent_type shape mismatch.")
        if self.map_polylines.ndim != 3 or self.map_polylines.shape[2] != 5:
            raise ValueError("map_polylines must have shape [M, L, 5].")
        if self.map_mask.shape != self.map_polylines.shape[:2]:
            raise ValueError("map_mask shape mismatch.")
        if self.map_type.shape != self.map_polylines.shape[:1]:
            raise ValueError("map_type shape mismatch.")
        if self.route_polyline.ndim != 2 or self.route_polyline.shape[1] != 4:
            raise ValueError("route_polyline must have shape [R, 4].")
        if self.route_mask.shape != self.route_polyline.shape[:1]:
            raise ValueError("route_mask shape mismatch.")
        if self.expert_future.ndim != 2 or self.expert_future.shape[1] != 4:
            raise ValueError("expert_future must have shape [T, 4].")
        if self.expert_future_mask.shape != self.expert_future.shape[:1]:
            raise ValueError("expert_future_mask shape mismatch.")
        numeric = (
            self.ego_history,
            self.agent_history,
            self.map_polylines,
            self.route_polyline,
            self.expert_future,
        )
        if not all(np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("TensorizedScene contains NaN or Inf.")


def stack_tensorized_scenes(scenes: list[TensorizedScene]) -> dict[str, np.ndarray]:
    if not scenes:
        raise ValueError("At least one scene is required.")
    for scene in scenes:
        scene.validate()
    tensor_fields = (
        "ego_history",
        "ego_history_mask",
        "agent_history",
        "agent_history_mask",
        "agent_type",
        "map_polylines",
        "map_mask",
        "map_type",
        "route_polyline",
        "route_mask",
        "expert_future",
        "expert_future_mask",
    )
    result = {name: np.stack([getattr(scene, name) for scene in scenes]) for name in tensor_fields}
    result["scenario_id"] = np.asarray([scene.scenario_id for scene in scenes])
    result["timestamp_us"] = np.asarray([scene.timestamp_us for scene in scenes], dtype=np.int64)
    result["schema_version"] = np.asarray([scene.schema_version for scene in scenes])
    return result

