from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..interfaces import SceneSample


@dataclass(frozen=True)
class NuPlanExtractionConfig:
    history_steps: int = 5
    history_horizon_s: float = 2.0
    future_steps: int = 8
    future_horizon_s: float = 4.0
    max_agents: int = 64


def extract_scene_sample(
    scenario: Any,
    *,
    iteration: int = 0,
    map_polylines_world: Iterable[tuple[int, np.ndarray]] = (),
    route_polyline_world: np.ndarray | None = None,
    config: NuPlanExtractionConfig | None = None,
) -> SceneSample:
    """Extract one devkit-independent sample without reading future agents."""
    cfg = config or NuPlanExtractionConfig()
    current = scenario.get_ego_state_at_iteration(iteration)
    origin = current.rear_axle

    past_ego = list(
        scenario.get_ego_past_trajectory(
            iteration, cfg.history_horizon_s, num_samples=max(0, cfg.history_steps - 1)
        )
    )
    ego_states = (past_ego + [current])[-cfg.history_steps :]
    ego_history = np.zeros((cfg.history_steps, 4), dtype=np.float64)
    ego_mask = np.zeros(cfg.history_steps, dtype=np.bool_)
    ego_offset = cfg.history_steps - len(ego_states)
    for index, state in enumerate(ego_states, start=ego_offset):
        ego_history[index] = _ego_state_features(state, origin)
        ego_mask[index] = True

    past_detections = list(
        scenario.get_past_tracked_objects(
            iteration, cfg.history_horizon_s, num_samples=max(0, cfg.history_steps - 1)
        )
    )
    current_detections = scenario.get_tracked_objects_at_iteration(iteration)
    detections = (past_detections + [current_detections])[-cfg.history_steps :]
    current_objects = _tracked_objects(current_detections)
    current_objects.sort(key=lambda obj: (_distance(obj.center, origin), str(obj.track_token)))
    selected = current_objects[: cfg.max_agents]
    track_ids = [str(obj.track_token) for obj in selected]
    track_to_row = {track_id: row for row, track_id in enumerate(track_ids)}
    agent_history = np.zeros((len(selected), cfg.history_steps, 7), dtype=np.float64)
    agent_mask = np.zeros((len(selected), cfg.history_steps), dtype=np.bool_)
    agent_type = np.zeros(len(selected), dtype=np.int32)
    for row, obj in enumerate(selected):
        agent_type[row] = _enum_value(obj.tracked_object_type)
    detection_offset = cfg.history_steps - len(detections)
    for time_index, detection in enumerate(detections, start=detection_offset):
        for obj in _tracked_objects(detection):
            row = track_to_row.get(str(obj.track_token))
            if row is None:
                continue
            agent_history[row, time_index] = _agent_features(obj, origin)
            agent_mask[row, time_index] = True

    future_states = list(
        scenario.get_ego_future_trajectory(
            iteration, cfg.future_horizon_s, num_samples=cfg.future_steps
        )
    )[: cfg.future_steps]
    expert_future = np.zeros((cfg.future_steps, 4), dtype=np.float64)
    expert_mask = np.zeros(cfg.future_steps, dtype=np.bool_)
    for index, state in enumerate(future_states):
        expert_future[index] = _ego_state_features(state, origin)
        expert_mask[index] = True

    transformed_map = []
    for kind, world_points in map_polylines_world:
        local = _transform_polyline(np.asarray(world_points), origin, output_dims=5)
        distance = float(np.min(np.linalg.norm(local[:, :2], axis=1))) if len(local) else math.inf
        transformed_map.append((distance, int(kind), local))
    transformed_map.sort(key=lambda item: (item[0], item[1]))
    map_values = [item[2] for item in transformed_map]
    map_types = [item[1] for item in transformed_map]

    route = None
    route_mask = None
    if route_polyline_world is not None:
        route = _transform_polyline(np.asarray(route_polyline_world), origin, output_dims=4)
        route_mask = np.ones(len(route), dtype=np.bool_)

    sample = SceneSample(
        scenario_id=str(scenario.token),
        timestamp_us=int(scenario.get_time_point(iteration).time_us),
        ego_history=ego_history,
        ego_history_mask=ego_mask,
        agent_history=agent_history,
        agent_history_mask=agent_mask,
        agent_type=agent_type,
        agent_track_ids=track_ids,
        map_polylines=map_values,
        map_mask=[np.ones(len(item), dtype=np.bool_) for item in map_values],
        map_type=map_types,
        route_polyline=route,
        route_mask=route_mask,
        expert_future=expert_future,
        expert_future_mask=expert_mask,
        metadata={
            "log_name": str(scenario.log_name),
            "scenario_type": str(scenario.scenario_type),
            "map_name": str(scenario.map_api.map_name),
            "source": "nuplan",
        },
    )
    sample.validate()
    return sample


def _ego_state_features(state: Any, origin: Any) -> np.ndarray:
    x, y = _to_local(state.rear_axle.x, state.rear_axle.y, origin)
    return np.asarray(
        [
            x,
            y,
            _wrap(state.rear_axle.heading - origin.heading),
            state.dynamic_car_state.rear_axle_velocity_2d.magnitude(),
        ],
        dtype=np.float64,
    )


def _agent_features(obj: Any, origin: Any) -> np.ndarray:
    x, y = _to_local(obj.center.x, obj.center.y, origin)
    vx, vy = _rotate(obj.velocity.x, obj.velocity.y, -origin.heading)
    return np.asarray(
        [x, y, _wrap(obj.center.heading - origin.heading), vx, vy, obj.box.length, obj.box.width],
        dtype=np.float64,
    )


def _transform_polyline(points: np.ndarray, origin: Any, output_dims: int) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("World polylines must have shape [N, >=2].")
    result = np.zeros((len(points), output_dims), dtype=np.float64)
    for index, point in enumerate(points):
        result[index, :2] = _to_local(point[0], point[1], origin)
        if points.shape[1] >= 3:
            result[index, 2] = _wrap(point[2] - origin.heading)
    if output_dims >= 5 and len(points):
        result[:, 3] = np.cos(result[:, 2])
        result[:, 4] = np.sin(result[:, 2])
    return result


def _tracked_objects(detections: Any) -> list[Any]:
    return list(detections.tracked_objects.tracked_objects)


def _distance(point: Any, origin: Any) -> float:
    return math.hypot(point.x - origin.x, point.y - origin.y)


def _to_local(x: float, y: float, origin: Any) -> tuple[float, float]:
    return _rotate(x - origin.x, y - origin.y, -origin.heading)


def _rotate(x: float, y: float, yaw: float) -> tuple[float, float]:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return cosine * x - sine * y, sine * x + cosine * y


def _wrap(yaw: float) -> float:
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


def _enum_value(value: Any) -> int:
    raw = getattr(value, "value", value)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0
