from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interfaces import SceneSample, TensorizedScene


@dataclass(frozen=True)
class TensorizerConfig:
    history_steps: int = 5
    future_steps: int = 8
    max_agents: int = 64
    max_map_polylines: int = 128
    max_map_points: int = 20
    max_route_points: int = 64


def tensorize_scene(sample: SceneSample, config: TensorizerConfig | None = None) -> TensorizedScene:
    sample.validate()
    cfg = config or TensorizerConfig()
    ego = _pad_2d(sample.ego_history, cfg.history_steps, 4)
    ego_mask = _pad_mask_1d(sample.ego_history_mask, cfg.history_steps, sample.ego_history)
    agents = _pad_3d(sample.agent_history, cfg.max_agents, cfg.history_steps, 7)
    agent_mask = _pad_mask_2d(
        sample.agent_history_mask,
        cfg.max_agents,
        cfg.history_steps,
        sample.agent_history,
    )
    agent_type = np.zeros(cfg.max_agents, dtype=np.int32)
    if sample.agent_type is not None:
        count = min(cfg.max_agents, sample.agent_type.shape[0])
        agent_type[:count] = sample.agent_type[:count]

    map_values = np.zeros((cfg.max_map_polylines, cfg.max_map_points, 5), dtype=np.float32)
    map_mask = np.zeros((cfg.max_map_polylines, cfg.max_map_points), dtype=np.bool_)
    for idx, polyline in enumerate((sample.map_polylines or [])[: cfg.max_map_polylines]):
        points = min(cfg.max_map_points, polyline.shape[0])
        dims = min(5, polyline.shape[1])
        map_values[idx, :points, :dims] = polyline[:points, :dims]
        if sample.map_mask is None:
            map_mask[idx, :points] = True
        else:
            map_mask[idx, :points] = sample.map_mask[idx][:points]
    map_type = np.zeros(cfg.max_map_polylines, dtype=np.int32)
    if sample.map_type is not None:
        count = min(cfg.max_map_polylines, len(sample.map_type))
        map_type[:count] = np.asarray(sample.map_type[:count], dtype=np.int32)

    route = _pad_2d(sample.route_polyline, cfg.max_route_points, 4)
    route_mask = _pad_mask_1d(sample.route_mask, cfg.max_route_points, sample.route_polyline)
    expert = _pad_2d(sample.expert_future, cfg.future_steps, 4)
    expert_mask = _pad_mask_1d(sample.expert_future_mask, cfg.future_steps, sample.expert_future)
    result = TensorizedScene(
        scenario_id=sample.scenario_id,
        timestamp_us=sample.timestamp_us,
        ego_history=ego,
        ego_history_mask=ego_mask,
        agent_history=agents,
        agent_history_mask=agent_mask,
        agent_type=agent_type,
        map_polylines=map_values,
        map_mask=map_mask,
        map_type=map_type,
        route_polyline=route,
        route_mask=route_mask,
        expert_future=expert,
        expert_future_mask=expert_mask,
    )
    result.validate()
    return result


def _pad_2d(value, rows: int, columns: int) -> np.ndarray:
    output = np.zeros((rows, columns), dtype=np.float32)
    if value is not None:
        count, dims = min(rows, value.shape[0]), min(columns, value.shape[1])
        output[:count, :dims] = value[:count, :dims]
    return output


def _pad_3d(value, first: int, second: int, third: int) -> np.ndarray:
    output = np.zeros((first, second, third), dtype=np.float32)
    if value is not None:
        a, h, dims = min(first, value.shape[0]), min(second, value.shape[1]), min(third, value.shape[2])
        output[:a, :h, :dims] = value[:a, :h, :dims]
    return output


def _pad_mask_1d(mask, rows: int, value) -> np.ndarray:
    output = np.zeros(rows, dtype=np.bool_)
    if mask is not None:
        output[: min(rows, mask.shape[0])] = mask[:rows]
    elif value is not None:
        output[: min(rows, value.shape[0])] = True
    return output


def _pad_mask_2d(mask, first: int, second: int, value) -> np.ndarray:
    output = np.zeros((first, second), dtype=np.bool_)
    if mask is not None:
        a, h = min(first, mask.shape[0]), min(second, mask.shape[1])
        output[:a, :h] = mask[:a, :h]
    elif value is not None:
        output[: min(first, value.shape[0]), : min(second, value.shape[1])] = True
    return output

