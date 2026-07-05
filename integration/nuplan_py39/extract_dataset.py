"""Extract deterministic nuPlan subsets into devkit-independent NumPy shards."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_sequential import Sequential

from safety_planner_adapter import _rotate, _to_ego, _wrap


SCHEMA_VERSION = "1.0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/data/nuplan/v1.1")
    parser.add_argument("--map-root", default="/data/nuplan/maps")
    parser.add_argument("--db-files", default="/data/nuplan/v1.1/splits/trainval")
    parser.add_argument("--map-version", default="nuplan-maps-v1.0")
    parser.add_argument("--output", default="/data/nuplan/processed/safety_planner/v1")
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--closed-loop-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--index-only", action="store_true")
    return parser.parse_args()


def build_scenarios(args):
    for path in (args.data_root, args.map_root, args.db_files):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    builder = NuPlanScenarioBuilder(
        data_root=args.data_root,
        map_root=args.map_root,
        sensor_root=args.data_root,
        db_files=args.db_files,
        map_version=args.map_version,
        include_cameras=False,
        max_workers=1,
        verbose=True,
    )
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=True,
        shuffle=False,
    )
    return builder.get_scenarios(scenario_filter, Sequential())


def deterministic_subsets(scenarios, counts, seed):
    by_id = {}
    for scenario in scenarios:
        if scenario.token in by_id:
            raise ValueError("Duplicate scenario token: %s" % scenario.token)
        by_id[scenario.token] = scenario
    required = sum(counts.values())
    if len(by_id) < required:
        raise ValueError("Need %d scenarios, got %d" % (required, len(by_id)))
    ranked = sorted(
        by_id,
        key=lambda token: hashlib.sha256((str(seed) + ":" + token).encode("utf-8")).digest(),
    )
    result = {}
    offset = 0
    for split in ("train", "val", "closed_loop"):
        result[split] = [by_id[token] for token in ranked[offset : offset + counts[split]]]
        offset += counts[split]
    return result


def extract_scenario(scenario):
    iteration = 0
    history_steps = 5
    future_steps = 8
    current = scenario.get_ego_state_at_iteration(iteration)
    origin = current.rear_axle
    ego_states = list(scenario.get_ego_past_trajectory(iteration, 2.0, num_samples=4)) + [current]
    ego_states = ego_states[-history_steps:]
    ego = np.zeros((history_steps, 4), dtype=np.float32)
    ego_mask = np.zeros(history_steps, dtype=np.bool_)
    offset = history_steps - len(ego_states)
    for index, state in enumerate(ego_states, start=offset):
        ego[index] = _ego_features(state, origin)
        ego_mask[index] = True

    past = list(scenario.get_past_tracked_objects(iteration, 2.0, num_samples=4))
    current_detection = scenario.get_tracked_objects_at_iteration(iteration)
    detections = (past + [current_detection])[-history_steps:]
    current_objects = list(current_detection.tracked_objects.tracked_objects)
    current_objects.sort(
        key=lambda obj: (math.hypot(obj.center.x - origin.x, obj.center.y - origin.y), str(obj.track_token))
    )
    selected = current_objects[:64]
    track_rows = {str(obj.track_token): index for index, obj in enumerate(selected)}
    agents = np.zeros((64, history_steps, 7), dtype=np.float32)
    agent_mask = np.zeros((64, history_steps), dtype=np.bool_)
    agent_type = np.zeros(64, dtype=np.int32)
    for row, obj in enumerate(selected):
        agent_type[row] = int(obj.tracked_object_type.value)
    offset = history_steps - len(detections)
    for time_index, detection in enumerate(detections, start=offset):
        for obj in detection.tracked_objects.tracked_objects:
            row = track_rows.get(str(obj.track_token))
            if row is None:
                continue
            agents[row, time_index] = _agent_features(obj, origin)
            agent_mask[row, time_index] = True

    map_values, map_mask, map_type = _extract_map(scenario, origin)
    route, route_mask = _extract_route(scenario, origin)

    expert = np.zeros((future_steps, 4), dtype=np.float32)
    expert_mask = np.zeros(future_steps, dtype=np.bool_)
    future = list(scenario.get_ego_future_trajectory(iteration, 4.0, num_samples=future_steps))
    for index, state in enumerate(future[:future_steps]):
        expert[index] = _ego_features(state, origin)
        expert_mask[index] = True

    return {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "scenario_id": np.asarray(str(scenario.token)),
        "timestamp_us": np.asarray(scenario.get_time_point(iteration).time_us, dtype=np.int64),
        "scenario_type": np.asarray(str(scenario.scenario_type)),
        "source_log": np.asarray(str(scenario.log_name)),
        "ego_history": ego,
        "ego_history_mask": ego_mask,
        "agent_history": agents,
        "agent_history_mask": agent_mask,
        "agent_type": agent_type,
        "map_polylines": map_values,
        "map_mask": map_mask,
        "map_type": map_type,
        "route_polyline": route,
        "route_mask": route_mask,
        "expert_future": expert,
        "expert_future_mask": expert_mask,
    }


def _extract_map(scenario, origin):
    values = np.zeros((128, 20, 5), dtype=np.float32)
    masks = np.zeros((128, 20), dtype=np.bool_)
    types = np.zeros(128, dtype=np.int32)
    layers = (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR)
    objects = scenario.map_api.get_proximal_map_objects(Point2D(origin.x, origin.y), 80.0, list(layers))
    rows = []
    for kind, layer in enumerate(layers, start=1):
        for obj in objects.get(layer, []):
            path = list(obj.baseline_path.discrete_path)
            distance = min(
                (math.hypot(point.x - origin.x, point.y - origin.y) for point in path),
                default=float("inf"),
            )
            rows.append((distance, kind, str(obj.id), path))
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    for row, (_, kind, _, path) in enumerate(rows[:128]):
        for column, point in enumerate(path[:20]):
            x, y = _to_ego(point.x, point.y, origin.x, origin.y, origin.heading)
            yaw = _wrap(point.heading - origin.heading)
            values[row, column] = [x, y, yaw, math.cos(yaw), math.sin(yaw)]
            masks[row, column] = True
        types[row] = kind
    return values, masks, types


def _extract_route(scenario, origin):
    values = np.zeros((64, 4), dtype=np.float32)
    mask = np.zeros(64, dtype=np.bool_)
    points = []
    for roadblock_id in scenario.get_route_roadblock_ids():
        roadblock = scenario.map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
        if roadblock is None:
            roadblock = scenario.map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        if roadblock is not None:
            edges = list(roadblock.interior_edges)
            if edges:
                edge = min(
                    edges,
                    key=lambda item: min(
                        (
                            math.hypot(point.x - origin.x, point.y - origin.y)
                            for point in item.baseline_path.discrete_path
                        ),
                        default=float("inf"),
                    ),
                )
                points.extend(edge.baseline_path.discrete_path)
    for index, point in enumerate(points[:64]):
        x, y = _to_ego(point.x, point.y, origin.x, origin.y, origin.heading)
        values[index, :3] = [x, y, _wrap(point.heading - origin.heading)]
        mask[index] = True
    return values, mask


def _ego_features(state, origin):
    x, y = _to_ego(state.rear_axle.x, state.rear_axle.y, origin.x, origin.y, origin.heading)
    return [
        x,
        y,
        _wrap(state.rear_axle.heading - origin.heading),
        state.dynamic_car_state.rear_axle_velocity_2d.magnitude(),
    ]


def _agent_features(obj, origin):
    x, y = _to_ego(obj.center.x, obj.center.y, origin.x, origin.y, origin.heading)
    vx, vy = _rotate(obj.velocity.x, obj.velocity.y, -origin.heading)
    return [x, y, _wrap(obj.center.heading - origin.heading), vx, vy, obj.box.length, obj.box.width]


def write_split(split, scenarios, output, shard_size, config_hash):
    rows = []
    for shard_index, start in enumerate(range(0, len(scenarios), shard_size)):
        selected = scenarios[start : start + shard_size]
        samples = [extract_scenario(scenario) for scenario in selected]
        arrays = {key: np.stack([sample[key] for sample in samples]) for key in samples[0]}
        shard_name = "%s-%05d.npz" % (split, shard_index)
        np.savez_compressed(output / shard_name, **arrays)
        for index, sample in enumerate(samples):
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "scenario_id": str(sample["scenario_id"]),
                    "scenario_type": str(sample["scenario_type"]),
                    "source_log": str(sample["source_log"]),
                    "timestamp_us": int(sample["timestamp_us"]),
                    "split": split,
                    "shard": shard_name,
                    "index": index,
                    "config_hash": config_hash,
                }
            )
    _write_manifest(rows, output / (split + "-manifest"))


def _write_manifest(rows, path):
    path.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), path.with_suffix(".parquet"))


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    counts = {"train": args.train_count, "val": args.val_count, "closed_loop": args.closed_loop_count}
    config_hash = hashlib.sha256(json.dumps(vars(args), sort_keys=True).encode("utf-8")).hexdigest()
    subsets = deterministic_subsets(build_scenarios(args), counts, args.seed)
    index_rows = []
    for split, scenarios in subsets.items():
        for scenario in scenarios:
            index_rows.append(
                {
                    "scenario_id": str(scenario.token),
                    "scenario_type": str(scenario.scenario_type),
                    "source_log": str(scenario.log_name),
                    "split": split,
                    "seed": args.seed,
                    "config_hash": config_hash,
                }
            )
    _write_manifest(index_rows, output / "subset-index")
    if not args.index_only:
        for split, scenarios in subsets.items():
            write_split(split, scenarios, output, args.shard_size, config_hash)


if __name__ == "__main__":
    main()
