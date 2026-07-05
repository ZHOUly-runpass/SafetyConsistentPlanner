"""Standalone Python 3.9 nuPlan adapter for the Python 3.10 planner worker."""

import math
import struct
import subprocess
import uuid

import msgpack
import numpy as np

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.training.preprocessing.features.trajectory_utils import transform_predictions_to_states


class SafetyConsistentPlannerAdapter(AbstractPlanner):
    def __init__(self, worker_command):
        self._worker_command = list(worker_command)
        self._worker = None
        self._initialization = None

    @classmethod
    def observation_type(cls):
        return DetectionsTracks

    def name(self):
        return "SafetyConsistentPlanner"

    def initialize(self, initialization):
        self._initialization = initialization
        self._worker = subprocess.Popen(
            self._worker_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def compute_planner_trajectory(self, current_input):
        if self._worker is None or self._worker.poll() is not None:
            raise RuntimeError("Planner worker is not running.")
        request_id = str(uuid.uuid4())
        scene = tensorize_planner_input(current_input, self._initialization)
        _write_message(
            self._worker.stdin,
            {"schema_version": "1.0", "request_id": request_id, "scene": scene},
        )
        response = _read_message(self._worker.stdout)
        if response.get("request_id") != request_id:
            raise RuntimeError("Planner worker response id mismatch.")
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Planner worker failed."))
        relative = np.asarray(response["selected_trajectory"], dtype=np.float64)
        ego_states = current_input.history.ego_states
        states = transform_predictions_to_states(
            relative[:, :3], ego_states, future_horizon=4.0, step_interval=0.5
        )
        return InterpolatedTrajectory(states)

    def close(self):
        if self._worker is not None:
            self._worker.terminate()
            self._worker.wait(timeout=10)
            self._worker = None


def tensorize_planner_input(current_input, initialization):
    ego_states = list(current_input.history.ego_states)[-5:]
    current = ego_states[-1]
    origin_x = current.rear_axle.x
    origin_y = current.rear_axle.y
    origin_yaw = current.rear_axle.heading
    ego = np.zeros((5, 4), dtype=np.float32)
    ego_mask = np.zeros(5, dtype=np.bool_)
    offset = 5 - len(ego_states)
    for index, state in enumerate(ego_states):
        x, y = _to_ego(state.rear_axle.x, state.rear_axle.y, origin_x, origin_y, origin_yaw)
        ego[offset + index] = [
            x,
            y,
            _wrap(state.rear_axle.heading - origin_yaw),
            state.dynamic_car_state.rear_axle_velocity_2d.magnitude(),
        ]
        ego_mask[offset + index] = True

    agents = np.zeros((64, 5, 7), dtype=np.float32)
    agent_mask = np.zeros((64, 5), dtype=np.bool_)
    agent_type = np.zeros(64, dtype=np.int32)
    observation = current_input.history.current_state[1]
    tracked = list(observation.tracked_objects.tracked_objects)
    tracked.sort(key=lambda obj: obj.center.distance_to(current.rear_axle))
    for index, obj in enumerate(tracked[:64]):
        x, y = _to_ego(obj.center.x, obj.center.y, origin_x, origin_y, origin_yaw)
        vx, vy = _rotate(obj.velocity.x, obj.velocity.y, -origin_yaw)
        agents[index, -1] = [
            x,
            y,
            _wrap(obj.center.heading - origin_yaw),
            vx,
            vy,
            obj.box.length,
            obj.box.width,
        ]
        agent_mask[index, -1] = True
        agent_type[index] = int(obj.tracked_object_type.value)

    map_values = np.zeros((128, 20, 5), dtype=np.float32)
    map_mask = np.zeros((128, 20), dtype=np.bool_)
    map_type = np.zeros(128, dtype=np.int32)
    map_objects = initialization.map_api.get_proximal_map_objects(
        Point2D(origin_x, origin_y),
        80.0,
        [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
    )
    polylines = []
    for layer_index, layer in enumerate((SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR)):
        for obj in map_objects.get(layer, []):
            polylines.append((layer_index + 1, obj.baseline_path.discrete_path))
    polylines.sort(key=lambda item: _path_distance(item[1], origin_x, origin_y))
    for row, (kind, path) in enumerate(polylines[:128]):
        for column, pose in enumerate(path[:20]):
            x, y = _to_ego(pose.x, pose.y, origin_x, origin_y, origin_yaw)
            map_values[row, column, :3] = [x, y, _wrap(pose.heading - origin_yaw)]
            map_mask[row, column] = True
        map_type[row] = kind

    route = np.zeros((64, 4), dtype=np.float32)
    route_mask = np.zeros(64, dtype=np.bool_)
    route_points = []
    for roadblock_id in initialization.route_roadblock_ids:
        roadblock = initialization.map_api.get_map_object(roadblock_id, SemanticMapLayer.ROADBLOCK)
        if roadblock is None:
            roadblock = initialization.map_api.get_map_object(
                roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR
            )
        if roadblock is not None:
            for edge in roadblock.interior_edges:
                route_points.extend(edge.baseline_path.discrete_path)
    route_points.sort(key=lambda pose: math.hypot(pose.x - origin_x, pose.y - origin_y))
    for index, pose in enumerate(route_points[:64]):
        x, y = _to_ego(pose.x, pose.y, origin_x, origin_y, origin_yaw)
        route[index, :3] = [x, y, _wrap(pose.heading - origin_yaw)]
        route_mask[index] = True

    return {
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
    }


def _to_ego(x, y, origin_x, origin_y, origin_yaw):
    return _rotate(x - origin_x, y - origin_y, -origin_yaw)


def _rotate(x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x - s * y, s * x + c * y


def _wrap(yaw):
    return (yaw + math.pi) % (2.0 * math.pi) - math.pi


def _path_distance(path, x, y):
    return min((math.hypot(pose.x - x, pose.y - y) for pose in path), default=float("inf"))


def _write_message(stream, payload):
    data = msgpack.packb(_encode(payload), use_bin_type=True)
    stream.write(struct.pack(">I", len(data)))
    stream.write(data)
    stream.flush()


def _read_message(stream):
    header = stream.read(4)
    if len(header) != 4:
        raise EOFError("Planner worker closed the response stream.")
    size = struct.unpack(">I", header)[0]
    body = stream.read(size)
    if len(body) != size:
        raise EOFError("Planner worker returned a truncated response.")
    return _decode(msgpack.unpackb(body, raw=False))


def _encode(value):
    if isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": value.dtype.str,
            "shape": value.shape,
            "data": value.tobytes(),
        }
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    return value


def _decode(value):
    if isinstance(value, dict) and value.get("__ndarray__") is True:
        return np.frombuffer(value["data"], dtype=np.dtype(value["dtype"])).reshape(value["shape"])
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value

