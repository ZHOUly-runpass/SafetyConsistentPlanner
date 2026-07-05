from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.safety_planner.datasets.nuplan_extraction import (
    NuPlanExtractionConfig,
    extract_scene_sample,
)


def _point(x, y, heading=0.0):
    return SimpleNamespace(x=x, y=y, heading=heading)


def _state(x, y, heading, speed):
    return SimpleNamespace(
        rear_axle=_point(x, y, heading),
        dynamic_car_state=SimpleNamespace(
            rear_axle_velocity_2d=SimpleNamespace(magnitude=lambda: speed)
        ),
    )


def _agent(token, x, y, kind=1):
    center = _point(x, y, 0.0)
    return SimpleNamespace(
        track_token=token,
        center=center,
        velocity=SimpleNamespace(x=1.0, y=0.0),
        box=SimpleNamespace(length=4.0, width=2.0),
        tracked_object_type=SimpleNamespace(value=kind),
    )


def _detections(*objects):
    return SimpleNamespace(tracked_objects=SimpleNamespace(tracked_objects=list(objects)))


class _Scenario:
    token = "token-1"
    log_name = "log-1"
    scenario_type = "lane_following"
    map_api = SimpleNamespace(map_name="test-map")

    def __init__(self):
        self.current = _state(10.0, 20.0, np.pi / 2.0, 3.0)
        self.agent_future_requested = False

    def get_ego_state_at_iteration(self, iteration):
        assert iteration == 0
        return self.current

    def get_ego_past_trajectory(self, iteration, horizon, num_samples):
        assert num_samples == 3
        return iter([_state(10.0, 18.0, np.pi / 2.0, 2.0)])

    def get_past_tracked_objects(self, iteration, horizon, num_samples):
        return iter([_detections(_agent("near", 10.0, 22.0))])

    def get_tracked_objects_at_iteration(self, iteration):
        return _detections(_agent("far", 10.0, 30.0), _agent("near", 10.0, 23.0))

    def get_future_tracked_objects(self, *args, **kwargs):
        self.agent_future_requested = True
        raise AssertionError("Future agents must not be read.")

    def get_ego_future_trajectory(self, iteration, horizon, num_samples):
        return iter([_state(10.0, 24.0, np.pi / 2.0, 4.0)])

    def get_time_point(self, iteration):
        return SimpleNamespace(time_us=123456)


def test_extract_scene_uses_rear_axle_frame_and_left_padding():
    scenario = _Scenario()
    config = NuPlanExtractionConfig(history_steps=4, future_steps=2, max_agents=1)
    sample = extract_scene_sample(
        scenario,
        config=config,
        map_polylines_world=[(2, np.asarray([[10.0, 25.0, np.pi / 2.0]]))],
        route_polyline_world=np.asarray([[10.0, 26.0, np.pi / 2.0]]),
    )

    np.testing.assert_array_equal(sample.ego_history_mask, [False, False, True, True])
    np.testing.assert_allclose(sample.ego_history[-1], [0.0, 0.0, 0.0, 3.0], atol=1e-7)
    np.testing.assert_allclose(sample.ego_history[-2, :2], [-2.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(sample.expert_future[0, :2], [4.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(sample.map_polylines[0][0, :3], [5.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(sample.route_polyline[0, :3], [6.0, 0.0, 0.0], atol=1e-7)
    assert scenario.agent_future_requested is False


def test_agent_selection_is_current_distance_based_and_history_aligned():
    sample = extract_scene_sample(
        _Scenario(),
        config=NuPlanExtractionConfig(history_steps=4, future_steps=2, max_agents=1),
    )

    assert sample.agent_track_ids == ["near"]
    np.testing.assert_array_equal(sample.agent_history_mask[0], [False, False, True, True])
    np.testing.assert_allclose(sample.agent_history[0, -1, :2], [3.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(sample.agent_history[0, -2, :2], [2.0, 0.0], atol=1e-7)
