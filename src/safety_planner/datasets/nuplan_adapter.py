from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..interfaces.scene import SceneSample


class NuPlanAdapter:
    def __init__(self, config_path: str | None = None) -> None:
        self._config_path = config_path
        self._fixture_scenes = self._load_fixture_scenes(config_path)

    def load_scene(self, scenario_id: str, timestamp_us: int) -> SceneSample:
        if scenario_id in self._fixture_scenes:
            payload = dict(self._fixture_scenes[scenario_id])
            payload.setdefault("scenario_id", scenario_id)
            payload.setdefault("timestamp_us", timestamp_us)
            return self._scene_from_payload(payload)

        try:
            import nuplan  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "nuPlan devkit is not installed. Install it on a Linux "
                "development machine, or pass a JSON fixture file for local "
                "interface tests."
            ) from exc
        raise RuntimeError(
            "Live nuPlan loading must be implemented against the exact devkit "
            "version and database paths on the Linux development machine."
        )

    def get_scenario_list(self, split: str = "trainval") -> list[str]:
        if self._fixture_scenes:
            return sorted(self._fixture_scenes.keys())
        try:
            import nuplan  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "nuPlan devkit is not installed and no local fixture scenes were provided."
            ) from exc
        raise RuntimeError(
            f"Scenario indexing for split {split!r} depends on configured nuPlan database paths."
        )

    @staticmethod
    def _load_fixture_scenes(config_path: str | None) -> dict[str, dict[str, Any]]:
        if config_path is None:
            return {}
        path = Path(config_path)
        if not path.exists() or path.suffix.lower() != ".json":
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        scenes = payload.get("scenes", payload)
        if not isinstance(scenes, dict):
            raise ValueError("Fixture JSON must contain a mapping of scenario ids to scenes.")
        return scenes

    @staticmethod
    def _scene_from_payload(payload: dict[str, Any]) -> SceneSample:
        def array(name: str, dtype: Any = np.float64):
            value = payload.get(name)
            return None if value is None else np.asarray(value, dtype=dtype)

        def list_of_arrays(name: str, dtype: Any = np.float64):
            value = payload.get(name)
            return None if value is None else [np.asarray(item, dtype=dtype) for item in value]

        sample = SceneSample(
            scenario_id=str(payload.get("scenario_id", "")),
            timestamp_us=int(payload.get("timestamp_us", 0)),
            ego_history=array("ego_history"),
            ego_history_mask=array("ego_history_mask", np.bool_),
            agent_history=array("agent_history"),
            agent_history_mask=array("agent_history_mask", np.bool_),
            agent_type=array("agent_type", np.int32),
            agent_track_ids=payload.get("agent_track_ids"),
            map_polylines=list_of_arrays("map_polylines"),
            map_mask=list_of_arrays("map_mask", np.bool_),
            map_type=payload.get("map_type"),
            route_polyline=array("route_polyline"),
            route_mask=array("route_mask", np.bool_),
            expert_future=array("expert_future"),
            expert_future_mask=array("expert_future_mask", np.bool_),
            metadata=payload.get("metadata"),
        )
        sample.validate()
        return sample
