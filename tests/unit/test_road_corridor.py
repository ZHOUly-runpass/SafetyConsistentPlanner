from __future__ import annotations

import numpy as np
import pytest

from src.safety_planner.interfaces import RoadCorridor


def test_road_corridor_validates_linear_bounds() -> None:
    corridor = RoadCorridor(
        normals=np.tile(np.array([0.0, 1.0]), (4, 1)),
        lower_bounds=np.full(4, -2.0),
        upper_bounds=np.full(4, 2.0),
        valid_mask=np.ones(4, dtype=np.bool_),
    )
    corridor.validate(num_intervals=3)


def test_road_corridor_rejects_inverted_bounds() -> None:
    corridor = RoadCorridor(
        normals=np.tile(np.array([0.0, 1.0]), (3, 1)),
        lower_bounds=np.array([-2.0, 3.0, -2.0]),
        upper_bounds=np.array([2.0, 2.0, 2.0]),
        valid_mask=np.ones(3, dtype=np.bool_),
    )
    with pytest.raises(ValueError, match="lower bound"):
        corridor.validate(num_intervals=2)

