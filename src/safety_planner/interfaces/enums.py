from __future__ import annotations

from enum import IntEnum


class MpcStateIndex(IntEnum):
    X = 0
    Y = 1
    YAW = 2
    VELOCITY = 3


class MpcControlIndex(IntEnum):
    ACCELERATION = 0
    STEERING = 1


class TrajectoryIndex(IntEnum):
    X = 0
    Y = 1
    YAW = 2
    VELOCITY = 3
    ACCELERATION = 4
    CURVATURE = 5
