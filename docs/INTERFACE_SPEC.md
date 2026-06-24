# Interface Specification

## Overview

This document defines all Python data interfaces used at module boundaries. All interfaces use named dataclasses with explicit shape annotations and schema versioning.

## Coordinate & Time Configuration

| Parameter | Value |
|-----------|-------|
| Reference point | Ego rear-axle center |
| x-axis | Forward |
| y-axis | Left |
| Yaw positive | Counter-clockwise |
| Yaw range | [-pi, pi) |
| History horizon | 2.0 s, dt=0.5 s, 5 points |
| Planner horizon | 4.0 s, dt=0.5 s, 8 future points |
| MPC horizon | 3.0 s, dt=0.2 s, N=15 intervals |
| Replan period | 0.5 s |

## Interface Modules

### Module: `interfaces/enums.py`
- `MpcStateIndex`: X=0, Y=1, YAW=2, VELOCITY=3
- `MpcControlIndex`: ACCELERATION=0, STEERING=1
- `TrajectoryIndex`: X=0, Y=1, YAW=2, VELOCITY=3, ACCEL=4, CURVATURE=5

### Module: `interfaces/trajectory.py`
- `VehicleState`: [x, y, yaw, velocity] in SI units
- `VehicleControl`: [acceleration, steering] in SI units
- `Trajectory`: timestamps [T], states [T, D], valid_mask [T]

### Module: `interfaces/obstacle.py`
- `PredictedObstacle`: centers [Nm+1, 2], yaws/widths/lengths [Nm+1], velocities [Nm+1, 2], valid_mask [Nm+1]

### Module: `interfaces/scene.py`
- `SceneSample`: ego_history [H, D_ego], agent_history [A, H, D_agent], map_polylines, route_polyline, expert_future [Np, 4]

### Module: `interfaces/mpc.py`
- `MpcRequest`: initial_state [4], reference_states [Nm+1, 4], reference_timestamps [Nm+1], predicted_obstacles
- `MpcResult`: safe_states [Nm+1, 4], controls [Nm, 2], h_values [A, Nm+1], cbf_residuals [A, Nm], slack [A, Nm], first_control [2]

### Module: `interfaces/planner.py`
- `PlannerOutput`: base_trajectory [B, Np, 4], candidate_trajectories [B, G, Np, 4], safety predictions [B, G]

### Module: `interfaces/ranking.py`
- `RankingFeatures`: per-candidate analytical cost, predicted h_min, feasibility probability, correction, risk, confidence [G]
- `RankingResult`: sorted candidate_ids [G], scores [G], selected_candidate_id

### Module: `interfaces/pseudo_label.py`
- `PseudoLabelRecord`: dual-grid trajectories, h_values, slack, feasible, quality_grade (A/B/C/D)

### Module: `interfaces/execution.py`
- `FallbackMode`: none, next_candidate, reduced_speed, conservative_reference, emergency_stop
- `ExecutionOutput`: selected_candidate_id, first_control, fallback_mode, diagnostics

## Shape Conventions

| Tensor | Shape | Dtype |
|--------|-------|-------|
| Planner candidates | [B, G, Np, 4] | float32 |
| MPC reference states | [Nm+1, 4] | float64 |
| MPC controls | [Nm, 2] | float64 |
| Obstacle centers | [A, Nm+1, 2] | float64 |
| h values | [A, Nm+1] | float64 |
| CBF residuals/slack | [A, Nm] | float64 |
| First control | [2] | float64 |

## Solver Backends

- `NumpyVehicleDcbfMpcSolver`: local reference backend. It tracks the provided reference, derives controls, computes ellipse barrier values, D-CBF residuals, slack, and complete `MpcResult` diagnostics without CasADi/IPOPT.
- `CasadiVehicleDcbfMpcSolver`: development-machine backend entry. In non-strict mode it falls back to the NumPy backend when CasADi is missing. In strict mode it raises a clear dependency error.

## Dtype Policy

- Solver (CasADi/IPOPT): float64
- Model (PyTorch training): float32
- Storage/pseudo labels: float64

## Versioning

All core dataclasses contain a `schema_version` field. When the schema changes, increment the version and update this document.
