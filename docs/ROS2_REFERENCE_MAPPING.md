# ROS2 D-CBF-MPC Reference Mapping

## Purpose

Map the upstream ROS2 D-CBF-MPC baseline modules to the new vehicle-oriented interface in this workshop.

## Upstream Modules -> Workshop Counterparts

| ROS2 Baseline (TurtleBot3) | Workshop (Vehicle) |
|----------------------------|---------------------|
| Unicycle model | Kinematic bicycle model |
| `local_planner/controller.py` (TurtleBot3 PID) | `interfaces/execution.py` (ExecutionOutput + first control) |
| `local_planner/local_planner.py` (CasADi MPC) | `dcbf_mpc/solver.py` (CasadiVehicleDcbfMpcSolver) |
| `local_planner/usage_casadi.py` (CBF helpers) | `dcbf_mpc/barrier.py` (ellipse barrier + D-CBF residual) |
| `local_map/ellipse.hpp` (obstacle ellipse fitting) | `vehicle/geometry.py` (obstacle ellipse params) |
| `local_map/local_map.cpp` (PointCloud2 -> grid -> ellipses) | Not ported; nuPlan provides agent states directly |
| `obs_param/kalman.cpp` (Kalman prediction) | Not ported; use nuPlan agent futures / constant velocity |
| `global_path_publisher/` (straight line) | Not ported; route comes from nuPlan |
| `scene/` (Gazebo simulation) | nuPlan closed-loop simulator (server) |
| `linear_path_publisher/` (goal-driven path) | Not ported |

## Key Differences

| Aspect | ROS2 Baseline | Workshop |
|--------|--------------|----------|
| Robot type | TurtleBot3 (unicycle) | Vehicle (kinematic bicycle) |
| State | [x, y, yaw] (3D) + separate velocity | [x, y, yaw, velocity] (4D) |
| Control | [v, omega] | [acceleration, steering] |
| Obstacle source | LiDAR -> ellipses (online fit) | nuPlan agent predictions (offline + online) |
| Horizon | ROS param configurable | Fixed config in YAML |
| Data flow | ROS2 topics (Float32MultiArray) | Named typed dataclasses |
| Execution | ROS2 node in loop | Python function call in loop |

## Reused Concepts (with adaptation)

1. **Ellipse barrier formulation**: Same math, adapted from `usage_casadi.py` to `barrier.py` for the bicycle model.
2. **D-CBF constraint**: `h(k+1) >= (1-gamma)*h(k) - slack(k)` preserved exactly.
3. **IPOPT solver options**: Adapted from `controller.py` to `solver.yaml` and `solver.py`.
4. **Slack penalty**: Same quadratic penalty, extracted to `diagnostics.py`.
5. **Obstacle prediction**: Ported the concept of future ellipse tracking, adapted to nuPlan agent trajectories.
