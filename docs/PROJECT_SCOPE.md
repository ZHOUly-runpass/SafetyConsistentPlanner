# Project Scope: Safety-Consistent Learning-Based Trajectory Planning

## Objective

Build a learning-based multi-candidate trajectory planner with a D-CBF-MPC safety teacher (offline) and safety execution layer (online) for autonomous vehicle trajectory planning, using the nuPlan dataset.

## Architecture Pipeline

```
nuPlan scene input
  -> unified ego-centric scene adapter
  -> vector-only scene encoder (Workshop v1)
  -> IL base trajectory
  -> B-spline continuous multi-candidate trajectories
  -> safety prediction heads (h_min, feasibility, correction, risk)
  -> hard constraint filtering
  -> analytical cost + CBF/MPC consistency features
  -> lightweight ranker (GBDT or small MLP)
  -> Top-K candidates
  -> vehicle D-CBF-MPC
  -> first control execution
  -> closed-loop advance with fallback
```

## Workshop v1 Scope

### Included
- Vector-only scene encoding
- IL base trajectory regression
- B-spline multi-candidate generation
- Multi-attribute safety prediction
- Lightweight GBDT/MLP ranker
- Vehicle kinematic bicycle model D-CBF-MPC
- Fallback strategy (next candidate, reduced speed, conservative reference, emergency stop)
- nuPlan closed-loop integration

### Excluded (deferred)
- Reward World Model
- GRPO training
- Camera/image encoding
- Chance-constrained CBF
- Differentiable MPC
- Learned residual dynamics
- Real vehicle deployment

## Repository Structure

```
03_SafetyConsistentPlanner_Workshop/
  configs/          YAML configuration files
  docs/             Project documentation
  environment/      Environment setup scripts
  scripts/          Local and server scripts
  src/safety_planner/  All source code
  tests/            Unit, integration, golden tests
  data/             (empty locally; data on server)
  outputs/          (empty locally; outputs on server)
```

## Key Conventions

- Coordinate: ego rear-axle center, x forward, y left, yaw CCW positive
- Units: meter, second, radian, m/s, m/s^2
- MPC state: [x, y, yaw, velocity]  (float64)
- MPC control: [acceleration, steering]  (float64)
- Trajectory: [x, y, yaw, velocity, acceleration, curvature]  (float32)
- Planner candidates: [B, G, Np, 4]
- MPC reference: [Nm+1, 4], controls: [Nm, 2]
