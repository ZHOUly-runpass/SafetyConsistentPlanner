# Local Code Preparation Guide

## Overview

This guide explains how to prepare code on a local Windows machine and transfer it to a Linux development machine for training and closed-loop simulation.

## What is Prepared Locally

### Completed Files

| Category | Files | Description |
|----------|-------|-------------|
| Interfaces | `src/safety_planner/interfaces/` | 10 data interfaces including ranking, shape checks, and schema_version |
| Vehicle | `src/safety_planner/vehicle/` | kinematic bicycle model, coordinate transforms, geometry |
| D-CBF-MPC | `src/safety_planner/dcbf_mpc/` | barrier, CBF residual, resampling, diagnostics, NumPy reference solver, CasADi backend entry |
| Planning | `src/safety_planner/planning/` | candidate filtering, ranking, and fallback execution |
| Safety teacher | `src/safety_planner/safety_teacher/` | MpcResult to pseudo-label conversion |
| Evaluation | `src/safety_planner/evaluation/` | ADE, FDE, yaw, and velocity metrics |
| Models | `src/safety_planner/models/` | scene encoder, IL head, B-spline head, safety heads, ranker |
| Training | `src/safety_planner/training/` | loss functions and training entry points |
| Configs | `configs/` | 10 YAML configuration files |
| Tests | `tests/unit/` | 10 test files, 40 local test cases |
| Docs | `docs/` | project scope, interface spec, ROS2 reference mapping |

### What is NOT Done Locally

- Strict CasADi/IPOPT nonlinear optimization backend (local NumPy reference backend is available)
- Live nuPlan data loading (local JSON fixtures are supported; no automatic downloads)
- GPU training pipeline and large-scale training results
- GPU training, closed-loop simulation

## Transfer to Linux Dev Machine

### Steps

1. Package the workspace:
   ```bash
   # Windows PowerShell
   Compress-Archive -Path "D:\E2Eproject_MPC\03_SafetyConsistentPlanner_Workshop\*" -DestinationPath "workshop.zip"
   ```

2. Upload and extract:
   ```bash
   scp workshop.zip user@dev-machine:/home/user/
   ssh user@dev-machine
   cd /home/user/
   unzip workshop.zip -d 03_SafetyConsistentPlanner_Workshop
   ```

3. Verify:
   ```bash
   cd 03_SafetyConsistentPlanner_Workshop
   python -m pytest tests/unit/ -v
   ```

4. Set up environment:
   ```bash
   conda create -n safety_planner python=3.10 -y
   conda activate safety_planner
   pip install -r environment/learning/requirements.txt
   pip install -r environment/solver/requirements.txt
   ```

### Important Notes

- Local code does not depend on nuPlan, ROS2, or CUDA
- Update hardcoded paths in `configs/paths/server.example.yaml` to actual paths
- IPOPT solver parameters can be adjusted in `configs/mpc/solver.yaml`
- Vehicle parameters should eventually match nuPlan specifications; `configs/mpc/vehicle_bicycle.yaml` provides initial reasonable defaults
