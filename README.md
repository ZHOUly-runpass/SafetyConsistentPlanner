# Safety Consistent Planner

Workshop implementation of a multi-candidate trajectory planner with safety
prediction, candidate ranking, and a D-CBF-MPC execution interface.

## Environment

The validated development environment uses Python 3.10. Install the package
and test dependencies with:

```bash
python -m pip install -e ".[dev,learning,solver]"
python -m pytest -q -rs
```

For the development machine, use the pinned files under `requirements/` and
the environment snapshot under `outputs/environment/`. Install the CUDA build
of PyTorch from a source compatible with the NVIDIA driver; the validated
server environment uses PyTorch 2.7.0 with CUDA 12.6.

## Data And Paths

Copy `configs/paths/server.example.yaml` to `configs/paths/server.yaml` and set
machine-specific paths there. nuPlan data and maps are expected outside the
repository and are not downloaded by setup scripts.

## Current Boundary

The JSON fixture adapter, NumPy reference tracker, model forward paths,
candidate execution, fallback behavior, and training dry-runs are covered by
unit tests. `CasadiVehicleDcbfMpcSolver.solve()` still delegates to the NumPy
reference tracker; a constrained nonlinear D-CBF-MPC implementation and real
nuPlan closed-loop integration remain future work.

