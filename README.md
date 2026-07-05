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

The repository now includes a real CasADi/IPOPT D-CBF-MPC backend, typed road
corridors, fixed tensorized scene shards, a mask-aware vector model, executable
IL/safety/ranker training paths, pseudo-label generation, the persistent
MessagePack planner worker, and evaluation report generation.

The standalone Python 3.9 nuPlan adapter is implemented under `integration/`,
but real trainval extraction and the 20-scenario official closed-loop run still
require licensed nuPlan data and maps at the configured server paths. Full
preprocessing and multi-worker training remain blocked until the development
machine's CPU/RAM stability issue is resolved.
