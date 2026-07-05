# End-to-End Runbook

## Gates

1. Run all Python commands through `scripts/safety_python.sh` while the host CPU
   issue remains unresolved.
2. Put licensed nuPlan v1.1 trainval data in `/data/nuplan/v1.1` and maps in
   `/data/nuplan/maps`.
3. Run `scripts/check_data_gate.py`. Full preprocessing additionally requires
   the administrator-created hardware stability marker.

## Environments

- `safety_planner_py310`: model, CasADi/IPOPT, training, worker and reports.
- `nuplan_py39`: official nuPlan 1.2.2 extraction and simulation adapter only.

Create the nuPlan environment in explicit stages so dependency failures remain
diagnosable:

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
taskset -c 0 bash scripts/install_nuplan_py39.sh
```

The base environment uses a current Python 3.9-compatible pip. The upstream
runtime pins remain unchanged, while `environment/nuplan_py39_constraints.txt`
freezes packages that the 2022 devkit left unbounded. The installer uses exact
URLs for the old CUDA wheels because pip 21.2.4 can no longer parse the current
PyTorch wheel index reliably.

The two environments communicate through the versioned MessagePack worker
protocol. They must not share site-packages because nuPlan pins NumPy 1.23.4.

## Small Reproducible Milestone

- Deterministically select 1000 train, 200 validation and 20 closed-loop IDs.
- Write fixed tensors in 512-sample NPZ shards and both JSONL and Parquet manifests.
- Train IL for five epochs, generate MPC pseudo-labels, train safety heads and ranker.
- Run non-reactive official closed-loop evaluation on the 20-ID manifest.
- Preserve configs, source commit, environment lock, checkpoints and reports.

Full trainval preprocessing and multi-worker training remain blocked until the
CPU/RAM stability issue is resolved.

## Implemented Validation

- Local suite: 61 passed; CasADi integration skipped when CasADi is unavailable.
- Development machine: 64 passed with the real CasADi/IPOPT backend.
- GPU smoke training: CUDA available, one epoch completed, best checkpoint
  loaded successfully.
- External blocker: `/data/nuplan/v1.1` and `/data/nuplan/maps` are not present.
- Hardware blocker: the first nuPlan dependency install ended in a segmentation
  fault. A later pip invocation failed while compiling a standard-library regex,
  which is consistent with the unresolved host stability issue. Do not approve
  full preprocessing until repeated environment validation is clean.
