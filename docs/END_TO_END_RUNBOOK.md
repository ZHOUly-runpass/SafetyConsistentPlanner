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
