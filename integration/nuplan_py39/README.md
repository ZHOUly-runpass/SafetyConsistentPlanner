# nuPlan Python 3.9 Adapter

This directory is intentionally standalone. Copy it into the isolated official
nuPlan 1.2.2 environment and instantiate `SafetyConsistentPlannerAdapter` with
the persistent Python 3.10 worker command, for example:

```python
SafetyConsistentPlannerAdapter([
    "/home/zhou/.local/bin/safety-planner-python",
    "/home/zhou/SafetyConsistentPlanner/scripts/planner_worker.py",
    "--checkpoint",
    "/home/zhou/SafetyConsistentPlanner/outputs/checkpoints/safety_small/last.pt",
])
```

The adapter uses only Python 3.9 syntax and does not import the main package.

After the licensed trainval data and maps are present, create the deterministic
scenario index first:

```bash
taskset -c 0 python integration/nuplan_py39/extract_dataset.py --index-only
```

Remove `--index-only` only for the fixed 1000/200/20 milestone. The command
writes fixed-shape NPZ shards plus JSONL and Parquet manifests under
`/data/nuplan/processed/safety_planner/v1`. It never downloads source data.
