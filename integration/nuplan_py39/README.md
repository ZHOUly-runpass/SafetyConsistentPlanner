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
