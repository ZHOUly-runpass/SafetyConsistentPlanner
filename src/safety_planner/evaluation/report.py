from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ScenarioMetrics:
    scenario_id: str
    collision: bool
    off_road: bool
    progress: float
    comfort: float
    minimum_distance: float
    h_min: float
    slack_max: float
    infeasible: bool
    correction_l2: float
    solver_time_ms: float
    planning_latency_ms: float
    fallback_used: bool


def aggregate_metrics(metrics: list[ScenarioMetrics]) -> dict:
    if not metrics:
        raise ValueError("At least one scenario metric is required.")
    solver_times = np.asarray([item.solver_time_ms for item in metrics], dtype=np.float64)
    latencies = np.asarray([item.planning_latency_ms for item in metrics], dtype=np.float64)
    return {
        "num_scenarios": len(metrics),
        "collision_rate": float(np.mean([item.collision for item in metrics])),
        "off_road_rate": float(np.mean([item.off_road for item in metrics])),
        "mean_progress": float(np.mean([item.progress for item in metrics])),
        "mean_comfort": float(np.mean([item.comfort for item in metrics])),
        "minimum_distance": float(np.min([item.minimum_distance for item in metrics])),
        "h_min": float(np.min([item.h_min for item in metrics])),
        "slack_max": float(np.max([item.slack_max for item in metrics])),
        "infeasible_rate": float(np.mean([item.infeasible for item in metrics])),
        "mean_correction_l2": float(np.mean([item.correction_l2 for item in metrics])),
        "solver_time_ms": _percentiles(solver_times),
        "planning_latency_ms": _percentiles(latencies),
        "fallback_frequency": float(np.mean([item.fallback_used for item in metrics])),
    }


def write_evaluation_report(metrics: list[ScenarioMetrics], output_dir: str | Path) -> dict:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in metrics]
    aggregate = aggregate_metrics(metrics)
    (root / "per_scenario.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (root / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(rows), root / "per_scenario.parquet")
    except ImportError:
        pass
    lines = [
        "# SafetyConsistentPlanner Evaluation",
        "",
        f"- Scenarios: {aggregate['num_scenarios']}",
        f"- Collision rate: {aggregate['collision_rate']:.4f}",
        f"- Off-road rate: {aggregate['off_road_rate']:.4f}",
        f"- Mean progress: {aggregate['mean_progress']:.4f}",
        f"- Infeasible rate: {aggregate['infeasible_rate']:.4f}",
        f"- Fallback frequency: {aggregate['fallback_frequency']:.4f}",
        f"- Solver P50/P90/P95 (ms): {aggregate['solver_time_ms']['p50']:.2f} / "
        f"{aggregate['solver_time_ms']['p90']:.2f} / {aggregate['solver_time_ms']['p95']:.2f}",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate


def _percentiles(values: np.ndarray) -> dict[str, float]:
    p50, p90, p95 = np.percentile(values, [50, 90, 95])
    return {"p50": float(p50), "p90": float(p90), "p95": float(p95)}

