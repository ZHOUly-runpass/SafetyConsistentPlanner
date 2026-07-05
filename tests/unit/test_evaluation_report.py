from __future__ import annotations

import json

from src.safety_planner.evaluation.report import (
    ScenarioMetrics,
    aggregate_metrics,
    write_evaluation_report,
)


def _metric(index: int) -> ScenarioMetrics:
    return ScenarioMetrics(
        scenario_id=f"scene-{index}",
        collision=index == 0,
        off_road=False,
        progress=float(index),
        comfort=1.0,
        minimum_distance=2.0 + index,
        h_min=0.5,
        slack_max=0.0,
        infeasible=False,
        correction_l2=0.1,
        solver_time_ms=10.0 + index,
        planning_latency_ms=20.0 + index,
        fallback_used=index == 1,
    )


def test_aggregate_metrics_computes_rates_and_percentiles() -> None:
    aggregate = aggregate_metrics([_metric(0), _metric(1)])
    assert aggregate["collision_rate"] == 0.5
    assert aggregate["fallback_frequency"] == 0.5
    assert aggregate["solver_time_ms"]["p95"] > aggregate["solver_time_ms"]["p50"]


def test_write_evaluation_report_creates_required_artifacts(tmp_path) -> None:
    write_evaluation_report([_metric(0), _metric(1)], tmp_path)
    assert (tmp_path / "per_scenario.json").exists()
    assert (tmp_path / "aggregate.json").exists()
    assert (tmp_path / "report.md").exists()
    assert json.loads((tmp_path / "aggregate.json").read_text())["num_scenarios"] == 2
