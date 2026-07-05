from __future__ import annotations

import json
from pathlib import Path

from ..interfaces import MpcRequest, MpcResult, PseudoLabelRecord
from ..dcbf_mpc import DcbfMpcSolver
import numpy as np
from ..dcbf_mpc import resample_trajectory_to_mpc_grid
from ..interfaces import PredictedObstacle


def grade_mpc_result(
    result: MpcResult,
    safe_margin_a: float = 0.5,
    safe_margin_b: float = 0.0,
    slack_tolerance: float = 1e-6,
) -> str:
    if not result.feasible or result.timed_out:
        return "C"
    if result.h_min >= safe_margin_a and result.slack_max <= slack_tolerance:
        return "A"
    if result.h_min >= safe_margin_b:
        return "B"
    return "C"


def build_pseudo_label_record(
    request: MpcRequest,
    result: MpcResult,
    code_commit: str = "",
) -> PseudoLabelRecord:
    grade = grade_mpc_result(result)
    failure_reason = "" if grade in ("A", "B") else result.metadata.get("reason", "unsafe_or_infeasible") if result.metadata else "unsafe_or_infeasible"
    record = PseudoLabelRecord(
        scenario_id=request.scenario_id,
        candidate_id=request.candidate_id,
        ref_mpc_states=request.reference_states,
        safe_mpc_states=result.safe_states,
        controls=result.controls,
        h_values=result.h_values,
        cbf_residuals=result.cbf_residuals,
        slack=result.obstacle_slack,
        feasible=result.feasible,
        solver_status=result.solver_status,
        objective_total=result.objective_total,
        objective_tracking=result.objective_tracking,
        objective_control=result.objective_control,
        objective_smoothness=result.objective_smoothness,
        objective_slack=result.objective_slack,
        correction_l2=result.correction_l2,
        correction_max=result.correction_max,
        solve_time_ms=result.solve_time_ms,
        quality_grade=grade,
        failure_reason=failure_reason,
        solver_config_hash=request.solver_config_hash,
        code_commit=code_commit,
        prediction_source=request.prediction_source,
    )
    record.validate()
    return record


def generate_pseudo_labels(
    requests: list[MpcRequest],
    solver: DcbfMpcSolver,
    output_dir: str | Path,
    code_commit: str = "",
) -> list[PseudoLabelRecord]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records: list[PseudoLabelRecord] = []
    manifest_rows: list[dict] = []
    for index, request in enumerate(requests):
        try:
            result = solver.solve(request)
            record = build_pseudo_label_record(request, result, code_commit=code_commit)
        except Exception as exc:
            record = PseudoLabelRecord(
                scenario_id=request.scenario_id,
                candidate_id=request.candidate_id,
                feasible=False,
                solver_status=-1,
                quality_grade="D",
                failure_reason=f"{type(exc).__name__}:{exc}",
                solver_config_hash=request.solver_config_hash,
                code_commit=code_commit,
                prediction_source=request.prediction_source,
            )
        filename = f"record-{index:08d}.json"
        record.save_json(root / filename)
        records.append(record)
        manifest_rows.append(
            {
                "schema_version": record.schema_version,
                "scenario_id": record.scenario_id,
                "candidate_id": record.candidate_id,
                "quality_grade": record.quality_grade,
                "feasible": record.feasible,
                "solver_status": record.solver_status,
                "failure_reason": record.failure_reason,
                "path": filename,
                "solver_config_hash": record.solver_config_hash,
                "code_commit": record.code_commit,
            }
        )
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    write_pseudo_label_shards(records, root, manifest_rows=manifest_rows)
    return records


def write_pseudo_label_shards(
    records: list[PseudoLabelRecord],
    output_dir: str | Path,
    shard_size: int = 512,
    manifest_rows: list[dict] | None = None,
) -> list[dict]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in manifest_rows] if manifest_rows is not None else []
    if rows and len(rows) != len(records):
        raise ValueError("manifest_rows must match records.")
    if not rows:
        rows = [_record_manifest_row(record) for record in records]
    array_fields = (
        "ref_planner_states",
        "safe_planner_states",
        "ref_mpc_states",
        "safe_mpc_states",
        "controls",
        "h_values",
        "cbf_residuals",
        "slack",
    )
    for shard_index, offset in enumerate(range(0, len(records), shard_size)):
        chunk = records[offset : offset + shard_size]
        shard_name = f"pseudo-labels-{shard_index:05d}.npz"
        arrays: dict[str, np.ndarray] = {
            "scenario_id": np.asarray([record.scenario_id for record in chunk]),
            "candidate_id": np.asarray([record.candidate_id for record in chunk], dtype=np.int32),
            "feasible": np.asarray([record.feasible for record in chunk], dtype=np.bool_),
            "solver_status": np.asarray([record.solver_status for record in chunk], dtype=np.int32),
            "quality_grade": np.asarray([record.quality_grade for record in chunk]),
            "failure_reason": np.asarray([record.failure_reason for record in chunk]),
            "objective_total": np.asarray([record.objective_total for record in chunk]),
            "objective_tracking": np.asarray([record.objective_tracking for record in chunk]),
            "objective_control": np.asarray([record.objective_control for record in chunk]),
            "objective_smoothness": np.asarray([record.objective_smoothness for record in chunk]),
            "objective_slack": np.asarray([record.objective_slack for record in chunk]),
            "correction_l2": np.asarray([record.correction_l2 for record in chunk]),
            "correction_max": np.asarray([record.correction_max for record in chunk]),
            "solve_time_ms": np.asarray([record.solve_time_ms for record in chunk]),
        }
        for field in array_fields:
            values, mask = _pad_record_arrays([getattr(record, field) for record in chunk])
            arrays[field] = values
            arrays[field + "_mask"] = mask
        np.savez_compressed(root / shard_name, **arrays)
        for index, row in enumerate(rows[offset : offset + len(chunk)]):
            row["shard"] = shard_name
            row["index"] = index
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for pseudo-label manifests.") from exc
    pq.write_table(pa.Table.from_pylist(rows), root / "manifest.parquet")
    return rows


def _record_manifest_row(record: PseudoLabelRecord) -> dict:
    return {
        "schema_version": record.schema_version,
        "scenario_id": record.scenario_id,
        "candidate_id": record.candidate_id,
        "quality_grade": record.quality_grade,
        "feasible": record.feasible,
        "solver_status": record.solver_status,
        "failure_reason": record.failure_reason,
        "solver_config_hash": record.solver_config_hash,
        "code_commit": record.code_commit,
        "prediction_source": record.prediction_source,
    }


def _pad_record_arrays(values: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray]:
    present = [np.asarray(value, dtype=np.float64) for value in values if value is not None]
    rank = max((value.ndim for value in present), default=1)
    shape = tuple(max((value.shape[axis] if axis < value.ndim else 1 for value in present), default=0) for axis in range(rank))
    output = np.zeros((len(values),) + shape, dtype=np.float64)
    mask = np.zeros((len(values),) + shape, dtype=np.bool_)
    for row, value in enumerate(values):
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float64)
        slices = (row,) + tuple(slice(0, size) for size in array.shape)
        output[slices] = array
        mask[slices] = True
    return output, mask


def build_candidate_requests(
    scenario_id: str,
    candidate_trajectories: np.ndarray,
    planner_timestamps: np.ndarray,
    initial_state: np.ndarray,
    predicted_obstacles: list[PredictedObstacle],
    mpc_num_intervals: int = 15,
    mpc_dt: float = 0.2,
    prediction_source: str = "unknown",
    solver_config_hash: str = "",
) -> list[MpcRequest]:
    if candidate_trajectories.ndim != 3 or candidate_trajectories.shape[2] != 4:
        raise ValueError("candidate_trajectories must have shape [G, T, 4].")
    requests = []
    for candidate_id, candidate in enumerate(candidate_trajectories):
        mpc_timestamps, reference, valid = resample_trajectory_to_mpc_grid(
            planner_timestamps,
            candidate,
            np.ones(candidate.shape[0], dtype=np.bool_),
            mpc_num_intervals,
            mpc_dt,
            current_time=float(planner_timestamps[0]),
        )
        if not np.all(valid):
            raise ValueError("Candidate does not cover the complete MPC horizon.")
        requests.append(
            MpcRequest(
                schema_version="1.0",
                scenario_id=scenario_id,
                candidate_id=candidate_id,
                initial_state=np.asarray(initial_state, dtype=np.float64),
                reference_states=reference,
                reference_timestamps=mpc_timestamps,
                predicted_obstacles=predicted_obstacles,
                prediction_source=prediction_source,
                solver_config_hash=solver_config_hash,
            )
        )
    return requests


def records_to_safety_targets(
    records: list[PseudoLabelRecord],
    scenario_ids: list[str],
    num_candidates: int,
) -> dict[str, np.ndarray]:
    grouped = {(record.scenario_id, record.candidate_id): record for record in records}
    shape = (len(scenario_ids), num_candidates)
    h_min = np.full(shape, -1e3, dtype=np.float32)
    feasible = np.zeros(shape, dtype=np.float32)
    correction = np.zeros(shape, dtype=np.float32)
    risk = np.full(shape, 1e3, dtype=np.float32)
    for scene_index, scenario_id in enumerate(scenario_ids):
        for candidate_id in range(num_candidates):
            record = grouped.get((scenario_id, candidate_id))
            if record is None:
                continue
            if record.h_values is not None and record.h_values.size:
                finite = record.h_values[np.isfinite(record.h_values)]
                h_min[scene_index, candidate_id] = float(np.min(finite)) if finite.size else 1e3
            feasible[scene_index, candidate_id] = float(record.feasible)
            correction[scene_index, candidate_id] = float(record.correction_l2)
            slack_max = float(np.max(record.slack)) if record.slack is not None and record.slack.size else 0.0
            risk[scene_index, candidate_id] = max(0.0, -h_min[scene_index, candidate_id]) + slack_max
    return {
        "target_h_min": h_min,
        "target_feasible": feasible,
        "target_correction": correction,
        "target_risk": risk,
    }


def write_ranker_features(records: list[PseudoLabelRecord], output_path: str | Path) -> dict:
    rows = []
    labels = []
    for record in records:
        finite_h = (
            record.h_values[np.isfinite(record.h_values)]
            if record.h_values is not None
            else np.asarray([], dtype=np.float64)
        )
        h_min = float(np.min(finite_h)) if finite_h.size else -1e3
        slack_max = (
            float(np.max(record.slack))
            if record.slack is not None and record.slack.size
            else 0.0
        )
        rows.append(
            [
                record.objective_total,
                h_min,
                float(record.feasible),
                record.correction_l2,
                slack_max,
            ]
        )
        labels.append(-record.objective_total - (0.0 if record.feasible else 1e6))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    features = np.asarray(rows, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    np.savez(output, features=features, labels=target)
    return {"num_rows": len(records), "num_features": features.shape[1] if features.size else 5}
