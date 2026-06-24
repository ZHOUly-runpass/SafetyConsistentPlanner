from __future__ import annotations

from ..interfaces import MpcRequest, MpcResult, PseudoLabelRecord


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
