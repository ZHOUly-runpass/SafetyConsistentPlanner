from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from ..interfaces.mpc import MpcRequest, MpcResult
from ..vehicle.coordinate_transform import wrap_yaw
from .barrier import compute_dcbf_residual, compute_ellipse_barrier
from .diagnostics import populate_diagnostics


class DcbfMpcSolver(ABC):
    @abstractmethod
    def solve(self, request: MpcRequest) -> MpcResult:
        ...


class NumpyVehicleDcbfMpcSolver(DcbfMpcSolver):
    """Deterministic local safety-layer implementation.

    This class is intentionally dependency-light. It validates the request,
    follows the provided reference, derives a control sequence, and computes the
    same safety diagnostics expected from the CasADi/IPOPT backend. It is useful
    for local tests and synthetic development before the Linux solver stack is
    available.
    """

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}

    def solve(self, request: MpcRequest) -> MpcResult:
        num_intervals = request.reference_states.shape[0] - 1
        request.validate(num_intervals=num_intervals)

        timestamps = request.reference_timestamps.astype(np.float64)
        dt_values = np.diff(timestamps)
        dt = float(np.median(dt_values))
        if dt <= 0.0:
            raise ValueError("MPC timestamps must have positive spacing.")

        safe_states = request.reference_states.astype(np.float64).copy()
        safe_states[0] = request.initial_state.astype(np.float64)
        controls = self._derive_controls(safe_states, dt)

        h_values = self._compute_h_values(safe_states, request)
        if h_values.shape[0] == 0:
            cbf_residuals = np.zeros((0, num_intervals), dtype=np.float64)
            obstacle_slack = np.zeros((0, num_intervals), dtype=np.float64)
        else:
            gamma = self._dcbf_config().get("gamma", 0.2)
            raw_residuals = compute_dcbf_residual(
                h_values[:, 1:],
                h_values[:, :-1],
                gamma=float(gamma),
                slack=None,
            )
            obstacle_slack = np.maximum(0.0, -raw_residuals)
            cbf_residuals = raw_residuals + obstacle_slack

        result = MpcResult(
            safe_states=safe_states,
            controls=controls,
            h_values=h_values,
            cbf_residuals=cbf_residuals,
            obstacle_slack=obstacle_slack,
            feasible=self._is_feasible(h_values, obstacle_slack),
            timed_out=False,
            solver_status=0,
            iterations=1,
            objective_tracking=0.0,
            objective_control=float(np.sum(controls**2)),
            objective_smoothness=self._smoothness_cost(controls),
            objective_slack=float(np.sum(obstacle_slack**2)),
            solve_time_ms=0.0,
            first_control=controls[0].copy() if num_intervals > 0 else np.zeros(2, dtype=np.float64),
            metadata={
                "backend": "numpy_reference_tracker",
                "dt": dt,
                "num_obstacles": len(request.predicted_obstacles),
            },
        )
        result.objective_total = (
            result.objective_tracking
            + result.objective_control
            + result.objective_smoothness
            + result.objective_slack
        )
        populate_diagnostics(
            result,
            h_values=h_values,
            slack=obstacle_slack,
            reference_states=request.reference_states,
            safe_states=safe_states,
        )
        return result

    def _vehicle_config(self) -> dict:
        return self._config.get("vehicle", self._config)

    def _dcbf_config(self) -> dict:
        return self._config.get("dcbf", self._config)

    def _derive_controls(self, states: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
        num_intervals = states.shape[0] - 1
        controls = np.zeros((num_intervals, 2), dtype=np.float64)
        if num_intervals == 0:
            return controls

        vehicle = self._vehicle_config()
        wheelbase = float(vehicle.get("wheelbase", 3.0))
        acceleration_bounds = vehicle.get("acceleration", [-5.0, 3.0])
        steering_bounds = vehicle.get("steering", [-0.5, 0.5])

        dv = np.diff(states[:, 3]) / dt
        dyaw = wrap_yaw(np.diff(states[:, 2]))
        v_mid = np.maximum(np.abs(states[:-1, 3]), 1e-3)
        steering = np.arctan((dyaw / dt) * wheelbase / v_mid)

        controls[:, 0] = np.clip(dv, acceleration_bounds[0], acceleration_bounds[1])
        controls[:, 1] = np.clip(steering, steering_bounds[0], steering_bounds[1])
        return controls

    def _compute_h_values(
        self,
        states: NDArray[np.float64],
        request: MpcRequest,
    ) -> NDArray[np.float64]:
        num_obstacles = len(request.predicted_obstacles)
        num_points = states.shape[0]
        h_values = np.zeros((num_obstacles, num_points), dtype=np.float64)
        if num_obstacles == 0:
            return h_values

        vehicle = self._vehicle_config()
        dcbf = self._dcbf_config()
        ego_length = float(vehicle.get("length", 4.8))
        ego_width = float(vehicle.get("width", 2.0))
        inflation = float(dcbf.get("inflation", 0.0))

        for obstacle_idx, obstacle in enumerate(request.predicted_obstacles):
            valid_h = compute_ellipse_barrier(
                states[:, 0],
                states[:, 1],
                states[:, 2],
                ego_length,
                ego_width,
                obstacle.centers[:, 0],
                obstacle.centers[:, 1],
                obstacle.yaws,
                obstacle.lengths,
                obstacle.widths,
                inflation=inflation,
            )
            h_values[obstacle_idx] = np.where(obstacle.valid_mask, valid_h, np.inf)
        return h_values

    def _is_feasible(
        self,
        h_values: NDArray[np.float64],
        obstacle_slack: NDArray[np.float64],
    ) -> bool:
        dcbf = self._dcbf_config()
        min_margin = float(dcbf.get("min_margin", 0.0))
        slack_tolerance = float(dcbf.get("slack_tolerance", 1e-6))
        if h_values.size == 0:
            return True
        finite_h = h_values[np.isfinite(h_values)]
        if finite_h.size == 0:
            return True
        return bool(np.min(finite_h) >= min_margin and np.max(obstacle_slack) <= slack_tolerance)

    @staticmethod
    def _smoothness_cost(controls: NDArray[np.float64]) -> float:
        if controls.shape[0] < 2:
            return 0.0
        delta = np.diff(controls, axis=0)
        return float(np.sum(delta**2))


class CasadiVehicleDcbfMpcSolver(DcbfMpcSolver):
    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}

    def solve(self, request: MpcRequest) -> MpcResult:
        strict = bool(self._config.get("strict_casadi", False))
        try:
            import casadi  # noqa: F401
        except ImportError as exc:
            if strict:
                raise RuntimeError(
                    "CasADi/IPOPT is not installed. Install it on the Linux "
                    "development machine or set strict_casadi=false to use the "
                    "deterministic NumPy backend for local validation."
                ) from exc
            return NumpyVehicleDcbfMpcSolver(self._config).solve(request)

        # The workshop repository keeps the solver contract executable without
        # forcing IPOPT into local development. The same diagnostics are returned
        # here; strict nonlinear optimization can replace this method on the
        # Linux machine without changing callers.
        result = NumpyVehicleDcbfMpcSolver(self._config).solve(request)
        result.metadata = result.metadata or {}
        result.metadata["backend"] = "casadi_available_numpy_tracker"
        return result
