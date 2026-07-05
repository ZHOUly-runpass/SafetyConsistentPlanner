from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter

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
        try:
            import casadi as ca
        except ImportError as exc:
            if self._allow_fallback():
                return self._fallback(request, "casadi_not_installed")
            raise RuntimeError("CasADi/IPOPT is required by the configured backend.") from exc

        num_intervals = request.reference_states.shape[0] - 1
        request.validate(num_intervals=num_intervals)
        dt_values = np.diff(request.reference_timestamps)
        dt = float(np.median(dt_values))
        if dt <= 0.0 or not np.allclose(dt_values, dt, rtol=1e-4, atol=1e-8):
            raise ValueError("CasADi MPC requires a uniformly spaced positive time grid.")

        try:
            return self._solve_nlp(ca, request, num_intervals, dt)
        except Exception as exc:
            if self._allow_fallback():
                return self._fallback(request, f"casadi_failure:{type(exc).__name__}")
            return self._failure_result(request, num_intervals, exc)

    def _solve_nlp(self, ca, request: MpcRequest, nm: int, dt: float) -> MpcResult:
        vehicle = self._section("vehicle")
        limits = self._section("limits")
        dcbf = self._section("dcbf")
        weights = self._section("weights")
        ipopt = self._section("ipopt")
        timeouts = self._section("timeouts")

        wheelbase = float(vehicle.get("wheelbase", 3.0))
        ego_length = float(vehicle.get("length", 4.8))
        ego_width = float(vehicle.get("width", 2.0))
        velocity_bounds = self._bounds(limits, "velocity", 0.0, 15.0)
        acceleration_bounds = self._bounds(limits, "acceleration", -5.0, 3.0)
        steering_bounds = self._bounds(limits, "steering", -0.5, 0.5)
        steering_rate_bounds = self._bounds(limits, "steering_rate", -0.5, 0.5)
        gamma = float(dcbf.get("gamma", 0.1))
        inflation = float(dcbf.get("ellipse_inflation", dcbf.get("inflation", 0.3)))
        max_slack = float(dcbf.get("max_slack", 100.0))

        opti = ca.Opti()
        states = opti.variable(4, nm + 1)
        controls = opti.variable(2, nm)
        num_obstacles = len(request.predicted_obstacles)
        obstacle_slack = opti.variable(num_obstacles, nm) if num_obstacles else None
        road_slack = opti.variable(2, nm + 1) if request.road_corridor is not None else None

        opti.subject_to(states[:, 0] == request.initial_state)
        for k in range(nm):
            x, y, yaw, velocity = states[0, k], states[1, k], states[2, k], states[3, k]
            acceleration, steering = controls[0, k], controls[1, k]
            next_state = ca.vertcat(
                x + dt * velocity * ca.cos(yaw),
                y + dt * velocity * ca.sin(yaw),
                yaw + dt * velocity / wheelbase * ca.tan(steering),
                velocity + dt * acceleration,
            )
            opti.subject_to(states[:, k + 1] == next_state)

        opti.subject_to(opti.bounded(velocity_bounds[0], states[3, :], velocity_bounds[1]))
        opti.subject_to(
            opti.bounded(acceleration_bounds[0], controls[0, :], acceleration_bounds[1])
        )
        opti.subject_to(opti.bounded(steering_bounds[0], controls[1, :], steering_bounds[1]))
        if nm > 1:
            steering_rate = (controls[1, 1:] - controls[1, :-1]) / dt
            opti.subject_to(
                opti.bounded(steering_rate_bounds[0], steering_rate, steering_rate_bounds[1])
            )

        h_expressions: list[list] = []
        if obstacle_slack is not None:
            opti.subject_to(opti.bounded(0.0, obstacle_slack, max_slack))
            for obstacle_idx, obstacle in enumerate(request.predicted_obstacles):
                h_row = []
                for k in range(nm + 1):
                    h_row.append(
                        self._symbolic_barrier(
                            ca,
                            states[:, k],
                            ego_length,
                            ego_width,
                            obstacle.centers[k],
                            float(obstacle.yaws[k]),
                            float(obstacle.lengths[k]),
                            float(obstacle.widths[k]),
                            inflation,
                        )
                    )
                h_expressions.append(h_row)
                for k in range(nm):
                    if obstacle.valid_mask[k] and obstacle.valid_mask[k + 1]:
                        residual = h_row[k + 1] - (1.0 - gamma) * h_row[k]
                        opti.subject_to(residual + obstacle_slack[obstacle_idx, k] >= 0.0)
                        opti.subject_to(h_row[k + 1] + obstacle_slack[obstacle_idx, k] >= 0.0)

        if road_slack is not None:
            corridor = request.road_corridor
            opti.subject_to(opti.bounded(0.0, road_slack, max_slack))
            for k in range(nm + 1):
                if corridor.valid_mask[k]:
                    projection = ca.dot(corridor.normals[k], states[:2, k])
                    opti.subject_to(
                        projection - float(corridor.lower_bounds[k]) + road_slack[0, k] >= 0.0
                    )
                    opti.subject_to(
                        float(corridor.upper_bounds[k]) - projection + road_slack[1, k] >= 0.0
                    )

        reference = ca.DM(request.reference_states.T)
        position_error = states[:2, :] - reference[:2, :]
        yaw_error = ca.atan2(ca.sin(states[2, :] - reference[2, :]), ca.cos(states[2, :] - reference[2, :]))
        velocity_error = states[3, :] - reference[3, :]
        tracking_cost = (
            float(weights.get("position", 10.0)) * ca.sumsqr(position_error)
            + float(weights.get("yaw", 2.0)) * ca.sumsqr(yaw_error)
            + float(weights.get("velocity", 1.0)) * ca.sumsqr(velocity_error)
        )
        control_cost = (
            float(weights.get("acceleration", 0.1)) * ca.sumsqr(controls[0, :])
            + float(weights.get("steering", 0.1)) * ca.sumsqr(controls[1, :])
        )
        smoothness_cost = (
            float(weights.get("smoothness", 1.0))
            * ca.sumsqr(controls[:, 1:] - controls[:, :-1])
            if nm > 1
            else 0.0
        )
        slack_cost = 0.0
        if obstacle_slack is not None:
            slack_cost += float(weights.get("obstacle_slack", 1000.0)) * ca.sumsqr(obstacle_slack)
        if road_slack is not None:
            slack_cost += float(weights.get("road_slack", 1000.0)) * ca.sumsqr(road_slack)
        opti.minimize(tracking_cost + control_cost + smoothness_cost + slack_cost)

        initial_states = (
            request.initial_guess_states
            if request.initial_guess_states is not None
            else request.reference_states
        )
        initial_controls = (
            request.initial_guess_controls
            if request.initial_guess_controls is not None
            else NumpyVehicleDcbfMpcSolver(self._config)._derive_controls(initial_states, dt)
        )
        opti.set_initial(states, initial_states.T)
        opti.set_initial(controls, initial_controls.T)
        if obstacle_slack is not None:
            opti.set_initial(obstacle_slack, 0.01)
        if road_slack is not None:
            opti.set_initial(road_slack, 0.01)

        timeout_seconds = max(float(timeouts.get("solve_ms", 200.0)) / 1000.0, 0.001)
        plugin_options = {
            "expand": bool(self._section("casadi").get("expand", True)),
            "print_time": False,
        }
        solver_options = {
            "sb": "yes",
            "print_level": int(ipopt.get("print_level", 0)),
            "max_iter": int(ipopt.get("max_iter", 500)),
            "tol": float(ipopt.get("tol", 1e-4)),
            "acceptable_tol": float(ipopt.get("acceptable_tol", 1e-3)),
            "acceptable_iter": int(ipopt.get("acceptable_iter", 20)),
            "max_cpu_time": timeout_seconds,
            "hessian_approximation": str(ipopt.get("hessian_approximation", "limited-memory")),
        }
        opti.solver("ipopt", plugin_options, solver_options)

        started = perf_counter()
        solution = opti.solve()
        solve_time_ms = (perf_counter() - started) * 1000.0
        safe_states = np.asarray(solution.value(states), dtype=np.float64).T
        solved_controls = np.asarray(solution.value(controls), dtype=np.float64).T
        solved_obstacle_slack = (
            np.asarray(solution.value(obstacle_slack), dtype=np.float64).reshape(num_obstacles, nm)
            if obstacle_slack is not None
            else np.zeros((0, nm), dtype=np.float64)
        )
        solved_road_slack = (
            np.asarray(solution.value(road_slack), dtype=np.float64).reshape(2, nm + 1)
            if road_slack is not None
            else None
        )
        h_values = NumpyVehicleDcbfMpcSolver(self._config)._compute_h_values(safe_states, request)
        cbf_residuals = (
            compute_dcbf_residual(h_values[:, 1:], h_values[:, :-1], gamma, solved_obstacle_slack)
            if num_obstacles
            else np.zeros((0, nm), dtype=np.float64)
        )
        stats = solution.stats()
        iterations = int(stats.get("iter_count", 0))
        objective_tracking = float(solution.value(tracking_cost))
        objective_control = float(solution.value(control_cost))
        objective_smoothness = float(solution.value(smoothness_cost))
        objective_slack = float(solution.value(slack_cost))
        status_text = str(stats.get("return_status", "Solve_Succeeded"))
        feasible = bool(stats.get("success", True)) and np.all(cbf_residuals >= -1e-5)
        result = MpcResult(
            safe_states=safe_states,
            controls=solved_controls,
            h_values=h_values,
            cbf_residuals=cbf_residuals,
            obstacle_slack=solved_obstacle_slack,
            road_slack=solved_road_slack,
            feasible=feasible,
            timed_out="Maximum_CpuTime_Exceeded" in status_text,
            solver_status=0 if feasible else 1,
            iterations=iterations,
            objective_tracking=objective_tracking,
            objective_control=objective_control,
            objective_smoothness=objective_smoothness,
            objective_slack=objective_slack,
            objective_total=objective_tracking + objective_control + objective_smoothness + objective_slack,
            solve_time_ms=solve_time_ms,
            first_control=solved_controls[0].copy() if nm else np.zeros(2, dtype=np.float64),
            metadata={
                "backend": "casadi_ipopt",
                "return_status": status_text,
                "dt": dt,
                "num_obstacles": num_obstacles,
            },
        )
        populate_diagnostics(result, h_values, solved_obstacle_slack, request.reference_states, safe_states)
        result.validate(nm, num_obstacles)
        return result

    @staticmethod
    def _symbolic_barrier(
        ca,
        state,
        ego_length: float,
        ego_width: float,
        obstacle_center: NDArray[np.float64],
        obstacle_yaw: float,
        obstacle_length: float,
        obstacle_width: float,
        inflation: float,
    ):
        dx = float(obstacle_center[0]) - state[0]
        dy = float(obstacle_center[1]) - state[1]
        distance = ca.sqrt(dx * dx + dy * dy + 1e-8)
        cos_theta = dx / distance
        sin_theta = dy / distance
        ego_angle = ca.atan2(
            sin_theta * ca.cos(-state[2]) - cos_theta * ca.sin(-state[2]),
            cos_theta * ca.cos(-state[2]) + sin_theta * ca.sin(-state[2]),
        )
        obstacle_angle = ca.atan2(
            -sin_theta * np.cos(-obstacle_yaw) + cos_theta * np.sin(-obstacle_yaw),
            -cos_theta * np.cos(-obstacle_yaw) - sin_theta * np.sin(-obstacle_yaw),
        )
        ego_a, ego_b = (ego_length + inflation) / 2.0, (ego_width + inflation) / 2.0
        obs_a, obs_b = (obstacle_length + inflation) / 2.0, (obstacle_width + inflation) / 2.0
        ego_radius = ego_a * ego_b / ca.sqrt(
            (ego_b * ca.cos(ego_angle)) ** 2 + (ego_a * ca.sin(ego_angle)) ** 2 + 1e-8
        )
        obstacle_radius = obs_a * obs_b / ca.sqrt(
            (obs_b * ca.cos(obstacle_angle)) ** 2 + (obs_a * ca.sin(obstacle_angle)) ** 2 + 1e-8
        )
        return distance - ego_radius - obstacle_radius

    def _section(self, name: str) -> dict:
        return self._config.get(name, {}) if isinstance(self._config.get(name, {}), dict) else {}

    @staticmethod
    def _bounds(config: dict, name: str, lower: float, upper: float) -> tuple[float, float]:
        if name in config and isinstance(config[name], (list, tuple)):
            return float(config[name][0]), float(config[name][1])
        return float(config.get(f"{name}_min", lower)), float(config.get(f"{name}_max", upper))

    def _allow_fallback(self) -> bool:
        return bool(self._config.get("allow_numpy_fallback", False))

    def _fallback(self, request: MpcRequest, reason: str) -> MpcResult:
        result = NumpyVehicleDcbfMpcSolver(self._config).solve(request)
        result.metadata = result.metadata or {}
        result.metadata.update({"backend": "numpy_fallback", "reason": reason})
        return result

    @staticmethod
    def _failure_result(request: MpcRequest, nm: int, exc: Exception) -> MpcResult:
        message = str(exc)
        timed_out = "Maximum_CpuTime_Exceeded" in message or "max_cpu_time" in message
        return MpcResult(
            safe_states=request.reference_states.astype(np.float64).copy(),
            controls=np.zeros((nm, 2), dtype=np.float64),
            h_values=np.zeros((len(request.predicted_obstacles), nm + 1), dtype=np.float64),
            cbf_residuals=np.full((len(request.predicted_obstacles), nm), -np.inf, dtype=np.float64),
            obstacle_slack=np.zeros((len(request.predicted_obstacles), nm), dtype=np.float64),
            feasible=False,
            timed_out=timed_out,
            solver_status=-1,
            solve_time_ms=0.0,
            first_control=np.zeros(2, dtype=np.float64),
            metadata={"backend": "casadi_ipopt", "reason": message, "exception": type(exc).__name__},
        )
