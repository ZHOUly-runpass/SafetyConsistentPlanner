from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np

from safety_planner.planning.ipc import read_message, write_message


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--solver-config")
    args = parser.parse_args()
    command = [
        sys.executable,
        "scripts/planner_worker.py",
        "--checkpoint",
        args.checkpoint,
        "--device",
        args.device,
    ]
    if args.solver_config:
        command.extend(["--solver-config", args.solver_config])
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    try:
        request = {
            "schema_version": "1.0",
            "request_id": "smoke-1",
            "scenario_id": "worker-smoke",
            "timestamp_us": 0,
            "scene": _empty_scene(),
            "initial_state": np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            "planner_timestamps": np.arange(1, 9, dtype=np.float64) * 0.5,
            "predicted_obstacles": [],
        }
        write_message(process.stdin, request)
        response = read_message(process.stdout)
        if response is None or not response.get("ok"):
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError("Worker smoke failed: %r\n%s" % (response, stderr))
        _validate_response(response)
        print(
            json.dumps(
                {
                    "selected_candidate_id": response["selected_candidate_id"],
                    "fallback": response["fallback"],
                    "mpc": response["mpc"],
                    "trajectory_shape": list(response["selected_trajectory"].shape),
                    "first_control": response["first_control"].tolist(),
                },
                indent=2,
            )
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _empty_scene() -> dict:
    return {
        "ego_history": np.zeros((5, 4), dtype=np.float32),
        "ego_history_mask": np.ones(5, dtype=np.bool_),
        "agent_history": np.zeros((64, 5, 7), dtype=np.float32),
        "agent_history_mask": np.zeros((64, 5), dtype=np.bool_),
        "agent_type": np.zeros(64, dtype=np.int32),
        "map_polylines": np.zeros((128, 20, 5), dtype=np.float32),
        "map_mask": np.zeros((128, 20), dtype=np.bool_),
        "map_type": np.zeros(128, dtype=np.int32),
        "route_polyline": np.zeros((64, 4), dtype=np.float32),
        "route_mask": np.zeros(64, dtype=np.bool_),
    }


def _validate_response(response: dict) -> None:
    if response.get("request_id") != "smoke-1":
        raise RuntimeError("Worker response id mismatch.")
    if np.asarray(response.get("first_control")).shape != (2,):
        raise RuntimeError("Worker response has invalid first_control.")
    trajectory = np.asarray(response.get("selected_trajectory"))
    if trajectory.ndim != 2 or trajectory.shape[1] != 4:
        raise RuntimeError("Worker response has invalid selected_trajectory.")
    if not isinstance(response.get("mpc"), dict) or "h_min" not in response["mpc"]:
        raise RuntimeError("Worker response is missing MPC diagnostics.")
    if response.get("fallback") not in {
        "none",
        "next_candidate",
        "reduced_speed",
        "conservative_reference",
        "emergency_stop",
    }:
        raise RuntimeError("Worker response has invalid fallback mode.")


if __name__ == "__main__":
    main()
