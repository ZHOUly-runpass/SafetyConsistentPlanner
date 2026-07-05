from __future__ import annotations

import argparse
import sys

from safety_planner.planning.ipc import serve_messages
from safety_planner.planning.worker import PlannerWorker
from safety_planner.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--solver-config")
    args = parser.parse_args()
    solver_config = load_config(args.solver_config) if args.solver_config else {}
    if isinstance(solver_config.get("solver"), dict):
        solver_config = {**solver_config, **solver_config["solver"]}
    worker = PlannerWorker(args.checkpoint, args.device, solver_config=solver_config)
    serve_messages(sys.stdin.buffer, sys.stdout.buffer, worker.handle)


if __name__ == "__main__":
    main()
