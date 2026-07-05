from __future__ import annotations

import argparse
import sys

from safety_planner.planning.ipc import serve_messages
from safety_planner.planning.worker import PlannerWorker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    worker = PlannerWorker(args.checkpoint, args.device)
    serve_messages(sys.stdin.buffer, sys.stdout.buffer, worker.handle)


if __name__ == "__main__":
    main()

