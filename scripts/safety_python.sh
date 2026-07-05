#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${SAFETY_PLANNER_PYTHON:-/home/zhou/software/miniconda3/envs/safety_planner_py310/bin/python}"
exec taskset -c "${SAFETY_PLANNER_CPUSET:-0}" "$PYTHON_BIN" "$@"

