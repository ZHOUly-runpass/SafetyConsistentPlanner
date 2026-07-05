#!/usr/bin/env bash
set -euo pipefail

DEVKIT_COMMIT="e9241677997dd86bfc0bcd44817ab04fe631405b"
ROOT="${SAFETY_PLANNER_ROOT:-$HOME/SafetyConsistentPlanner}"
MAMBA="${MICROMAMBA_BIN:-$HOME/.local/bin/micromamba}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/software/micromamba-root}"
ENV_PREFIX="$MAMBA_ROOT_PREFIX/envs/nuplan_py39"
PYTHON="$ENV_PREFIX/bin/python"
CPUSET="${SAFETY_PLANNER_CPUSET:-0}"
CONSTRAINTS="$ROOT/environment/nuplan_py39_constraints.txt"
UPSTREAM="https://raw.githubusercontent.com/motional/nuplan-devkit/$DEVKIT_COMMIT"

run_python() {
  taskset -c "$CPUSET" "$PYTHON" "$@"
}

echo "[1/5] Creating the isolated Python 3.9 base environment"
taskset -c "$CPUSET" "$MAMBA" create -y -f "$ROOT/environment/nuplan_py39.yml"

echo "[2/5] Installing exact official CUDA 11.1 PyTorch wheels"
run_python -m pip install --no-deps \
  "https://download.pytorch.org/whl/cu111/torch-1.9.0%2Bcu111-cp39-cp39-linux_x86_64.whl" \
  "https://download.pytorch.org/whl/cu111/torchvision-0.10.0%2Bcu111-cp39-cp39-linux_x86_64.whl" \
  "https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl"

echo "[3/5] Installing official Python dependencies with compatibility constraints"
run_python -m pip install -c "$CONSTRAINTS" \
  -r "$UPSTREAM/requirements_torch.txt" \
  -r "$UPSTREAM/requirements.txt" \
  msgpack==1.1.1

echo "[4/5] Installing the pinned nuPlan devkit without re-resolving dependencies"
run_python -m pip install --no-deps \
  "git+https://github.com/motional/nuplan-devkit.git@$DEVKIT_COMMIT"

echo "[5/5] Validating imports and writing the resolved environment lock"
run_python -c "import msgpack, nuplan, numpy, torch; print(numpy.__version__, torch.__version__)"
run_python -m pip check
run_python -m pip freeze > "$ROOT/environment/nuplan_py39.lock.txt"
