#!/usr/bin/env bash
# Run baselines_main.py for every ADVERSARIAL_CONFIGS grid preset (ADV_PRESET),
# plus the no-attack defense matrix.
# Usage (from repo root):
#   ./scripts/run_baselines_grid.sh
#   ./scripts/run_baselines_grid.sh seed=123
#
# Default Hydra overrides match: algo=sac_algo env.name=antmaze-v1
# Edit DEFAULT_HYDRA below if needed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/scripts${PYTHONPATH:+:$PYTHONPATH}"

DEFAULT_HYDRA=(algo=sac_algo env.name=antmaze-v1)

PRESETS="$(python3 -c "from ADVERSARIAL_CONFIGS import GRID_PRESET_NAMES, NO_ATTACK_PRESET_NAMES; print(' '.join(NO_ATTACK_PRESET_NAMES + GRID_PRESET_NAMES))")"

for p in ${PRESETS}; do
  echo "========================================"
  echo "ADV_PRESET=${p}"
  echo "========================================"
  ADV_PRESET="${p}" python3 "${REPO_ROOT}/scripts/baselines_main.py" "${DEFAULT_HYDRA[@]}" "$@"
done

echo "All grid presets finished."
