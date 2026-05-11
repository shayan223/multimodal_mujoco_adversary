#!/usr/bin/env bash
# Run one data-collection rollout and save benign/adversarial pairs for
# FGSM epsilons 0.007 and 0.015.
# Usage (from repo root):
#   ./scripts/run_collect_fgsm_dataset.sh
#   ./scripts/run_collect_fgsm_dataset.sh seed=123

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/scripts${PYTHONPATH:+:$PYTHONPATH}"

python3 "${REPO_ROOT}/scripts/collect_adversarial_dataset.py" \
  --attack-choice FGSM \
  --collection-attacks FGSM:0.007,FGSM:0.015 \
  --experiment-name fgsm_collection \
  "$@"
