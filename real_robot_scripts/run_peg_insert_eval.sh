#!/usr/bin/env bash
# Step-by-step runner for the real-robot FORGE peg-insert evaluation.
#
# Usage:
#   real_robot_scripts/run_peg_insert_eval.sh <step> <checkpoint_dir> [extra args...]
#
#   steps:
#     dry        off-hardware mock dry-run of calibrate + eval (no robot, no wandb)
#     calibrate  capture + write the hole goal into eval_config.yaml (on robot)
#     readstate  print the assembled 24-D observation for one live pose (on robot)
#     smoke      2-episode eval, no wandb — confirm force levels / stability (on robot)
#     full       full eval batch, logs Eval_Core/* to wandb (on robot)
#
# Env: the ON-ROBOT steps need an environment with BOTH skrl (policy) and
# pylibfranka (FR3 driver). Set $EVAL_ENV to that conda env (see README).
# The 'dry' step only needs skrl+torch (the training env works).
set -euo pipefail

STEP="${1:-}"
CKPT="${2:-}"
shift $(( $# >= 2 ? 2 : $# )) || true
EXTRA=("$@")

CONFIG="real_robot_scripts/eval_config.yaml"
EVAL_ENV="${EVAL_ENV:-fail}"          # env with skrl (+ pylibfranka for on-robot)
AGENT="${AGENT:-0}"
NUM_EPISODES="${NUM_EPISODES:-20}"

run() { echo "+ conda run -n ${EVAL_ENV} --no-capture-output python $*"; conda run -n "${EVAL_ENV}" --no-capture-output python "$@"; }

case "${STEP}" in
  dry)
    echo "### Mock dry-run (no hardware) ###"
    run real_robot_scripts/calibrate_goal.py --config "${CONFIG}" \
        --override robot.use_mock=true --dry --no_verify --no_write
    run real_robot_scripts/peg_insert_eval.py --checkpoint "${CKPT}" --agent "${AGENT}" \
        --config "${CONFIG}" --override robot.use_mock=true --no_wandb --num_episodes 2 "${EXTRA[@]}"
    ;;
  calibrate)
    echo "### Goal calibration (on robot) ###"
    run real_robot_scripts/calibrate_goal.py --config "${CONFIG}" "${EXTRA[@]}"
    ;;
  readstate)
    echo "### Read-state obs cross-check (on robot) ###"
    run real_robot_scripts/peg_insert_eval.py --checkpoint "${CKPT}" --agent "${AGENT}" \
        --config "${CONFIG}" --read_state "${EXTRA[@]}"
    ;;
  smoke)
    echo "### Smoke eval: 2 episodes, no wandb (on robot) ###"
    run real_robot_scripts/peg_insert_eval.py --checkpoint "${CKPT}" --agent "${AGENT}" \
        --config "${CONFIG}" --no_wandb --num_episodes 2 "${EXTRA[@]}"
    ;;
  full)
    echo "### Full eval: ${NUM_EPISODES} episodes -> wandb (on robot) ###"
    run real_robot_scripts/peg_insert_eval.py --checkpoint "${CKPT}" --agent "${AGENT}" \
        --config "${CONFIG}" --num_episodes "${NUM_EPISODES}" "${EXTRA[@]}"
    ;;
  *)
    echo "usage: $0 {dry|calibrate|readstate|smoke|full} <checkpoint_dir> [extra args]"
    exit 1
    ;;
esac
