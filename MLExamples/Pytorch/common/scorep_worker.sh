#!/bin/bash
# Per-rank Score-P worker, launched by torchrun --no-python (see scorep_launch.sh).
#
# torchrun sets RANK / LOCAL_RANK / WORLD_SIZE in the environment for each worker;
# we give every rank its own Score-P experiment directory so they do not collide,
# then exec the training script under the Score-P Python instrumenter.
#
#   $SCOREP_EXP_BASE   base dir for per-rank experiments (default: scorep_run)
#   $SCOREP_PY_ARGS    extra args passed to `python -m scorep`
#                      (default: "--nocompiler --mpp=none")
set -u

RANK_ID="${RANK:-${LOCAL_RANK:-0}}"
export SCOREP_EXPERIMENT_DIRECTORY="${SCOREP_EXP_BASE:-scorep_run}/rank_${RANK_ID}"

exec python -m scorep ${SCOREP_PY_ARGS:---nopython --nocompiler --mpp=none} "$@"
