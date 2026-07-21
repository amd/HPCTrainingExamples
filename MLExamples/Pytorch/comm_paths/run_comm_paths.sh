#!/bin/bash
# Launch the 4-way all-reduce comparison, one rank per GPU.
#   ./run_comm_paths.sh          # 2 ranks
#   NRANKS=8 ./run_comm_paths.sh # 8 ranks
set -u

NRANKS=${NRANKS:-2}

# Only load what is not already loaded (avoids reloading the heavy pytorch env).
if ! module -t list 2>&1 | grep -q "^rocm/"; then
  module load rocm
fi
for m in pytorch mpi4py cupy; do
  if ! module -t list 2>&1 | grep -q "^${m}/"; then
    module load "${m}"
  fi
done

HERE=$(dirname "$(readlink -f "$0")")

if command -v mpirun >/dev/null 2>&1; then
  mpirun -np "${NRANKS}" python3 "${HERE}/allreduce_comm_paths.py"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
  srun -n "${NRANKS}" python3 "${HERE}/allreduce_comm_paths.py"
else
  echo "need mpirun or an srun allocation to launch ${NRANKS} ranks" >&2
  exit 1
fi
