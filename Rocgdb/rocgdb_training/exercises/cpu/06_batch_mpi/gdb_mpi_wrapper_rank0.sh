#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# Per-rank (roc)gdb wrapper for MPI jobs - debug RANK 0 ONLY.
#
# Rank 0 runs under the debugger; every other rank runs the application
# normally. Wrap your executable with this script under your MPI launcher:
#
#   <mpi-launcher> ... ./gdb_mpi_wrapper_rank0.sh ./my_app [app args ...]
#
# e.g.  srun -n 512 ./gdb_mpi_wrapper_rank0.sh ./my_app
#       mpirun -np 512 ./gdb_mpi_wrapper_rank0.sh ./my_app
#
# Why rank 0 only:
#   - When you already know the bug hits EVERY rank, one rank's log is enough,
#     and you avoid a flood of hundreds of identical per-rank logs.
#   - Running only rank 0 under the debugger perturbs timing, which can trigger
#     or mask race conditions (a useful diagnostic in itself).
# To debug every rank instead, use the companion gdb_mpi_wrapper_all_ranks.sh.
#
# Debugger: rocgdb if it is on PATH, else gdb. Override with DBG=/path/to/dbg.
#
# Courtesy reference (independent prior art for this per-rank batch technique):
#   HLRS wiki "ROCgdb" - https://kb.hlrs.de/platforms/index.php/ROCgdb

set -uo pipefail

# Discover THIS process's MPI rank from the launcher's environment. The variable
# is launcher-specific; the first one that is set wins. Extend for other sites.
discover_rank() {
    if   [ -n "${OMPI_COMM_WORLD_RANK:-}" ]; then printf '%s' "$OMPI_COMM_WORLD_RANK"  # Open MPI (mpirun)
    elif [ -n "${PMIX_RANK:-}" ];            then printf '%s' "$PMIX_RANK"             # PMIx (OpenPMIx; PBS, OMPI>=4, ...)
    elif [ -n "${PMI_RANK:-}" ];             then printf '%s' "$PMI_RANK"              # PMI-1/2 (MPICH, Cray MPI, Intel MPI)
    elif [ -n "${SLURM_PROCID:-}" ];         then printf '%s' "$SLURM_PROCID"          # Slurm (srun)
    fi
}

rank="$(discover_rank)"
if [ -z "$rank" ]; then
    echo "gdb_mpi_wrapper: could not discover the MPI rank from the environment." >&2
    echo "  Expected one of OMPI_COMM_WORLD_RANK / PMIX_RANK / PMI_RANK / SLURM_PROCID." >&2
    echo "  Add your launcher's rank variable to discover_rank()." >&2
    exit 2
fi

# Every rank other than 0 runs the application normally, no debugger.
if [ "$rank" -ne 0 ]; then
    exec "$@"
fi

# Rank 0 only, from here on.
DBG="${DBG:-$(command -v rocgdb >/dev/null 2>&1 && echo rocgdb || echo gdb)}"
cmds="gdb_commands_${rank}.txt"

{
    echo "set pagination off"
    echo "set confirm off"
    echo "set breakpoint pending on"
    echo "catch signal SIGSEGV"
    echo "catch signal SIGABRT"
    echo "catch signal SIGBUS"
    [ "$(basename "$DBG")" = rocgdb ] && echo "set amdgpu precise-memory"
    echo "run"
    echo "thread apply all bt full"
    if [ "$(basename "$DBG")" = rocgdb ]; then
        echo "info queues"
        echo "info dispatches"
    fi
} > "$cmds"

# One per-rank log holding BOTH gdb's own output and the program's stdout/stderr,
# so nothing interleaves across ranks on the shared terminal.
exec "$DBG" -batch -x "$cmds" --args "$@" > "gdb_out_${rank}.txt" 2>&1
