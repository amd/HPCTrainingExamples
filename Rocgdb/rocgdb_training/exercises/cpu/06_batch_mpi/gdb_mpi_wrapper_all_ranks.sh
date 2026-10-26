#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# Per-rank (roc)gdb wrapper for MPI jobs - debug ALL ranks.
#
# Each rank writes its OWN gdb command file and its OWN log, so the ranks'
# output does not interleave (no race on stdout). Wrap your executable with
# this script under your MPI launcher:
#
#   <mpi-launcher> ... ./gdb_mpi_wrapper_all_ranks.sh ./my_app [app args ...]
#
# e.g.  srun -n 8 ./gdb_mpi_wrapper_all_ranks.sh ./my_app
#       mpirun -np 8 ./gdb_mpi_wrapper_all_ranks.sh ./my_app
#
# Debugger: rocgdb if it is on PATH, else gdb. Override with DBG=/path/to/dbg.
#
# Courtesy reference (independent prior art for this per-rank batch technique):
#   HLRS wiki "ROCgdb" - https://kb.hlrs.de/platforms/index.php/ROCgdb
#
# Inspect the per-rank gdb_out_<rank>.txt files afterwards (start at the end of
# each: the last dispatched kernel / most recent frames are there).
#
# For BIG jobs where the bug reproduces on every rank, prefer the companion
# script gdb_mpi_wrapper_rank0.sh, which debugs rank 0 only and lets the other
# ranks run normally - far less output, and it perturbs timing (which can
# trigger or mask races).

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

DBG="${DBG:-$(command -v rocgdb >/dev/null 2>&1 && echo rocgdb || echo gdb)}"
cmds="gdb_commands_${rank}.txt"

# Build this rank's command file. GPU-only commands are added only for rocgdb.
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
