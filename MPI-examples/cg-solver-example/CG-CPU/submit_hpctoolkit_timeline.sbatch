#!/bin/bash
#SBATCH --job-name=cg_hpctoolkit
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --ntasks=4
#SBATCH --time=00:30:00
#SBATCH --output=hpctoolkit_%j.out
#SBATCH --error=hpctoolkit_%j.out
set -u
date; echo "Node(s): ${SLURM_JOB_NODELIST:-$(hostname)}"
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

# HPCToolkit call-path sampling of the CPU CG solver.  Two measurements from one
# build (see docs/profilers/hpctoolkit.md):
#   * REALTIME  -> wall-clock timeline (fills the trace view, incl. MPI waits)
#   * CPUTIME   -> clean on-CPU call-path profile (which routine costs the CPU)
# hpcrun needs libpapi.so.7.1, which is present only on the *compute* node -- so
# this must run inside an allocation (this batch job), not on the login node.
module purge 2>/dev/null
module load "${ROCM_MODULE:-rocm/7.2.4}"
module load "${OPENMPI_MODULE:-openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4}"
module load "${HPCTOOLKIT_MODULE:-hpctoolkit/2025.1.2}"

echo "=== build (plain -g -O3) ==="
make clean >/dev/null 2>&1; make 2>&1 | tail -2 || { echo BUILD FAILED; exit 1; }

SEED=12345
# A larger matrix -> a longer, many-iteration solve that fills the timeline. The
# 3D-Poisson matrix ships with the GPU example (generate it with gen_poisson.py).
MAT="${MAT:-../CG-GPU/src/poisson80.pm}"
[ -f "$MAT" ] || MAT=src/Dubcova2.pm
# `--mca pml ob1 --mca btl vader,self,tcp` uses OpenMPI's shared-memory path
# instead of UCX, which keeps the trace to the 4 rank threads + one MPI progress
# thread each (UCX adds several ucs_* helper threads that clutter the timeline).
MPIRUN=(mpirun -n 4 --oversubscribe --mca pml ob1 --mca btl vader,self,tcp)

# hpcstruct on the compute node so its runtime libs are present.  hpcprof's
# `-R <build>=<build>/.` silences the "struct file partial match" remap warning.
BUILD="$PWD"

echo "=== 1) REALTIME timeline (trace view) ==="
rm -rf cg_cpu_rt.m cg_cpu_rt.d
CG_SEED=$SEED "${MPIRUN[@]}" hpcrun -e REALTIME@200 -t -o cg_cpu_rt.m ./cg_cpu "$MAT" 2>&1 | tail -3
hpcstruct ./cg_cpu >/dev/null 2>&1
hpcprof -S cg_cpu.hpcstruct -R "$BUILD"="$BUILD/." -o cg_cpu_rt.d cg_cpu_rt.m 2>&1 | grep -iE 'error' || true
echo "  -> cg_cpu_rt.d/trace.db = $(stat -c %s cg_cpu_rt.d/trace.db 2>/dev/null) bytes"

echo "=== 2) CPUTIME profile (call-path breakdown) ==="
rm -rf cg_cpu.m cg_cpu.d
CG_SEED=$SEED "${MPIRUN[@]}" hpcrun -e CPUTIME@500 -t -o cg_cpu.m ./cg_cpu "$MAT" 2>&1 | tail -3
hpcstruct ./cg_cpu >/dev/null 2>&1
hpcprof -S cg_cpu.hpcstruct -R "$BUILD"="$BUILD/." -o cg_cpu.d cg_cpu.m 2>&1 | grep -iE 'error' || true
echo "  -> cg_cpu.d/{meta,profile,trace}.db built"

date
echo "DONE.  Open in hpcviewer:"
echo "    hpcviewer cg_cpu_rt.d     # Trace tab -> per-rank timeline"
echo "    hpcviewer cg_cpu.d        # Bottom-up view -> CPU call-path breakdown"
echo "(headless PNGs: ./capture_hpcviewer.sh <db> out.png -- see the guide)"
