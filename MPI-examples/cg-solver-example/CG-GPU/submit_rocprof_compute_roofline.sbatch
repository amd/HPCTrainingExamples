#!/bin/bash
#SBATCH --job-name=cg_rpc_roof
#SBATCH --partition=SH5_MI300A_SPX
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=rpc_roof_%j.out
#SBATCH --error=rpc_roof_%j.out
set -u
date; echo "Node(s): ${SLURM_JOB_NODELIST:-$(hostname)}"
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

# Standalone roofline of cg_gpu with rocprof-compute (--roof-only): it runs the
# roofline benchmark for the GPU's peaks, then a fast kernel pass, and emits
# empirRoof_*.pdf plots in the workload dir.  See docs/profilers/rocprof-compute.md.
# Run on an *SPX* MI300A node so the whole device is one GPU (1 rank == 1 GPU).
module purge 2>/dev/null
module load "${ROCM_MODULE:-rocm/7.2.4}"
export HSA_XNACK=1

echo "=== build ==="; make clean >/dev/null 2>&1
make 2>&1 | tail -2 || { echo BUILD FAILED; exit 1; }

METHOD="${METHOD:-rccl}"
MAT="${MAT:-src/Dubcova2.pm}"
SEED="${CG_SEED:-12345}"
NAME="${NAME:-cg_roof}"
rm -rf "workloads/$NAME"

# cg_gpu is an MPI program; launch through `mpirun -n 1` so MPI_Init succeeds
# (a bare srun/direct launch aborts without PMI).  rocprof-compute wraps the
# whole launch line after `--`.  --kernel-names labels the dots; -R FP32 FP64
# renders both single- and double-precision roofline PDFs.
rocprof-compute profile -n "$NAME" --roof-only --kernel-names \
  -R FP32 FP64 -- \
  mpirun -n 1 ./cg_gpu "$MAT" "$METHOD" "$SEED"

echo "=== roofline artifacts ==="
ls -la workloads/"$NAME"/MI*/ 2>/dev/null | grep -Ei 'roof|\.pdf|\.csv' | head -40
find workloads/"$NAME" -name 'empirRoof*.pdf' -o -name 'roofline*.csv' 2>/dev/null

date
echo "DONE -> workloads/$NAME/  (empirRoof_*.pdf = roofline plots)"
