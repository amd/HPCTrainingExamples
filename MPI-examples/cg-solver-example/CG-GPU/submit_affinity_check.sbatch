#!/bin/bash
#SBATCH --job-name=cg_affinity
#SBATCH --partition=1CN192C4G1H_MI300A_Ubuntu22
#SBATCH --exclusive
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --time=00:15:00
#SBATCH --output=cg_affinity_%j.out
#SBATCH --error=cg_affinity_%j.out

set -u
date
echo "Node(s): $SLURM_JOB_NODELIST"

module load rocm/6.4.1
module load openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4 2>/dev/null || module load openmpi
module list 2>&1

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
export OMPI_CXX=hipcc

echo "=============================================================="
echo " GPU topology (rocm-smi)"
echo "=============================================================="
rocm-smi --showtopo 2>/dev/null | head -60 || true
echo "Visible GPUs to the batch shell: ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"

echo "=============================================================="
echo " Build diagnostic"
echo "=============================================================="
mpicxx -O2 -std=c++17 check_affinity.cpp -o check_affinity \
       -I"$ROCM_PATH/include" -L"$ROCM_PATH/lib" -lamdhip64 || exit 1

RANKS=${SLURM_NTASKS:-4}

echo "=============================================================="
echo " Config A: mpirun --bind-to none --oversubscribe  (what test_cg_gpu.sh uses)"
echo "=============================================================="
mpirun --bind-to none --oversubscribe -n "$RANKS" ./check_affinity

echo "=============================================================="
echo " Config B: mpirun --map-by numa --bind-to core  (NUMA-local CPU binding)"
echo "=============================================================="
mpirun --map-by numa --bind-to core -n "$RANKS" ./check_affinity 2>&1 || \
   echo "(Config B failed -- allocation may not expose enough cores for core binding)"

echo "=============================================================="
echo " Config C: srun  (let SLURM place ranks / bind GPUs)"
echo "=============================================================="
srun -n "$RANKS" ./check_affinity 2>&1 || echo "(srun launch failed)"

echo "=============================================================="
echo " Config D: mpirun + gpu_bind.sh wrapper  (1 GPU + NUMA-local CPUs per rank)"
echo "=============================================================="
chmod +x gpu_bind.sh
mpirun --bind-to none -n "$RANKS" ./gpu_bind.sh ./check_affinity 2>&1

rm -f check_affinity
date
