#!/bin/bash
#SBATCH --job-name=cg_roofline
#SBATCH --partition=SH5_MI300A_SPX
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=roofline_%j.out
#SBATCH --error=roofline_%j.out
set -u
date; echo "Node(s): ${SLURM_JOB_NODELIST:-$(hostname)}"
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

# Percent-of-peak roofline of cg_gpu with roofline-extractor (drives rocprofv3:
# 4 counter passes + 1 kernel-trace pass, then computes AI / roofline %).
# See docs/profilers/roofline-extractor.md.  Run on an *SPX* MI300A node so the
# GPU arch string matches a known entry (a CPX virtual GPU may not -- pin
# --arch MI300A regardless).
module purge 2>/dev/null
module load "${ROCM_MODULE:-rocm/7.2.4}"
module load "${OPENMPI_MODULE:-openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4}"
module load "${ROOFLINE_MODULE:-roofline-extractor/dev}"
export HSA_XNACK=1

echo "=== build ==="; make clean >/dev/null 2>&1
make 2>&1 | tail -2 || { echo BUILD FAILED; exit 1; }

METHOD="${METHOD:-rccl}"
MAT="${MAT:-src/Dubcova2.pm}"
OUT="${OUT:-rex_out}"
rm -rf "$OUT"

# cg_gpu is an MPI program; launch it through `mpirun -n 1` so MPI_Init succeeds
# (a bare srun/direct launch aborts without PMI).  roofline-extractor detects the
# launcher and runs rocprofv3 *inside* it, once per rank.
CG_SEED="${CG_SEED:-12345}" roofline-extractor-profile -o "$OUT" --arch MI300A -- \
  mpirun -n 1 ./cg_gpu "$MAT" "$METHOD"

echo "=== headline ==="
grep -E "percent of (linear|curved) roofline|Total application GPU time" "$OUT"/profile_*.log | tail -4
date
echo "DONE -> $OUT/  (open counters.html for the roofline plot; *_EXTRACTED*.csv for the numbers)"
