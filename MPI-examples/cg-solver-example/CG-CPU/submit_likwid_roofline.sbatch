#!/bin/bash
#SBATCH -p PPAC_MI300A_SPX
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:15:00
#SBATCH -J cg_likwid_roofline
#SBATCH -o cg_likwid_roofline_%j.out
#
# Build a likwid roofline for the CG-CPU solver on ONE MI300A APU core.
#
# CEILINGS come from likwid-bench (timer-based, needs no PMU).
# APP POINT: likwid >= 5.5.2 recognizes the MI300A core (AMD K19 / Zen4) and
#   reads FLOPS_DP + L2/L3 natively via perf_event.  The MEM/MEM_DP (UMC uncore)
#   group still reads 0 on this node (no amd_umc/amd_df PMU, no /dev/cpu/*/msr),
#   so the L1 and DRAM byte traffic still come from cachegrind.  The numbers are
#   baked into render_likwid_roofline.py; the runs below regenerate them.
set -e
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load rocm/7.2.4 2>/dev/null || true
module load likwid 2>/dev/null || true      # 5.5.2 (MI300A-aware); highest version wins
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=^ucx
SEED=12345
MAT=src/Dubcova2.pm
OUT=roofline_meas; mkdir -p "$OUT"
flt() { grep -vE 'topology_rocm|hipGetDevice|Cannot open ROCm|Failed to obtaian|No HIP capabale'; }

echo "== likwid-bench ceilings (single core) =="
./measure_likwid_bench.sh

echo "== likwid-perfctr native application point =="
for G in FLOPS_DP L2 L3 MEM_DP; do
  echo "---- group=$G ----"
  timeout -s KILL 90 likwid-perfctr -C 0 -g "$G" -- ./cg_cpu "$MAT" "$SEED" 2>&1 \
    | flt | tee "$OUT/likwid_native_$G.txt" \
    | grep -iE 'iterations|solve time|RETIRED_SSE_AVX_FLOPS|Runtime \(RDTSC\)|DP \[MFLOP|bandwidth|data volume' || true
done

echo "== render =="
python3 render_likwid_roofline.py ../docs/profilers/figs/cg_likwid_roofline_cpu_n1.png
echo "Done."
