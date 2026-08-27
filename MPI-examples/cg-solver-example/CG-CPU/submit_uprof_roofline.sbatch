#!/bin/bash
#SBATCH -p PPAC_MI300A_SPX
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:15:00
#SBATCH -J cg_uprof_roofline
#SBATCH -o cg_uprof_roofline_%j.out
#
# Build an AMD uProf "Classic Roofline" point for the CG-CPU solver.
#
# NOTE on why this is not a bare `AMDuProfPcm roofline` run:
#   `AMDuProfPcm roofline` needs the data-fabric (DF/uncore) counters to
#   measure DRAM bandwidth.  On these MI300A nodes those counters require the
#   amd_uncore module / root (`--msr`), which participants do not have, and the
#   MI300A APU core (family 0x19, model 0x90) has no shipped RL_ config anyway.
#   So we measure the two roofline inputs with tools that DO work unprivileged:
#     * FP work  -> AMDuProfPcm -m fp   (retired SSE/AVX FLOPs)
#     * DRAM bytes -> valgrind cachegrind (deterministic last-level-D misses)
#   and plot with uProf's own AMDuProfModelling.py.
set -e
export PATH=$PATH:/nfsapps/ubuntu-24.04-nightlies/opt/AMDuProf_5.3-518/bin
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4 2>/dev/null || true
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

export CG_SEED=12345
# Force ob1 PML so single-rank MPI_Init succeeds inside the Slurm allocation.
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=^ucx
RUN="mpirun --oversubscribe -n 1 ./cg_cpu src/Dubcova2.pm 12345"
OUT=roofline_meas ; mkdir -p "$OUT"
make >/dev/null 2>&1 || true

echo "== 1. solve time =="
$RUN 2>/dev/null | tee "$OUT/prog.txt"
SOLVE=$(awk '/CG solve time:/{print $4}' "$OUT/prog.txt")

echo "== 2. uProf FP (retired SSE/AVX FLOPs, whole app) =="
AMDuProfPcm -m fp -a -C -o "$OUT/fp.csv" -- $RUN >/dev/null 2>&1
GFLOP=$(awk -F, '/Retired SSE\/AVX Flops/{print $2}' "$OUT/fp.csv")

echo "== 3. cachegrind DRAM bytes (last-level D misses x 64) =="
valgrind --tool=cachegrind --cache-sim=yes \
  --cachegrind-out-file="$OUT/cg.cg.out" ./cg_cpu src/Dubcova2.pm 12345 \
  2> "$OUT/cg.cg.log" >/dev/null
LLD=$(grep 'LLd misses:' "$OUT/cg.cg.log" | head -1 | sed -E 's/.*LLd misses:[[:space:]]*//' | awk '{print $1}' | tr -d ',')

# Derived roofline point (see comments above for the formulas).
python3 - "$GFLOP" "$SOLVE" "$LLD" "$OUT/cg_cpu_roofline.csv" <<'PY'
import sys
gflop, solve, lld = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
dram_bytes = lld * 64.0
tp  = gflop / solve                      # achieved GFLOP/s over the solve
bw  = dram_bytes / 1e9 / solve           # achieved DRAM GB/s over the solve
ai  = gflop*1e9 / dram_bytes             # FLOP / byte  (= tp/bw)
with open(sys.argv[4], "w") as f:
    f.write("AMDuProfPcm Report\n")
    for k,v in [("Family","0x19"),("Model","0x90"),("Sockets","1"),
                ("CoreComplex","1"),("Threads","192"),("Cores","96"),
                ("CpuFreqMHz","3700.45"),("MemoryChannels","1"),
                ("ModelName","AMD Instinct MI300A Accelerator"),
                ("Type","app"),("Name","cg_cpu"),
                ("Throughput","%.4f"%tp),("Bandwidth","%.4f"%bw)]:
        f.write("%s,%s\n"%(k,v))
print("FLOPs=%.3f GFLOP  solve=%.4f s  DRAM=%.2f MB  ->  %.2f GFLOPS, %.3f GB/s, AI=%.2f FLOP/B"
      % (gflop, solve, dram_bytes/1e6, tp, bw, ai))
PY

echo "== 4. render with uProf AMDuProfModelling.py =="
python3 render_cpu_roofline.py "$OUT/cg_cpu_roofline.csv" "$OUT" \
  ../docs/profilers/figs/cg_uprof_roofline_cpu_n1.png "cg_cpu (Dubcova2)"
echo "Done."
