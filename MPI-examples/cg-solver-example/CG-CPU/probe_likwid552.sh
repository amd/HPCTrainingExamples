#!/bin/bash
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load rocm/7.2.4 2>/dev/null
module load likwid/5.5.2 2>/dev/null || module load likwid 2>/dev/null
source /etc/profile.d/lmod.sh 2>/dev/null
module load openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4 2>/dev/null || true
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=^ucx
export CG_SEED=12345
echo "== which/version =="; which likwid-perfctr; likwid-perfctr --version 2>&1 | head -1
echo "== CPU info =="; likwid-perfctr -i 2>&1 | grep -iE "CPU name|CPU type|CPU short|family|model" | head
echo "== FLOPS_DP on cg_cpu (rank0 pinned to core 0) =="
mpirun --oversubscribe -n 1 likwid-perfctr -f -C 0 -g FLOPS_DP ./cg_cpu src/Dubcova2.pm 12345 2>&1 \
  | grep -viE "HIP capab|pml framework|No components|installed|shared lib|Host:|Framework:|Sometimes|This typically|^-----" \
  | grep -iE "iterations|residual|solve|DP MFLOP|MFLOP|AVX|Runtime|Region|Metric|\|" | head -40
echo "== MEM_DP on cg_cpu =="
mpirun --oversubscribe -n 1 likwid-perfctr -f -C 0 -g MEM_DP ./cg_cpu src/Dubcova2.pm 12345 2>&1 \
  | grep -iE "Memory bandwidth|Memory data|MFLOP|Runtime|Operational intensity|\|" | head -40
