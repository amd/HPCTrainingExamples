#!/bin/bash
#SBATCH --job-name=cg_tau
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --exclusive
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --time=00:20:00
#SBATCH --output=tau_%j.out
#SBATCH --error=tau_%j.out
set -u
date; echo "Node(s): ${SLURM_JOB_NODELIST:-$(hostname)}"
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
module purge 2>/dev/null
# TAU on AAC6 is built with -perfetto against rocprofiler-sdk; the tau module is
# published under the rocm/7.2.4 modulepath (see `man aac6_tau`).
module load "${ROCM_MODULE:-rocm/7.2.4}"
module load "${OPENMPI_MODULE:-openmpi}"
module load "${TAU_MODULE:-tau/dev}"
chmod +x gpu_bind.sh 2>/dev/null

# A plain build is enough: TAU intercepts MPI (MPI_Allreduce / MPI_Waitall /
# MPI_Isend) and, with `tau_exec -rocm`, the HIP API + GPU kernels + RCCL. No
# roctx instrumentation is needed -- TAU labels the MPI calls itself.
echo "=== build (plain) ==="; make clean >/dev/null 2>&1
make 2>&1 | tail -2 || { echo BUILD FAILED; exit 1; }

# TAU perfetto trace switches (see docs/profilers/tau.md):
#   TAU_TRACE=1               enable tracing (profile-only otherwise)
#   TAU_TRACE_FORMAT=perfetto emit a Perfetto protobuf (tau.rank_<n>.perfetto)
#                             instead of the classic TAU .trc/.edf files
#   TAU_PERFETTO_KEEP_FILES=1 keep the per-rank files after the merge so a single
#                             readable rank-0 trace can be opened on its own
export HSA_XNACK=1
export TAU_TRACE=1
export TAU_TRACE_FORMAT=perfetto
export TAU_PERFETTO_KEEP_FILES=1

OUT=tau_timeline; rm -rf "$OUT"; mkdir -p "$OUT"
SEED=12345
MAT=src/Dubcova2.pm

# One 4-rank trace per transport so the host-side comparison (§4 of the guide) is
# reproducible: isend (GPU-Aware MPI), staged (host-path), rccl (GPU-native).
# `-mca coll_ucc_enable 0` avoids a UCC collective abort seen under tau_exec's
# LD_PRELOAD interception; OpenMPI's built-in collectives are used instead.
for METHOD in ${METHODS:-isend staged rccl}; do
  echo "=== 4-rank $METHOD under tau_exec -rocm -> perfetto ==="
  D="$OUT/${METHOD}_run"; mkdir -p "$D"
  export TRACEDIR="$PWD/$D" PROFILEDIR="$PWD/$D"
  CG_SEED=$SEED mpirun -n 4 --oversubscribe -mca coll_ucc_enable 0 \
    ./gpu_bind.sh tau_exec -rocm ./cg_gpu "$MAT" "$METHOD" > "$D/run.log" 2>&1
  echo "rc=$?"; grep -E "solve time|comm total|halo exch|dot allreduce" "$D/run.log"
  # Keep the readable rank-0 trace; drop the other ranks + merged file to save space.
  if [ -f "$D/tau.rank_0.perfetto" ]; then
    cp "$D/tau.rank_0.perfetto" "$OUT/cg_tau_${METHOD}_n4.perfetto"
    echo "  cg_tau_${METHOD}_n4.perfetto ($(stat -c %s "$D/tau.rank_0.perfetto") bytes)"
  fi
  rm -f "$D"/tau.rank_[123].perfetto "$D"/tau.perfetto.gz 2>/dev/null
done

echo "=== labels captured (isend) ==="
strings "$OUT/cg_tau_isend_n4.perfetto" 2>/dev/null | \
  grep -oE 'MPI_(Allreduce|Waitall|Isend|Irecv)|csrmvn_general_kernel|ncclDevKernel_Generic_|hipMemcpy(Async)?|hipLaunchKernel' | \
  sort | uniq -c | sort -rn | head
date
echo "DONE -> $OUT/  (open cg_<method>_n4.perfetto in Perfetto; search MPI_Allreduce / MPI_Waitall / a kernel)"
