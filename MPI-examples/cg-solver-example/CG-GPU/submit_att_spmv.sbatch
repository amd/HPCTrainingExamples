#!/bin/bash
#SBATCH --job-name=cg_att_spmv
#SBATCH --partition=SH5_MI300A_SPX
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=att_spmv_%j.out
#SBATCH --error=att_spmv_%j.out
set -u
date; echo "Node(s): ${SLURM_JOB_NODELIST:-$(hostname)}"
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1

# Advanced Thread Trace (ATT) of the cg_gpu SpMV kernel on ONE compute unit of
# ONE rank, decoded to ISA-instruction granularity.  See
# docs/profilers/Advanced_Thread_Trace.md.
#
# ATT *collection* is built into rocprofv3; *decoding* needs the closed-source
# rocprof-trace-decoder library, which ships with rocm >= 7.12 only.  Run on an
# *SPX* MI300A node (gfx942) so the gfx9 ATT options apply.
module purge 2>/dev/null
module load "${ROCM_MODULE:-rocm/7.13.0}"
module load "${OPENMPI_MODULE:-openmpi/5.0.10-ucc1.6.0-ucx1.19.1-xpmem-2.7.4}"
export HSA_XNACK=1

# Decoder-library preflight: fail early with a clear message on rocm < 7.12.
DECODER="$ROCM_PATH/lib/librocprof-trace-decoder.so"
if [ ! -e "$DECODER" ]; then
  echo "FATAL: $DECODER not found -- ATT decode needs rocm >= 7.12 (7.12.0/7.13.0)." >&2
  echo "       Set ROCM_MODULE=rocm/7.13.0 and resubmit." >&2
  exit 1
fi
echo "[ok] decoder: $(readlink -f "$DECODER")"

echo "=== build ==="; make clean >/dev/null 2>&1
make 2>&1 | tail -2 || { echo BUILD FAILED; exit 1; }

METHOD="${METHOD:-rccl}"
MAT="${MAT:-src/Dubcova2.pm}"
OUT="${OUT:-att_spmv}"
KERNEL_REGEX="${KERNEL_REGEX:-csrmv}"
TARGET_CU="${TARGET_CU:-1}"
SIMD_SELECT="${SIMD_SELECT:-0xF}"
SE_MASK="${SE_MASK:-0x1}"
ACTIVITY="${ACTIVITY:-8}"          # AMD-recommended gfx9 sampling period
NKERNELS="${NKERNELS:-1}"
rm -rf "$OUT"

# cg_gpu is an MPI program; launch through `mpirun -n 1` (a bare srun/direct
# launch aborts in MPI_Init without PMI).  gpu_bind.sh pins the single rank to
# one GPU + NUMA node so the ATT target is unambiguous.
echo "=== ATT collect + decode (kernel=$KERNEL_REGEX cu=$TARGET_CU activity=$ACTIVITY) ==="
CG_SEED="${CG_SEED:-12345}" mpirun -n 1 --oversubscribe ./gpu_bind.sh \
  rocprofv3 --att \
    --att-library-path "$ROCM_PATH/lib" \
    --att-target-cu "$TARGET_CU" \
    --att-shader-engine-mask "$SE_MASK" \
    --att-simd-select "$SIMD_SELECT" \
    --att-activity "$ACTIVITY" \
    --att-consecutive-kernels "$NKERNELS" \
    --kernel-include-regex "$KERNEL_REGEX" \
    -d "$OUT" \
    -- ./cg_gpu "$MAT" "$METHOD"

echo "=== decoded ISA output ==="
# The ATT viewer data lands under ui_output_agent_*/ as JSON.  The decoded ISA
# with per-instruction Hit/Latency/Stall counts is in code.json (schema:
# ISA,_,LineNumber,Source,Codeobj,Vaddr,Hit,Latency,Stall,Idle).  The
# *_gfx942_code_object_id_*.out files are the raw ELF code objects the viewer
# disassembles -- not human-readable.  Summarize code.json into ISA_top.txt.
VIEW=$(find "$OUT" -maxdepth 2 -type d -name 'ui_output_agent_*' 2>/dev/null | head -1)
if [ -z "$VIEW" ] || [ ! -f "$VIEW/code.json" ]; then
  echo "WARNING: no ui_output_agent_*/code.json found under $OUT (decode may have failed)." >&2
  find "$OUT" -maxdepth 2 -type f | head -40
else
  echo "  viewer data: $VIEW"
  python3 - "$VIEW/code.json" "$OUT/ISA_top.txt" <<'PY'
import json,sys,re
from collections import defaultdict
code=json.load(open(sys.argv[1]))["code"]
kname=code[0][3] if code and len(code[0])>3 else "(unknown)"
def num(x):
    try:return int(x)
    except:return 0
ins=[r for r in code[1:] if isinstance(r[0],str) and r[0] and not r[0].startswith(';')]
tot=sum(num(r[7]) for r in ins) or 1
def cls(op):
    o=op.split()[0]
    if o.startswith('s_waitcnt'):return 'wait'
    if re.match(r'(global_|flat_|buffer_|scratch_)(load|store|atomic)',o):return 'vmem'
    if o.startswith('ds_'):return 'lds'
    if o.startswith('v_'):return 'valu'
    if o.startswith('s_'):return 'salu'
    return 'other'
by=defaultdict(lambda:[0,0])
for r in ins:
    c=cls(r[0]); by[c][0]+=num(r[7]); by[c][1]+=num(r[8])
with open(sys.argv[2],'w') as f:
    f.write("# ATT decoded ISA for the SpMV kernel (rocprofv3 --att)\n")
    f.write("# kernel: %s\n"%kname)
    f.write("# schema: Hit  Latency  Stall  <instruction>  (cycles; 1 CU)\n\n")
    f.write("## cycles by instruction class (%% of total latency)\n")
    for c,v in sorted(by.items(),key=lambda x:-x[1][0]):
        f.write("  %-6s Latency=%-9d Stall=%-9d (%.1f%%)\n"%(c,v[0],v[1],100*v[0]/tot))
    f.write("\n## top 12 instructions by latency (cycles)\n")
    for r in sorted(ins,key=lambda r:-num(r[7]))[:12]:
        f.write("  %-6d %-9d %-9d  %s\n"%(num(r[6]),num(r[7]),num(r[8]),r[0]))
print("wrote",sys.argv[2])
PY
  echo "--- $OUT/ISA_top.txt ---"; cat "$OUT/ISA_top.txt"
fi
echo "=== viewer folder(s) + rocpd db ==="
find "$OUT" -maxdepth 2 \( -type d -name 'ui_output_agent_*' -o -name '*_results.db' \) 2>/dev/null
date
echo "DONE -> $OUT/  (open ui_output_agent_* in the ROCm ATT viewer; ISA_top.txt for the text)"
