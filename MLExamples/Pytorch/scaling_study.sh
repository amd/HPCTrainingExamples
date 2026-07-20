#!/bin/bash -l
# Strong-scaling study of the three PyTorch examples on 1/2/4 GPUs.
# Strong scaling => the GLOBAL batch is fixed and split across GPUs
# (per-GPU batch = GLOBAL / N), so more GPUs share the same total work.
#
# Reports per run (RESULT line): throughput, peak_mem_mb, and rccl_s
# (uniform RCCL kernel device time/step from torch.profiler).
#
# fp32 throughout (no --amp / --mixed-precision) to avoid the known ROCm 7.2.x
# bf16 hipBLASLt transformer hang and to keep the comparison on one footing.
set -u
cd "$(dirname "$(readlink -fm "$0")")" || exit 1
ROOT="$PWD"

unset VIRTUAL_ENV PYTHONHOME
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v 'venv' | paste -sd:)
module purge >/dev/null 2>&1
module load rocm/${ROCM_VER:-7.2.3} openmpi pytorch/${PT_VER:-2.12.0} 2>/dev/null
export UPSTREAM_MINGPT="$HOME/pytorch_examples/distributed/minGPT-ddp/mingpt"
export UPSTREAM_FSDP="$HOME/pytorch_examples/distributed/FSDP2"
export MIOPEN_FIND_MODE=FAST HSA_XNACK=1 NCCL_DEBUG=WARN

python -c 'import torch;print("Torch",torch.__version__,"HIP",getattr(torch.version,"hip",None),"count",torch.cuda.device_count())' || exit 1
echo "GLOBAL batches: imagenet=${GB_IMG:-256} minGPT=${GB_GPT:-32} fsdp2=${GB_FSDP:-32}"
echo "======================================================================="

GB_IMG=${GB_IMG:-256}; GB_GPT=${GB_GPT:-32}; GB_FSDP=${GB_FSDP:-32}
GPUS="${GPUS:-1 2 4}"

run_one () {   # $1=label $2=dir $3=script $4=extra-args-with-{B}
  local label="$1" dir="$2" script="$3" argtmpl="$4"
  echo; echo "############################## $label ##############################"
  for N in $GPUS; do
    local perN
    case "$label" in
      imagenet) perN=$(( GB_IMG / N )); local args="${argtmpl//\{B\}/$perN}";;
      minGPT)   perN=$(( GB_GPT / N )); local args="${argtmpl//\{B\}/$perN}";;
      FSDP2)    perN=$(( GB_FSDP / N )); local args="${argtmpl//\{B\}/$perN}";;
    esac
    echo "---- $label  N=$N  per_gpu_batch=$perN ----"
    ( cd "$dir" && timeout 600 torchrun --standalone --nproc_per_node=$N $script $args \
        2>&1 | grep -aE "^RESULT|^# |Error|error|Traceback|CUDA|hang" ) \
      || echo "[warn] $label N=$N returned nonzero"
  done
}

run_one imagenet "$ROOT/imagenet/benchmarks"    ddp_resnet_bench.py \
        "-a resnet50 -b {B} --warmup 10 --iters 40 --rccl-time"

UPSTREAM="$UPSTREAM_MINGPT" run_one minGPT "$ROOT/minGPT-ddp" ddp_gpt_bench.py \
        "--n-layer 12 --n-embd 768 --n-head 12 --block-size 512 --batch-size {B} --warmup 10 --iters 40 --rccl-time"

UPSTREAM="$UPSTREAM_FSDP" run_one FSDP2 "$ROOT/FSDP2" fsdp2_bench.py \
        "--n-layers 16 --dim 1024 --n-heads 16 --seq-len 512 --batch-size {B} --warmup 5 --iters 20 --rccl-time"

echo; echo "======================= DONE ======================="
