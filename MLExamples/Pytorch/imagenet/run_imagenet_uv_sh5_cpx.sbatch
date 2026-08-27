#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=6
#SBATCH --partition=SH5_MI300A_CPX
#SBATCH --time=02:00:00
#SBATCH --job-name=imagenet_uv_sh5_cpx
#SBATCH --output=run_imagenet_uv_sh5_cpx-%j.out
#SBATCH --error=run_imagenet_uv_sh5_cpx-%j.out
#
# RCCL weak-scaling sweep on ONE MI300A in CPX mode. In CPX each of the APU's
# 6 XCDs is exposed as its own HIP device, so this node presents 6 "virtual GPUs"
# backed by a single physical MI300A. All 6 ranks all-reduce over the on-package
# Infinity Fabric, so RCCL cost stays very small here (contrast with the 24-GPU
# PPAC_MI300A_CPX sweep, which crosses physical APUs).
#
# Notes vs the SPX (full-APU) sweep:
#  - GPU_LIST goes up to 6 (one XCD .. the whole chip), not 4.
#  - PERGPU_BATCH is smaller: a CPX partition has ~1/6 the memory/compute of a
#    full APU, so the large SPX per-GPU batch of 128 can OOM or run very slowly.
#    Tune PERGPU_BATCH up until peak_mem_mb approaches the partition limit.

GPU_LIST="${GPU_LIST:-1 2 4 6}"
PERGPU_BATCH="${PERGPU_BATCH:-32}"

if [ -d imagenet_test ]; then
   rm -rf imagenet_test
fi

BASE_DIR=`pwd`
WORK_DIR=$(mktemp -d)
cd $WORK_DIR
trap 'rm -rf ${WORK_DIR}' EXIT
cp ${BASE_DIR}/* .

uv init imagenet_test
cd imagenet_test
uv venv --system-site-packages
module load rocm openmpi pytorch
# Activate AFTER module load so .venv/bin is first on PATH (module load prepends its own python).
source .venv/bin/activate

git clone --depth=1 https://github.com/pytorch/examples.git ./pytorch_examples
cp pytorch_examples/imagenet/* .

# Modifications to the example source code -- single source of truth in
# apply_basic_edits.sh (sourced so its COMMON_DIR/HSA_XNACK exports propagate to
# the runs below, and its sed commands operate on main.py in the current work
# dir). Relies on $BASE_DIR (set above) to locate helpers via $BASE_DIR/../common.
source "$BASE_DIR/apply_basic_edits.sh"

# Verify the edits actually landed and abort loudly if they did not (guards
# against a silently-unmodified upstream main.py running the full epoch).
grep -q 'i >= 100' main.py && grep -q 'PEAK_MEM_MB' main.py || {
  echo "ERROR: apply_basic_edits.sh did not instrument main.py; aborting." >&2
  exit 1
}

export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
export MIOPEN_LOG_LEVEL=3
export KINETO_LOG_LEVEL=3
mkdir -p "$MIOPEN_USER_DB_PATH"
trap 'rm -rf "$MIOPEN_USER_DB_PATH"' EXIT

python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'

# Warm the MIOpen cache once on a single partition, at the per-GPU batch the
# sweep actually uses (MIOpen caches per input shape).
HIP_VISIBLE_DEVICES=0 python -c "import torch,torchvision.models as M; \
   d=torch.device('cuda'); \
   n=M.resnet50().to(d); \
   c=torch.nn.CrossEntropyLoss().to(d); \
   x=torch.randn($PERGPU_BATCH,3,224,224,device=d); \
   y=torch.randint(0,1000,($PERGPU_BATCH,),device=d); \
   [c(n(x),y).backward() for _ in range(3)];  \
   torch.cuda.synchronize(); \
   print('warm done')"

# Weak-scaling sweep: constant per-GPU batch, global batch = N * PERGPU_BATCH.
# main.py's --multiprocessing-distributed divides -b (global) across the N ranks.
for N in $GPU_LIST; do
  DEV=$(seq -s, 0 $((N - 1)))
  GLOBAL=$((N * PERGPU_BATCH))
  echo "=== $N GPU(s) [$DEV]  global_batch=$GLOBAL  per_gpu=$PERGPU_BATCH ==="
  HIP_VISIBLE_DEVICES=$DEV python main.py -a resnet50 --dummy --dist-url 'tcp://127.0.0.1:23456' \
	--dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 \
	-b $GLOBAL -p 20 --epochs 1 |& tee run_${N}.log
done

# .to (copy) vs migrate staging comparison on a SINGLE partition: compare STAGE_MS_PER_STEP
STAGE=copy    HIP_VISIBLE_DEVICES=0 python main.py -a resnet50 --dummy --dist-url 'tcp://127.0.0.1:23456' \
	--dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b $PERGPU_BATCH -p 20 --epochs 1 |& tee stage_copy.log
STAGE=migrate HIP_VISIBLE_DEVICES=0 python main.py -a resnet50 --dummy --dist-url 'tcp://127.0.0.1:23456' \
	--dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b $PERGPU_BATCH -p 20 --epochs 1 |& tee stage_migrate.log

# Scaling runs are run_<N>.log; staging runs are stage_{copy,migrate}.log, so a
# plain glob keeps the two reports separate. Lines are self-describing (gpus=N).
echo "=== RCCL total time (per GPU count) ==="
grep -h RCCL_TOTAL_MS run_*.log | sort -t= -k2 -n

echo "=== Host->device staging: .to (copy) vs .migrate ==="
grep -h STAGE_MS_PER_STEP stage_*.log

echo "=== Calculating the performance =="
PERGPU_BATCH=$PERGPU_BATCH $BASE_DIR/images_per_sec.sh

deactivate
