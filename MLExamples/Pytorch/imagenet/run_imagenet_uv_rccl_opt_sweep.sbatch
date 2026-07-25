#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --time=08:00:00
#SBATCH --job-name=imagenet_uv_spx_rccl_opt
#SBATCH --output=run_imagenet_uv_spx-%j.out
#SBATCH --error=run_imagenet_uv_spx-%j.out

# Save directory for calling scripts
BASE_DIR=`pwd`

# Create a work directory in /tmp and set trap to automatically cleanup
#   Can work in imagenet directory if working interactive
WORK_DIR=$(mktemp -d)
cd $WORK_DIR
trap 'rm -rf ${WORK_DIR}' EXIT

# Copy files over to temp directory
cp ${BASE_DIR}/* .

# Set up virtual environment to avod scattering python packages across system
#    and for more repeatability
uv init imagenet_test
cd imagenet_test
uv venv --system-site-packages
source .venv/bin/activate
# Use pre-installed module versions to avoid downloading large wheels.
module load rocm openmpi pytorch

# Get example and copy to current working directory
git clone --depth=1 https://github.com/pytorch/examples.git ./pytorch_examples
cp pytorch_examples/imagenet/* .

# Apply the source edits to main.py.
#   apply_basic_edits.sh lives in the repo dir ($BASE_DIR). It runs `sed -i` on
#   ./main.py in the CURRENT dir and references $BASE_DIR internally, so we copy
#   it here (we are in $WORK_DIR/imagenet_test, one level below where the earlier
#   `cp ${BASE_DIR}/*` landed it), export BASE_DIR, and run it.  The script has no
#   `set -e`, so previously a wrong-directory invocation failed silently and the
#   UNMODIFIED upstream main.py ran (no RCCL_TOTAL_MS, full 2503-iter epochs).
#   Verify the edits actually landed and abort loudly if they did not.
cp "${BASE_DIR}/apply_basic_edits.sh" .
chmod +x apply_basic_edits.sh
BASE_DIR="${BASE_DIR}" ./apply_basic_edits.sh
grep -q 'i >= 100' main.py && grep -q 'RCCL_TOTAL_MS' main.py || {
  echo "ERROR: apply_basic_edits.sh did not instrument main.py; aborting." >&2
  exit 1
}

# Full MIOpen startup can take many minutes. Setting fast mode to shorten the startup time
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
# Suppress some warning noise
export MIOPEN_LOG_LEVEL=3
export KINETO_LOG_LEVEL=3
# MIOPEN_USER_DB_PATH and MIOPEN_CUSTOM_CACHE_DIR are set in the PyTorch module to a /tmp directory
# Create the directory and set the automatic removal at end of job
mkdir -p "$MIOPEN_USER_DB_PATH"
trap 'rm -rf "$MIOPEN_USER_DB_PATH"' EXIT

# Check that torch is available and that GPUs are visible. Must be on a compute node
python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'

# Just a warm-up to get better performance numbers. Do a one GPU to avoid lock collisions on database. Full
# runs will just read from the database and should not have lock collisions in FAST mode.
HIP_VISIBLE_DEVICES=0 python -c "import torch,torchvision.models as M; \
   d=torch.device('cuda'); \
   n=M.resnet50().to(d); \
   c=torch.nn.CrossEntropyLoss().to(d); \
   x=torch.randn(256,3,224,224,device=d); \
   y=torch.randint(0,1000,(256,),device=d); \
   [c(n(x),y).backward() for _ in range(3)];  \
   torch.cuda.synchronize(); \
   print('warm done')"

# RCCL sweep (4 GPUs, b=512). Each factor is varied on its own, with the other
# NCCL knobs left at their defaults, by using a per-command env assignment (which
# scopes the variable to that single run only -- `export` leaked settings from one
# phase into the next).  Every run gets its own log so the final summary can grep
# RCCL_TOTAL_MS per configuration.

# Phase 1: NCCL algorithm.  Tree is latency-optimized (best once the all-reduce
# crosses APUs); Ring is bandwidth-optimized.
for ALGO in Tree Ring; do
  NCCL_ALGO=$ALGO \
  HIP_VISIBLE_DEVICES=0,1,2,3 python main.py -a resnet50 --dummy \
    --dist-url 'tcp://127.0.0.1:23456' --dist-backend nccl \
    --multiprocessing-distributed --world-size 1 --rank 0 -b 512 -p 20 --epochs 1 \
    |& tee run_algo_${ALGO}.log
done

# Phase 2: NCCL protocol.  LL/LL128 are the low-latency protocols (LL128 is the
# sweet spot for medium messages on high-bandwidth coherent links); Simple is the
# bandwidth-optimized fallback.
for PROTO in LL LL128 Simple; do
  NCCL_PROTO=$PROTO \
  HIP_VISIBLE_DEVICES=0,1,2,3 python main.py -a resnet50 --dummy \
    --dist-url 'tcp://127.0.0.1:23456' --dist-backend nccl \
    --multiprocessing-distributed --world-size 1 --rank 0 -b 512 -p 20 --epochs 1 \
    |& tee run_proto_${PROTO}.log
done

# Phase 3: NCCL channel count.
for NCH in 1 2 4 8 16 32; do
  NCCL_MIN_NCHANNELS=$NCH NCCL_MAX_NCHANNELS=$NCH \
  HIP_VISIBLE_DEVICES=0,1,2,3 python main.py -a resnet50 --dummy \
    --dist-url 'tcp://127.0.0.1:23456' --dist-backend nccl \
    --multiprocessing-distributed --world-size 1 --rank 0 -b 512 -p 20 --epochs 1 \
    |& tee run_nch_${NCH}.log
done

echo "=== RCCL total time by NCCL algorithm (protocol=default) ==="
for ALGO in Tree Ring; do
  printf "algo=%-8s " "$ALGO"; grep -h RCCL_TOTAL_MS run_algo_${ALGO}.log | tail -1
done

echo "=== RCCL total time by NCCL protocol (algorithm=default) ==="
for PROTO in LL LL128 Simple; do
  printf "proto=%-8s " "$PROTO"; grep -h RCCL_TOTAL_MS run_proto_${PROTO}.log | tail -1
done

echo "=== RCCL total time by channel count ==="
for NCH in 1 2 4 8 16 32; do
  printf "channels=%-3s " "$NCH"; grep -h RCCL_TOTAL_MS run_nch_${NCH}.log | tail -1
done

deactivate
