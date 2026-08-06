#!/bin/bash
### Simple step-by-step (manual sweep)

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

# Modifications to the example source code.
#
# All source edits live in apply_basic_edits.sh (single source of truth). We
# *source* it (not run it in a subshell) so its `export COMMON_DIR=...` and
# `export HSA_XNACK=1` propagate to the runs below, and so its `sed` commands
# operate on the main.py in the current work dir. It relies on $BASE_DIR (set
# above) to locate the shared helpers via $BASE_DIR/../common.

source "$BASE_DIR/apply_basic_edits.sh"

# Verify the edits actually landed and abort loudly if they did not (guards
# against a silently-unmodified upstream main.py running the full epoch).
grep -q 'i >= 100' main.py && grep -q 'PEAK_MEM_MB' main.py || {
  echo "ERROR: apply_basic_edits.sh did not instrument main.py; aborting." >&2
  exit 1
}

# Full MIOpen startup can take many minutes. Setting fast mode to shorten the startup time
export MIOPEN_FIND_MODE=FAST
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

# Run the three jobs for the scaling study on 1, 2, and 4 GPUs. The number of GPUS is determined by HIP_VISIBLE_DEVICES
HIP_VISIBLE_DEVICES=0       python main.py -a resnet50 --dummy --dist-url 'tcp://127.0.0.1:23456' \
	--dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 128  -p 20 --epochs 1 |& tee run_1.log
HIP_VISIBLE_DEVICES=0,1     python main.py -a resnet50 --dummy --dist-url 'tcp://127.0.0.1:23456' \
	--dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 256  -p 20 --epochs 1 |& tee run_2.log
HIP_VISIBLE_DEVICES=0,1,2,3 python main.py -a resnet50 --dummy --dist-url 'tcp://127.0.0.1:23456' \
	--dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 512  -p 20 --epochs 1 |& tee run_4.log

# The MI300A APU has a unified memory and does not need to copy the data, just the pointer. Other GPUS can emulate APU behavior
#   Requires HSA_XNACK 1 to be set. Set earlier in script
#   .to (copy) vs .migrate staging comparison (4 GPUs): compare STAGE_MS_PER_STEP in final report
STAGE=copy    HIP_VISIBLE_DEVICES=0,1,2,3 python main.py -a resnet50 --dummy --dist-url 'tcp://127.0.0.1:23456' \
	--dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 128  -p 20 --epochs 1 |& tee stage_copy.log
STAGE=migrate HIP_VISIBLE_DEVICES=0,1,2,3 python main.py -a resnet50 --dummy --dist-url 'tcp://127.0.0.1:23456' \
	--dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 128  -p 20 --epochs 1 |& tee stage_migrate.log

# Scaling runs are run_<N>.log; staging runs are stage_{copy,migrate}.log, so a
# grep keeps the two reports separate. Lines are self-describing (gpus=N).
echo "=== RCCL total time (per GPU count) ==="
grep -h RCCL_TOTAL_MS run_*.log | sort -t= -k2 -n

echo "=== Host->device staging: .to (copy) vs .migrate ==="
grep -h STAGE_MS_PER_STEP stage_*.log

echo "=== Calculating the performance =="
$BASE_DIR/images_per_sec.sh

deactivate
