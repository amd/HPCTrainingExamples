#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --time=02:00:00
#SBATCH --job-name=imagenet_uv_spx
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

# Upstream main.py inits the NCCL process group but never destroys it, so PyTorch
# warns at exit ("destroy_process_group() was not called ... can leak resources").
# Register an atexit handler right after init_process_group so every worker
# (incl. mp.spawn children) cleans up on a normal exit.
sed -i '/world_size=args.world_size, rank=args.rank)/a\        import atexit as _ax, torch.distributed as _d; _ax.register(lambda: _d.destroy_process_group() if _d.is_initialized() else None)' main.py

# Print per-GPU peak memory once at the end of train() (matches README peak_mem_mb)
sed -i '/^def validate(/i\    torch.cuda.is_available() and getattr(args,"rank",0)<=0 and print(f"PEAK_MEM_MB {torch.cuda.max_memory_allocated()/1e6:.0f}")' main.py

# --- Demo instrumentation: total RCCL time + .to vs .migrate staging time ---
# The migrate path (STAGE=migrate) aliases the batch instead of copying it; it
# needs these and HSA_XNACK=1 (exported below).
export COMMON_DIR="$BASE_DIR/../common"
export HSA_XNACK=1

# 1) Start a profiler and set up staging counters at the top of train().
sed -i '/^    model.train()/a\
    import torch.profiler as _tp\
    _prof = _tp.profile(activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA]); _prof.start()\
    _stage_ms = 0.0; _stage_n = 0; _stager = None\
    if os.environ.get("STAGE") == "migrate":\
        import sys; sys.path.insert(0, os.environ["COMMON_DIR"]); from zerocopy import Stager\
        _stager = Stager(device, method="register")' main.py

# 2) Keep the demo (and the profiler trace) short.
sed -i '/^        data_time.update(time.time() - end)/a\
        if i >= 100: break' main.py

# 3) Time the host->device staging, but only when STAGE is set (copy vs migrate).
sed -i '/^        images = images.to(device, non_blocking=True)/c\
        if os.environ.get("STAGE"):\
            _e0 = torch.cuda.Event(enable_timing=True); _e1 = torch.cuda.Event(enable_timing=True)\
            _e0.record()\
            images = _stager.to_device(images) if _stager is not None else images.to(device, non_blocking=True)\
            _e1.record(); torch.cuda.synchronize()\
            _stage_ms += _e0.elapsed_time(_e1); _stage_n += 1\
        else:\
            images = images.to(device, non_blocking=True)' main.py

# 4) register-migrate needs pageable (non-pinned) host memory.
sed -i 's/pin_memory=True/pin_memory=False/g' main.py

# 5) Stop the profiler and print total RCCL kernel time (+ staging time if timed).
sed -i '/^def validate(/i\
    _prof.stop()\
    _rccl_ms = sum(e.self_device_time_total for e in _prof.key_averages() if "nccl" in e.key.lower())/1e3\
    _ws = getattr(args,"world_size","?"); _stg = os.environ.get("STAGE","")\
    getattr(args,"rank",0)<=0 and print(f"RCCL_TOTAL_MS {_rccl_ms:.3f} gpus={_ws}")\
    getattr(args,"rank",0)<=0 and _stage_n and print(f"STAGE_MS_PER_STEP {_stage_ms/_stage_n:.4f} gpus={_ws} stage={_stg}")' main.py

# Full MIOpen startup can take many minutes. Setting fast mode to shorten the startup time
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
# Suppress some warning noise
export MIOPEN_LOG_LEVEL=3
export KINETO_LOG_LEVEL=3
# MIOPEN_USER_DB_PATH and MIOPEN_CUSTOM_CACHE_DIR are set in the python module to a /tmp directory
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
# plain glob keeps the two reports separate. Lines are self-describing (gpus=N).
echo "=== RCCL total time (per GPU count) ==="
grep -h RCCL_TOTAL_MS run_*.log | sort -t= -k2 -n

echo "=== Host->device staging: .to (copy) vs .migrate ==="
grep -h STAGE_MS_PER_STEP stage_*.log

echo "=== Calculating the performance =="
$BASE_DIR/images_per_sec.sh

deactivate
