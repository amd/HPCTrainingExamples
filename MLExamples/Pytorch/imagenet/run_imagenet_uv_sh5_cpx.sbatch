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
source .venv/bin/activate
module load rocm openmpi pytorch

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
