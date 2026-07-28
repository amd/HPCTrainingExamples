#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --job-name=imagenet_roofline_capture
#SBATCH --output=capture_roofline-%j.out
#SBATCH --error=capture_roofline-%j.out
#
# Capture a REAL per-kernel roofline for README section 15 using the system
# roofline-extractor module (by Andrew Chisolm / Noah Wolfe).
#
# A roofline is a per-kernel / per-GPU compute analysis, so this profiles a
# SINGLE-GPU, short ResNet-50 --dummy run of main.py. roofline-extractor-profile
# wraps the command in rocprofv3 five times (4 counter passes + 1 kernel trace),
# post-processes, and runs rooflineExtractor to emit:
#   counters.html               -> interactive D3 roofline plot (screenshot for the README)
#   counters_EXTRACTED.csv      -> per-dispatch roofline metrics
#   counters_EXTRACTED_AGG.csv  -> aggregated (per-kernel) roofline metrics
#   profile_<timestamp>.log     -> full guided per-kernel text analysis
#
# All artifacts are persisted to profiling/roofline/ in the repo.
#
# Submit from the imagenet dir:  sbatch profiling/capture_roofline.sbatch

# Where to write results back to (the repo imagenet dir).
OUT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
PROFILING_DIR="${OUT_DIR}/profiling"
ROOFLINE_DIR="${PROFILING_DIR}/roofline"
mkdir -p "${ROOFLINE_DIR}"

echo "=== capture_roofline: OUT_DIR=${OUT_DIR} on $(hostname) ==="

# Disposable work dir in /tmp.
WORK_DIR=$(mktemp -d)
cd "${WORK_DIR}" || exit 1

# uv venv with system site packages (matches capture_torch.sbatch).
uv init imagenet_test >/dev/null 2>&1
cd imagenet_test || exit 1
uv venv --system-site-packages
source .venv/bin/activate

# Default rocm + pytorch (the tutorial's actual stack) plus the roofline module,
# which puts roofline-extractor-{profile,extract} on $PATH (prereq: rocm).
module load rocm openmpi pytorch roofline-extractor

# MIOpen: fast solver + a writable cache dir; warm it before the counted passes so
# autotuning is out of the rocprofv3 measurement window. Clean up at exit.
export MIOPEN_FIND_MODE=${MIOPEN_FIND_MODE:-FAST}
export MIOPEN_LOG_LEVEL=3
export MIOPEN_USER_DB_PATH="${WORK_DIR}/miopen"
mkdir -p "$MIOPEN_USER_DB_PATH"
trap 'rm -rf "${WORK_DIR}" "${MIOPEN_USER_DB_PATH}"' EXIT

# Upstream example.
git clone --depth=1 https://github.com/pytorch/examples.git ./pytorch_examples
cp pytorch_examples/imagenet/* .

# ---------------------------------------------------------------------------
# Instrument main.py: clean process-group teardown + short run (10 steps) so
# each of the five rocprofv3 passes finishes quickly.
# ---------------------------------------------------------------------------
sed -i '/world_size=args.world_size, rank=args.rank)/a\        import atexit as _ax, torch.distributed as _d; _ax.register(lambda: _d.destroy_process_group() if _d.is_initialized() else None)' main.py
sed -i '/^        data_time.update(time.time() - end)/a\
        if i >= 10: break' main.py

# Sanity: GPU visible?
python3 -c 'import torch; print("cuda?", torch.cuda.is_available(), "devs", torch.cuda.device_count())'
which roofline-extractor-profile || echo "WARN: roofline-extractor-profile not on PATH"

# Warm the MIOpen cache single-process so autotuning is cached before profiling.
HIP_VISIBLE_DEVICES=0 python -c "import torch,torchvision.models as M; \
   d=torch.device('cuda'); n=M.resnet50().to(d); c=torch.nn.CrossEntropyLoss().to(d); \
   x=torch.randn(256,3,224,224,device=d); y=torch.randint(0,1000,(256,),device=d); \
   [c(n(x),y).backward() for _ in range(3)]; torch.cuda.synchronize(); print('warm done')"

# ---------------------------------------------------------------------------
# Capture: automated roofline profiling on ONE GPU (non-distributed main.py).
# roofline-extractor-profile writes all outputs into ${ROOFLINE_DIR}.
# ---------------------------------------------------------------------------
echo "=== roofline-extractor-profile (1 GPU, ResNet-50 --dummy) ==="
HIP_VISIBLE_DEVICES=0 roofline-extractor-profile -o "${ROOFLINE_DIR}" --arch MI300A -- \
  python main.py -a resnet50 --dummy --gpu 0 -b 256 -p 10 --epochs 1 \
  |& tee "${OUT_DIR}/capture_run_roofline.log"

# ---------------------------------------------------------------------------
# Report what landed in profiling/roofline/.
# ---------------------------------------------------------------------------
echo "=== DONE ==="
echo "Roofline artifacts in ${ROOFLINE_DIR}:"
ls -la "${ROOFLINE_DIR}" 2>/dev/null
echo
echo "Roofline plot (screenshot for the README): ${ROOFLINE_DIR}/counters.html"
echo "Per-kernel aggregated metrics: ${ROOFLINE_DIR}/counters_EXTRACTED_AGG.csv"
echo "Guided per-kernel text analysis: ${ROOFLINE_DIR}/profile_*.log"
deactivate 2>/dev/null || true
