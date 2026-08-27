#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=PPAC_MI300A_SPX
#SBATCH --time=02:00:00
#SBATCH --job-name=imagenet_uv_spx_compute_opt
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
# Use pre-installed module versions to avoid downloading large wheels.
module load rocm openmpi pytorch
# Activate AFTER module load so .venv/bin is first on PATH (module load prepends its own python).
source .venv/bin/activate

# Get example and copy to current working directory
git clone --depth=1 https://github.com/pytorch/examples.git ./pytorch_examples
cp pytorch_examples/imagenet/* .

# Apply the source edits to main.py.
#   apply_basic_edits.sh lives in the repo dir ($BASE_DIR). It runs `sed -i` on
#   ./main.py in the CURRENT dir and references $BASE_DIR internally, so we copy
#   it here (we are in $WORK_DIR/imagenet_test, one level below where the earlier
#   `cp ${BASE_DIR}/*` landed it), export BASE_DIR, and run it.  The script has no
#   `set -e`, so previously a wrong-directory invocation failed silently and the
#   UNMODIFIED upstream main.py ran (no PEAK_MEM_MB, full 2503-iter epochs).
#   Verify the edits actually landed and abort loudly if they did not.
cp "${BASE_DIR}/apply_basic_edits.sh" .
chmod +x apply_basic_edits.sh
BASE_DIR="${BASE_DIR}" ./apply_basic_edits.sh
grep -q 'i >= 100' main.py && grep -q 'PEAK_MEM_MB' main.py || {
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

# Baseline (eager) run.  Keep its own log so it is not clobbered by the compiled run.
# Print every 10 steps (-p 10) so the steady-state median below has enough samples
# once the first (warmup) step is dropped.
HIP_VISIBLE_DEVICES=0 python main.py -a resnet50 --dummy --dist-url 'tcp://127.0.0.1:23456' \
	--dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 512  -p 10 --epochs 1 |& tee run_baseline.log

# Apply the torch compile compute optimizations
sed -i '/model = torch.nn.parallel.DistributedDataParallel(model, device_ids=\[args.gpu\])/a\                model = torch.compile(model)' main.py
grep -q 'model = torch.compile(model)' main.py || {
  echo "ERROR: torch.compile edit did not apply to main.py; aborting." >&2
  exit 1
}

# torch.compile run (own log for comparison against the baseline).
HIP_VISIBLE_DEVICES=0 python main.py -a resnet50 --dummy --dist-url 'tcp://127.0.0.1:23456' \
	--dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 -b 512  -p 10 --epochs 1 |& tee run_compile.log

# Compare eager vs torch.compile on a STEADY-STATE median, not the running average.
#   The upstream "Epoch:" line prints two numbers after "Time": the instantaneous
#   per-step time and a running average in parentheses.  The running average folds
#   in the very first step, which for torch.compile includes the one-time graph
#   compile (tens of seconds) -- so averaging over a short (100-iter) run makes
#   compile look slower than eager.  Instead we collect the *instantaneous* times,
#   drop the first (warmup / compile) sample, and report the median.  img/s is the
#   global batch (b) divided by that median step time.  Both runs are 1 GPU, b=512.
summarize () {  # $1=label  $2=logfile
  awk -v label="$1" -v b=512 '
    /^Epoch:/ {
      line=$0; gsub(/[()]/," ",line); nf=split(line,f," ");
      for (i=1;i<=nf;i++) if (f[i]=="Time") { t[++n]=f[i+1]; break }
    }
    /^PEAK_MEM_MB/ { m=$2 }
    END {
      if (n<=1) { printf "%-10s (no steady-state Time samples in log)\n", label; exit }
      # drop the first sample (iteration 1: warmup / one-time torch.compile build)
      k=0; for (i=2;i<=n;i++) s[++k]=t[i];
      asort(s);
      med = (k%2) ? s[(k+1)/2] : (s[k/2]+s[k/2+1])/2;
      printf "%-10s img/s=%.0f  step=%.4fs  peak_mem_mb=%s  (steady-state median of %d steps, warmup dropped)\n", \
             label, b/med, med, m, k;
    }
  ' "$2"
}
echo "=== compute optimization: eager vs torch.compile (1 GPU, b=512) ==="
summarize eager   run_baseline.log
summarize compile run_compile.log

deactivate
