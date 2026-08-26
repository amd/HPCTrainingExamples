#!/bin/bash
#SBATCH --job-name=sw-nov-3-profile
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=profile_%j.out

set -e
STAGE_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
SW_ROOT="$(cd "${STAGE_DIR}/../.." && pwd)"
source "${SW_ROOT}/env.sh"

make clean && make

rocprofv3 --pmc VALUBusy -T --output-format csv -d outdir -o shallow -- ./shallow

rocprofv3 --pmc OccupancyPercent -T --output-format csv -d outdir -o shallow -- ./shallow

if [ "${ROOFLINE_TOOL:-extractor}" = rocprof-compute ]; then
    rocprof-compute profile -n 3_block_32x32 --overwrite --roof-only --device 0 -k compute_rhs \
        --iteration-multiplexing -- ./shallow
    rocprof-compute analyze -p "${STAGE_DIR}/workloads/3_block_32x32/0"
else
    : "${ROOFLINE_EXTRACTOR:?Set ROOFLINE_EXTRACTOR in env.sh to your rooflineExtractor checkout}"

    module use /nfsapps/ubuntu-24.04/modules/base 2>/dev/null || true
    module load roofline-extractor/dev 2>/dev/null || true

    python3 "${ROOFLINE_EXTRACTOR}/profile_app.py" -o roofline_out --arch MI300A -- ./shallow
fi
