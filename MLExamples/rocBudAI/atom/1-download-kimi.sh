#!/usr/bin/env bash
# Download Kimi-K3 (~1.5 TB) into a shared HF cache via a SLURM batch job, so the
# login node is never hogged. Skip if reusing the shared cache. ./1-download-kimi.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
mkdir -p "${LOGDIR}"

if [[ -d "${HF_HOME}/hub/models--moonshotai--Kimi-K3/snapshots" ]]; then
    echo "Model already cached at ${HF_HOME} — nothing to do."
    exit 0
fi

echo "### downloader venv (huggingface_hub + hf_transfer) ###"
DLVENV="${BASE}/dl-venv"
if [[ ! -x "${DLVENV}/bin/hf" ]]; then
    python3 -m venv "${DLVENV}"
    "${DLVENV}/bin/pip" install -q --upgrade pip "huggingface_hub>=2.2" hf_transfer
fi

echo "### submit CPU-only download job ###"
SB="${BASE}/download-kimi-k3.sbatch"
cat > "${SB}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=dl-kimi-k3
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=${LOGDIR}/dl-kimi-k3-%j.log
set -euo pipefail
umask 022                                   # world-readable for reuse
export HF_HOME=${HF_HOME} HF_HUB_ENABLE_HF_TRANSFER=1
"${DLVENV}/bin/hf" download "${MODEL}" --max-workers 16   # resumable
chmod -R a+rX "${HF_HOME}" || true
EOF
sbatch "${SB}"
echo "Watch:  squeue -u \$USER   /   tail -f ${LOGDIR}/dl-kimi-k3-*.log   (~50 min)"
