#!/usr/bin/env bash
# Download Muse-Spark-Open (~510 GB MXFP4) + the DSpark draft into a shared HF
# cache via a SLURM batch job, so the login node is never hogged. Skip if reusing
# the shared cache. ./1-download-muse.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
mkdir -p "${LOGDIR}"

_model_dir="${HF_HOME}/hub/models--$(echo "${MODEL_HF}" | sed 's#/#--#g')"
# Skip only if the snapshot exists AND no partial (*.incomplete) blobs remain: a
# snapshots/ dir alone appears after the FIRST file, so checking it by itself
# would treat an interrupted download as complete. `hf download` is resumable, so
# if anything is missing we just re-run the job and it picks up where it stopped.
_incomplete="$(find "${_model_dir}/blobs" -name '*.incomplete' -print -quit 2>/dev/null || true)"
if [[ -d "${_model_dir}/snapshots" && -z "${_incomplete}" ]]; then
    echo "Model already cached (complete) at ${HF_HOME} — nothing to do."
    exit 0
fi

echo "### downloader venv (huggingface_hub + hf_transfer) ###"
DLVENV="${BASE}/dl-venv"
if [[ ! -x "${DLVENV}/bin/hf" ]]; then
    python3 -m venv "${DLVENV}"
    "${DLVENV}/bin/pip" install -q --upgrade pip "huggingface_hub>=2.2" hf_transfer
fi

echo "### submit CPU-only download job (checkpoint + DSpark draft) ###"
SB="${BASE}/download-muse.sbatch"
cat > "${SB}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=dl-muse-spark
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=${LOGDIR}/dl-muse-spark-%j.log
set -euo pipefail
umask 022                                   # world-readable for reuse
# These checkpoints are served via HF Xet; HF_XET_HIGH_PERFORMANCE enables the
# fast Xet path (the older HF_HUB_ENABLE_HF_TRANSFER is now a no-op for Xet repos
# but kept for non-Xet files / older hub versions).
export HF_HOME=${HF_HOME} HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1
"${DLVENV}/bin/hf" download "${MODEL_HF}" --max-workers 16      # resumable
"${DLVENV}/bin/hf" download "${DSPARK_MODEL}" --max-workers 16  # DSpark draft
chmod -R a+rX "${HF_HOME}" || true
EOF
sbatch "${SB}"
echo "Watch:  squeue -u \$USER   /   tail -f ${LOGDIR}/dl-muse-spark-*.log"
