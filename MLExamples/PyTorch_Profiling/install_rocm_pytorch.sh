#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# install_rocm_pytorch.sh  --  run this ON A LOGIN NODE (one with internet).
#
# Builds a self-contained nightly ROCm + PyTorch venv under VENV_BASE, exposes
# it as an Lmod module, and pre-stages everything the compute-node test jobs
# need (many clusters give compute nodes no direct internet):
#   1. venv + pip install nightly rocm/torch/torchvision/transformers
#   2. rocm-sdk init (extract ROCm SDK)
#   3. generate the Lmod modulefile with resolved absolute paths
#   4. clone amd/HPCTrainingExamples and overlay these scripts
#   5. pre-stage rooflineExtractor + its pip deps
#   6. pre-create the rocprof-compute analyze venv (numpy 1.26.4)
#   7. pre-download the CIFAR-100 dataset
#
# Idempotent: re-running reuses existing pieces. All site-specific settings come
# from PyTorch_Profiling/local.env (copy local.env.example and edit it).
# ---------------------------------------------------------------------------
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Works both from inside MLExamples/PyTorch_Profiling and from a separate
# working copy that keeps the scripts in a PyTorch_Profiling/ subdirectory.
if [[ -d "${SELF_DIR}/PyTorch_Profiling" ]]; then
    OVERLAY_DIR="${SELF_DIR}/PyTorch_Profiling"
else
    OVERLAY_DIR="${SELF_DIR}"
fi
# shellcheck source=PyTorch_Profiling/env.sh
source "${OVERLAY_DIR}/env.sh"

for _required in GPU_ARCH ROCM_VERSION VENV_BASE; do
    if [[ -z "${!_required}" ]]; then
        echo "ERROR: ${_required} is not set. Configure it in ${SITE_ENV}." >&2
        exit 1
    fi
done

echo "=========================================================================="
echo " ROCm nightly build"
echo "   PROJECT       = ${PROJECT:-<cluster default>}"
echo "   VENV          = ${VENV}"
echo "   GPU_ARCH      = ${GPU_ARCH}"
echo "   ROCM_VERSION  = ${ROCM_VERSION}"
echo "   MODULE        = ${MODULE_NAME}/${MODULE_VERSION}"
echo "   MODULEROOT    = ${MODULEROOT}"
echo "   EXAMPLES_TOP  = ${EXAMPLES_TOP}"
echo "   host          = $(hostname)"
echo "=========================================================================="

# Only takes effect on sites that define PROXY.
export_proxy

# --- 0. base python to create the venv ------------------------------------
# PYTHON_MODULE lets Cray-style sites pull in a newer interpreter (cray-python);
# BASE_PYTHON overrides the choice outright. Otherwise system python3 is used.
ensure_module_cmd
if [[ -n "${PYTHON_MODULE}" ]]; then
    module load "${PYTHON_MODULE}" 2>/dev/null \
        || echo "WARN: could not 'module load ${PYTHON_MODULE}'; using system python3"
fi
BASE_PY="${BASE_PYTHON:-$(command -v python3)}"
echo "Base python: ${BASE_PY} ($(${BASE_PY} --version 2>&1))"

# --- 1. venv + nightly ROCm/PyTorch ---------------------------------------
mkdir -p "$(dirname "${VENV}")"
if [[ ! -x "${VENV}/bin/python3" ]]; then
    echo "Creating venv at ${VENV}"
    "${BASE_PY}" -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python3 -m pip install --upgrade pip

echo "Installing nightly ROCm + PyTorch (${ROCM_VERSION}, ${GPU_ARCH}) ..."
python3 -m pip install \
    --index-url "${ROCM_INDEX_URL}" \
    "rocm[profiler,devel,libraries,device-${GPU_ARCH}]==${ROCM_VERSION}" \
    "torch[device-${GPU_ARCH}]" \
    "torchvision[device-${GPU_ARCH}]"

# transformers is required by train_cifar_100.py.
python3 -m pip install transformers

# --- 2. extract the ROCm SDK ----------------------------------------------
echo "Running rocm-sdk init ..."
rocm-sdk init

# Resolve the extracted SDK path (python minor version independent).
SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
DEVEL="${SITE_PACKAGES}/_rocm_sdk_devel"
if [[ ! -d "${DEVEL}" ]]; then
    echo "ERROR: expected ROCm SDK not found at ${DEVEL}" >&2
    exit 1
fi
echo "ROCm SDK: ${DEVEL}"

# --- 3. generate the Lmod modulefile --------------------------------------
MODDIR="${MODULEROOT}/${MODULE_NAME}"
mkdir -p "${MODDIR}"
MODFILE="${MODDIR}/${MODULE_VERSION}.lua"
sed -e "s#@MODULE_NAME@#${MODULE_NAME}#g" \
    -e "s#@VERSION@#${MODULE_VERSION}#g" \
    -e "s#@GPU_ARCH@#${GPU_ARCH}#g" \
    -e "s#@VENV@#${VENV}#g" \
    -e "s#@DEVEL@#${DEVEL}#g" \
    "${OVERLAY_DIR}/modulefiles/${MODULE_NAME}/TEMPLATE.lua" > "${MODFILE}"
echo "Wrote modulefile: ${MODFILE}"
echo "  ( module use ${MODULEROOT} && module load ${MODULE_NAME}/${MODULE_VERSION} )"

# --- 4. make sure the examples tree is in place ---------------------------
if [[ "${OVERLAY_DIR}" == "${EXAMPLES_TOP}" ]]; then
    echo "Running inside the examples tree (${EXAMPLES_TOP}); no clone needed."
else
    if [[ ! -d "${EXAMPLES_REPO}/.git" ]]; then
        echo "Cloning amd/HPCTrainingExamples -> ${EXAMPLES_REPO}"
        git clone https://github.com/amd/HPCTrainingExamples.git "${EXAMPLES_REPO}"
    else
        echo "Examples repo present; pulling latest"
        git -C "${EXAMPLES_REPO}" pull --ff-only || echo "WARN: git pull skipped"
    fi
    echo "Overlaying scripts into ${EXAMPLES_TOP}"
    mkdir -p "${EXAMPLES_TOP}"
    cp -rv "${OVERLAY_DIR}/." "${EXAMPLES_TOP}/"
    chmod +x "${EXAMPLES_TOP}"/*/*.sh "${EXAMPLES_TOP}"/setup_rocm.sh 2>/dev/null || true

    # The job scripts source env.sh from the deployed tree, where it resolves
    # local.env by default. Install the site config actually used for this build
    # under that name so jobs cannot pick up another cluster's settings.
    if [[ -f "${SITE_ENV}" ]]; then
        cp -v "${SITE_ENV}" "${EXAMPLES_TOP}/local.env"
    fi
fi

# --- 5. pre-stage rooflineExtractor + deps --------------------------------
RE_DIR="${EXAMPLES_TOP}/roofline-extractor/rooflineExtractor"
if [[ ! -d "${RE_DIR}" ]]; then
    echo "Cloning rooflineExtractor -> ${RE_DIR}"
    git clone https://github.com/AMD-HPC/rooflineExtractor.git "${RE_DIR}"
fi
echo "Installing rooflineExtractor requirements into the shared venv"
python3 -m pip install -r "${RE_DIR}/requirements.txt"

# --- 6. pre-create the rocprof-compute analyze venv (numpy 1.26.4) --------
ROCM_PROFILER="${SITE_PACKAGES}/_rocm_profiler"
REQ_FILE="${ROCM_PROFILER}/libexec/rocprofiler-compute/requirements.txt"
if [[ -f "${REQ_FILE}" ]]; then
    if [[ ! -x "${ANALYZE_VENV}/bin/python3" ]]; then
        echo "Creating analyze venv at ${ANALYZE_VENV}"
        "${VENV}/bin/python3" -m venv "${ANALYZE_VENV}"
    fi
    "${ANALYZE_VENV}/bin/python3" -m pip install --upgrade pip
    "${ANALYZE_VENV}/bin/python3" -m pip install -r "${REQ_FILE}"
    cp "${REQ_FILE}" "${ANALYZE_VENV}/.rocprof_compute_reqs_installed"
    echo "Analyze venv ready: ${ANALYZE_VENV}"
else
    echo "WARN: rocprof-compute requirements.txt not found at ${REQ_FILE}; skipping analyze venv"
fi

# --- 7. pre-download the CIFAR-100 dataset --------------------------------
# Use download_only_nogpus.py: unlike train_cifar_100.py --download-only, it does
# NOT call dist.init_process_group("nccl", ...), which would hang on a login node
# that has no GPU. This runs purely on torchvision on the login node.
echo "Pre-downloading CIFAR-100 dataset"
python3 "${EXAMPLES_TOP}/download_only_nogpus.py" \
    --data-path "${EXAMPLES_TOP}/data"

echo "=========================================================================="
echo " Build complete."
echo "   Load with : module use ${MODULEROOT} && module load ${MODULE_NAME}/${MODULE_VERSION}"
echo "   Tests in  : ${EXAMPLES_TOP}"
echo "   Submit    : bash ${SELF_DIR}/run_all.sh"
echo "=========================================================================="
