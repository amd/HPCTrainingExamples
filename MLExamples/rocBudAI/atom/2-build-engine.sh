#!/usr/bin/env bash
# Install ATOM + a newer AITER into layer dirs on top of the vllm module.
# No container, no source build. ./2-build-engine.sh
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
mkdir -p "${BASE}" "${UV_CACHE_DIR}"

# Reproduction builds into YOUR BASE (not the shared prebuilt defaults).
ATOM_PYLIB="${BASE}/atom-pylib"
AITER_DIR="${BASE}/aiter-wheel"

# One installer, uv if present else pip. Extra args (e.g. --no-deps) pass through.
pyinstall() {  # pyinstall TARGET PKG... [--flags]
    local target="$1"; shift
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$(command -v python3)" --target "$target" "$@"
    else
        python3 -m pip install --no-cache-dir --target "$target" "$@"
    fi
}

echo "### ATOM (pure-python, pinned ${ATOM_COMMIT}) ###"
ATOM_SRC="${BASE}/ATOM"
if [[ ! -d "${ATOM_SRC}/.git" ]]; then
    git init -q "${ATOM_SRC}"
    git -C "${ATOM_SRC}" remote add origin "${ATOM_REPO}"
fi
# Fetch just the pinned commit (GitHub allows fetch-by-SHA); fall back to a full
# fetch if the server refuses a bare SHA, then check it out.
git -C "${ATOM_SRC}" fetch --depth 1 origin "${ATOM_COMMIT}" 2>/dev/null \
    || git -C "${ATOM_SRC}" fetch origin
git -C "${ATOM_SRC}" checkout -q "${ATOM_COMMIT}"

# Apply recipe patches (Triton-ABI fixes against this ATOM commit). Idempotent:
# a patch already applied (reverse-applies cleanly) is skipped, so re-runs work.
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)/patches"
if [[ -d "${PATCH_DIR}" ]]; then
    for p in "${PATCH_DIR}"/*.patch; do
        [[ -e "$p" ]] || continue
        if git -C "${ATOM_SRC}" apply --reverse --check "$p" 2>/dev/null; then
            echo "-- patch already applied: $(basename "$p")"
        else
            git -C "${ATOM_SRC}" apply "$p" && echo "-- patch applied: $(basename "$p")"
        fi
    done
fi
pyinstall "${ATOM_PYLIB}" "${ATOM_SRC}" einops

echo "### resolve AITER release wheel (${AITER_RELEASE_TAG}, ${AITER_ROCM}, ${PYTAG}) ###"
WHEEL_URL="$(curl -fsSL "https://api.github.com/repos/ROCm/aiter/releases/tags/${AITER_RELEASE_TAG}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(next(a['browser_download_url'] for a in d['assets'] if '${AITER_ROCM}' in a['name'] and '${PYTAG}' in a['name']))")"
echo "-> $(basename "${WHEEL_URL}")"

echo "### download (~600 MB) + install AITER into its own layer ###"
WHL="${BASE}/aiter.whl"; [[ -f "${WHL}" ]] || curl -fsSL "${WHEEL_URL}" -o "${WHL}"
pyinstall "${AITER_DIR}" --no-deps "${WHL}"

grep -q topk_select "${AITER_DIR}/aiter/__init__.py" \
    && echo "OK: topk_select present ($(ls "${AITER_DIR}"/aiter/jit/*.so 2>/dev/null | wc -l) prebuilt kernels)" \
    || { echo "ERROR: topk_select missing"; exit 1; }

chmod -R a+rX "${BASE}" 2>/dev/null &   # world-readable; NFS-slow, so background it
echo "Done. To use YOUR build, export these before serving (aiter first):"
echo "  export ATOM_PYLIB=${ATOM_PYLIB}"
echo "  export AITER_DIR=${AITER_DIR}"
echo "Next: ./3-run-serve-test.sh"
