#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.
#
# install-rocbudai-ollama.sh — install rocBudAI in user mode, as an unprivileged
# user, against a personal ollama daemon. It clones rocBudAI at a pinned ref,
# applies the user-mode patch, stages opencode and ripgrep (checksum-verified
# against rocBudAI's provenance manifests), uses or downloads ollama, and
# writes a modulefile (Lmod .lua or classic Tcl, auto-detected).

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PATCH="${SELF_DIR}/patches/rocbudai-ollama.patch"
MODELFILE_SRC="${SELF_DIR}/Modelfile.laguna-xs-2.1-q4_K_M.rocbudai"

# The patch is authored against ROCBUDAI_REF.
ROCBUDAI_REPO_URL="https://github.com/AMD-HPC/rocBudAI.git"
ROCBUDAI_REF="01303e7649774d46e427bdb6e3be98de51ba1199"
OPENCODE_VERSION="1.14.28"
RG_VERSION="15.1.0"
# Laguna tool calling needs ollama >= 0.32.3.
OLLAMA_VERSION="0.34.4"
OLLAMA_MIN_VERSION="0.32.3"

PREFIX="${HOME}/rocbudai-ollama"
MODEL="laguna-xs-2.1:q4_K_M"
OLLAMA_ADDR="127.0.0.1:11434"
MODELS_DIR=""
GFX_ARCH=""
OLLAMA_BIN=""
DOWNLOAD_OLLAMA=0
ROCBUDAI_SRC=""
FORCE=0
DRY_RUN=0
ALLOW_NO_ALLOCATION=0

if [ -t 1 ]; then
    C_TAG=$'\033[1;36m'; C_ERR=$'\033[1;31m'; C_WARN=$'\033[1;33m'; C_OFF=$'\033[0m'
else
    C_TAG=''; C_ERR=''; C_WARN=''; C_OFF=''
fi

err()  { echo -e "${C_ERR}[install-rocbudai-ollama] error:${C_OFF} $*" >&2; }
warn() { echo -e "${C_WARN}[install-rocbudai-ollama] warning:${C_OFF} $*" >&2; }
info() { echo -e "${C_TAG}[install-rocbudai-ollama]${C_OFF} $*"; }
run()  { if [[ $DRY_RUN -eq 1 ]]; then echo "  + $*"; else "$@"; fi; }

usage() {
    cat <<EOF
Usage: install-rocbudai-ollama.sh [options]

  --prefix DIR         install root (default: \$HOME/rocbudai-ollama)
  --model NAME         ollama model the agent runs (default: ${MODEL})
  --ollama-host H:P    address of your ollama daemon (default: ${OLLAMA_ADDR})
  --models-dir DIR     where ollama stores models (default: ollama's ~/.ollama/models);
                       the default model needs about 20 GB
  --gfx-arch ARCH      pin the profiling-target persona: gfx90a | mi300x | gfx950 | mi300a.
                       Omit it to let the launcher detect the GPU; pin it where rocminfo
                       does not report an "MI300X" marketing name (it then picks mi300a)
  --ollama-bin PATH    use this ollama binary instead of the one on PATH
  --download-ollama    download ollama ${OLLAMA_VERSION} into the prefix even if one is
                       on PATH (needs zstd; about 2.5 GB with the ROCm libraries)
  --rocbudai-ref REF   git ref to pin (default: the ref the patch targets)
  --rocbudai-src PATH  use a local rocBudAI checkout instead of cloning
  --force              overwrite an existing --prefix
  --allow-no-allocation  skip the check that, on a Slurm system, we run on a node
                       of our own allocation (for a compute node reached another way)
  --dry-run            print actions without changing anything
  --help               this message

NO airgap is enforced by this install: do not use it on code or data that must
stay private. It is only for trying rocBudAI; production use must deploy it per
https://github.com/AMD-HPC/rocBudAI/blob/main/INSTALL.md to get the airgap.
EOF
}

need() { [[ $# -ge 2 && -n "$2" ]] || { err "$1 requires a value"; exit 2; }; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)          need "$@"; PREFIX="$2"; shift 2 ;;
        --model)           need "$@"; MODEL="$2"; shift 2 ;;
        --ollama-host)     need "$@"; OLLAMA_ADDR="$2"; shift 2 ;;
        --models-dir)      need "$@"; MODELS_DIR="$2"; shift 2 ;;
        --gfx-arch)        need "$@"; GFX_ARCH="$2"; shift 2 ;;
        --ollama-bin)      need "$@"; OLLAMA_BIN="$2"; shift 2 ;;
        --download-ollama) DOWNLOAD_OLLAMA=1; shift ;;
        --rocbudai-ref)    need "$@"; ROCBUDAI_REF="$2"; shift 2 ;;
        --rocbudai-src)    need "$@"; ROCBUDAI_SRC="$2"; shift 2 ;;
        --force)           FORCE=1; shift ;;
        --allow-no-allocation) ALLOW_NO_ALLOCATION=1; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        --help|-h)         usage; exit 0 ;;
        *)                 err "unknown argument: $1"; usage; exit 2 ;;
    esac
done

OLLAMA_ADDR="${OLLAMA_ADDR#http://}"
OLLAMA_ADDR="${OLLAMA_ADDR%/}"

PERSONA=""
case "${GFX_ARCH}" in
    "")            ;;
    gfx90a)        PERSONA="AGENTS-gfx90a.md" ;;
    gfx942|mi300x) PERSONA="AGENTS-gfx942-mi300x.md" ;;
    gfx950)        PERSONA="AGENTS-gfx950.md" ;;
    mi300a)        PERSONA="AGENTS-default.md" ;;
    *) err "unsupported --gfx-arch '${GFX_ARCH}' (use gfx90a|mi300x|gfx950|mi300a)"; exit 2 ;;
esac

# A bare `salloc` shell sets SLURM_JOB_ID but often stays on the login node, so
# we also require this host to be one of the job's nodes.
if [[ ${DRY_RUN} -eq 0 && ${ALLOW_NO_ALLOCATION} -eq 0 ]] && command -v squeue >/dev/null 2>&1; then
    if [[ -z "${SLURM_JOB_ID:-}" ]] \
        || ! scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>/dev/null | grep -qx "$(hostname -s)"; then
        err "run this on a compute node of a Slurm allocation, not on a login node, e.g.:"
        err "  srun -N1 --gres=gpu:1 -t 01:00:00 --pty bash"
        err "on a compute node reached another way (e.g. ssh into your allocation), pass --allow-no-allocation."
        exit 1
    fi
fi

[[ -f "${PATCH}" ]]         || { err "patch not found: ${PATCH}"; exit 1; }
[[ -f "${MODELFILE_SRC}" ]] || { err "Modelfile not found: ${MODELFILE_SRC}"; exit 1; }
for tool in curl tar sha256sum; do
    command -v "${tool}" >/dev/null 2>&1 || { err "${tool} not found on PATH."; exit 1; }
done

warn "user mode: NO airgap is enforced. Nothing blocks network egress from rocBudAI"
warn "sessions started with this install; do not use it on code or data that must stay private."
warn "This install is only for trying rocBudAI. Production use must deploy it per"
warn "https://github.com/AMD-HPC/rocBudAI/blob/main/INSTALL.md to get the airgap and Slurm gating."

version_of() { "$1" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | tail -n1; }
version_ge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]; }
if [[ ${DOWNLOAD_OLLAMA} -eq 0 ]]; then
    [[ -n "${OLLAMA_BIN}" ]] || OLLAMA_BIN="$(command -v ollama 2>/dev/null || true)"
    if [[ -n "${OLLAMA_BIN}" ]]; then
        [[ -x "${OLLAMA_BIN}" ]] || { err "ollama binary not executable: ${OLLAMA_BIN}"; exit 1; }
        _have="$(version_of "${OLLAMA_BIN}")"
        if [[ -z "${_have}" ]] || ! version_ge "${_have}" "${OLLAMA_MIN_VERSION}"; then
            err "${OLLAMA_BIN} is ollama '${_have:-unknown}'; Laguna needs >= ${OLLAMA_MIN_VERSION}."
            err "pass --download-ollama to install ${OLLAMA_VERSION} into the prefix instead."
            exit 1
        fi
        info "using ollama ${_have} at ${OLLAMA_BIN}"
    else
        DOWNLOAD_OLLAMA=1
    fi
fi
if [[ ${DOWNLOAD_OLLAMA} -eq 1 ]]; then
    command -v zstd >/dev/null 2>&1 || {
        err "no usable ollama found and zstd (needed to unpack ollama ${OLLAMA_VERSION}) is missing."
        err "install zstd, load a site ollama module (>= ${OLLAMA_MIN_VERSION}), or pass --ollama-bin."
        exit 1
    }
    OLLAMA_BIN="${PREFIX}/ollama/bin/ollama"
fi

if [[ -e "${PREFIX}" ]]; then
    if [[ ${FORCE} -eq 1 ]]; then
        info "removing existing ${PREFIX} (--force)"
        run rm -rf "${PREFIX}"
    else
        err "${PREFIX} already exists (pass --force to overwrite)."; exit 1
    fi
fi

if [[ -n "${ROCBUDAI_SRC}" ]]; then
    [[ -d "${ROCBUDAI_SRC}" ]] || { err "--rocbudai-src not a directory: ${ROCBUDAI_SRC}"; exit 1; }
    info "copying rocBudAI from ${ROCBUDAI_SRC}"
    run cp -r "${ROCBUDAI_SRC}" "${PREFIX}"
else
    command -v git >/dev/null 2>&1 || { err "git not found; use --rocbudai-src for an offline install."; exit 1; }
    info "cloning rocBudAI (${ROCBUDAI_REF})"
    run git clone --quiet "${ROCBUDAI_REPO_URL}" "${PREFIX}"
    run git -C "${PREFIX}" checkout --quiet "${ROCBUDAI_REF}"
fi

info "applying user-mode patch"
if [[ $DRY_RUN -eq 0 ]]; then
    if ! git -C "${PREFIX}" apply --check "${PATCH}" 2>/dev/null; then
        err "patch does not apply cleanly to this rocBudAI tree."
        err "pin the matching ref (--rocbudai-ref ${ROCBUDAI_REF}) or re-cut the patch."
        exit 1
    fi
    git -C "${PREFIX}" apply "${PATCH}"
else
    echo "  + git -C ${PREFIX} apply ${PATCH}"
fi

info "installing ${MODELFILE_SRC##*/}"
run cp "${MODELFILE_SRC}" "${PREFIX}/deploy/ollama-models/"

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
fetch_verified() {
    local url="$1" manifest="$2"
    run curl -fsSL -o "${STAGE}/${url##*/}" "${url}"
    if [[ $DRY_RUN -eq 0 ]]; then
        (cd "${STAGE}" && sha256sum --quiet -c "${manifest}") \
            || { err "checksum mismatch for ${url##*/}"; exit 1; }
    fi
}

info "staging opencode ${OPENCODE_VERSION}"
fetch_verified "https://github.com/sst/opencode/releases/download/v${OPENCODE_VERSION}/opencode-linux-x64.tar.gz" \
    "${PREFIX}/archive/opencode-${OPENCODE_VERSION}-provenance/SHA256SUMS.txt"
run mkdir -p "${PREFIX}/opencode"
run tar -xzf "${STAGE}/opencode-linux-x64.tar.gz" -C "${PREFIX}/opencode"

info "staging ripgrep ${RG_VERSION}"
_rg="ripgrep-${RG_VERSION}-x86_64-unknown-linux-musl"
fetch_verified "https://github.com/BurntSushi/ripgrep/releases/download/${RG_VERSION}/${_rg}.tar.gz" \
    "${PREFIX}/archive/ripgrep-${RG_VERSION}-provenance/SHA256SUMS.txt"
run tar -xzf "${STAGE}/${_rg}.tar.gz" -C "${STAGE}"
run install -m 0755 "${STAGE}/${_rg}/rg" "${PREFIX}/bin/rg"

if [[ ${DOWNLOAD_OLLAMA} -eq 1 ]]; then
    info "downloading ollama ${OLLAMA_VERSION} with ROCm libraries (about 2.5 GB)"
    run mkdir -p "${PREFIX}/ollama"
    for _pkg in ollama-linux-amd64 ollama-linux-amd64-rocm; do
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  + curl .../v${OLLAMA_VERSION}/${_pkg}.tar.zst | zstd -d | tar -x -C ${PREFIX}/ollama"
        else
            curl -fsSL "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/${_pkg}.tar.zst" \
                | zstd -dq | tar -x -C "${PREFIX}/ollama"
        fi
    done
fi

MODDIR="${PREFIX}/modulefiles/rocbudai"
MODULE_FLAVOR="${ROCBUDAI_MODULE_FLAVOR:-}"
if [[ -z "${MODULE_FLAVOR}" ]]; then
    if [[ -n "${LMOD_CMD:-}${LMOD_VERSION:-}${LMOD_DIR:-}" ]] || command -v lmod >/dev/null 2>&1; then
        MODULE_FLAVOR="lua"
    elif [[ -n "${MODULES_CMD:-}${MODULESHOME:-}" ]]; then
        MODULE_FLAVOR="tcl"
    else
        MODULE_FLAVOR="lua"
        info "no module system detected; defaulting to an Lmod (.lua) modulefile (override with ROCBUDAI_MODULE_FLAVOR=tcl)"
    fi
fi
case "${MODULE_FLAVOR}" in
    lua) MODFILE="${MODDIR}/ollama.lua" ;;
    tcl) MODFILE="${MODDIR}/ollama" ;;
    *)   err "invalid ROCBUDAI_MODULE_FLAVOR='${MODULE_FLAVOR}' (expected 'lua' or 'tcl')"; exit 1 ;;
esac

# "op name value" rows. The context length must match the opencode limit in the
# patched templates, not ollama's VRAM-based default.
MODROWS=(
    "prepend PATH ${PREFIX}/bin"
    "prepend PATH $(dirname "${OLLAMA_BIN}")"
    "setenv ROCBUDAI_ROOT ${PREFIX}"
    "setenv ROCBUDAI_USER_MODE 1"
    "setenv ROCBUDAI_OPENCODE_BIN ${PREFIX}/opencode/opencode"
    "setenv ROCBUDAI_OPENCODE_DIR ${PREFIX}/opencode"
    "setenv ROCBUDAI_KB_INPUTS_DIR ${PREFIX}/share/rocbudai/kb"
    "setenv ROCBUDAI_OLLAMA_HOST http://${OLLAMA_ADDR}"
    "setenv OLLAMA_HOST ${OLLAMA_ADDR}"
    "setenv OLLAMA_CONTEXT_LENGTH 262144"
    "setenv OLLAMA_FLASH_ATTENTION 1"
    "setenv OLLAMA_KV_CACHE_TYPE q8_0"
    "setenv OLLAMA_IGPU_ENABLE 1"
    "setenv OLLAMA_NO_CLOUD 1"
    "setenv OLLAMA_NOPRUNE 1"
    "setenv OPENCODE_DISABLE_AUTOUPDATE 1"
    "setenv OPENCODE_DISABLE_LSP_DOWNLOAD 1"
    "setenv OPENCODE_DISABLE_MODELS_FETCH 1"
    "setenv OPENCODE_DISABLE_EXTERNAL_SKILLS 1"
    "setenv OPENCODE_DISABLE_SHARE 1"
    "setenv OPENCODE_DISABLE_MOUSE 1"
)
[[ -n "${MODELS_DIR}" ]] && MODROWS+=("setenv OLLAMA_MODELS ${MODELS_DIR}")
[[ -n "${PERSONA}" ]] && MODROWS+=("setenv ROCBUDAI_AGENTS_TEMPLATE ${PREFIX}/share/rocbudai/${PERSONA}")

HELP_LINES=(
    "rocBudAI (user mode) — AMD GPU profiling/optimisation AI assistant."
    ""
    "NO airgap is enforced: nothing blocks network egress from these sessions,"
    "so do not use them on code or data that must stay private. This module is"
    "only for trying rocBudAI; production use must deploy it per rocBudAI's"
    "INSTALL.md to get the airgap and Slurm gating."
    ""
    "  ollama serve > ~/ollama-rocbudai.log 2>&1 &   # your own daemon"
    "  cd <project-dir> && rocbudai                  # launch the OpenCode TUI"
    "  rocbudai --continue                           # resume the last session"
)

info "writing ${MODULE_FLAVOR} modulefile ${MODFILE}"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "  + generate ${MODFILE} (flavor=${MODULE_FLAVOR}, model=${MODEL}, ollama=${OLLAMA_ADDR})"
else
    mkdir -p "${MODDIR}"
    find "${MODDIR}" -mindepth 1 -maxdepth 1 ! -name "$(basename "${MODFILE}")" -exec rm -rf {} +
    if [[ "${MODULE_FLAVOR}" == "lua" ]]; then
        {
            echo "-- rocbudai/ollama — rocBudAI in user mode (generated by install-rocbudai-ollama.sh)"
            echo "-- NO airgap is enforced. Loading sets the environment only; run 'rocbudai' to start."
            echo
            echo 'help([['
            printf '%s\n' "${HELP_LINES[@]}"
            echo ']])'
            echo
            echo 'whatis("Name        : rocbudai (user mode)")'
            echo 'whatis("Description : AMD GPU profiling AI assistant (OpenCode + personal ollama, no airgap)")'
            echo
            for row in "${MODROWS[@]}"; do
                read -r op name value <<<"${row}"
                case "${op}" in
                    prepend) echo "prepend_path(\"${name}\", \"${value}\")" ;;
                    setenv)  echo "setenv(\"${name}\", \"${value}\")" ;;
                esac
            done
            echo
            echo "-- Honour a user export of ROCBUDAI_MODEL; the allow-list follows the model."
            echo 'local model = os.getenv("ROCBUDAI_MODEL")'
            echo "if model == nil or model == \"\" then model = \"${MODEL}\" end"
            echo 'setenv("ROCBUDAI_MODEL", model)'
            echo 'setenv("ROCBUDAI_ALLOWED_MODELS", model)'
        } > "${MODFILE}"
    else
        {
            echo "#%Module1.0"
            echo "## rocbudai/ollama — rocBudAI in user mode (generated by install-rocbudai-ollama.sh)"
            echo "## NO airgap is enforced. Loading sets the environment only; run 'rocbudai' to start."
            echo
            echo "proc ModulesHelp { } {"
            for line in "${HELP_LINES[@]}"; do echo "    puts stderr {${line}}"; done
            echo "}"
            echo
            echo "module-whatis {Name        : rocbudai (user mode)}"
            echo "module-whatis {Description : AMD GPU profiling AI assistant (OpenCode + personal ollama, no airgap)}"
            echo
            for row in "${MODROWS[@]}"; do
                read -r op name value <<<"${row}"
                case "${op}" in
                    prepend) echo "prepend-path ${name} \"${value}\"" ;;
                    setenv)  echo "setenv ${name} \"${value}\"" ;;
                esac
            done
            echo
            echo "# Honour a user export of ROCBUDAI_MODEL; the allow-list follows the model."
            echo "if {[info exists env(ROCBUDAI_MODEL)] && \$env(ROCBUDAI_MODEL) ne {}} {"
            echo "    set model \$env(ROCBUDAI_MODEL)"
            echo "} else {"
            echo "    set model \"${MODEL}\""
            echo "}"
            echo "setenv ROCBUDAI_MODEL \$model"
            echo "setenv ROCBUDAI_ALLOWED_MODELS \$model"
        } > "${MODFILE}"
    fi
fi

MODELFILE_DST="${PREFIX}/deploy/ollama-models/${MODELFILE_SRC##*/}"
echo
info "done. Everything was installed under ${PREFIX} (remove it with: rm -rf ${PREFIX})"
echo
echo "To use it (on the GPU node where the model will run):"
echo "  module use ${PREFIX}/modulefiles"
echo "  module load rocbudai/ollama"
echo "  ollama serve > ~/ollama-rocbudai.log 2>&1 &     # multi-GPU nodes: see the README"
echo "  ollama pull ${MODEL}"
if [[ "${MODEL}" == "laguna-xs-2.1:q4_K_M" ]]; then
    echo "  ollama create ${MODEL} -f ${MODELFILE_DST}"
fi
echo "  cd <your-project-dir> && rocbudai"
