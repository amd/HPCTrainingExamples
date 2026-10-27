#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# Use of this source code is governed by an MIT-style license that can be
# found in the LICENSE file or at https://opensource.org/licenses/MIT.
#
# install-rocbudai-atom.sh — install rocBudAI as a thin client of a Kimi-K3
# model served locally by ATOM (AMD's AiTer-Optimized, vLLM-like, OpenAI-
# compatible engine) on an MI355X (gfx950) node. It clones rocBudAI at a pinned
# ref, applies the vLLM/ATOM backend patch, drops in the no-GPU addendum and the
# serve helper, and writes a modulefile (Lmod .lua or classic Tcl, auto-detected
# from the host's module system) so users get the stock `module load rocbudai`
# experience.
#
# The model (Kimi-K3) runs on a compute node; the agent (OpenCode TUI) runs
# wherever the module is loaded — normally a LOGIN node, so it does not contend
# for CPU with the model. On load the modulefile finds the compute node of the
# user's live Slurm allocation and points the agent at it automatically.
#
# The rocBudAI session picker, naming, banner, resume, auto-nudge, and
# rocbudai-submit paths are reused unchanged; only backend/model resolution, the
# opencode provider config, and the no-GPU override come from the patch.
#
# NOTE: this installs the AGENT (client) side. Build the ATOM engine + download
# the weights first with the sibling kit scripts (1-download-kimi.sh,
# 2-build-engine.sh); see README.md.

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PATCH="${SELF_DIR}/patches/rocbudai-atom.patch"
ADDENDUM="${SELF_DIR}/no-gpu-addendum.md"
SERVE_SRC="${SELF_DIR}/rocbudai-serve-kimi.sh"

# Pinned rocBudAI commit the patch is authored against. Override with
# --rocbudai-ref (a clean apply is required, so only move this when the patch
# is re-cut) or bypass git entirely with --rocbudai-src for an air-gapped host.
ROCBUDAI_REPO_URL="https://github.com/AMD-HPC/rocBudAI.git"
ROCBUDAI_REF="5d695c8917f08f50716fab50ad5e2ed1ee264836"

# Install root. Defaults to <ROCBUDAI_PREFIX_ROOT>/kimi so sibling variants
# (kimi, muse, ...) share one parent dir; override the base at install time with
# the ROCBUDAI_PREFIX_ROOT env var, or the whole path with --prefix.
PREFIX="${ROCBUDAI_PREFIX_ROOT:-${HOME}/opt/rocbudai}/kimi"
GFX_ARCH="gfx950"
MODEL="moonshotai/Kimi-K3"
CONTEXT_LIMIT="131072"
PORT="8000"
PARTITION="mi355x"
ROCM_MODULE="rocm/7.2.2"
ENDPOINT=""
HF_HOME_DIR=""
ATOM_PYLIB_DIR=""
AITER_DIR=""
OPENCODE_BIN_OVR=""
ROCBUDAI_SRC=""
FORCE=0
DRY_RUN=0

# Colored, spaced status output; disabled when stdout is not a TTY so redirected
# or piped logs stay plain.
if [ -t 1 ]; then
    C_TAG=$'\033[1;36m'; C_HEAD=$'\033[1;36m'; C_ERR=$'\033[1;31m'; C_OFF=$'\033[0m'
else
    C_TAG=''; C_HEAD=''; C_ERR=''; C_OFF=''
fi

err()  { echo -e "${C_ERR}[install-rocbudai-atom] error:${C_OFF} $*" >&2; }
info() { echo -e "${C_TAG}[install-rocbudai-atom]${C_OFF} $*"; }
head() { echo -e "${C_HEAD}$*${C_OFF}"; }
run()  { if [[ $DRY_RUN -eq 1 ]]; then echo "  + $*"; else "$@"; fi; }

usage() {
    cat <<EOF
Usage: install-rocbudai-atom.sh [options]

Client (agent):
  --prefix DIR          install root (default: \${ROCBUDAI_PREFIX_ROOT:-\$HOME/opt/rocbudai}/kimi;
                        set ROCBUDAI_PREFIX_ROOT to relocate the shared parent).
  --gfx-arch ARCH       profiling-target persona to pin: gfx90a | mi300x | gfx950
                        | mi300a (default: gfx950, this being an MI355X recipe).
  --model NAME          served model id (default: ${MODEL}).
  --context-limit N     context window opencode advertises; must match the
                        server's --max-model-len / RBK_MAXLEN (default: ${CONTEXT_LIMIT}).
  --port N              port the ATOM server listens on (default: ${PORT}).
  --partition P         Slurm partition for node auto-detect + rocbudai-submit
                        (default: ${PARTITION}).
  --rocm-module M       ROCm module the modulefile loads (default: ${ROCM_MODULE}).
  --endpoint URL        bake a fixed OpenAI base URL (e.g. http://host:${PORT}/v1) and
                        skip the live Slurm-allocation auto-detection.
  --opencode-bin PATH   opencode binary (default: first 'opencode' on PATH).
  --rocbudai-ref REF    git ref to pin (default: the ref the patch targets).
  --rocbudai-src PATH   use a local rocBudAI checkout instead of cloning.
  --force               overwrite an existing --prefix.
  --dry-run             print actions without changing anything.
  --help                this message.

Engine (serve helper defaults, baked into \${PREFIX}/bin/rocbudai-serve-kimi):
  --hf-home DIR         shared HF weights cache (also baked as module HF_HOME).
  --atom-pylib DIR      ATOM pure-python install dir (from 2-build-engine.sh).
  --aiter DIR           AITER wheel dir (from 2-build-engine.sh).
EOF
}

need() { [[ $# -ge 2 && -n "$2" ]] || { err "$1 requires a value"; exit 2; }; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)        need "$@"; PREFIX="$2"; shift 2 ;;
        --gfx-arch)      need "$@"; GFX_ARCH="$2"; shift 2 ;;
        --model)         need "$@"; MODEL="$2"; shift 2 ;;
        --context-limit) need "$@"; CONTEXT_LIMIT="$2"; shift 2 ;;
        --port)          need "$@"; PORT="$2"; shift 2 ;;
        --partition)     need "$@"; PARTITION="$2"; shift 2 ;;
        --rocm-module)   need "$@"; ROCM_MODULE="$2"; shift 2 ;;
        --endpoint)      need "$@"; ENDPOINT="$2"; shift 2 ;;
        --opencode-bin)  need "$@"; OPENCODE_BIN_OVR="$2"; shift 2 ;;
        --hf-home)       need "$@"; HF_HOME_DIR="$2"; shift 2 ;;
        --atom-pylib)    need "$@"; ATOM_PYLIB_DIR="$2"; shift 2 ;;
        --aiter)         need "$@"; AITER_DIR="$2"; shift 2 ;;
        --rocbudai-ref)  need "$@"; ROCBUDAI_REF="$2"; shift 2 ;;
        --rocbudai-src)  need "$@"; ROCBUDAI_SRC="$2"; shift 2 ;;
        --force)         FORCE=1; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --help|-h)       usage; exit 0 ;;
        *)               err "unknown argument: $1"; usage; exit 2 ;;
    esac
done

# Resolve the persona to pin from the arch.
case "${GFX_ARCH}" in
    gfx90a)        PERSONA="AGENTS-gfx90a.md" ;;
    gfx942|mi300x) PERSONA="AGENTS-gfx942-mi300x.md" ;;
    gfx950)        PERSONA="AGENTS-gfx950.md" ;;
    mi300a)        PERSONA="AGENTS-default.md" ;;
    *) err "unsupported --gfx-arch '${GFX_ARCH}' (use gfx90a|mi300x|gfx950|mi300a)"; exit 2 ;;
esac

[[ -f "${PATCH}" ]]     || { err "patch not found: ${PATCH}"; exit 1; }
[[ -f "${ADDENDUM}" ]]  || { err "no-gpu addendum not found: ${ADDENDUM}"; exit 1; }
[[ -f "${SERVE_SRC}" ]] || { err "serve helper not found: ${SERVE_SRC}"; exit 1; }

# Prepare the install prefix.
if [[ -e "${PREFIX}" ]]; then
    if [[ ${FORCE} -eq 1 ]]; then
        info "removing existing ${PREFIX} (--force)"
        run rm -rf "${PREFIX}"
    else
        err "${PREFIX} already exists (pass --force to overwrite)."; exit 1
    fi
fi

# Obtain the rocBudAI tree at the pinned ref, in place at the prefix.
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

# Apply the vLLM/ATOM backend patch (must apply cleanly against the pinned ref).
info "applying vLLM/ATOM backend patch"
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

# Drop the no-GPU addendum where the launcher looks for it.
info "installing no-gpu addendum"
run cp "${ADDENDUM}" "${PREFIX}/share/rocbudai/no-gpu-addendum.md"

# Install the serve helper and bake any engine-path overrides into its defaults.
info "installing serve helper (bin/rocbudai-serve-kimi)"
if [[ $DRY_RUN -eq 0 ]]; then
    cp "${SERVE_SRC}" "${PREFIX}/bin/rocbudai-serve-kimi"
    chmod 0755 "${PREFIX}/bin/rocbudai-serve-kimi"
    _sed_serve() { sed -i -E "s|^(RBK_$1=\"\\\$\\{RBK_$1:-)[^}]*(\\}\")|\\1$2\\2|" "${PREFIX}/bin/rocbudai-serve-kimi"; }
    [[ -n "${HF_HOME_DIR}"   ]] && _sed_serve HF_HOME   "${HF_HOME_DIR}"
    [[ -n "${ATOM_PYLIB_DIR}" ]] && _sed_serve ATOM_PYLIB "${ATOM_PYLIB_DIR}"
    [[ -n "${AITER_DIR}"     ]] && _sed_serve AITER     "${AITER_DIR}"
else
    echo "  + cp ${SERVE_SRC} ${PREFIX}/bin/rocbudai-serve-kimi (+ bake HF/ATOM/AITER defaults)"
fi

# Confirm the pinned persona exists in the tree.
if [[ $DRY_RUN -eq 0 && ! -f "${PREFIX}/share/rocbudai/${PERSONA}" ]]; then
    err "persona ${PERSONA} not found under ${PREFIX}/share/rocbudai (ref mismatch?)."
    exit 1
fi

# Locate opencode for the modulefile (raw opencode honours OPENCODE_CONFIG).
OPENCODE_BIN="${OPENCODE_BIN_OVR:-$(command -v opencode 2>/dev/null || true)}"
OPENCODE_DIR=""
if [[ -n "${OPENCODE_BIN}" ]]; then
    OPENCODE_DIR="$(dirname "${OPENCODE_BIN}")"
else
    err "opencode not found on PATH; install it (npm i -g opencode-ai@1.14.28) or pass --opencode-bin before launching."
fi

# Pick the modulefile flavor from the host's module system. Lmod advertises
# itself via LMOD_*; classic Tcl environment-modules exports only
# MODULES_CMD/MODULESHOME and cannot read .lua files, so it needs a Tcl
# modulefile. Override the auto-detection with ROCBUDAI_MODULE_FLAVOR=lua|tcl.
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
    lua) MODFILE="${MODDIR}/kimi.lua" ;;
    tcl) MODFILE="${MODDIR}/kimi" ;;
    *)   err "invalid ROCBUDAI_MODULE_FLAVOR='${MODULE_FLAVOR}' (expected 'lua' or 'tcl')"; exit 1 ;;
esac

# Drop rocBudAI's stock modulefiles (the ollama-based rocbudai/dev, any
# default-version markers, and the other-flavor file from a prior install) so
# `module use` exposes only the one ATOM client we generate below.
if [[ -d "${MODDIR}" ]]; then
    info "removing rocBudAI's stock modulefiles (keeping only rocbudai/$(basename "${MODFILE}"))"
    run find "${MODDIR}" -mindepth 1 -maxdepth 1 ! -name "$(basename "${MODFILE}")" -exec rm -rf {} +
fi

info "writing ${MODULE_FLAVOR} modulefile ${MODFILE}"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "  + generate ${MODFILE} (flavor=${MODULE_FLAVOR}, backend=vllm, arch=${GFX_ARCH}, model=${MODEL})"
    echo
    info "dry-run complete; nothing was written."
    exit 0
fi

mkdir -p "${MODDIR}"
SALLOC="salloc -p ${PARTITION} --exclusive --gres=gpu:amd_instinct_mi355_oam:8 --mem=0 -t 2:00:00"

if [[ "${MODULE_FLAVOR}" == "lua" ]]; then
    cat > "${MODFILE}" <<EOF
-- rocbudai/kimi — rocBudAI over Kimi-K3 served by ATOM (generated by install-rocbudai-atom.sh)
-- The model runs on a compute node; the agent (OpenCode TUI) runs wherever this
-- module is loaded (normally a LOGIN node). On load it finds the compute node of
-- your live Slurm allocation and points the agent there; with no allocation it
-- errors out. On an interactive TTY it auto-launches the TUI.

help([[
rocBudAI — AMD GPU profiling/optimisation AI assistant (Kimi-K3 on ATOM / MI355X).

Workflow (model on a compute node, agent on the login node):

  # 1. In a compute-node allocation, start the Kimi-K3 server (full path: the
  #    module is NOT loaded on the compute node):
  ${SALLOC}
  ${PREFIX}/bin/rocbudai-serve-kimi   # wait for "Application startup complete"

  # 2. Back on the LOGIN node, just load — it finds your allocation's node:
  cd <project-dir>
  module load rocbudai         # auto-points at the model and launches the TUI

To override the auto-detected endpoint:  export ROCBUDAI_OPENAI_BASE=http://host:${PORT}/v1

Project repo: https://github.com/AMD-HPC/rocBudAI
]])

whatis("Name        : rocbudai (ATOM client)")
whatis("Version     : atom (${MODEL} / ATOM / ${GFX_ARCH})")
whatis("Description : AMD GPU profiling/optimisation AI assistant (OpenCode + ATOM)")
whatis("URL         : https://github.com/AMD-HPC/rocBudAI")

-- ROCm gives us rocminfo/rocm-smi on PATH (read-only pre-approved tools).
load("${ROCM_MODULE}")

local root = "${PREFIX}"
prepend_path("PATH", pathJoin(root, "bin"))
setenv("ROCBUDAI_ROOT", root)
EOF

    [[ -n "${OPENCODE_BIN}" ]] && echo "setenv(\"ROCBUDAI_OPENCODE_BIN\", \"${OPENCODE_BIN}\")" >> "${MODFILE}"
    [[ -n "${OPENCODE_DIR}" ]] && echo "setenv(\"ROCBUDAI_OPENCODE_DIR\", \"${OPENCODE_DIR}\")" >> "${MODFILE}"

    cat >> "${MODFILE}" <<EOF

-- Point the agent's knowledge base at the cloned default KB; honour a user export.
if os.getenv("ROCBUDAI_KB_INPUTS_DIR") == nil or os.getenv("ROCBUDAI_KB_INPUTS_DIR") == "" then
    setenv("ROCBUDAI_KB_INPUTS_DIR", pathJoin(root, "share/rocbudai/kb"))
end

-- Backend: OpenAI-compatible vLLM/ATOM instead of Ollama.
setenv("ROCBUDAI_BACKEND", "vllm")
setenv("ROCBUDAI_OPENAI_PROVIDER", "vllm")

-- Context window opencode advertises for the model. Must match the server's
-- --max-model-len (rocbudai-serve-kimi RBK_MAXLEN) so the agent does not compact
-- early or overshoot and get a 400.
if os.getenv("ROCBUDAI_CONTEXT_LIMIT") == nil or os.getenv("ROCBUDAI_CONTEXT_LIMIT") == "" then
    setenv("ROCBUDAI_CONTEXT_LIMIT", "${CONTEXT_LIMIT}")
end
EOF

    if [[ -n "${ENDPOINT}" ]]; then
        cat >> "${MODFILE}" <<EOF

-- Fixed endpoint baked at install time (--endpoint); auto-detection skipped.
if os.getenv("ROCBUDAI_OPENAI_BASE") == nil or os.getenv("ROCBUDAI_OPENAI_BASE") == "" then
    setenv("ROCBUDAI_OPENAI_BASE", "${ENDPOINT}")
end
EOF
    else
        cat >> "${MODFILE}" <<EOF

-- Point the agent at the Kimi-K3 server. If the user set ROCBUDAI_OPENAI_BASE to
-- a real remote endpoint themselves, respect it. Otherwise (empty, a value we
-- auto-set on a previous load, or a stale localhost default) auto-discover the
-- compute node of their live Slurm allocation and build the endpoint. No
-- allocation, or more than one, is a hard error — nothing unambiguous to use.
local _base = os.getenv("ROCBUDAI_OPENAI_BASE")
local _manual = _base ~= nil and _base ~= ""
    and os.getenv("ROCBUDAI_OPENAI_BASE_AUTO") ~= "1"
    and _base:find("127.0.0.1", 1, true) == nil
    and _base:find("localhost", 1, true) == nil
if mode() == "load" and not _manual then
    local port = os.getenv("ROCBUDAI_OPENAI_PORT")
    if port == nil or port == "" then port = "${PORT}" end
    local nodes = {}
    local out = capture("squeue -h -u \"\$USER\" -t R -o '%N' 2>/dev/null"
                        .. " | xargs -r -n1 scontrol show hostnames 2>/dev/null | sort -u")
    for line in (out or ""):gmatch("[^\n]+") do
        if line ~= "" then nodes[#nodes + 1] = line end
    end
    if #nodes == 0 then
        LmodError("rocBudAI: no live Slurm allocation found — nothing to connect to.\n"
            .. "Start the Kimi-K3 server first (on a compute node; full path, the\n"
            .. "module is not on PATH there):\n"
            .. "  ${SALLOC}\n"
            .. "  ${PREFIX}/bin/rocbudai-serve-kimi   # wait for 'Application startup complete'\n"
            .. "then, back on the login node:  module load rocbudai\n")
    elseif #nodes > 1 then
        LmodError("rocBudAI: you have allocations on multiple nodes ("
            .. table.concat(nodes, ", ") .. ").\n"
            .. "Pick the one running the server, then load:\n"
            .. "  export ROCBUDAI_OPENAI_BASE=http://<node>:" .. port .. "/v1\n")
    else
        setenv("ROCBUDAI_OPENAI_BASE", "http://" .. nodes[1] .. ":" .. port .. "/v1")
        setenv("ROCBUDAI_OPENAI_BASE_AUTO", "1")   -- so a later load re-detects
        LmodMessage("[rocBudAI] model node: " .. nodes[1]
            .. "  (ROCBUDAI_OPENAI_BASE=http://" .. nodes[1] .. ":" .. port .. "/v1)")
    end
end
EOF
    fi

    cat >> "${MODFILE}" <<EOF

-- opencode phone-home suppression (soft airgap layer).
setenv("OPENCODE_DISABLE_AUTOUPDATE",      "1")
setenv("OPENCODE_DISABLE_LSP_DOWNLOAD",    "1")
setenv("OPENCODE_DISABLE_MODELS_FETCH",    "1")
setenv("OPENCODE_DISABLE_EXTERNAL_SKILLS", "1")
setenv("OPENCODE_DISABLE_SHARE",           "1")
setenv("OPENCODE_DISABLE_MOUSE",           "1")

-- Model pin + allow-list. Users may override ROCBUDAI_MODEL only to another
-- entry in the allow-list.
if os.getenv("ROCBUDAI_MODEL") == nil or os.getenv("ROCBUDAI_MODEL") == "" then
    setenv("ROCBUDAI_MODEL", "${MODEL}")
end
setenv("ROCBUDAI_ALLOWED_MODELS", "${MODEL}")

-- Pinned profiling-target persona (honour a user export to force one).
setenv("ROCBUDAI_GFX_ARCH", "${GFX_ARCH}")
if os.getenv("ROCBUDAI_AGENTS_TEMPLATE") == nil or os.getenv("ROCBUDAI_AGENTS_TEMPLATE") == "" then
    setenv("ROCBUDAI_AGENTS_TEMPLATE", pathJoin(root, "share/rocbudai/${PERSONA}"))
end

-- Partition for node auto-detect + where rocbudai-submit dispatches GPU jobs.
setenv("ROCBUDAI_SPX_PARTITIONS", "${PARTITION}")
setenv("ROCBUDAI_SUBMIT_PARTITION", "${PARTITION}")

-- The model owns all 8 GPUs on its node, so do NOT fence the agent's shell onto
-- a "bench GPU" (there isn't a spare one, and the agent is on a login node).
setenv("ROCBUDAI_NO_FENCING", "1")
EOF

    [[ -n "${HF_HOME_DIR}" ]] && cat >> "${MODFILE}" <<EOF

-- Shared, world-readable HF cache (weights downloaded once for everyone).
setenv("HF_HOME", "${HF_HOME_DIR}")
EOF

    cat >> "${MODFILE}" <<EOF

-- Auto-launch the TUI on load (recursion-guarded by ROCBUDAI_ACTIVE in the hook).
if mode() == "load" then
    execute{
        cmd = "bash " .. pathJoin(root, "libexec/rocbudai-load-hook.sh"),
        modeA = {"load"},
    }
end
EOF

else
    # Classic Tcl environment-modules: same variables in Tcl idioms.
    cat > "${MODFILE}" <<EOF
#%Module1.0
## rocbudai/kimi — rocBudAI over Kimi-K3 served by ATOM (generated by install-rocbudai-atom.sh)
## The model runs on a compute node; the agent runs wherever this module loads.

proc ModulesHelp { } {
    puts stderr {rocBudAI — AMD GPU profiling/optimisation AI assistant (Kimi-K3 on ATOM / MI355X).}
    puts stderr {}
    puts stderr {  # On a compute node, start the server (full path):}
    puts stderr {  ${SALLOC}}
    puts stderr {  ${PREFIX}/bin/rocbudai-serve-kimi}
    puts stderr {  # Back on the login node:}
    puts stderr {  cd <project-dir>; module load rocbudai}
}

module-whatis {Name        : rocbudai (ATOM client)}
module-whatis {Description : AMD GPU profiling AI assistant (OpenCode + ATOM)}

module load ${ROCM_MODULE}

set root "${PREFIX}"
prepend-path PATH \$root/bin
setenv ROCBUDAI_ROOT \$root
EOF

    [[ -n "${OPENCODE_BIN}" ]] && echo "setenv ROCBUDAI_OPENCODE_BIN \"${OPENCODE_BIN}\"" >> "${MODFILE}"
    [[ -n "${OPENCODE_DIR}" ]] && echo "setenv ROCBUDAI_OPENCODE_DIR \"${OPENCODE_DIR}\"" >> "${MODFILE}"

    cat >> "${MODFILE}" <<EOF

if {![info exists env(ROCBUDAI_KB_INPUTS_DIR)] || \$env(ROCBUDAI_KB_INPUTS_DIR) eq {}} {
    setenv ROCBUDAI_KB_INPUTS_DIR \$root/share/rocbudai/kb
}

setenv ROCBUDAI_BACKEND vllm
setenv ROCBUDAI_OPENAI_PROVIDER vllm

if {![info exists env(ROCBUDAI_CONTEXT_LIMIT)] || \$env(ROCBUDAI_CONTEXT_LIMIT) eq {}} {
    setenv ROCBUDAI_CONTEXT_LIMIT ${CONTEXT_LIMIT}
}
EOF

    if [[ -n "${ENDPOINT}" ]]; then
        cat >> "${MODFILE}" <<EOF

if {![info exists env(ROCBUDAI_OPENAI_BASE)] || \$env(ROCBUDAI_OPENAI_BASE) eq {}} {
    setenv ROCBUDAI_OPENAI_BASE "${ENDPOINT}"
}
EOF
    else
        cat >> "${MODFILE}" <<EOF

# Auto-discover the compute node of the user's live Slurm allocation.
set _base ""
if {[info exists env(ROCBUDAI_OPENAI_BASE)]} { set _base \$env(ROCBUDAI_OPENAI_BASE) }
set _auto ""
if {[info exists env(ROCBUDAI_OPENAI_BASE_AUTO)]} { set _auto \$env(ROCBUDAI_OPENAI_BASE_AUTO) }
set _manual 0
if {\$_base ne {} && \$_auto ne {1} && [string first 127.0.0.1 \$_base] < 0 && [string first localhost \$_base] < 0} {
    set _manual 1
}
if {[module-info mode load] && !\$_manual} {
    set _port ${PORT}
    if {[info exists env(ROCBUDAI_OPENAI_PORT)] && \$env(ROCBUDAI_OPENAI_PORT) ne {}} { set _port \$env(ROCBUDAI_OPENAI_PORT) }
    set _nodes {}
    if {![catch {exec sh -c {squeue -h -u "\$USER" -t R -o '%N' 2>/dev/null | xargs -r -n1 scontrol show hostnames 2>/dev/null | sort -u}} _out]} {
        foreach _l [split \$_out "\n"] { if {\$_l ne {}} { lappend _nodes \$_l } }
    }
    if {[llength \$_nodes] == 0} {
        puts stderr "rocBudAI: no live Slurm allocation found — start rocbudai-serve-kimi on a compute node first."
    } elseif {[llength \$_nodes] > 1} {
        puts stderr "rocBudAI: allocations on multiple nodes ([join \$_nodes {, }]); export ROCBUDAI_OPENAI_BASE=http://<node>:\$_port/v1"
    } else {
        set _n [lindex \$_nodes 0]
        setenv ROCBUDAI_OPENAI_BASE "http://\$_n:\$_port/v1"
        setenv ROCBUDAI_OPENAI_BASE_AUTO 1
        puts stderr "\[rocBudAI\] model node: \$_n  (ROCBUDAI_OPENAI_BASE=http://\$_n:\$_port/v1)"
    }
}
EOF
    fi

    cat >> "${MODFILE}" <<EOF

setenv OPENCODE_DISABLE_AUTOUPDATE      1
setenv OPENCODE_DISABLE_LSP_DOWNLOAD    1
setenv OPENCODE_DISABLE_MODELS_FETCH    1
setenv OPENCODE_DISABLE_EXTERNAL_SKILLS 1
setenv OPENCODE_DISABLE_SHARE           1
setenv OPENCODE_DISABLE_MOUSE           1

if {![info exists env(ROCBUDAI_MODEL)] || \$env(ROCBUDAI_MODEL) eq {}} {
    setenv ROCBUDAI_MODEL ${MODEL}
}
setenv ROCBUDAI_ALLOWED_MODELS ${MODEL}

setenv ROCBUDAI_GFX_ARCH ${GFX_ARCH}
if {![info exists env(ROCBUDAI_AGENTS_TEMPLATE)] || \$env(ROCBUDAI_AGENTS_TEMPLATE) eq {}} {
    setenv ROCBUDAI_AGENTS_TEMPLATE \$root/share/rocbudai/${PERSONA}
}

setenv ROCBUDAI_SPX_PARTITIONS ${PARTITION}
setenv ROCBUDAI_SUBMIT_PARTITION ${PARTITION}
setenv ROCBUDAI_NO_FENCING 1
EOF

    [[ -n "${HF_HOME_DIR}" ]] && echo "setenv HF_HOME \"${HF_HOME_DIR}\"" >> "${MODFILE}"

    cat >> "${MODFILE}" <<EOF

# Auto-launch the TUI on load (recursion-guarded by ROCBUDAI_ACTIVE); printed to
# stdout because the module function evals a modulefile's stdout.
if {[module-info mode load] && [module-info shelltype] eq {sh}} {
    puts stdout "bash \$root/libexec/rocbudai-load-hook.sh;"
}
EOF
fi

echo
info "done. Everything was installed under: ${C_HEAD}${PREFIX}${C_OFF}"
echo "  To remove this install later: rm -rf ${PREFIX}"
echo
head "Before you load (one-time): build the ATOM engine + download Kimi-K3 weights"
head "with the sibling kit scripts (see README.md):"
echo "  ./1-download-kimi.sh      # ~1.5 TB, shared HF cache"
echo "  ./2-build-engine.sh       # ATOM (pinned) + AITER wheel + patch"
echo
head "To use it:"
echo "  # 1) On a compute node, start the server:"
echo "  ${SALLOC}"
echo "  ${PREFIX}/bin/rocbudai-serve-kimi"
echo "  # 2) On the login node:"
echo "  module use ${PREFIX}/modulefiles"
echo "  cd <your-project-dir>"
echo "  module load rocbudai"
