#!/usr/bin/env bash
# Build a muse-capable vLLM (+ AITER) from source at the merge commit — no
# container, no prebuilt wheel (upstream ROCm wheels bundle their own torch and
# lack a muse-capable torch-2.14 build). Wraps AMD's HPCTrainingDock vllm_setup.sh,
# which compiles vLLM against a `pytorch` Lmod module, builds AITER from source, and
# emits a vllm/<ver>-pt<ptver> module (pinned to VLLM_BUILD_REF); the serve command
# loads it via config.sh's VLLM_MODULE / VLLM_MODULEPATH.
#
# Prereqs: a `pytorch` module providing torch 2.14 for this ROCm (HPCTrainingDock
# pytorch_setup.sh), git, and a ROCm C/C++/HIP toolchain. Long compile — run in an
# allocation or on a build node. See README for the full flow.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
# The builder drops the modulefile at ${VLLM_MODULEPATH}/vllm/<token>.lua so it
# loads as vllm/<token> when the serve command does `module use ${VLLM_MODULEPATH}`.
MOD_DIR="${VLLM_MODULEPATH}/vllm"
mkdir -p "${BASE}" "${LOGDIR}" "${VLLM_BUILD_PREFIX}" "${MOD_DIR}"

echo "### fetch AMD builder: ${VLLM_SETUP_URL} ###"
SETUP="${BASE}/vllm_setup.sh"
curl -fsSL "${VLLM_SETUP_URL}" -o "${SETUP}"
chmod +x "${SETUP}"

# ── Pin the exact ref (branch | tag | commit SHA) ─────────────────────────────
# AMD's script clones with `git clone --depth 1 --branch <ref>`, which accepts a
# branch or tag but NOT a bare commit SHA. To pin ANY ref uniformly (including the
# merge commit), we shallow-fetch it into a local repo, put a branch on it, and
# hand that local repo to the builder via file:// + --vllm-ref <branch>. GitHub
# allows fetching by SHA, so this works for commits, branches and tags alike.
PIN="${BASE}/vllm-pin"
PIN_BRANCH="muse-pin"
echo "### pin vLLM @ ${VLLM_BUILD_REF} (local branch ${PIN_BRANCH}) ###"
rm -rf "${PIN}"
git init -q "${PIN}"
git -C "${PIN}" remote add origin "${VLLM_REPO}"
git -C "${PIN}" fetch -q --depth 1 origin "${VLLM_BUILD_REF}"
git -C "${PIN}" branch -f "${PIN_BRANCH}" FETCH_HEAD
echo "pinned $(git -C "${PIN}" rev-parse --short "${PIN_BRANCH}")"

# ── Invoke AMD's builder ──────────────────────────────────────────────────────
# --install-path/--module-path point under /shared so the builder's write-probe
# drops sudo; --vllm-repo file://PIN + --vllm-ref muse-pin gives the exact commit;
# --vllm-version bypasses the torch->vLLM auto-derive gate (its table has no torch
# 2.14 mapping) and labels the install dir + module; AITER is built from source,
# pinned to AITER_RELEASE_TAG. The builder NOOPs (rc 43) if already built and skips
# (rc 42) if the pytorch module is missing — surface both clearly.
echo "### build vLLM + AITER from source (long compile) ###"
# We pin vLLM by SHA into a tagless local repo, so setuptools-scm (vLLM's dynamic
# version backend) has no tag to read and would stamp the wheel 0.0.0. Hand it the
# intended version so the built wheel is labelled correctly. Package-scoped var
# wins over the generic one; set both for older setuptools-scm.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="${VLLM_BUILD_VERSION}"
export SETUPTOOLS_SCM_PRETEND_VERSION="${VLLM_BUILD_VERSION}"
set +e
"${SETUP}" \
    --build-vllm 1 \
    --vllm-repo "file://${PIN}" \
    --vllm-ref "${PIN_BRANCH}" \
    --vllm-version "${VLLM_BUILD_VERSION}" \
    --amdgpu-gfxmodel "${GFX_ARCH}" \
    --rocm-version "${ROCM_MODULE#rocm/}" \
    --pytorch-module "${PYTORCH_MODULE}" \
    --with-aiter 1 \
    --aiter-ref "${AITER_RELEASE_TAG}" \
    --install-path "${VLLM_BUILD_PREFIX}" \
    --module-path "${MOD_DIR}" \
    2>&1 | tee "${LOGDIR}/build-from-source.log"
rc=${PIPESTATUS[0]}
set -e
case "${rc}" in
    0)  echo "build OK" ;;
    43) echo "NOTE: vLLM already installed for this (version,pytorch) pair (builder NOOP). Pass --replace via the builder to rebuild." ;;
    42) echo "ERROR: the '${PYTORCH_MODULE}' Lmod module is missing for ${ROCM_MODULE}. Build it first (HPCTrainingDock pytorch_setup.sh) or set PYTORCH_MODULE." >&2; exit 42 ;;
    *)  echo "ERROR: vLLM source build failed (rc=${rc}); see ${LOGDIR}/build-from-source.log" >&2; exit "${rc}" ;;
esac

# ── Apply the post-install ROCm workaround patch ──────────────────────────────
# vLLM's env_override promotes libtorch symbols RTLD_GLOBAL on ROCm (for GPU
# profiling), which makes Triton's libtriton bind LLVM against torch's copy and
# SEGV on `import vllm`. The patch gates that behind VLLM_NO_ROCM_SYMBOL_PROMOTION
# (which the serve command sets). Applied to the installed tree; idempotent.
# Only engine patches (patches/vllm-*.patch) are applied here; the agent-side
# rocbudai-*.patch lives in the same dir but targets the rocBudAI tree and is
# applied by the installer, not against the vLLM install.
echo "### apply ROCm workaround patch(es) to the installed vLLM ###"
for PF in patches/vllm-*.patch; do
    [[ -e "${PF}" ]] || continue
    EO="$(find "${VLLM_BUILD_PREFIX}" -path '*/dist-packages/vllm/env_override.py' \
            -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)"
    if [[ -z "${EO}" ]]; then
        echo "WARNING: installed vllm/env_override.py not found; cannot apply ${PF} (server may SEGV on import)." >&2
        break
    fi
    DIST_DIR="${EO%/vllm/env_override.py}"
    if grep -q "VLLM_NO_ROCM_SYMBOL_PROMOTION" "${EO}"; then
        echo "  already applied: $(basename "${PF}")"
    elif patch -p1 -d "${DIST_DIR}" --forward --silent < "${PF}"; then
        echo "  applied: $(basename "${PF}") -> ${EO}"
    else
        echo "ERROR: failed to apply ${PF} to ${EO}" >&2; exit 1
    fi
done

# ── Resolve the generated module token and tell the user how to use it ─────────
# vllm_setup.sh drops <token>.lua into ${MOD_DIR} (= ${VLLM_MODULEPATH}/vllm), so
# the loadable name is vllm/<token> once ${VLLM_MODULEPATH} is on MODULEPATH.
MOD_LUA="$(ls -t "${MOD_DIR}"/*.lua 2>/dev/null | head -1 || true)"
if [[ -n "${MOD_LUA}" ]]; then
    MOD_TOKEN="vllm/$(basename "${MOD_LUA}" .lua)"
    echo
    echo "Done. The serve command loads this via config.sh's VLLM_MODULE / VLLM_MODULEPATH."
    echo "If your build differs from the defaults, export these before serving:"
    echo "  export VLLM_MODULEPATH=${VLLM_MODULEPATH}"
    echo "  export VLLM_MODULE=${MOD_TOKEN}"
    echo "Next: ./3-run-serve-test.sh"
else
    echo "WARNING: no modulefile found under ${MOD_DIR}; check the build log." >&2
fi
