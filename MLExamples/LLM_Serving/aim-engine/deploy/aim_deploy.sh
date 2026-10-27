#!/bin/bash

# --------------------------------------------------------------------------
# Incremental AIM deployment into an EXISTING Kubernetes cluster. Pick a level:
# higher levels assume less is already installed and do more, and each level
# preflights its preconditions before acting. This is a thin dispatcher over the
# sibling scripts (aim_base_check.sh, aim_engine_setup.sh, aim_prereqs_setup.sh);
# it holds no install logic of its own beyond applying the AIMService.
#
#   L1 base-image serve : deploy the model as a plain Deployment (no operator).
#                         Assumes present: cluster + amd.com/gpu.
#   L2 operator serve   : deploy an AIMService.
#                         Assumes present: AIM Engine operator + the 7 prereqs.
#   L3 operator install : install AIM Engine, then serve (L2).
#                         Assumes present: the 7 prereqs.
#   L4 prereqs install  : install the 7 add-ons + AIM Engine, then serve (L2).
#                         Assumes present: the AMD GPU Operator (amd.com/gpu).
#
# Level 1 runs an AIM container directly (the microservice); levels 2-4 use AIM
# Engine (the operator). For how the two relate, see the reference stacks
# overview: https://enterprise-ai.docs.amd.com/en/latest/reference-stacks.html
#
# Cluster creation ("from zero") is intentionally out of scope: use
# aim_engine_test.sh for a throwaway kind cluster, or AMD's Cluster Bloom/Forge
# for a real bare-metal install.
# --------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

: ${LEVEL:=}
# Ungated model-specific image for the operator levels (2-4), so no token is needed.
: ${AIM_MODEL_IMAGE:=amdenterpriseai/aim-qwen-qwen3-32b:0.13.0}
: ${NAMESPACE:=default}
# Predictor resource floor. AIM Engine merges spec.resources over the profile's
# computed resources but replaces the requests/limits maps wholesale, so we must
# restate every key we care about: memory (auto-generated profiles omit it,
# leaving KServe's 2Gi default that OOM-kills weight loading) AND amd.com/gpu
# (omitting it drops the GPU limit, so the pod gets no isolated device and, on a
# bare-passthrough node, sees every host GPU, which breaks in-container rocminfo).
# The GPU count tracks AIM_ACCELERATOR_COUNT. Defaults suit the 32B image above.
: ${AIM_CPU_REQUEST:=8}
: ${AIM_MEM_REQUEST:=64Gi}
: ${AIM_CPU_LIMIT:=16}
: ${AIM_MEM_LIMIT:=128Gi}
# GPU count the resolved profile should use (tensor-parallel size). 1 keeps the
# smoke test on a single GPU, the predictor configuration we verified end to end;
# raise it (2, 4, 8) for multi-GPU profiles. If resolution reports ProfileNotFound
# the model may not ship a profile at this count, so try another value.
: ${AIM_ACCELERATOR_COUNT:=1}
# Inference-engine arg overrides baked into spec.profileOverrides.engineArgs, so they
# survive re-runs (which regenerate the AIMService) without a manual kubectl patch.
# AIM_MAX_MODEL_LEN is a shortcut for the max-model-len (context window) key; add
# any other key=value pairs to AIM_ENGINE_ARGS (space-separated) or with repeated
# --engine-arg. Only emitted on the operator levels (2-4), where a profile exists.
: ${AIM_MAX_MODEL_LEN:=}
: ${AIM_ENGINE_ARGS:=}
# Path to a YAML merge patch applied over the generated AIMService on every run,
# for any spec field the shortcuts above do not reach (env, resources, replicas,
# containerEnv, adapters, autoscaling, ...). Re-applied each run, so it too
# survives regeneration. Merge semantics are RFC 7386: nested maps merge by key,
# lists and scalars replace wholesale.
: ${AIM_OVERRIDES_FILE:=}
# One-shot auto-nudge that clears an AIM Engine reconcile gap (see below). 0 off.
: ${AIM_AUTO_NUDGE:=1}
: ${HF_TOKEN:=}
ENGINE_ARGS=()  # key=value engine-arg overrides collected from env and flags
BASE_ARGS=()    # forwarded to aim_base_check.sh on level 1 only
SETUP_ARGS=()   # forwarded to aim_engine_setup.sh on levels 3 and 4

usage()
{
   echo "Usage:"
   echo "  --level [ 1|2|3|4 ] deployment level (required):"
   echo "     1: base-image serve (plain Deployment; assumes cluster + amd.com/gpu)"
   echo "     2: operator serve (AIMService; assumes operator + prereqs installed)"
   echo "     3: install AIM Engine, then serve (assumes the 7 prereqs installed)"
   echo "     4: install prereqs + AIM Engine, then serve (assumes GPU Operator)"
   echo "  --model-image [ IMAGE ] AIM model image for levels 2-4, default ${AIM_MODEL_IMAGE}"
   echo "                          (level 1 uses aim_base_check.sh's own ungated default)"
   echo "  --max-model-len [ N ] (levels 2-4) context window; shortcut for --engine-arg max-model-len=N"
   echo "  --engine-arg [ KEY=VALUE ] (levels 2-4) inference-engine arg override, repeatable;"
   echo "                             baked into spec.profileOverrides.engineArgs (survives re-runs)"
   echo "  --overrides-file [ PATH ] (levels 2-4) YAML merge patch applied over the AIMService"
   echo "                            each run, for any spec field the flags above do not reach"
   echo "  --kubeconfig [ PATH ] kubeconfig to use for this run (exports KUBECONFIG;"
   echo "                        inherited by the setup scripts this dispatches to)"
   echo "  --namespace [ NS ] namespace for the AIMService, default ${NAMESPACE}"
   echo "  --keep [ 0|1 ] (level 1 only) leave the Deployment/Service running"
   echo "  --verbose [ 0|1 ] (level 1 only) tail recent pod logs while waiting"
   echo "  --replace [ 0|1 ] (levels 3-4) uninstall an existing AIM Engine release first"
   echo "  --aim-version [ VER ] (levels 3-4) AIM Engine chart version, default latest"
   echo "  --crds-chart [ REF ] (levels 3-4) AIM CRDs chart override"
   echo "  --chart [ REF ] (levels 3-4) AIM Engine chart override"
   echo "  --help: print this usage information"
   echo ""
   echo "Env: HF_TOKEN (gated models), AIM_MODEL_IMAGE, AIM_CPU_REQUEST,"
   echo "  AIM_MEM_REQUEST, AIM_CPU_LIMIT, AIM_MEM_LIMIT (predictor resources),"
   echo "  AIM_ACCELERATOR_COUNT (GPU/tensor-parallel size, default 1),"
   echo "  AIM_MAX_MODEL_LEN (context window), AIM_ENGINE_ARGS (space-separated"
   echo "  key=value engine-arg overrides), AIM_OVERRIDES_FILE (YAML merge patch"
   echo "  over the AIMService), AIM_AUTO_NUDGE (0 disables the auto-nudge)."
}

# Colored, spaced status output; disabled when stdout is not a TTY so redirected
# or piped logs stay plain.
if [ -t 1 ]; then
   C_TAG=$'\033[1;36m'; C_HEAD=$'\033[1;36m'; C_ERR=$'\033[1;31m'; C_OFF=$'\033[0m'
else
   C_TAG=''; C_HEAD=''; C_ERR=''; C_OFF=''
fi
say() { echo -e "${C_TAG}[deploy]${C_OFF} ${*}"; }

# Print the reason AFTER the usage block (usage() does not exit) so it is the
# last line the user sees, then fail.
send-error() { usage; echo -e "\nError: ${@}" >&2; exit 1; }
# State/environment failures: print the reason plainly, no usage block.
fatal() { echo -e "\n${C_ERR}[deploy] ERROR:${C_OFF} ${@}" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
   case "${1}" in
      "--level")       shift; LEVEL=${1} ;;
      "--kubeconfig")  shift; [ -f "${1}" ] || send-error "kubeconfig file not found :: ${1}"; export KUBECONFIG="${1}" ;;
      "--model-image") shift; AIM_MODEL_IMAGE=${1} ;;
      "--max-model-len") shift; AIM_MAX_MODEL_LEN=${1} ;;
      "--engine-arg")  shift; [[ "${1}" == *=* ]] || send-error "--engine-arg expects KEY=VALUE :: ${1}"; ENGINE_ARGS+=("${1}") ;;
      "--overrides-file") shift; [ -f "${1}" ] || send-error "overrides file not found :: ${1}"; AIM_OVERRIDES_FILE="${1}" ;;
      "--namespace")   shift; NAMESPACE=${1} ;;
      "--keep")        shift; BASE_ARGS+=(--keep "${1}") ;;
      "--verbose")     shift; BASE_ARGS+=(--verbose "${1}") ;;
      "--replace")     shift; SETUP_ARGS+=(--replace "${1}") ;;
      "--aim-version") shift; SETUP_ARGS+=(--aim-version "${1}") ;;
      "--crds-chart")  shift; SETUP_ARGS+=(--crds-chart "${1}") ;;
      "--chart")       shift; SETUP_ARGS+=(--chart "${1}") ;;
      "--help")        usage; exit 0 ;;
      *)               send-error "Unsupported argument :: ${1}" ;;
   esac
   shift
done

[ -n "${LEVEL}" ] || send-error "select a level with --level [1|2|3|4]."
command -v kubectl >/dev/null 2>&1 || send-error "kubectl not found on PATH."
# An expired OIDC token makes the first kubectl call block on a browser re-login;
# on a headless host that looks like a silent hang, so warn up front.
say "checking cluster access (if this hangs, your login likely expired: run 'kubectl get nodes' to re-authenticate, then retry)"
# Key off returned node names, not the exit code: OIDC discovery bursts 401 then
# recover, so retry; only if none ever come back do we surface the real reason.
reachable=""
for _ in $(seq 10); do
   [ -n "$(kubectl get nodes --request-timeout=15s -o name 2>/dev/null)" ] && { reachable=1; break; }
   sleep 2
done
[ -n "${reachable}" ] \
   || send-error "cannot reach a Kubernetes cluster (check KUBECONFIG / login) :: $(kubectl get nodes 2>&1 >/dev/null)"

operator_present() { kubectl get crd aimservices.aim.eai.amd.com >/dev/null 2>&1; }

# L2 action: wire an optional HF token, apply an AIMService, report how to check it.
serve_operator()
{
   operator_present || fatal "AIM Engine operator not installed in this cluster (CRD aimservices.aim.eai.amd.com missing).
Level 2 assumes the operator and its prerequisites are already present. To install:
  --level 3   install AIM Engine, then serve (assumes the 7 prerequisites)
  --level 4   install the 7 prerequisites + AIM Engine, then serve (assumes the GPU Operator)"
   if [ -n "${HF_TOKEN}" ]; then
      say "configuring Hugging Face token (secret + default AIMRuntimeConfig)"
      kubectl create secret generic hf-token -n "${NAMESPACE}" \
         --from-literal=hf-token="${HF_TOKEN}" --dry-run=client -o yaml | kubectl apply --request-timeout=30s --server-side --force-conflicts --validate=false -f -
      kubectl apply --request-timeout=30s --server-side --force-conflicts --validate=false -f - <<EOF
apiVersion: aim.eai.amd.com/v1alpha1
kind: AIMRuntimeConfig
metadata:
  name: default
  namespace: ${NAMESPACE}
spec:
  env:
  - name: HF_TOKEN
    valueFrom:
      secretKeyRef:
        name: hf-token
        key: hf-token
EOF
   fi
   # Assemble engine-arg overrides from the shortcut, the space-separated env var,
   # and any --engine-arg flags, then render a spec.profileOverrides.engineArgs
   # block. Values are quoted and passed verbatim to the inference-engine CLI (vLLM
   # today; the field is engine-agnostic). Empty when no
   # overrides are set, so the manifest is unchanged for a plain run.
   [ -n "${AIM_MAX_MODEL_LEN}" ] && ENGINE_ARGS+=("max-model-len=${AIM_MAX_MODEL_LEN}")
   for kv in ${AIM_ENGINE_ARGS}; do
      [[ "${kv}" == *=* ]] || fatal "AIM_ENGINE_ARGS entries must be KEY=VALUE :: ${kv}"
      ENGINE_ARGS+=("${kv}")
   done
   profile_overrides=""
   if [ "${#ENGINE_ARGS[@]}" -gt 0 ]; then
      profile_overrides=$'  profileOverrides:\n    engineArgs:\n'
      for kv in "${ENGINE_ARGS[@]}"; do
         profile_overrides+="      ${kv%%=*}: \"${kv#*=}\""$'\n'
      done
      say "baking engine-arg overrides into profileOverrides: ${ENGINE_ARGS[*]}"
   fi
   say "applying AIMService for ${AIM_MODEL_IMAGE}"
   aimservice=$(cat <<EOF
apiVersion: aim.eai.amd.com/v1alpha2
kind: AIMService
metadata:
  name: aim-smoke
  namespace: ${NAMESPACE}
  annotations:
    aim.eai.amd.com/reconciler-pipeline: profile
spec:
  model:
    image: ${AIM_MODEL_IMAGE}
  resources:
    requests:
      cpu: "${AIM_CPU_REQUEST}"
      memory: ${AIM_MEM_REQUEST}
      amd.com/gpu: "${AIM_ACCELERATOR_COUNT}"
    limits:
      cpu: "${AIM_CPU_LIMIT}"
      memory: ${AIM_MEM_LIMIT}
      amd.com/gpu: "${AIM_ACCELERATOR_COUNT}"
  profile:
    selector:
      acceleratorCount: ${AIM_ACCELERATOR_COUNT}
      minimumType: any
${profile_overrides}
EOF
)
   # Server-side apply (no client-side GET or OpenAPI download) so the server
   # computes the merge; retried because OIDC discovery round-trips can 401.
   applied=""
   for _ in $(seq 10); do
      printf '%s\n' "${aimservice}" \
         | kubectl apply --request-timeout=30s --server-side --force-conflicts --validate=false -f - && { applied=1; break; }
      sleep 2
   done
   [ -n "${applied}" ] || fatal "AIMService apply failed."
   # Overlay the user's YAML merge patch (any spec field) on top of the base apply.
   # Applied every run so it survives regeneration; same retry rationale as above.
   if [ -n "${AIM_OVERRIDES_FILE}" ]; then
      say "applying overrides from ${AIM_OVERRIDES_FILE}"
      patched=""
      for _ in $(seq 10); do
         kubectl patch aimservice aim-smoke -n "${NAMESPACE}" --request-timeout=30s \
            --type merge --patch-file "${AIM_OVERRIDES_FILE}" && { patched=1; break; }
         sleep 2
      done
      [ -n "${patched}" ] || fatal "AIMService overrides patch failed."
   fi
   # AIM Engine sometimes fails to re-queue the AIMService when its AIMProfileCache
   # object reaches Ready, leaving it parked in Starting with no InferenceService.
   # A detached one-shot watcher forces a reconcile (annotation bump) once the cache
   # is Ready, so users need not nudge by hand. Disable with AIM_AUTO_NUDGE=0.
   if [ "${AIM_AUTO_NUDGE}" = "1" ]; then
      (
         # Gate only on the AIMService's own conditions: a namespaced identity that
         # cannot list aimprofilecache/inferenceservice (common on multi-tenant
         # clusters) can still read these, so the nudge fires regardless of RBAC.
         for _ in $(seq 1 240); do
            sleep 15
            conds=$(kubectl get aimservice aim-smoke -n "${NAMESPACE}" \
               -o jsonpath='{range .status.conditions[*]}|{.type}={.status}{end}|' 2>/dev/null)
            case "${conds}" in
               *"|Ready=True|"*) exit 0 ;;                 # serving; nothing left to do
               *InferenceServicePodsReady=*) exit 0 ;;     # InferenceService exists; stall already cleared
            esac
            kubectl annotate aimservice aim-smoke -n "${NAMESPACE}" \
               kick="$(date +%s)" --overwrite >/dev/null 2>&1
         done
      ) >/tmp/aim-nudge.log 2>&1 &
      disown 2>/dev/null || true
      say "auto-nudge watcher running (pid $!): clears the known cache-ready stall. Disable with AIM_AUTO_NUDGE=0."
   fi
   echo ""
   say "AIMService applied. Verify level ${LEVEL}:"
   echo ""
   echo -e "${C_HEAD}1) Check readiness${C_OFF} (first pull can take many minutes):"
   cat <<EOF
  kubectl get aimservice aim-smoke -n ${NAMESPACE} -o custom-columns='NAME:.metadata.name,READY:.status.conditions[?(@.type=="Ready")].status,REASON:.status.conditions[?(@.type=="Ready")].reason'
  kubectl describe aimservice aim-smoke -n ${NAMESPACE} | grep -iE 'reason:|message:' | tail -n2
EOF
   echo ""
   echo -e "${C_HEAD}2) Wait for Ready, then run an inference${C_OFF} (served under the real model id, not aim-smoke):"
   cat <<EOF
  kubectl wait --for=condition=Ready aimservice/aim-smoke -n ${NAMESPACE} --timeout=1800s
  svc=\$(kubectl get svc -n ${NAMESPACE} -l aim.eai.amd.com/service.name=aim-smoke,component=predictor -o name | head -n1)
  kubectl port-forward -n ${NAMESPACE} \$svc 8080:80 >/tmp/pf.log 2>&1 &
  sleep 3
  model=\$(curl -sS localhost:8080/v1/models | grep -o '"id":"[^"]*"' | head -n1 | cut -d'"' -f4)
  curl -sS localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"\$model\",\"messages\":[{\"role\":\"user\",\"content\":\"What is ROCm?\"}],\"max_tokens\":200}"
EOF
   echo ""
   echo -e "${C_HEAD}3) Confirm the GPU is in use${C_OFF}:"
   cat <<EOF
  pod=\$(kubectl get pods -n ${NAMESPACE} -l aim.eai.amd.com/service.name=aim-smoke -o name | head -n1)
  kubectl exec -n ${NAMESPACE} \$pod -- rocm-smi
EOF
   echo ""
   echo -e "${C_HEAD}4) Clean up when done${C_OFF}:"
   cat <<EOF
  kubectl delete aimservice aim-smoke -n ${NAMESPACE}
EOF
   echo ""
}

case "${LEVEL}" in
   1) HF_TOKEN="${HF_TOKEN}" exec "${HERE}/aim_base_check.sh" --namespace "${NAMESPACE}" "${BASE_ARGS[@]}" ;;
   2) serve_operator ;;
   3) "${HERE}/aim_engine_setup.sh" "${SETUP_ARGS[@]}" || exit $?; serve_operator ;;
   4) "${HERE}/aim_engine_setup.sh" --install-prereqs 1 "${SETUP_ARGS[@]}" || exit $?; serve_operator ;;
   *) send-error "invalid level '${LEVEL}'; choose 1, 2, 3, or 4." ;;
esac
