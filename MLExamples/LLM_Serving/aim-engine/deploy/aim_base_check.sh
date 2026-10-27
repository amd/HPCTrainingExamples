#!/bin/bash

# --------------------------------------------------------------------------
# Minimal, operator-less AIM serving check. Runs an AIM container as a plain
# Deployment + Service instead of an AIMService, to prove a cluster's GPU can
# serve a model BEFORE committing to the full AIM Engine operator stack
# (aim_engine_setup.sh). The default is deliberately ungated: the generic base
# image plus a small open model (Qwen2.5-1.5B-Instruct), so no Hugging Face token
# is ever required. Point --image at a model-specific image (dropping --model-id)
# to run the exact container the operator's by-image predictor would run.
#
# Scope: this matches the operator only on the RUNTIME (image, GPU detection,
# weight download, engine start). It does NOT exercise the operator's profile
# RESOLUTION, which keys off node labels (aim-accelerator.<MODEL> AND
# partitioning-scheme.default). A green check here is necessary but not
# sufficient: the label-driven resolution is validated only by the full operator
# flow (aim_engine_test.sh --auto-run 1).
#
# It needs only its own small prerequisites: a reachable cluster and a node that
# advertises amd.com/gpu. It uses NO CRDs, NO operator, and NO accelerator node
# labels; the image detects the GPU in-container. HF_TOKEN is used only if set,
# for the rare case of deliberately choosing a gated model.
#
# This runs an AIM container directly (the microservice), not AIM Engine (the
# operator). For how the two relate, see the reference stacks overview:
#   https://enterprise-ai.docs.amd.com/en/latest/reference-stacks.html
# --------------------------------------------------------------------------

# Ungated by default: generic base image + a small open model, so no token is needed.
: ${IMAGE:=amdenterpriseai/aim-base:0.11}
# Model id for a base image (AIM_MODEL_ID). Leave empty when --image is a
# model-specific image, since those set AIM_ID themselves.
: ${MODEL_ID:=Qwen/Qwen2.5-1.5B-Instruct}
: ${HF_TOKEN:=}
: ${NAMESPACE:=default}
# Plain assignment, not ": ${NAME:=...}", so a stray NAME in the environment
# cannot shadow it (only the --name flag overrides). Kubernetes names must be
# lowercase RFC1123, which a hostname-derived value would violate.
NAME="aim-base-check"
: ${GPU_COUNT:=1}
: ${MEM_REQUEST:=16Gi}
: ${CPU_REQUEST:=4}
: ${READY_TIMEOUT:=1800}   # seconds; first-run weight download can be slow
: ${PROGRESS_INTERVAL:=15} # seconds between progress lines while waiting
KEEP=0
VERBOSE=0

usage()
{
   echo "Usage:"
   echo "  --image [ IMAGE ] AIM image to serve, default ${IMAGE}"
   echo "  --model-id [ ORG/MODEL ] only for a bare aim-base image: HF model to serve"
   echo "  --namespace [ NS ] namespace to deploy into, default ${NAMESPACE}"
   echo "  --name [ NAME ] deployment/service name, default ${NAME}"
   echo "  --keep [ 0|1 ] 1: leave the Deployment/Service running, default 0 (clean up)"
   echo "  --verbose [ 0|1 ] 1: also tail recent pod logs in progress output, default 0"
   echo "  --kubeconfig [ PATH ] kubeconfig to use for this run (exports KUBECONFIG)"
   echo "  --help: print this usage information"
   echo ""
   echo "Env: HF_TOKEN (gated models), IMAGE, MODEL_ID, GPU_COUNT, READY_TIMEOUT, PROGRESS_INTERVAL."
}

# Print the reason AFTER the usage block (usage() does not exit) so it is the
# last line the user sees, then fail.
send-error() { usage; echo -e "\nError: ${@}" >&2; exit 1; }

# Colored, spaced status output; disabled when stdout is not a TTY so redirected
# or piped logs stay plain.
if [ -t 1 ]; then
   C_TAG=$'\033[1;36m'; C_HEAD=$'\033[1;36m'; C_OK=$'\033[1;32m'; C_ERR=$'\033[1;31m'; C_OFF=$'\033[0m'
else
   C_TAG=''; C_HEAD=''; C_OK=''; C_ERR=''; C_OFF=''
fi
say() { echo -e "${C_TAG}[base-check]${C_OFF} ${*}"; }

while [[ $# -gt 0 ]]; do
   case "${1}" in
      "--image")     shift; IMAGE=${1} ;;
      "--model-id")  shift; MODEL_ID=${1} ;;
      "--namespace") shift; NAMESPACE=${1} ;;
      "--name")      shift; NAME=${1} ;;
      "--keep")      shift; KEEP=${1} ;;
      "--verbose")   shift; VERBOSE=${1} ;;
      "--kubeconfig") shift; [ -f "${1}" ] || send-error "kubeconfig file not found :: ${1}"; export KUBECONFIG="${1}" ;;
      "--help")      usage; exit 0 ;;
      *)             send-error "Unsupported argument :: ${1}" ;;
   esac
   shift
done

command -v kubectl >/dev/null 2>&1 || send-error "kubectl not found on PATH."
# An expired OIDC token makes the first kubectl call block on a browser re-login;
# on a headless host that looks like a silent hang, so warn up front.
say "checking cluster access (if this hangs, your login likely expired: run 'kubectl get nodes' to re-authenticate, then retry)"
# Key off returned node names, not the exit code: OIDC discovery bursts 401 then
# recover, so retry; only if none ever come back do we surface the real reason
# (expired login, wrong KUBECONFIG) instead of masking it as a missing GPU.
nodes=""
for _ in $(seq 10); do
   nodes=$(kubectl get nodes --request-timeout=15s -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)
   [ -n "${nodes}" ] && break
   sleep 2
done
[ -n "${nodes}" ] \
   || send-error "cannot reach a Kubernetes cluster (check KUBECONFIG / login) :: $(kubectl get nodes 2>&1 >/dev/null)"

# Prerequisite for this simple check: some node advertises amd.com/gpu.
gpu_ok=0
for node in ${nodes}; do
   q=$(kubectl get node "${node}" --request-timeout=15s -o jsonpath='{.status.allocatable.amd\.com/gpu}' 2>/dev/null)
   [ -n "${q}" ] && [ "${q}" != "0" ] && { gpu_ok=1; break; }
done
[ "${gpu_ok}" = "1" ] \
   || send-error "no node advertises amd.com/gpu; install the AMD GPU device plugin (or GPU Operator) first."

kubectl get namespace "${NAMESPACE}" >/dev/null 2>&1 || kubectl create namespace "${NAMESPACE}"

env_block=$'\n        env:'
[ -n "${MODEL_ID}" ] && env_block+=$'\n        - name: AIM_MODEL_ID\n          value: "'"${MODEL_ID}"'"'
if [ -n "${HF_TOKEN}" ]; then
   say "configuring HF_TOKEN secret for gated model access"
   kubectl create secret generic "${NAME}-hf" -n "${NAMESPACE}" \
      --from-literal=hf-token="${HF_TOKEN}" --dry-run=client -o yaml | kubectl apply --request-timeout=30s --server-side --force-conflicts --validate=false -f -
   env_block+=$'\n        - name: HF_TOKEN\n          valueFrom:\n            secretKeyRef:\n              name: '"${NAME}"$'-hf\n              key: hf-token'
fi
[ "${env_block}" = $'\n        env:' ] && env_block=""

cleanup() {
   # Always drop the transient port-forward, even with --keep, so it does not
   # linger holding port 8000 after we exit.
   [ -n "${pf}" ] && kill "${pf}" >/dev/null 2>&1 || true
   [ "${KEEP}" = "1" ] && { say "--keep set; leaving ${NAME} running in ${NAMESPACE}"; return; }
   say "cleaning up ${NAME} in ${NAMESPACE}"
   kubectl delete deployment,service "${NAME}" -n "${NAMESPACE}" >/dev/null 2>&1 || true
   [ -n "${HF_TOKEN}" ] && kubectl delete secret "${NAME}-hf" -n "${NAMESPACE}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

say "deploying ${IMAGE} (amd.com/gpu=${GPU_COUNT}${MODEL_ID:+, AIM_MODEL_ID=${MODEL_ID}})"
manifest=$(cat <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${NAME}
  labels: { app: ${NAME} }
spec:
  progressDeadlineSeconds: ${READY_TIMEOUT}
  replicas: 1
  selector:
    matchLabels: { app: ${NAME} }
  template:
    metadata:
      labels: { app: ${NAME} }
    spec:
      containers:
      - name: ${NAME}
        image: "${IMAGE}"
        imagePullPolicy: Always${env_block}
        ports:
        - { name: http, containerPort: 8000 }
        resources:
          requests: { memory: "${MEM_REQUEST}", cpu: "${CPU_REQUEST}", amd.com/gpu: "${GPU_COUNT}" }
          limits:   { memory: "${MEM_REQUEST}", cpu: "${CPU_REQUEST}", amd.com/gpu: "${GPU_COUNT}" }
        startupProbe:
          httpGet: { path: /v1/models, port: http }
          periodSeconds: 10
          failureThreshold: $(( READY_TIMEOUT / 10 ))
        readinessProbe:
          httpGet: { path: /v1/models, port: http }
        livenessProbe:
          httpGet: { path: /health, port: http }
        volumeMounts:
        - { name: dshm, mountPath: /dev/shm }
      volumes:
      - name: dshm
        emptyDir: { medium: Memory, sizeLimit: 8Gi }
---
apiVersion: v1
kind: Service
metadata:
  name: ${NAME}
  labels: { app: ${NAME} }
spec:
  type: ClusterIP
  ports:
  - { name: http, port: 80, targetPort: 8000 }
  selector: { app: ${NAME} }
EOF
)
# Server-side apply (no client-side GET or OpenAPI download) so the server
# computes the merge; retried because OIDC discovery round-trips can 401.
applied=""
for _ in $(seq 10); do
   printf '%s\n' "${manifest}" \
      | kubectl apply --request-timeout=30s --server-side --force-conflicts --validate=false -n "${NAMESPACE}" -f - && { applied=1; break; }
   sleep 2
done
[ -n "${applied}" ] || send-error "deploy failed."

say "waiting for the model to become Ready (weight download can take a while)"
progress_start=$(date +%s)
progress_loop()
{
   while true; do
      sleep "${PROGRESS_INTERVAL}"
      el=$(( $(date +%s) - progress_start ))
      pod=$(kubectl get pod -n "${NAMESPACE}" -l app="${NAME}" --no-headers 2>/dev/null | head -n1)
      say "+${el}s :: ${pod:-<no pod scheduled yet>}"
      if [ "${VERBOSE}" = "1" ]; then
         kubectl logs -n "${NAMESPACE}" "deploy/${NAME}" --tail=2 2>/dev/null \
            | sed 's/^/[base-check]   log: /'
      fi
   done
}
progress_loop &
progress_pid=$!
stop-progress() { kill "${progress_pid}" 2>/dev/null; wait "${progress_pid}" 2>/dev/null; }

# Poll for readiness instead of `kubectl rollout status`: on a flaky OIDC cluster
# that single watch call can catch a transient 401 and exit early, which would
# look like a failed rollout. Treat an empty/failed read as "not ready yet" and
# only give up once READY_TIMEOUT actually elapses.
deadline=$(( $(date +%s) + READY_TIMEOUT ))
ready=""
while [ "$(date +%s)" -lt "${deadline}" ]; do
   avail=$(kubectl get deployment "${NAME}" -n "${NAMESPACE}" --request-timeout=15s \
             -o jsonpath='{.status.availableReplicas}' 2>/dev/null)
   [ "${avail}" = "1" ] && { ready=1; break; }
   sleep "${PROGRESS_INTERVAL}"
done
if [ -z "${ready}" ]; then
   stop-progress
   say "${C_ERR}pod did not become Ready within ${READY_TIMEOUT}s${C_OFF}; recent events and logs:"
   kubectl describe deployment/"${NAME}" -n "${NAMESPACE}" 2>/dev/null | sed -n '/Events:/,$p'
   kubectl logs -n "${NAMESPACE}" "deploy/${NAME}" --tail=50 2>/dev/null || true
   send-error "the deployment never became Ready."
fi
stop-progress

# Bring up the port-forward and wait until the endpoint actually answers. On a
# slow or flaky cluster the tunnel can take longer than a fixed sleep to bind, or
# die on a transient auth blip, so we poll /v1/models and re-establish if the
# forwarder has exited. Keep its log so a real failure is visible.
pf_log=$(mktemp)
pf=""
serve_ok=""
for _ in $(seq 1 12); do
   if [ -z "${pf}" ] || ! kill -0 "${pf}" 2>/dev/null; then
      kubectl port-forward -n "${NAMESPACE}" "svc/${NAME}" 8000:80 >"${pf_log}" 2>&1 &
      pf=$!
   fi
   sleep 5
   curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && { serve_ok=1; break; }
done
if [ -z "${serve_ok}" ]; then
   say "${C_ERR}could not reach http://localhost:8000 via port-forward${C_OFF}; port-forward log:"
   sed 's/^/[base-check]   pf: /' "${pf_log}" 2>/dev/null
   send-error "port-forward to the served endpoint failed."
fi

# The served model name is whatever the image resolved; read it from /v1/models.
served=$(curl -sS http://localhost:8000/v1/models | tr ',' '\n' | sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | head -n1)
[ -z "${served}" ] && served="${MODEL_ID}"
say "served model id: ${served:-<unknown>}"

resp=$(curl -sS http://localhost:8000/v1/completions \
   -H "Content-Type: application/json" \
   -d '{"model":"'"${served}"'","prompt":"San Francisco is a","max_tokens":7,"temperature":0}')

if echo "${resp}" | grep -q '"choices"'; then
   say "${C_OK}PASS:${C_OFF} model served a completion."
   echo "${resp}"
   say "confirming vLLM placed its KV cache on the GPU:"
   gpu_metric=$(curl -sS http://localhost:8000/metrics 2>/dev/null | grep -E 'cache_usage_perc' | grep -v '^#' | head -n1)
   if [ -n "${gpu_metric}" ]; then
      say "vLLM GPU KV-cache metric present: ${gpu_metric}"
   elif kubectl logs -n "${NAMESPACE}" "deploy/${NAME}" 2>/dev/null | grep -iE 'GPU KV cache|GPU blocks' | tail -n1; then
      :
   else
      say "NOTE: no vLLM GPU signal from /metrics or logs; GPU use unconfirmed."
   fi
   echo ""
   say "${C_OK}Level 1 verified${C_OFF}: inference returned a completion and vLLM is using the GPU."
   echo ""
   if [ "${KEEP}" = "1" ]; then
      echo -e "${C_HEAD}Probe the running deployment${C_OFF} (left up by --keep 1):"
   else
      echo -e "${C_HEAD}Probe it yourself${C_OFF} (re-run with --keep 1 first):"
   fi
   cat <<EOF
  kubectl port-forward -n ${NAMESPACE} svc/${NAME} 8000:80 >/tmp/pf.log 2>&1 &
  sleep 3
  curl -sS localhost:8000/metrics | grep cache_usage_perc
  curl -sS localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"${served}","messages":[{"role":"user","content":"What is ROCm?"}],"max_tokens":200}'
EOF
   echo ""
   if [ "${KEEP}" = "1" ]; then
      echo -e "${C_HEAD}Clean up when done${C_OFF}:"
      cat <<EOF
  kubectl delete deployment,service ${NAME} -n ${NAMESPACE}
EOF
      echo ""
   fi
   exit 0
else
   say "${C_ERR}FAIL:${C_OFF} no valid completion. Response: ${resp}"
   exit 1
fi
