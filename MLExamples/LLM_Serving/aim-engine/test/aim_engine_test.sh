#!/bin/bash

# --------------------------------------------------------------------------
# End-to-end correctness harness for the AIM Engine scripts on a throwaway
# `kind` (Kubernetes IN Docker) cluster with REAL AMD GPU passthrough.
#
# The host's /dev/kfd + /dev/dri are bind-mounted into the kind node (same
# idea as `docker run --device`), and the ROCm k8s device plugin advertises
# amd.com/gpu -- so preflight, install, and inference all run for real. A
# GPU host is required; there is no no-GPU mode.
#
# This exercises both layers: --base-image-only runs an AIM container directly
# (the microservice) on kind, the default flow drives AIM Engine (the operator).
# --container-only skips Kubernetes entirely and runs the AIM container via
# docker/podman, so users without sudo (and hosts on cgroup v1, where kind's
# rootless provider refuses to start) can still validate GPU serving of an AIM.
# For how AIMs and AIM Engine relate, see the reference stacks overview:
#   https://enterprise-ai.docs.amd.com/en/latest/reference-stacks.html
#
# Everything lives in a visible working dir that is removed on exit (when the
# interactive shell or the auto-run flow ends) so nothing is left in ~.
# --------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# The deployment scripts this harness exercises live in the sibling deploy/ dir.
DEPLOY_DIR="$(cd "${HERE}/../deploy" 2>/dev/null && pwd -P)"
[ -n "${DEPLOY_DIR}" ] || { echo "[test] ERROR: expected sibling deploy/ dir at ${HERE}/../deploy" >&2; exit 1; }

AUTO_RUN=0
BASE_IMAGE_ONLY=0
CONTAINER_ONLY=0
CLUSTER_NAME="aim-engine-test"
WORK_DIR="${HOME}/aim-engine-test"

: ${KIND_VERSION:=v0.27.0}       # ships a k8s 1.32 node image (AIM needs 1.32+)
: ${KUBECTL_VERSION:=v1.32.2}
: ${HELM_VERSION:=v3.16.2}
: ${AMDGPU_DP_URL:=https://raw.githubusercontent.com/ROCm/k8s-device-plugin/master/k8s-ds-amdgpu-dp.yaml}
# Model image for the operator inference smoke test. Ungated by default (no HF_TOKEN needed).
: ${AIM_TEST_MODEL_IMAGE:=amdenterpriseai/aim-qwen-qwen3-32b:0.13.0}
# In-cluster NFS provisioner: gives the throwaway cluster a dynamic ReadWriteMany
# default StorageClass, mirroring the NFS a real system provides. The AIM cache
# PVC is RWX, which the single-node local-path default cannot satisfy.
: ${NFS_CHART_REPO:=https://kubernetes-sigs.github.io/nfs-ganesha-server-and-external-provisioner/}
: ${NFS_CHART_VERSION:=}
: ${NFS_STORAGE_SIZE:=100Gi}
# Kubernetes-free (--container-only) path: run the AIM container directly. Small
# ungated defaults so it is fast and needs no token. Works rootless on cgroup v1.
: ${DIRECT_IMAGE:=amdenterpriseai/aim-base:0.11}
: ${DIRECT_MODEL_ID:=Qwen/Qwen2.5-0.5B-Instruct}
# Force tensor-parallel-size 1 so a small model's attention-head count (e.g. 14 or
# 12) need not divide the host's GPU count (often 8). AIM derives TP from its own
# accelerator detection, not from device/env visibility, so we set the runtime's
# AIM_ACCELERATOR_COUNT directly and pin execution to GPU 0.
: ${DIRECT_GPU_COUNT:=1}
: ${DIRECT_VISIBLE_DEVICES:=0}
# AIM accelerator model used to label the kind node so profile resolution matches
# (the bare device plugin doesn't set the label a real AMD GPU Operator would).
# Auto-detected from rocminfo if empty; override with --gpu-model.
: ${AIM_GPU_MODEL:=}
# Hugging Face token for gated models (Llama/Gemma). If set, the harness creates
# the secret + a default AIMRuntimeConfig that injects HF_TOKEN into the pods.
: ${HF_TOKEN:=}

usage()
{
   echo "Usage:"
   echo "  --auto-run [ 0|1 ] 0: bring up kind+GPU and print manual commands (default);"
   echo "                     1: run preflight/install/idempotency + inference, then clean up"
   echo "  --base-image-only [ 0|1 ] 1: skip the operator; run only the minimal"
   echo "                     aim_base_check.sh serve test (fast GPU-serves-a-model check)"
   echo "  --container-only [ 0|1 ] 1: no Kubernetes at all; run the AIM container"
   echo "                     directly via docker/podman (no sudo, works on cgroup v1)."
   echo "                     Validates GPU serving of an AIM, not the operator/k8s scripts."
   echo "                     Combine with --auto-run 1 to check and tear down; the"
   echo "                     default (0) leaves it serving and drops you into a shell."
   echo "  --cluster-name [ NAME ] kind cluster name, default $CLUSTER_NAME"
   echo "  --gpu-model [ MODEL ] AIM accelerator model to label the node with"
   echo "                        (e.g. MI355X, MI300X); auto-detected via rocminfo if unset"
   echo "  --help: print this usage information"
   echo ""
   echo "Env: HF_TOKEN (gated models), AIM_TEST_MODEL_IMAGE, AIM_GPU_MODEL,"
   echo "     DIRECT_IMAGE, DIRECT_MODEL_ID, DIRECT_VISIBLE_DEVICES (for --container-only)."
}

# Print the reason AFTER the usage block (usage() does not exit) so it is the
# last line the user sees, then fail.
send-error() { usage; echo -e "\nError: ${@}" >&2; exit 1; }
# Environment/runtime failures: print the reason plainly, no usage block.
fatal() { echo -e "\n[test] ERROR: ${@}" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
   case "${1}" in
      "--auto-run")     shift; AUTO_RUN=${1} ;;
      "--base-image-only") shift; BASE_IMAGE_ONLY=${1} ;;
      "--container-only") shift; CONTAINER_ONLY=${1} ;;
      "--cluster-name") shift; CLUSTER_NAME=${1} ;;
      "--gpu-model")    shift; AIM_GPU_MODEL=${1} ;;
      "--help")         usage; exit 0 ;;
      *)                send-error "Unsupported argument :: ${1}" ;;
   esac
   shift
done

# Pick a container runtime for kind: real Docker if present, else Podman. kind
# only uses Podman when KIND_EXPERIMENTAL_PROVIDER=podman, and Podman also needs
# cgroup v2 (rootless especially); on cgroup v1 kind cannot start a cluster.
if [ "${KIND_EXPERIMENTAL_PROVIDER}" = "podman" ]; then
   RUNTIME=podman
elif command -v docker >/dev/null 2>&1 && ! docker --version 2>/dev/null | grep -qi podman; then
   RUNTIME=docker
elif command -v podman >/dev/null 2>&1; then
   RUNTIME=podman; export KIND_EXPERIMENTAL_PROVIDER=podman
else
   fatal "no container runtime found; install Docker or Podman."
fi
echo "[test] container runtime: ${RUNTIME}"

# Kubernetes-free path: run the AIM container directly, no kind/kubectl/helm and
# no cgroup v2 requirement, so it works rootless on cgroup v1 without sudo. This
# validates the AIM microservice layer (GPU serves a model), not the operator.
if [ "${CONTAINER_ONLY}" = "1" ]; then
   [ -e /dev/kfd ] && [ -d /dev/dri ] || fatal "no AMD GPU on this host (/dev/kfd or /dev/dri missing)."
   cname="aim-direct-$$"
   grp=(--group-add video --group-add render)
   [ "${RUNTIME}" = "podman" ] && grp=(--group-add keep-groups)
   envs=(-e "AIM_MODEL_ID=${DIRECT_MODEL_ID}")
   # AIM derives tensor-parallel size from its own accelerator detection, which
   # device/env visibility does not constrain; set the count it uses directly.
   [ -n "${DIRECT_GPU_COUNT}" ] && envs+=(-e "AIM_ACCELERATOR_COUNT=${DIRECT_GPU_COUNT}")
   [ -n "${DIRECT_VISIBLE_DEVICES}" ] && envs+=(-e "HIP_VISIBLE_DEVICES=${DIRECT_VISIBLE_DEVICES}" -e "ROCR_VISIBLE_DEVICES=${DIRECT_VISIBLE_DEVICES}")
   [ -n "${HF_TOKEN}" ] && envs+=(-e "HF_TOKEN=${HF_TOKEN}")
   rm_container() { "${RUNTIME}" rm -f "${cname}" >/dev/null 2>&1 || true; }
   trap rm_container EXIT
   echo "[test] container-only: ${RUNTIME} run ${DIRECT_IMAGE} (AIM_MODEL_ID=${DIRECT_MODEL_ID}, AIM_ACCELERATOR_COUNT=${DIRECT_GPU_COUNT}), no Kubernetes"
   "${RUNTIME}" run -d --name "${cname}" \
      --device /dev/kfd --device /dev/dri "${grp[@]}" \
      --security-opt seccomp=unconfined -p 8000:8000 "${envs[@]}" "${DIRECT_IMAGE}" \
      || fatal "failed to start the AIM container with ${RUNTIME}."
   echo "[test] waiting for the model to serve on :8000 (weight download can take a while)"
   ok=0
   start=$(date +%s)
   for i in $(seq 1 60); do
      curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && { ok=1; break; }
      # Fail fast if the runtime crashed (e.g. a config validation error) instead
      # of waiting out the whole timeout on a container that will never serve.
      if [ "$("${RUNTIME}" inspect -f '{{.State.Running}}' "${cname}" 2>/dev/null)" != "true" ]; then
         "${RUNTIME}" logs --tail 80 "${cname}" 2>&1
         fatal "the AIM container exited before serving (see logs above)."
      fi
      last=$("${RUNTIME}" logs --tail 1 "${cname}" 2>&1 | tr -d '\r')
      echo "[test] +$(( $(date +%s) - start ))s waiting :: ${last}"
      sleep 30
   done
   [ "${ok}" = "1" ] || { "${RUNTIME}" logs --tail 80 "${cname}" 2>&1; fatal "the model never served /v1/models."; }
   served=$(curl -sS http://localhost:8000/v1/models | tr ',' '\n' | sed -n 's/.*"id":"\([^"]*\)".*/\1/p' | head -n1)
   [ -z "${served}" ] && served="${DIRECT_MODEL_ID}"
   echo "[test] served model id: ${served}"
   resp=$(curl -sS http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
      -d '{"model":"'"${served}"'","prompt":"San Francisco is a","max_tokens":7,"temperature":0}')
   if echo "${resp}" | grep -q '"choices"'; then
      echo "[test] PASS: model served a completion."
      echo "${resp}"
      echo "[test] asking the model 'What is ROCm?' via /v1/chat/completions:"
      chat=$(curl -sS http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
         -d '{"model":"'"${served}"'","messages":[{"role":"user","content":"What is ROCm?"}],"max_tokens":200,"temperature":0}')
      if command -v jq >/dev/null 2>&1; then
         echo "[test] answer: $(echo "${chat}" | jq -r '.choices[0].message.content')"
      else
         echo "${chat}"
      fi
      echo "[test] confirming vLLM placed its KV cache on the GPU:"
      gpu_metric=$(curl -sS http://localhost:8000/metrics 2>/dev/null | grep -E 'cache_usage_perc' | grep -v '^#' | head -n1)
      if [ -n "${gpu_metric}" ]; then
         echo "[test] vLLM GPU KV-cache metric present: ${gpu_metric}"
      elif "${RUNTIME}" logs "${cname}" 2>/dev/null | grep -iE 'GPU KV cache|GPU blocks' | tail -n1; then
         :
      else
         echo "[test] NOTE: no vLLM GPU signal from /metrics or logs; GPU use unconfirmed."
      fi
      # auto-run: report and tear down. Otherwise exec INTO the still-serving
      # container (a throwaway sandbox: edits and deletes stay inside it, the host
      # is untouched) so we can run our own inference; the EXIT trap removes the
      # container when the shell exits.
      [ "${AUTO_RUN}" = "1" ] && exit 0
      cat <<EOF

[test] The AIM model is serving on http://localhost:8000 (OpenAI-compatible API).
[test] Dropping you INTO the container: a throwaway sandbox, so nothing you do here
[test] touches the host, and it is all discarded when you exit. Try:
  curl -sS localhost:8000/v1/models
  curl -sS localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\
    -d '{"model":"${served}","messages":[{"role":"user","content":"What is ROCm?"}],"max_tokens":200}'
[test] 'exit' stops and removes the container.

EOF
      "${RUNTIME}" exec -it "${cname}" bash 2>/dev/null || "${RUNTIME}" exec -it "${cname}" sh
      exit 0
   fi
   echo "[test] FAIL: no valid completion. Response: ${resp}"; exit 1
fi

# cgroup v1: kind only rejects it for the ROOTLESS Podman provider; rootful
# Podman (running as root) still works on cgroup v1 with a deprecation warning
# (kind issue #3915). So we only hard-fail the non-root case here.
if [ "${RUNTIME}" = "podman" ] && [ ! -f /sys/fs/cgroup/cgroup.controllers ]; then
   if [ "$(id -u)" -ne 0 ]; then
      fatal "Rootless Podman + kind needs cgroup v2, but this host is on cgroup v1,
so kind cannot create a cluster here without root. For a no-sudo check, run the
Kubernetes-free container test instead (validates GPU serving of an AIM):
  ${0} --container-only 1
The full operator/k8s scripts need Docker, a cgroup v2 host, or an existing
cluster. See https://kind.sigs.k8s.io/docs/user/rootless/"
   fi
   echo "[test] cgroup v1 with rootful Podman: supported by kind ${KIND_VERSION} (expect a deprecation warning)."
fi
[ -e /dev/kfd ] && [ -d /dev/dri ] || send-error "no AMD GPU on this host (/dev/kfd or /dev/dri missing)."
case "$(uname -m)" in
   x86_64|amd64) ARCH=amd64 ;;
   aarch64|arm64) ARCH=arm64 ;;
   *) send-error "unsupported architecture $(uname -m)." ;;
esac

BIN_DIR="${WORK_DIR}/bin"
mkdir -p "${BIN_DIR}"
export PATH="${BIN_DIR}:${PATH}"
export KUBECONFIG="${WORK_DIR}/kubeconfig"

fetch() { echo "[test] downloading ${1}"; curl -fsSL "${1}" -o "${2}" || send-error "download failed: ${1}"; }

if ! command -v kind >/dev/null 2>&1; then
   fetch "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-${ARCH}" "${BIN_DIR}/kind"; chmod +x "${BIN_DIR}/kind"
fi
if ! command -v kubectl >/dev/null 2>&1; then
   fetch "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${ARCH}/kubectl" "${BIN_DIR}/kubectl"; chmod +x "${BIN_DIR}/kubectl"
fi
if ! command -v helm >/dev/null 2>&1; then
   fetch "https://get.helm.sh/helm-${HELM_VERSION}-linux-${ARCH}.tar.gz" "${BIN_DIR}/helm.tgz"
   tar -xzf "${BIN_DIR}/helm.tgz" -C "${BIN_DIR}" --strip-components=1 "linux-${ARCH}/helm"
   chmod +x "${BIN_DIR}/helm"; rm -f "${BIN_DIR}/helm.tgz"
fi

# kind node runs privileged, so bind-mounting the device nodes is enough.
# Also mount the aim-engine scripts and the kubectl/helm binaries (read-only)
# so they can be run from inside the node container as a throwaway sandbox.
cat > "${WORK_DIR}/kind-gpu.yaml" <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraMounts:
  - hostPath: /dev/kfd
    containerPath: /dev/kfd
  - hostPath: /dev/dri
    containerPath: /dev/dri
  - hostPath: ${DEPLOY_DIR}
    containerPath: /aim-engine
    readOnly: true
  - hostPath: ${BIN_DIR}
    containerPath: /aim-bin
    readOnly: true
EOF

teardown() {
   echo "[test] cleaning up (kind cluster + ${WORK_DIR})"
   kind delete cluster --name "${CLUSTER_NAME}" >/dev/null 2>&1 || true
   rm -rf "${WORK_DIR}"
}

if kind get clusters 2>/dev/null | grep -qx "${CLUSTER_NAME}"; then
   echo "[test] reusing existing kind cluster ${CLUSTER_NAME}"
   kind export kubeconfig --name "${CLUSTER_NAME}" --kubeconfig "${KUBECONFIG}" \
      || fatal "could not export kubeconfig for existing cluster ${CLUSTER_NAME}."
else
   echo "[test] creating kind cluster ${CLUSTER_NAME} (GPU passthrough)"
   kind create cluster --name "${CLUSTER_NAME}" --config "${WORK_DIR}/kind-gpu.yaml" \
      --kubeconfig "${KUBECONFIG}" \
      || fatal "kind create cluster failed (runtime: ${RUNTIME}); if using Podman, see the cgroup v2 notes above."
fi
# From here on any failure/exit tears the cluster (and work dir) down.
trap teardown EXIT

echo "[test] deploying the ROCm k8s device plugin"
curl -fsSL --retry 5 --retry-all-errors --retry-delay 3 -o "${WORK_DIR}/amd-gpu-dp.yaml" "${AMDGPU_DP_URL}" \
   || send-error "could not download the device plugin manifest from ${AMDGPU_DP_URL}"
kubectl apply -f "${WORK_DIR}/amd-gpu-dp.yaml" || send-error "device plugin apply failed."

echo "[test] waiting for a Kubernetes node to advertise amd.com/gpu"
for i in $(seq 1 30); do
   n=$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable.amd\.com/gpu}' 2>/dev/null)
   [ -n "${n}" ] && [ "${n}" != "0" ] && break
   sleep 5
done
[ -n "${n}" ] && [ "${n}" != "0" ] \
   || send-error "no amd.com/gpu became allocatable; check the device plugin pod and host GPU."
echo "[test] amd.com/gpu allocatable = ${n}"

# Base-image-only path: no operator, so skip the accelerator labels and the
# AIMRuntimeConfig (both are operator concepts). Bring up the cluster + device
# plugin above, then hand off to the minimal serve check.
if [ "${BASE_IMAGE_ONLY}" = "1" ]; then
   if [ "${AUTO_RUN}" = "1" ]; then
      echo ""; echo "[test] base-image-only: minimal serve check (no operator, no prereqs)"
      HF_TOKEN="${HF_TOKEN}" "${DEPLOY_DIR}/aim_base_check.sh"; exit $?
   fi
   cat <<EOF

[test] kind cluster '${CLUSTER_NAME}' is UP with amd.com/gpu=${n} (base-image-only).
[test] Dropping you INTO the Kubernetes node container -- a throwaway sandbox.
[test] Run the minimal, operator-less serve check:
  ./aim_base_check.sh
[test] 'exit' tears the whole cluster down and the host is untouched.

EOF
   "${RUNTIME}" exec -it \
      -e PATH="/aim-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
      -e KUBECONFIG=/etc/kubernetes/admin.conf \
      -e HF_TOKEN="${HF_TOKEN}" \
      -e AIM_TEST_MODEL_IMAGE="${AIM_TEST_MODEL_IMAGE}" \
      -w /aim-engine \
      "${CLUSTER_NAME}-control-plane" bash --norc -i
   exit 0
fi

# Give the throwaway cluster a dynamic ReadWriteMany default StorageClass, the
# way a real system's NFS would. The AIM operator's per-profile cache PVC is RWX,
# which the single-node local-path default cannot bind. We install an in-cluster
# NFS server + provisioner and make 'nfs' the default class BEFORE the operator
# creates any PVC, so the unmodified deployment scripts stay storage-agnostic.
echo "[test] installing in-cluster NFS dynamic provisioner (RWX default StorageClass)"
helm repo add nfs-ganesha "${NFS_CHART_REPO}" >/dev/null 2>&1 || true
helm repo update >/dev/null 2>&1 || true
helm upgrade --install nfs-server nfs-ganesha/nfs-server-provisioner \
   --namespace nfs-server --create-namespace \
   ${NFS_CHART_VERSION:+--version "${NFS_CHART_VERSION}"} \
   --set persistence.enabled=true \
   --set persistence.storageClass=standard \
   --set persistence.size="${NFS_STORAGE_SIZE}" \
   --set storageClass.create=true \
   --set storageClass.name=nfs \
   --set storageClass.defaultClass=true \
   --wait --timeout 5m \
   || fatal "NFS provisioner install failed."
# Drop local-path's default flag so 'nfs' is the sole default and RWX PVCs bind to it.
kubectl patch storageclass standard \
   -p '{"metadata":{"annotations":{"storageclass.kubernetes.io/is-default-class":"false"}}}' >/dev/null 2>&1 || true
echo "[test] default StorageClass is now 'nfs' (RWX via in-cluster NFS)"

# Label the node with the GPU model so AIM profile resolution matches. A real
# cluster gets this from the AMD GPU Operator's AcceleratorDetector; the bare
# device plugin used here does not, so we inject the equivalent label.
if [ -z "${AIM_GPU_MODEL}" ] && command -v rocminfo >/dev/null 2>&1; then
   AIM_GPU_MODEL=$(rocminfo 2>/dev/null | grep -m1 -oiE 'MI[0-9]{3}[A-Z]*' | tr 'a-z' 'A-Z')
fi
NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
if [ -n "${AIM_GPU_MODEL}" ]; then
   # The label VALUE must be the GPU count, not a boolean: v1alpha2 profile
   # resolution matches on Exists (value ignored), but the v1alpha1 service-template
   # GPUReady check parses the value as a count, so 'true' yields "Available: none".
   echo "[test] labeling node ${NODE} as accelerator ${AIM_GPU_MODEL}=${n} (unpartitioned)"
   kubectl label node "${NODE}" "feature.node.kubernetes.io/aim-accelerator.${AIM_GPU_MODEL}=${n}" --overwrite
   # v1alpha2 AND-s the model term with a partitioning-scheme term; the default
   # (unpartitioned) profile requires the 'default' sentinel the AcceleratorDetector
   # would stamp. Without it every profile reports HardwareNotAvailable.
   kubectl label node "${NODE}" "feature.node.kubernetes.io/aim-accelerator.partitioning-scheme.default=${n}" --overwrite
else
   echo "[test] WARNING GPU model unknown (pass --gpu-model); AIM profile matching will fail for inference."
fi

# Optional Hugging Face token for gated models: secret + default AIMRuntimeConfig.
if [ -n "${HF_TOKEN}" ]; then
   echo "[test] configuring Hugging Face token (secret + default AIMRuntimeConfig)"
   kubectl create secret generic hf-token -n default \
      --from-literal=hf-token="${HF_TOKEN}" --dry-run=client -o yaml | kubectl apply -f -
   kubectl apply -f - <<EOF
apiVersion: aim.eai.amd.com/v1alpha1
kind: AIMRuntimeConfig
metadata:
  name: default
  namespace: default
spec:
  env:
  - name: HF_TOKEN
    valueFrom:
      secretKeyRef:
        name: hf-token
        key: hf-token
EOF
fi

if [ "${AUTO_RUN}" != "1" ]; then
   cat <<EOF

[test] kind cluster '${CLUSTER_NAME}' is UP with amd.com/gpu=${n}.
[test] Dropping you INTO the Kubernetes node container -- a throwaway sandbox.
[test]   scripts:    /aim-engine  (read-only)
[test]   kubectl/helm: /aim-bin   (on PATH)
[test]   KUBECONFIG:  /etc/kubernetes/admin.conf (this cluster)
[test] Node labeled accelerator: ${AIM_GPU_MODEL:-<none: pass --gpu-model>}
[test] HF token: ${HF_TOKEN:+configured}${HF_TOKEN:-not set (gated models will not download; export HF_TOKEN)}
[test] Mess it up freely; 'exit' tears the whole cluster down and the host is untouched.
[test] This kind cluster is the "from zero" foundation; aim_deploy.sh runs levels
[test] 1-4 on top of it. Pick a level (higher does more, assumes less):

  # L1: base-image serve (no operator, no prereqs):
  ./aim_deploy.sh --level 1

  # L4: install prereqs + AIM Engine, then serve via the operator:
  ./aim_deploy.sh --level 4

  # inspect either way:
  kubectl get aimservice,inferenceservice,pods -n default

EOF
   "${RUNTIME}" exec -it \
      -e PATH="/aim-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
      -e KUBECONFIG=/etc/kubernetes/admin.conf \
      -e HF_TOKEN="${HF_TOKEN}" \
      -e AIM_TEST_MODEL_IMAGE="${AIM_TEST_MODEL_IMAGE}" \
      -w /aim-engine \
      "${CLUSTER_NAME}-control-plane" bash --norc -i
   exit 0
fi

# ------ auto-run: full flow with assertions, always cleaned up on exit ---------------
FAILED=0

echo ""; echo "[test] STEP 1: preflight with no prereqs (expect exit 42)"
"${DEPLOY_DIR}/aim_engine_setup.sh"; rc=$?
if [ "${rc}" -eq 42 ]; then echo "[test] PASS: missing prerequisites reported."
else echo "[test] FAIL: expected exit 42, got ${rc}."; FAILED=1; fi

echo ""; echo "[test] STEP 2: install prereqs + AIM Engine (expect success)"
if "${DEPLOY_DIR}/aim_engine_setup.sh" --install-prereqs 1; then echo "[test] PASS: installed."
else echo "[test] FAIL: install did not succeed."; FAILED=1; fi

echo ""; echo "[test] STEP 3: idempotency (re-run, expect success)"
if "${DEPLOY_DIR}/aim_engine_setup.sh"; then echo "[test] PASS: idempotent."
else echo "[test] FAIL: second run did not succeed."; FAILED=1; fi

echo ""; echo "[test] STEP 4: inference smoke test (${AIM_TEST_MODEL_IMAGE})"
kubectl apply -f - <<EOF
apiVersion: aim.eai.amd.com/v1alpha2
kind: AIMService
metadata:
  name: aim-smoke
  namespace: default
  annotations:
    aim.eai.amd.com/reconciler-pipeline: profile
spec:
  model:
    image: ${AIM_TEST_MODEL_IMAGE}
  profile:
    selector:
      minimumType: any
EOF
echo "[test] waiting for the InferenceService to become Ready (model download can take a while)"
isvc=""
for i in $(seq 1 120); do
   isvc=$(kubectl get inferenceservice -n default -l aim.eai.amd.com/service.name=aim-smoke -o name 2>/dev/null | head -n1)
   [ -n "${isvc}" ] && kubectl wait --for=condition=Ready "${isvc}" -n default --timeout=10s >/dev/null 2>&1 && break
   sleep 10
done
if [ -n "${isvc}" ] && kubectl get "${isvc}" -n default >/dev/null 2>&1; then
   kubectl port-forward -n default "svc/$(basename ${isvc})-predictor" 8080:80 >/dev/null 2>&1 &
   pf=$!; sleep 5
   resp=$(curl -sS http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"aim-smoke","messages":[{"role":"user","content":"Hello!"}]}')
   kill ${pf} >/dev/null 2>&1 || true
   if echo "${resp}" | grep -q '"choices"'; then echo "[test] PASS: inference returned a completion."
   else echo "[test] FAIL: no valid completion. Response: ${resp}"; FAILED=1; fi
else
   echo "[test] FAIL: InferenceService for aim-smoke never became Ready."; FAILED=1
fi

echo ""
if [ "${FAILED}" -eq 0 ]; then echo "[test] RESULT: all AIM Engine harness checks passed."; exit 0
else echo "[test] RESULT: one or more AIM Engine harness checks FAILED."; exit 1; fi
