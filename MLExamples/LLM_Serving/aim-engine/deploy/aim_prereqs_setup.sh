#!/bin/bash

# --------------------------------------------------------------------------
# Install the cluster-scoped add-ons that AIM Engine depends on, EXCEPT
# the AMD GPU Operator/driver (assumed already present on GPU nodes, the
# same way the other repo scripts assume base packages are installed).
#
# These add-ons back the AIM Engine (operator) layer. For how AIMs and AIM
# Engine fit the wider stack, see the reference stacks overview:
#   https://enterprise-ai.docs.amd.com/en/latest/reference-stacks.html
#
# Installs (all idempotent via `helm upgrade --install` / `kubectl apply`):
#   cert-manager, Gateway API CRDs, kgateway, KServe, KEDA,
#   keda-otel-add-on, OpenTelemetry Operator.
#
# Off by default in aim_engine_setup.sh (that script only checks); run this
# directly, or via `aim_engine_setup.sh --install-prereqs 1`, to install.
#
# Coordinates/versions verified against the AIM Engine docs and each chart's
# registry (2026-08-17). AIM floors: KServe v0.16.1, Gateway API v1.3.0,
# kgateway v2.0+, cert-manager v1.16+, KEDA 2.18+, OTel Operator 0.101+.
# Override any of these via environment.
# --------------------------------------------------------------------------

# Pinned versions (override via environment).
: ${CERT_MANAGER_VERSION:=v1.16.2}
: ${GATEWAY_API_VERSION:=v1.3.0}
: ${KGATEWAY_VERSION:=v2.4.2}
# AIM cites a v0.16.1 floor, but the only published 0.16 KServe OCI chart is v0.16.0.
: ${KSERVE_VERSION:=v0.16.0}
: ${KSERVE_NAMESPACE:=kserve-system}
# Standard mode (no Knative). v0.16 accepts the legacy value RawDeployment; the
# newer 'Standard' alias is rejected in this release.
: ${KSERVE_DEPLOYMENT_MODE:=RawDeployment}
: ${KEDA_CHART_VERSION:=2.18.3}
: ${KEDA_OTEL_VERSION:=v0.1.4}
: ${OTEL_OPERATOR_VERSION:=0.121.0}

DRY_RUN=0

usage()
{
   echo "Usage:"
   echo "  --dry-run [ 0|1 ] print the actions without applying, default $DRY_RUN"
   echo "  --kubeconfig [ PATH ] kubeconfig to use for this run (exports KUBECONFIG)"
   echo "  --help: print this usage information"
   echo ""
   echo "Versions/namespaces are set via env vars: CERT_MANAGER_VERSION,"
   echo "  GATEWAY_API_VERSION, KGATEWAY_VERSION, KSERVE_VERSION, KSERVE_NAMESPACE,"
   echo "  KSERVE_DEPLOYMENT_MODE, KEDA_CHART_VERSION, KEDA_OTEL_VERSION, OTEL_OPERATOR_VERSION."
}

# Print the reason AFTER the usage block (usage() does not exit) so it is the
# last line the user sees, then fail.
send-error() { usage; echo -e "\nError: ${@}" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
   case "${1}" in
      "--dry-run") shift; DRY_RUN=${1} ;;
      "--kubeconfig") shift; [ -f "${1}" ] || send-error "kubeconfig file not found :: ${1}"; export KUBECONFIG="${1}" ;;
      "--help")    usage; exit 0 ;;
      *)           send-error "Unsupported argument :: ${1}" ;;
   esac
   shift
done

command -v kubectl >/dev/null 2>&1 || send-error "kubectl not found on PATH."
command -v helm    >/dev/null 2>&1 || send-error "helm not found on PATH."
kubectl cluster-info >/dev/null 2>&1 || send-error "cannot reach a Kubernetes cluster (check KUBECONFIG / context)."

run() {
   echo "+ $*"
   [ "${DRY_RUN}" = "1" ] && return 0
   "$@"
}

ver_args() { [ -n "${1}" ] && printf -- '--version\n%s\n' "${1}"; }

echo "== [1/7] cert-manager ${CERT_MANAGER_VERSION} =="
run helm repo add jetstack https://charts.jetstack.io >/dev/null 2>&1 || true
run helm repo update jetstack >/dev/null 2>&1 || true
run helm upgrade --install cert-manager jetstack/cert-manager \
   --namespace cert-manager --create-namespace \
   --version "${CERT_MANAGER_VERSION}" --set crds.enabled=true

echo "== [2/7] Gateway API CRDs ${GATEWAY_API_VERSION} =="
run kubectl apply -f \
   "https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml"

echo "== [3/7] kgateway ${KGATEWAY_VERSION} =="
run helm upgrade --install kgateway-crds \
   oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds \
   --version "${KGATEWAY_VERSION}" --namespace kgateway-system --create-namespace
run helm upgrade --install kgateway \
   oci://cr.kgateway.dev/kgateway-dev/charts/kgateway \
   --version "${KGATEWAY_VERSION}" --namespace kgateway-system --create-namespace

# AIM Engine needs KServe in Standard mode (native Kubernetes Deployments, no
# Knative), since these prerequisites ship Gateway API + kgateway rather than
# Knative Serving. Without the mode set, the controller defaults to Serverless
# and rejects every InferenceService (ServerlessModeRejected: Knative Services
# not available). For external routing/localmodel tuning a real cluster may also
# apply AIM's kserve-values.yaml (kserve-configuration docs) via `-f`.
echo "== [4/7] KServe ${KSERVE_VERSION} (ns ${KSERVE_NAMESPACE}, mode ${KSERVE_DEPLOYMENT_MODE}) =="
run helm upgrade --install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd \
   --version "${KSERVE_VERSION}" --namespace "${KSERVE_NAMESPACE}" --create-namespace
run helm upgrade --install kserve oci://ghcr.io/kserve/charts/kserve \
   --version "${KSERVE_VERSION}" --namespace "${KSERVE_NAMESPACE}" --create-namespace \
   --set "kserve.controller.deploymentMode=${KSERVE_DEPLOYMENT_MODE}"

echo "== [5/7] KEDA (chart ${KEDA_CHART_VERSION}) =="
run helm repo add kedacore https://kedacore.github.io/charts >/dev/null 2>&1 || true
run helm repo update kedacore >/dev/null 2>&1 || true
run helm upgrade --install keda kedacore/keda \
   --namespace keda --create-namespace --version "${KEDA_CHART_VERSION}"

# kedify/otel-add-on; release name 'keda-otel-scaler' in the keda namespace
# is what AIM's scaleFromZero.scalerAddress (keda-otel-scaler.keda.svc) expects.
echo "== [6/7] keda-otel-add-on ${KEDA_OTEL_VERSION:-latest} =="
run helm upgrade --install keda-otel-scaler \
   oci://ghcr.io/kedify/charts/otel-add-on \
   --namespace keda --create-namespace $(ver_args "${KEDA_OTEL_VERSION}")

echo "== [7/7] OpenTelemetry Operator ${OTEL_OPERATOR_VERSION:-latest} =="
run helm repo add open-telemetry \
   https://open-telemetry.github.io/opentelemetry-helm-charts >/dev/null 2>&1 || true
run helm repo update open-telemetry >/dev/null 2>&1 || true
run helm upgrade --install opentelemetry-operator \
   open-telemetry/opentelemetry-operator \
   --namespace opentelemetry-operator-system --create-namespace \
   $(ver_args "${OTEL_OPERATOR_VERSION}") \
   --set "manager.collectorImage.repository=otel/opentelemetry-collector-k8s"

# Wait for the operators (esp. the OTel operator webhook) to be Ready, else a
# subsequent AIM Engine install racing the webhook fails with 'connection refused'.
echo "== waiting for operator deployments to become Available =="
for ns in cert-manager kgateway-system kserve-system keda opentelemetry-operator-system; do
   kubectl get ns "${ns}" >/dev/null 2>&1 || continue
   run kubectl wait --for=condition=Available deploy --all -n "${ns}" --timeout=240s || true
done

echo ""
echo "[aim-prereqs] done. Re-run aim_engine_setup.sh to preflight and install AIM Engine."
