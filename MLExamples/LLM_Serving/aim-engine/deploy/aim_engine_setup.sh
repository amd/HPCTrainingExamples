#!/bin/bash

# --------------------------------------------------------------------------
# AIM Engine setup: deploy AMD's AIM (AMD Inference Microservices) Engine
# Kubernetes operator onto a cluster that is ALREADY running Kubernetes.
#
# This is the AIM Engine layer (the operator that manages AIMs). For how AIMs
# and AIM Engine fit the wider stack, see the reference stacks overview:
#   https://enterprise-ai.docs.amd.com/en/latest/reference-stacks.html
#
# Unlike the module-based *_setup.sh scripts, this installs a cluster-scoped
# k8s operator (CRDs + Helm chart) via kubectl/helm -- there is no Lmod
# module. "Success" = AIM CRDs Established + operator pod Running.
#
# The 8 add-ons AIM depends on (GPU Operator, KServe, kgateway, Gateway API,
# cert-manager, KEDA, keda-otel-add-on, OpenTelemetry Operator) are NOT
# installed here by default: this script PREFLIGHT-CHECKS them and, if any
# is missing, prints the gap list and exits (MISSING_PREREQ_RC). Pass
# --install-prereqs 1 to run the sibling aim_prereqs_setup.sh first (which
# installs the add-ons EXCEPT the GPU Operator/driver, assumed present).
# --------------------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# AIM Engine chart coordinates (verified against release v0.2.5, 2026-08-17).
# Empty AIM_VERSION => helm resolves the latest published chart.
AIM_NAMESPACE="aim-system"
AIM_CRDS_CHART="oci://docker.io/amdenterpriseai/aim-engine-crds-chart"
AIM_CHART="oci://docker.io/amdenterpriseai/aim-engine-chart"
AIM_VERSION="0.2.5"
INSTALL_PREREQS=0
REPLACE=0

MISSING_PREREQ_RC=42

usage()
{
   echo "Usage:"
   echo "  --namespace [ AIM_NAMESPACE ] operator namespace, default $AIM_NAMESPACE"
   echo "  --aim-version [ AIM_VERSION ] AIM Engine chart version, default latest"
   echo "  --crds-chart [ AIM_CRDS_CHART ] default $AIM_CRDS_CHART"
   echo "  --chart [ AIM_CHART ] default $AIM_CHART"
   echo "  --install-prereqs [ 0|1 ] run aim_prereqs_setup.sh first, default $INSTALL_PREREQS"
   echo "  --replace [ 0|1 ] uninstall an existing AIM Engine release first, default $REPLACE"
   echo "  --kubeconfig [ PATH ] kubeconfig to use for this run (exports KUBECONFIG)"
   echo "  --help: print this usage information"
}

# Print the reason AFTER the usage block (usage() does not exit) so it is the
# last line the user sees, then fail.
send-error()
{
   usage
   echo -e "\nError: ${@}" >&2
   exit 1
}

while [[ $# -gt 0 ]]; do
   case "${1}" in
      "--namespace")        shift; AIM_NAMESPACE=${1} ;;
      "--aim-version")      shift; AIM_VERSION=${1} ;;
      "--crds-chart")       shift; AIM_CRDS_CHART=${1} ;;
      "--chart")            shift; AIM_CHART=${1} ;;
      "--install-prereqs")  shift; INSTALL_PREREQS=${1} ;;
      "--replace")          shift; REPLACE=${1} ;;
      "--kubeconfig")       shift; [ -f "${1}" ] || send-error "kubeconfig file not found :: ${1}"; export KUBECONFIG="${1}" ;;
      "--help")             usage; exit 0 ;;
      *)                    send-error "Unsupported argument :: ${1}" ;;
   esac
   shift
done

command -v kubectl >/dev/null 2>&1 || send-error "kubectl not found on PATH."
command -v helm    >/dev/null 2>&1 || send-error "helm not found on PATH."
kubectl cluster-info >/dev/null 2>&1 || send-error "cannot reach a Kubernetes cluster (check KUBECONFIG / context)."
# If AIM Engine is already installed (its CRD is present), there is nothing for
# levels 3/4 to install. Say so and skip the install path: serving needs only
# namespaced access, so a caller without cluster-admin can still proceed (when
# invoked through aim_deploy.sh it continues to the serve step after this
# returns). --replace forces a reinstall, so it does not short-circuit.
if [ "${REPLACE}" != "1" ] && kubectl get crd aimservices.aim.eai.amd.com >/dev/null 2>&1; then
   echo "[aim] AIM Engine is already installed on this cluster; skipping install."
   echo "[aim] nothing for levels 3/4 to do here (level 2 serves directly); pass --replace 1 to reinstall."
   exit 0
fi
# CRDs + cluster-scoped RBAC require cluster-admin (the k8s 'sudo').
kubectl auth can-i create customresourcedefinitions.apiextensions.k8s.io >/dev/null 2>&1 \
   || send-error "current context lacks cluster-admin (cannot create CRDs); AIM Engine needs cluster-admin."

if [ "${INSTALL_PREREQS}" = "1" ]; then
   echo "[aim] --install-prereqs 1: installing add-ons via aim_prereqs_setup.sh"
   "${HERE}/aim_prereqs_setup.sh" || send-error "aim_prereqs_setup.sh failed; see output above."
fi

# ------ Preflight: every add-on must be present (identified by a signature CRD
#    or workload). Collect ALL gaps, then report once. ------------------------------------------------------
MISSING=()

have_crd() { kubectl get crd "${1}" >/dev/null 2>&1; }
have_crd_like() { kubectl get crd -o name 2>/dev/null | grep -q "${1}"; }
have_gpu_node() {
   kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.amd\.com/gpu}{"\n"}{end}' 2>/dev/null \
      | grep -qE '^[1-9]'
}

have_gpu_node || MISSING+=("AMD GPU pool (no Kubernetes node advertises allocatable amd.com/gpu; is the AMD GPU Operator / device plugin deployed?)")

have_crd "certificates.cert-manager.io"            || MISSING+=("cert-manager")
have_crd "gateways.gateway.networking.k8s.io"      || MISSING+=("Gateway API")
have_crd_like "kgateway"                            || MISSING+=("kgateway")
have_crd "inferenceservices.serving.kserve.io"     || MISSING+=("KServe")
have_crd "scaledobjects.keda.sh"                    || MISSING+=("KEDA")
have_crd "opentelemetrycollectors.opentelemetry.io" || MISSING+=("OpenTelemetry Operator")
# keda-otel-add-on ships no CRD; identify it by its deployment.
kubectl get deploy -A -o name 2>/dev/null | grep -qiE 'otel.*(scaler|add-?on)' \
   || MISSING+=("keda-otel-add-on")

if [ "${#MISSING[@]}" -gt 0 ]; then
   echo ""
   echo "[aim missing-prereq] the following AIM Engine prerequisites are absent:"
   for m in "${MISSING[@]}"; do echo "  - ${m}"; done
   echo ""
   echo "Install them first (e.g. ${HERE}/aim_prereqs_setup.sh for the non-GPU add-ons,"
   echo "and the AMD GPU Operator separately), or re-run with --install-prereqs 1."
   exit ${MISSING_PREREQ_RC}
fi
echo "[aim] preflight OK: all prerequisites present."

# ------ --replace: uninstall an existing release before reinstalling ---------------------
if [ "${REPLACE}" = "1" ]; then
   echo "[aim --replace 1] removing existing AIM Engine releases in ${AIM_NAMESPACE}"
   helm uninstall aim-engine -n "${AIM_NAMESPACE}" 2>/dev/null || true
   helm uninstall aim-engine-crds -n "${AIM_NAMESPACE}" 2>/dev/null || true
fi

# ------ Install AIM Engine: CRDs chart first, then the operator ------------------------------------
VERSION_ARGS=()
[ -n "${AIM_VERSION}" ] && VERSION_ARGS=(--version "${AIM_VERSION}")

echo "[aim] installing AIM Engine CRDs"
helm upgrade --install aim-engine-crds "${AIM_CRDS_CHART}" "${VERSION_ARGS[@]}" \
   --namespace "${AIM_NAMESPACE}" --create-namespace \
   || send-error "installing AIM Engine CRDs failed."
kubectl wait --for=condition=Established crd --all --timeout=60s \
   || send-error "AIM Engine CRDs did not reach Established."

echo "[aim] installing AIM Engine operator"
helm upgrade --install aim-engine "${AIM_CHART}" "${VERSION_ARGS[@]}" \
   --namespace "${AIM_NAMESPACE}" --create-namespace \
   || send-error "installing the AIM Engine operator failed."

echo "[aim] waiting for the operator to become Ready"
kubectl rollout status deploy -n "${AIM_NAMESPACE}" -l control-plane=controller-manager --timeout=180s \
   || kubectl wait --for=condition=Available deploy --all -n "${AIM_NAMESPACE}" --timeout=180s \
   || send-error "AIM Engine operator did not become Ready; check 'kubectl get pods -n ${AIM_NAMESPACE}'."

echo ""
echo "[aim] AIM Engine installed in namespace ${AIM_NAMESPACE}."
echo "  Verify: kubectl get pods -n ${AIM_NAMESPACE}"
echo "  CRDs:   kubectl get crds | grep aim.eai.amd.com"
echo ""
