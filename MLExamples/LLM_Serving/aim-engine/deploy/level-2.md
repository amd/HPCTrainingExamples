# Level 2: operator serve

Level 2 serves a model through the AIM Engine operator by applying an
`AIMService`, installing nothing itself: it assumes the operator and its seven
prerequisites are present. This is the common case on a managed cluster,
including a laptop with a kubeconfig for a cluster an administrator already set
up. If the operator is absent it stops and points at [level 3](level-3.md) or
[level 4](level-4.md). This guide is the reference for the serve-and-verify flow
levels 3 and 4 reuse.

## Run

Set `KUBECONFIG` and `NAMESPACE` first (see
[Supplying a kubeconfig](README.md#supplying-a-kubeconfig) and
[Choosing a namespace](README.md#choosing-a-namespace)), then run the level:

```bash
export KUBECONFIG=/path/to/your/kubeconfig
export NAMESPACE=<your-namespace>
./aim_deploy.sh --level 2 --namespace "$NAMESPACE"
```

It applies an `AIMService` named `aim-smoke` (ungated model-specific image by
default) with a resource floor and a single-GPU selector, and starts a one-shot
watcher that clears a known reconcile stall. It then prints the steps below. To
serve a different catalog model pass `--model-image`, and to tune parameters such
as the context window, weight/KV-cache precision, and GPU count (via
`AIM_ACCELERATOR_COUNT`) see
[Model catalog](README.md#model-catalog) and
[Customizing runtime parameters](README.md#customizing-runtime-parameters). To
replace the running model with another or take it down, see
[Switching or stopping the served model](README.md#switching-or-stopping-the-served-model).

## Verify and run an inference

The first pull and weight download take many minutes. We check readiness, then
query the predictor, reading the model id from `/v1/models` (vLLM registers it
under the image or Hugging Face repo name, not the `AIMService` name):

```bash
kubectl get aimservice aim-smoke -n "$NAMESPACE" \
  -o custom-columns='NAME:.metadata.name,READY:.status.conditions[?(@.type=="Ready")].status,REASON:.status.conditions[?(@.type=="Ready")].reason'
# while not Ready, this says why:
kubectl describe aimservice aim-smoke -n "$NAMESPACE" | grep -iE 'reason:|message:' | tail -n2

kubectl wait --for=condition=Ready aimservice/aim-smoke -n "$NAMESPACE" --timeout=1800s
# find the predictor Service by label (no need to read the InferenceService,
# which a namespaced identity may lack RBAC to list):
svc=$(kubectl get svc -n "$NAMESPACE" -l aim.eai.amd.com/service.name=aim-smoke,component=predictor -o name | head -n1)
kubectl port-forward -n "$NAMESPACE" "$svc" 8080:80 >/tmp/pf.log 2>&1 &
sleep 3
model=$(curl -sS localhost:8080/v1/models | grep -o '"id":"[^"]*"' | head -n1 | cut -d'"' -f4)
curl -sS localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"What is ROCm?\"}],\"max_tokens\":200}"
```

## Drive it with an example app

Point the `icf_4agent` example (HPCTrainingExamples, `MLExamples/icf_4agent`) at
the predictor and reuse the served id; levels 3 and 4 reach the same endpoint.
The first time we build its virtualenv with `create_aivenv.sh` (which
`start_app.sh` then activates):

```bash
cd /path/to/HPCTrainingExamples/MLExamples/icf_4agent
./create_aivenv.sh
export ICF_BASE_URL=http://localhost:8080/v1
export ICF_MODEL="$model"
export ICF_API_KEY=unused
./start_app.sh
```

## Confirm the GPU and clean up

```bash
pod=$(kubectl get pods -n "$NAMESPACE" -l aim.eai.amd.com/service.name=aim-smoke -o name | head -n1)
kubectl exec -n "$NAMESPACE" $pod -- rocm-smi
kubectl delete aimservice aim-smoke -n "$NAMESPACE"
```

If it never reaches Ready, two causes are common. Storage: the cache PVC needs a
ReadWriteMany StorageClass. Or the reconcile stall: the download finished but the
service stays `Progressing` because the controller did not re-queue on the cache
becoming Ready, cleared with one `kubectl annotate ... kick=...` nudge. Both are
covered under Known limits in [`deploy/README.md`](README.md).
