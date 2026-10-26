# Level 1: base-image serve

Level 1 is the quickest way to prove a cluster's GPU serves a model: it deploys
the generic AIM base image as a plain Deployment plus Service, with no operator,
CRDs, or accelerator labels. It assumes only a reachable cluster and a node
advertising `amd.com/gpu`, and the default model is small and ungated
(Qwen2.5-1.5B-Instruct), so no Hugging Face token is needed.

## Run and keep it serving

A plain `--level 1` deploys, proves inference, then deletes everything on exit;
`--keep 1` leaves the Deployment and Service up so we can use them. Set
`KUBECONFIG` and `NAMESPACE` first (see
[Supplying a kubeconfig](README.md#supplying-a-kubeconfig) and
[Choosing a namespace](README.md#choosing-a-namespace)), then run the level:

```bash
export KUBECONFIG=/path/to/your/kubeconfig
export NAMESPACE=<your-namespace>
./aim_deploy.sh --level 1 --keep 1 --namespace "$NAMESPACE"
```

It waits for Ready (the first pull and weight download take a few minutes),
serves a test completion, confirms vLLM placed its KV cache on the GPU, and
leaves `aim-base-check` running.

## Use the served model

The script's own port-forward is transient, so we open our own against the
namespace we deployed into and query the OpenAI-compatible endpoint, reading the
model id from `/v1/models` rather than hardcoding it. We wrap the port-forward in
a short retry so a transient auth hiccup while the tunnel establishes (which some
OIDC clusters show) does not leave us with a dead endpoint:

```bash
for _ in $(seq 10); do
  kubectl port-forward -n "$NAMESPACE" svc/aim-base-check 8000:80 >/tmp/pf.log 2>&1 &
  pf=$!
  sleep 3
  curl -sf localhost:8000/v1/models >/dev/null 2>&1 && break
  kill "$pf" 2>/dev/null
done
MODEL=$(curl -sS localhost:8000/v1/models | grep -o '"id":"[^"]*"' | head -n1 | cut -d'"' -f4)
curl -sS localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is ROCm?\"}],\"max_tokens\":200}"
curl -sS localhost:8000/metrics | grep cache_usage_perc   # vLLM confirms a GPU KV cache
```

Any OpenAI-protocol client works, for example the `openai` Python package or a
LangChain client pointed at `http://localhost:8000/v1`.

## Drive it with an example app

The `icf_4agent` example in the HPCTrainingExamples repository
(`MLExamples/icf_4agent`) is configured through environment variables. The first
time we build its virtualenv with `create_aivenv.sh` (which `start_app.sh` then
activates), then we point it at the endpoint and reuse the served id:

```bash
cd /path/to/HPCTrainingExamples/MLExamples/icf_4agent
./create_aivenv.sh
export ICF_BASE_URL=http://localhost:8000/v1
export ICF_MODEL="$MODEL"
export ICF_API_KEY=unused
./start_app.sh
```

It reasons best with a capable model such as a gpt-oss variant; the small
level-1 default exercises the wiring but answers weakly.

## Serve a different model

To serve a larger open model, set the id the base check reads (inherited by the
underlying script); a gated model also needs `HF_TOKEN`:

```bash
MODEL_ID="Qwen/Qwen2.5-7B-Instruct" ./aim_deploy.sh --level 1 --keep 1 --namespace "$NAMESPACE"
```

For a model-specific AIM image and the operator path, use [level 4](level-4.md).

## Clean up

```bash
kill "$pf" 2>/dev/null
kubectl delete deployment,service aim-base-check -n "$NAMESPACE"
```

Level 1 matches the operator only on the runtime (image, GPU detection, weight
download, engine start), not its label-driven profile resolution, so a pass here
is necessary but not sufficient for the operator path; [level 2](level-2.md)
validates that.
