# Deploying AIM

`aim_deploy.sh` deploys AIM into an existing cluster, one level at a time. The
levels form a ladder: a higher level assumes less is installed and does more,
and each preflights its preconditions before acting. We pick a level with
`--level`; the script dispatches to the building-block scripts below.

```bash
./aim_deploy.sh --level 1         # base-image serve, no operator
./aim_deploy.sh --level 4         # install prerequisites + AIM Engine, then serve
```

Each level has a usage guide covering how to run it and use the served model:

- [Level 1: base-image serve](level-1.md), no operator.
- [Level 2: operator serve](level-2.md), on a cluster that already has AIM Engine.
- [Level 3: install the operator, then serve](level-3.md).
- [Level 4: install prerequisites and the operator, then serve](level-4.md).

## Which level do I want

| Situation | Level |
| --- | --- |
| Prove a cluster's GPU serves a model, with nothing installed | 1 |
| AIM Engine and its prerequisites are already installed; we only submit workloads | 2 |
| The prerequisites are managed for us, but installing the operator is our job | 3 |
| We own a fresh cluster and install everything ourselves | 4 |

Level 2 is the common case on a managed cluster, including a laptop with a
kubeconfig for a cluster an administrator already set up; level 4 is turnkey for
a cluster we own; level 3 is the increment between them; level 1 is a separate
no-operator flow for a first-contact GPU check.

## Supplying a kubeconfig

From a machine not already pointed at the cluster, first set `KUBECONFIG` to the
path of the kubeconfig an administrator gave us, substituting our own path for
the placeholder below. We then confirm the variable is set and resolves to a
real file before running anything, since a wrong path leaves `kubectl` falling
back to `localhost:8080` and reporting the cluster as unreachable:

```bash
export KUBECONFIG=/path/to/your/kubeconfig
echo "${KUBECONFIG}"; test -s "${KUBECONFIG}" && echo "kubeconfig found" || echo "set KUBECONFIG to a real file"
```

Exporting it once means our own later `kubectl` commands inherit it too. To pass
it per run instead, all four scripts accept `--kubeconfig`, which exports
`KUBECONFIG` for that run and is inherited by the setup scripts (they show it
under `--help`):

```bash
./aim_deploy.sh --level 4 --kubeconfig /path/to/your/kubeconfig
```

A kubeconfig that authenticates through an OIDC provider (common on managed
clusters) runs an exec credential plugin such as `kubectl oidc-login`, which
needs the kubelogin plugin installed on `PATH` as `kubectl-oidc_login`. The
first `kubectl` call then performs a browser login: on a headless host we open
the URL it prints manually, forwarding the OIDC callback port over SSH when the
redirect targets a `localhost` address. Avoid `--grant-type=authcode-keyboard` as
a shortcut: some providers (recent Keycloak) reject its out-of-band redirect with
"Invalid parameter: redirect_uri". Without the plugin, `kubectl` cannot obtain a
token and the scripts report that the cluster is unreachable.

The token expires after a while, so an occasional re-login is normal. When it
lapses, the next `kubectl` call (including a script's first one) blocks waiting
for a fresh browser login, which on a headless host looks like a silent hang.
If a run seems to stall before printing anything, refresh the login by running
`kubectl get nodes` on its own (complete the browser step) and then re-run. As a
safeguard, the deploy applies objects server-side and retries transient failures,
so if an OIDC control plane ever rejects an occasional request with
`Unauthorized`, a run rides over it rather than failing.

## Choosing a namespace

On a multi-tenant cluster we must have access to a project and pass its name as
the namespace with `--namespace`, or the deploy will not work. Our identity is
usually authorized only in that project's namespace, not `default`, but the
scripts default to `default`, so a run without `--namespace` fails while
retrieving or creating objects even though `kubectl get nodes` and `kubectl get
ns` succeed: cluster-scoped reads are allowed but namespaced writes are not, and
some clusters report the denial as `Unauthorized` rather than `Forbidden`. Find
a namespace we can deploy into and pass it with `--namespace`:

```bash
for ns in $(kubectl get ns -o name | cut -d/ -f2); do
   kubectl auth can-i create deployments -n "$ns" >/dev/null 2>&1 && echo "can deploy in: $ns"
done
./aim_deploy.sh --level 1 --keep 1 --namespace <namespace>
```

If nothing prints, our identity has no namespace yet and an administrator must
grant one. If a namespace exists but access is still denied, its RoleBinding may
name an identity (an email or a group) that differs from the one the API server
derives from our token's OIDC claims, which is a cluster-side fix rather than a
kubeconfig change.

## Forwarded options and tokens

`aim_deploy.sh` forwards a level's flags to its underlying script: `--keep` and
`--verbose` to `aim_base_check.sh` on level 1; `--replace`, `--aim-version`,
`--crds-chart`, and `--chart` to `aim_engine_setup.sh` on levels 3 and 4 (for
example `./aim_deploy.sh --level 3 --replace 1`). The default models are ungated,
so no token is needed; a gated model such as Llama or Gemma needs `HF_TOKEN`
exported.

## Model catalog

The AIM catalog is AMD's set of prebuilt, pre-tuned inference containers, one
per model, published on Docker Hub under `amdenterpriseai/aim-*` and browsable at
the [AMD Enterprise AI catalog](https://enterprise-ai.docs.amd.com/en/latest/aims/catalog/models.html).
Each image bundles the model plus validated profiles (tuned runtime settings) for
the discrete Instinct GPUs, so the operator can pick optimal parameters for our
hardware automatically. The deploy default (`AIM_MODEL_IMAGE`,
`amdenterpriseai/aim-qwen-qwen3-32b:0.13.0`) is one such catalog image; serve a
different one on levels 2 to 4 with `--model-image`:

```bash
./aim_deploy.sh --level 2 --namespace "$NAMESPACE" --model-image amdenterpriseai/aim-deepseek-ai-deepseek-r1:0.13.0
```

On a cluster that already has AIM Engine, the operator republishes the catalog as
cluster-scoped `AIMClusterModel` objects, so we list what *this* cluster offers
(rather than the full public catalog) with:

```bash
kubectl get aimclustermodels
```

Two values from this catalog go into a deploy: the image **tag** and the GPU
**count** (`AIM_ACCELERATOR_COUNT`). The tag is the AIM
release version and is uniform across images in a release (the default is
`:0.13.0`), so read it from the `image:` field:

```bash
kubectl get aimclustermodel <name> -o yaml    # image: has the tag; profiles list the valid GPU counts
```

The count is not free-form: each image ships validated **profiles**, one per (GPU
type × accelerator count), and the operator resolves the profile by AND-ing your
GPU type and count, so a count with no matching profile is what raises
`ProfileNotFound`. Read the supported counts from the same `-o yaml` (or the
per-model page in the [AMD Enterprise AI catalog](https://enterprise-ai.docs.amd.com/en/latest/aims/catalog/models.html)):
a large model such as Mistral Large usually offers only a full-node profile (8
GPUs), while a smaller one like gpt-oss-120b commonly has lower-count profiles
too. Do not assume the count; read it for the specific image.

## Switching or stopping the served model

`aim_deploy.sh` manages a single `AIMService` named `aim-smoke`, so re-running it
with a different `--model-image` replaces the current model rather than adding a
second: the server-side apply updates `spec.model.image` in place and the
operator rolls the predictor onto the new image. That is enough when the new
model fits within the GPUs the old one already holds.

When the new model needs those GPUs (for example moving from a single-GPU model
to one whose only profile wants a full eight-GPU node), we free them first by
deleting the running service and waiting for its pods to terminate, then deploy
the replacement:

```bash
kubectl delete aimservice aim-smoke -n "$NAMESPACE"
kubectl wait --for=delete pod -n "$NAMESPACE" \
  -l aim.eai.amd.com/service.name=aim-smoke --timeout=300s
./aim_deploy.sh --level 2 --namespace "$NAMESPACE" --model-image <new-image>
```

Deleting the `AIMService` also stops serving outright, which is how we take a
model down when we are done with it. Running two models at once is a separate
case: it needs two services with distinct names, and since the script always
uses `aim-smoke` we apply the second one by hand, copying the manifest that
`serve_operator` emits and changing `metadata.name`.

## Customizing runtime parameters

Levels 2 to 4 (the operator path) are where we tune, and this works for any
catalog model. The image's profile sets runtime parameters automatically; we
override per service through `spec.profileOverrides.engineArgs`, which
shallow-merges over the profile and is passed verbatim to the inference-engine
CLI (vLLM today, but the field is engine-agnostic, so the context window is
`max-model-len`), and `spec.env` for environment variables.

The deploy bakes overrides in for us, so they survive re-runs (which regenerate
the `AIMService`). For scalar engine args, `--max-model-len` sets the context
window and `--engine-arg KEY=VALUE` (repeatable, or the space-separated
`AIM_ENGINE_ARGS` env var) passes any other; both feed
`spec.profileOverrides.engineArgs`. For a 16k context with a memory cap:

```bash
./aim_deploy.sh --level 2 --namespace "$NAMESPACE" \
  --max-model-len 16384 --engine-arg gpu-memory-utilization=0.9
```

Those shortcuts only cover scalar `key=value` engine args. For engine args that
are flags or take list/object values, and for any other `spec` field — `env`,
`resources`, `replicas`, `profileOverrides.containerEnv`, adapters, autoscaling —
pass `--overrides-file`, a YAML merge patch applied over the generated
`AIMService` on every run (`engineArgs` is `x-kubernetes-preserve-unknown-fields`,
so it accepts arbitrarily typed values here). Merge semantics are RFC 7386: nested maps merge by
key; lists and scalars replace wholesale. For example, `overrides.yaml`:

```yaml
spec:
  replicas: 2
  env:
  - name: VLLM_USE_V1
    value: "1"
  resources:
    limits:
      memory: 200Gi
```

```bash
./aim_deploy.sh --level 2 --namespace "$NAMESPACE" --overrides-file overrides.yaml
```

These parameters fall into two groups with different change costs. Engine and
resource settings (the context window `max-model-len`, weight and KV-cache
precision, `gpu-memory-utilization`, GPU count, replicas) are read when the vLLM
process starts, so changing any of them — by re-running `aim_deploy.sh`, editing
an `--overrides-file`, or patching the object — rolls the predictor pod and
reloads the weights: effectively a once-per-deploy cost of minutes for a large
model.

Sampling parameters (temperature, `top_p`, `max_tokens`) travel in each
`/v1/chat/completions` request, so a client changes them live with no restart. We
can bake serve-time defaults with the object-valued `override-generation-config`
engine arg, passed through `--overrides-file` since it is not a scalar (for
example `{"temperature": 0.2}`). Changing that default rolls the pod like any
other engine arg, but an individual request always overrides it without one.

To change the live spec of a running service (still rolling the predictor when
the change is an engine arg), patch it directly; a later `aim_deploy.sh` run
overwrites this, so prefer the flags above for anything durable:

```bash
kubectl patch aimservice aim-smoke -n "$NAMESPACE" --type merge \
  -p '{"spec":{"profileOverrides":{"engineArgs":{"max-model-len":"16384"}}}}'
```

Tensor-parallel size (GPUs per replica) is `AIM_ACCELERATOR_COUNT` on the deploy
(it sets both the profile selector and the GPU resource), and replicas scale with
`spec.replicas`. Level 1 has none of this: it runs the container as a plain
Deployment with no profile machinery, so customization is limited to whatever
environment variables the image honors. Use level 1 for a first-contact GPU check
and levels 2 to 4 when we want tuned, managed serving.

## Building-block scripts

`aim_deploy.sh` is a thin wrapper over three scripts we can also run directly:

- `aim_engine_setup.sh`: preflights the eight prerequisites and, if they pass,
  Helm-installs the AIM Engine CRDs and operator, else exits 42 listing the gaps.
  Needs a reachable cluster, cluster-admin, and `kubectl` and `helm`; no local
  GPU. Level 3 calls it, level 4 with `--install-prereqs 1`.
- `aim_prereqs_setup.sh`: installs the seven add-ons AIM depends on
  (cert-manager, Gateway API, kgateway, KServe, KEDA, keda-otel-add-on, and the
  OpenTelemetry Operator). The AMD GPU Operator is assumed present. On shared
  clusters these are usually admin-managed, so we run this mainly on self-managed
  or test clusters.
- `aim_base_check.sh`: runs an AIM as a plain Deployment plus Service (not an
  `AIMService`), requests one GPU, and confirms a single request. Ungated by
  default, cleans up unless `--keep 1`. It matches the operator only on the
  runtime, not its label-driven profile resolution, so a pass is necessary but
  not sufficient. See [level 1](level-1.md).

## Verifying a level

Each level prints how to confirm it worked: the model answers a request and the
GPU is in use. Level 1 checks both automatically; levels 2 to 4 print commands
to watch the `AIMService` and `InferenceService` reach Ready, run an inference,
and check GPU use. One rule holds everywhere: vLLM registers the model under its
real id (the image or Hugging Face repo name), not the `AIMService` name, so we
read the id from `/v1/models` rather than hardcoding it.

## Known limits

- Profile resolution matches node labels the GPU Operator stamps
  (`aim-accelerator.<MODEL>` and `aim-accelerator.partitioning-scheme.default`,
  AND-ed by the v1alpha2 resolver) and succeeds only if the image ships a profile
  for that GPU. Profiles target discrete Instinct GPUs (MI300X, MI325X, MI350X,
  MI355X), so an MI300A APU has none and is not expected to serve.
- The default models (for example Qwen2.5-1.5B-Instruct and Qwen3-32B) are
  ungated convenience picks: they exercise the serving path only and are not
  representative for benchmarking.
- The operator requires at least an optimized (pre-tuned) profile by default,
  which is a property of the model's catalog entry, not the GPU, so a community
  model can report `ProfileNotFound` even on capable hardware. We set
  `spec.profile.selector.minimumType: any` to allow a generic profile; raise it
  back to `optimized` to require tuned ones.
- The operator's cache PVC is ReadWriteMany so replicas share one weight copy,
  and routing needs a Gateway or load balancer. A ReadWriteOnce default
  (local-path) leaves the PVC Pending and the service Starting, so the operator
  path needs an RWX StorageClass (NFS, CephFS, or Longhorn) and a load balancer,
  which we assume the cluster provides; the [test harness](../test/README.md)
  supplies both via an in-cluster NFS provisioner.
- KServe runs in Standard mode (`RawDeployment`), not the Knative-backed
  Serverless default, since the prerequisites ship Gateway API and kgateway.
  Override with `KSERVE_DEPLOYMENT_MODE`.
- A generated profile can omit a memory request, so KServe applies its `2Gi`
  default and the predictor is OOM-killed as it loads weights. We set an explicit
  `spec.resources` floor (overridable via `AIM_CPU_REQUEST`, `AIM_MEM_REQUEST`,
  `AIM_CPU_LIMIT`, and `AIM_MEM_LIMIT`). AIM Engine replaces the resource maps
  wholesale rather than deep-merging, so `spec.resources` must restate every key
  including `amd.com/gpu` (set to `AIM_ACCELERATOR_COUNT`); omitting it drops the
  GPU limit, the pod gets no isolated device, and in-container `rocminfo` then
  sees every host GPU and fails.
- The service controller does not always re-queue an `AIMService` when its
  `AIMProfileCache` reaches Ready, so it can sit in `Starting` with `READY False`
  and reason `Progressing` even though the weight download has finished. This is
  an operator bug we track upstream. `aim_deploy.sh` runs a detached watcher that
  forces a reconcile (an annotation bump) until the service moves past the stall,
  gating only on the `AIMService`'s own conditions so it still works for a
  namespaced identity that cannot read the cache object (disable with
  `AIM_AUTO_NUDGE=0`). If the watcher is not running (the deploy shell exited, or
  we applied the `AIMService` by hand) and a service stays `Progressing` after the
  download completes, we clear it with one manual nudge:

  ```bash
  kubectl annotate aimservice aim-smoke -n "$NAMESPACE" kick="$(date +%s)" --overwrite
  ```

For an overview of both directories, see the [top-level README](../README.md).
