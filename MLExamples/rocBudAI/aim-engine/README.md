# rocBudAI over AIM Engine

| At a glance | |
| --- | --- |
| Model | large ~120B, e.g. gpt-oss-120b (served remotely) |
| Engine | AIM Engine (vLLM on Kubernetes) |
| Needs | a cluster already running AIM Engine; `kubectl` or an endpoint; no local GPU |
| Setup | minutes if a model is already serving; add the weight download if you deploy one |

This directory installs [rocBudAI](../README.md) as a thin client of a model
already served by AIM (AMD Inference Microservices) Engine: the vLLM,
OpenAI-compatible endpoint stood up by
[`LLM_Serving/aim-engine/deploy`](../../LLM_Serving/aim-engine/deploy/README.md).
The experience is stock rocBudAI, unchanged (see
[What every recipe shares](../README.md#what-every-recipe-shares)); only the model
moves. Where the stock deployment serves the model locally with ollama and gates
the launcher behind a `--comment=ollama` Slurm allocation, here the model runs
remotely on the cluster, and the agent can run from a login node or a laptop with
no GPU of its own.

## Requirements and setup time

> This setup does not provide the stock rocBudAI airgap: it installs nothing that
> blocks the agent's traffic, and `rocbudai-airgap-check` only probes whether the
> host is restricted. Privacy rests on our assertion that the serving cluster is
> trusted, plus the checks in [The trust boundary](#the-trust-boundary). It is a
> preview, not the production airgap (see
> [What every recipe shares](../README.md#what-every-recipe-shares)).

Beyond that, we need the following before starting:

- A Kubernetes cluster with AMD Instinct GPUs where AIM Engine is already
  installed. Installing AIM Engine needs cluster-admin rights and is covered by
  levels 3 and 4 of
  [`LLM_Serving/aim-engine/deploy`](../../LLM_Serving/aim-engine/deploy/README.md).
- A namespace on that cluster, either with a capable model already serving or
  with the rights to create an `AIMService`, so that `--deploy` below can stand
  one up. gpt-oss-120b, for example, needs one MI300X-class GPU.
- A way to reach the model: `kubectl` with a kubeconfig for a port-forward (see
  [Supplying a kubeconfig](../../LLM_Serving/aim-engine/deploy/README.md#supplying-a-kubeconfig)),
  or a routable endpoint passed with `--endpoint`, which needs no `kubectl`.
- A place where the agent's GPU workloads run: the login node of a Slurm system
  whose compute nodes share its file system (see
  [GPU execution](#gpu-execution-where-the-workload-runs)).
- `opencode` on `PATH` (`npm i -g opencode-ai@1.14.28`). On first run opencode
  also fetches the `@ai-sdk/openai-compatible` provider package on demand. On an
  internet-severed cluster, pre-stage both during cluster prep.
- `git` to fetch rocBudAI, unless we pass `--rocbudai-src` with a local checkout.

Two serving properties matter for the persona to work. The model must be a large,
tools-strong model (about 120B class): small models follow the strict persona and
first-turn rules unreliably. And the context window must fit the persona, which
is about 28k tokens. A vLLM `--max-model-len` of 32k stalls on prefill, so we
prefer 64k or more. The launcher reads the served `max_model_len` and warns when
it is small.

With a model already serving, setup is short: authenticating `kubectl` and
running the installer, which clones rocBudAI and writes a modulefile. Deploying
the model at install time adds its image pull and weight download, which for
gpt-oss-120b (tens of GB) takes many minutes, and some catalog models exceed
600 GB. Without a cluster that already runs AIM Engine, this example is not the
place to start: the [`ollama`](../ollama/README.md) example needs only a GPU
node.

## How it works

`install-rocbudai-aim.sh` does four things:

1. it clones rocBudAI at a pinned ref;
2. it applies a small AIM backend patch;
3. it drops in the no-GPU addendum;
4. it writes an Lmod modulefile that reuses rocBudAI's own load hook.

We install into a prefix in our home directory by default and touch nothing
system-wide.

The patch (`patches/rocbudai-aim.patch`) adds one backend branch to the stock
tree, keyed on `ROCBUDAI_BACKEND=aim`, and leaves the ollama path untouched. When
the AIM backend is active the launcher:

1. asserts the cluster is trusted (`ROCBUDAI_AIM_TRUSTED=1`) and refuses
   otherwise;
2. resolves the endpoint, either a URL we pass or a `kubectl port-forward` to the
   predictor Service, and verifies it is in-boundary (loopback or RFC1918);
3. reads the served model id and `max_model_len` from `GET /v1/models`, warning
   if the context window is below 64k;
4. skips the ollama and Slurm gates, writes `opencode.json` with the AIM provider
   and launches raw opencode rather than `ollama launch`.

The `opencode.json` it writes uses the OpenAI-compatible provider with tool
calling on and `share` disabled, and inherits rocBudAI's ASK-mode permission
policy from the stock template.

The AIM model is served by a custom provider opencode has no metadata for, so the
launcher declares its served `max_model_len` as the model's `limit` in
`opencode.json` (see
[What every recipe shares](../README.md#what-every-recipe-shares)). Without it,
opencode can send a negative `max_tokens` once the window fills, which vLLM rejects
(`max_tokens must be at least 1`).

Everything else, the persona seeding and the whole session machinery, is stock
rocBudAI running unmodified. Because we drive raw opencode, which honors
`OPENCODE_CONFIG`, the recipe needs none of the global-`AGENTS.md` workaround the
ollama launch path requires.

## The trust boundary

rocBudAI's promise is that our code, prompts and model output never leave a
boundary we trust. With a local ollama model that boundary is the GPU node; over
AIM it becomes the cluster that serves the model. The recipe therefore treats
trust as an explicit, verifiable assertion rather than an assumption.

The installer asks us to confirm (or takes `--trust`) that the cluster is one we
control and that is severed from the internet. It bakes `ROCBUDAI_AIM_TRUSTED=1`
into the modulefile only if so. At launch the patched `rocbudai-tui` re-checks
that flag and refuses to start without it, and it resolves the endpoint only if
it points at an in-boundary address. `rocbudai-airgap-check` verifies the rest on
demand:

- the trust assertion and the in-boundary endpoint;
- opencode's phone-home suppression and `share: disabled`;
- an advisory host-egress probe;
- with `--deep`, that the predictor Service is a ClusterIP, not a publicly
  reachable LoadBalancer or NodePort.

This is a defensible ceiling, not a proof. The assertion is the human judgment
and the checks make it verifiable: they catch obvious leaks, not a determined
insider. Cluster admins are trusted, and login nodes often have open egress.

## Quick start

We install once, then run `module load rocbudai` from any project directory.

Step 1: authenticate `kubectl`. The launcher port-forwards to the predictor, so
`kubectl` must be pointed at the cluster and logged in first; otherwise endpoint
discovery waits on it and appears to hang. On a headless login node the OIDC
browser login redirects to a `localhost` callback the node cannot open. We
therefore SSH in with that port forwarded, run `kubectl` in the same session, and
open the printed URL in a laptop browser. The forwarded port must match
kubelogin's `--listen-address`, default `8000`:

```bash
ssh -L 8000:localhost:8000 <user>@<login-node>
# then, in that same session on the node:
export KUBECONFIG=/path/to/your/kubeconfig
kubectl get svc -n <your-namespace>     # open the printed URL in your browser
```

If `kubectl` runs on a host with its own browser, the SSH forward is unnecessary.
See [Supplying a kubeconfig](../../LLM_Serving/aim-engine/deploy/README.md#supplying-a-kubeconfig)
for the kubelogin plugin and OIDC details.

Confirm a model is already serving (skip this when deploying one at install time,
see [Serving the model at install time](#serving-the-model-at-install-time)):

```bash
kubectl get aimservices -n <your-namespace>            # an AIMService that is Ready
kubectl get svc -n <your-namespace> -l component=predictor
```

We do not pick the model by name: the launcher reads the served id from the
predictor's `/v1/models` and uses it. When a namespace has more than one
predictor Service, it selects the first one it lists, which may not be the one we
want. To pin a specific model, we either export
`ROCBUDAI_AIM_SERVICE=<predictor-svc>` before `module load rocbudai`, or install
with `--endpoint http://host:port/v1` pointing straight at it, which skips the
port-forward and the auto-pick.

Step 2: install. Pass the namespace, and the Slurm partition `rocbudai-submit`
should dispatch GPU jobs to (find it with `sinfo -o "%P %G"`):

```bash
./install-rocbudai-aim.sh --namespace <your-namespace> --submit-partition <gpu-partition>
```

With a routable endpoint already in hand (Ingress, LoadBalancer, NodePort, or our
own port-forward), we skip `kubectl` entirely and pass it directly:

```bash
./install-rocbudai-aim.sh --endpoint http://localhost:8080/v1 --submit-partition <gpu-partition>
```

The installer prints the lines that finish setup:

```bash
module use $HOME/rocbudai-aim/modulefiles
cd <your-project-dir>          # the code you want rocBudAI to profile
module load rocbudai
```

Loading the module launches the TUI (terminal user interface) exactly as every
recipe does (see
[What every recipe shares](../README.md#what-every-recipe-shares)). One detail is
specific here: on a fresh session the launcher first asks which GPU we are
profiling, since the agent host has no local GPU to detect it from, and loads the
matching persona; a resumed session keeps its persona and is not asked again.

## GPU execution: where the workload runs

Serving the model on AIM is separate from running our GPU code. The agent runs
wherever we load the module, often a login node or laptop with no local GPU, so
the no-GPU addendum engages and dispatches profiling jobs through `rocbudai-submit`
(see [What every recipe shares](../README.md#what-every-recipe-shares)).

The intended topology is a model served on the Kubernetes cluster and GPU
workloads submitted to a Slurm HPC system with `rocbudai-submit`. Today the recipe
targets running the agent on the HPC login node itself: Slurm is on `PATH` and the
compute nodes share the login node's filesystem, so the agent submits jobs
directly. `rocbudai-submit` comes with the rocBudAI tree the installer stages, so
it is always on `PATH`.

`rocbudai-submit` carries no hardcoded cluster information. It reads the target
partition from `ROCBUDAI_SUBMIT_PARTITION`, falling back to the Slurm cluster
default when unset, and `-p` overrides it per run. Passing `--submit-partition` at
install time bakes the site's GPU partition into the modulefile so dispatched
jobs land where they should. We leave it unset only if the cluster default
partition is already the right GPU partition.

## Options

| Flag | Meaning |
| --- | --- |
| `--prefix DIR` | install root (default `$HOME/rocbudai-aim`) |
| `--gfx-arch ARCH` | optionally pin the profiling-target persona (`gfx90a`, `mi300x`, `gfx950`, `mi300a`); omit it and the launcher asks per session |
| `--namespace NS` | AIM namespace; the launcher port-forwards the predictor |
| `--endpoint URL` | direct base URL ending in `/v1`; skips the port-forward |
| `--submit-partition P` | Slurm partition `rocbudai-submit` dispatches GPU jobs to (baked as `ROCBUDAI_SUBMIT_PARTITION`; empty means the cluster default) |
| `--trust` | assert the cluster is trusted and in-boundary (no prompt) |
| `--rocbudai-ref REF` | git ref to pin (default: the ref the patch targets) |
| `--rocbudai-src PATH` | use a local rocBudAI checkout instead of cloning (air-gapped) |
| `--force` | overwrite an existing `--prefix` |
| `--dry-run` | print actions without changing anything |
| `--help` | usage |

The persona is chosen for the profiling target we work on, independent of the GPU
serving the model, so one shared install drives every supported architecture: the
launcher asks at the start of each fresh session. `--gfx-arch` is for a
single-architecture site that prefers to pin one default (baked into the
modulefile as `ROCBUDAI_AGENTS_TEMPLATE`). Exporting `ROCBUDAI_GFX_ARCH=<arch>`
before `module load rocbudai` answers the prompt non-interactively.

## Serving the model at install time

If the model is not yet served, `--deploy` (with `--namespace`) both stands the
model up and installs the client in one run. The installer delegates serving to
[`LLM_Serving/aim-engine/deploy/aim_deploy.sh --level 2`](../../LLM_Serving/aim-engine/deploy/level-2.md).
It forwards `--model-image`, `--max-model-len` and repeatable
`--engine-arg KEY=VALUE` one-to-one. It also inherits the `AIM_ACCELERATOR_COUNT`,
`AIM_ENGINE_ARGS` and `AIM_MAX_MODEL_LEN` environment variables. Without
`--model-image` the deploy default is a small smoke model
(`amdenterpriseai/aim-qwen-qwen3-32b`), so we name the model we actually want.

Both examples below assume an authenticated `kubectl` (Quick start, step 1). The
`<tag>` in `--model-image` is the AIM release version, and
`AIM_ACCELERATOR_COUNT` must match a GPU count the image ships a profile for (a
wrong count reports `ProfileNotFound`). We get both the same way: list what the
cluster serves with `kubectl get aimclustermodels`, then read the exact tag and
the valid counts from `kubectl get aimclustermodel <name> -o yaml`. The
[Model catalog](../../LLM_Serving/aim-engine/deploy/README.md#model-catalog)
section walks through reading a tag and the supported counts; see
[Customizing runtime parameters](../../LLM_Serving/aim-engine/deploy/README.md#customizing-runtime-parameters)
for every other knob. The two examples already fill in the tag and count from the
current catalog, so they run as-is; we confirm they still match the cluster
before deploying.

### gpt-oss-120b with custom parameters

We serve gpt-oss-120b, tuning precision and GPU memory but leaving the context
window at the image default (no `--max-model-len`). The opencode `limit` the
launcher writes (see [How it works](#how-it-works)) makes long sessions
auto-compact instead of overflowing, so the default window still runs and the
`max_tokens must be at least 1` error does not occur. The default profile window
here is only 32k, and opencode compacts as a fraction of it, so this compacts
sooner than the stock path, which sees gpt-oss's full 128k. For stock-equivalent
headroom, add `--max-model-len 131072` as in the next example.

```bash
AIM_ACCELERATOR_COUNT=1 ./install-rocbudai-aim.sh --namespace <ns> --deploy \
    --submit-partition <gpu-partition> \
    --model-image amdenterpriseai/aim-openai-gpt-oss-120b:0.11.1 \
    --engine-arg gpu-memory-utilization=0.90 \
    --engine-arg kv-cache-dtype=fp8
```

`AIM_ACCELERATOR_COUNT=1` matches gpt-oss-120b's single-GPU MI300X profile; the
image also ships 2, 4 and 8 GPU profiles for more throughput. The weights are
already mxfp4, so the precision knob here is the fp8 KV (key/value) cache.

### gpt-oss-120b with the full context window and more GPUs

The same model, with the context window raised to gpt-oss-120b's native ceiling
and scaled to multiple GPUs. As in the previous example, the profile defaults
`max-model-len` to 32k (read it as `max_model_len` from `/v1/models`).
`--max-model-len 131072` matches the stock path's 128k budget, and the
28k-token persona then leaves ample room. `AIM_ACCELERATOR_COUNT` picks a
supported GPU count, `kv-cache-dtype=fp8` shrinks the KV cache, and
`gpu-memory-utilization` caps HBM use:

```bash
AIM_ACCELERATOR_COUNT=4 ./install-rocbudai-aim.sh --namespace <ns> --deploy \
    --submit-partition <gpu-partition> \
    --model-image amdenterpriseai/aim-openai-gpt-oss-120b:0.11.1 \
    --max-model-len 131072 \
    --engine-arg kv-cache-dtype=fp8 \
    --engine-arg gpu-memory-utilization=0.90
```

This image ships profiles for 1, 2, 4 and 8 GPUs. 131072 is gpt-oss-120b's native
ceiling: setting `--max-model-len` higher makes the engine refuse to start. One
MI355X (288 GB) holds the mxfp4 weights plus a 128k fp8 KV cache, so even a
single-GPU deploy can serve the full window.

### After deploying: wait for Ready, then load

`--deploy` applies the `AIMService` and returns immediately, and the client
install finishes right away too. The model, however, still has to download and
load its weights before it can serve. For a large model this takes minutes, not
seconds: gpt-oss-120b pulls tens of GB, and some catalog models exceed 600 GB. We
therefore wait for the service to reach Ready before `module load`; otherwise the
launcher finds no predictor and exits with `no predictor Service found`.

Check progress without blocking:

```bash
# READY flips False (Progressing) -> True once it can serve:
kubectl get aimservice aim-smoke -n <ns> \
  -o custom-columns='READY:.status.conditions[?(@.type=="Ready")].status,REASON:.status.conditions[?(@.type=="Ready")].reason'
# first-pull download progress for big models (prints "Progress: NN% (bytes)"):
kubectl logs -n <ns> $(kubectl get pods -n <ns> -o name | grep download | head -n1) --tail=5
```

Or block until it is Ready (raise the timeout for very large models):

```bash
kubectl wait --for=condition=Ready aimservice/aim-smoke -n <ns> --timeout=2400s
```

If it stays `Progressing` long after the download finishes, we check the
predictor pod: a `CrashLoopBackOff` means it starts but the engine aborts.

```bash
kubectl get pods -n <ns>                          # look for a ...-predictor-... pod
kubectl logs -n <ns> <predictor-pod> --previous --tail=40
```

A log line like `Specified profile ID '...' not found` (with `Model-specific
profiles: 0`) means the image tag does not ship the profile its catalog metadata
advertises, so the deploy cannot succeed on that tag. Not every published tag is
complete: we pick a tag or model that is `READY` in
`kubectl get aimclustermodels` and confirm it serves before relying on it.

If instead the service goes Ready but the TUI rejects prompts with `the model's
context length is only <N> tokens`, the profile's default `max-model-len` is
smaller than the 28k-token persona needs. We redeploy with `--max-model-len`
raised toward the model's native window, as in the larger-context example above.

See [Verify and run an inference](../../LLM_Serving/aim-engine/deploy/level-2.md#verify-and-run-an-inference)
in the level 2 guide for the full readiness and smoke-test steps. Then we finish
with the `module use` and `module load` lines the installer printed (see
[Quick start](#quick-start)), loading the module from the project directory.

Only scalar `key=value` engine args reach the installer. Object-valued knobs, such
as a serve-time `override-generation-config` temperature default, are not
scalars: we set those by deploying directly with `aim_deploy.sh --overrides-file`
(see [Customizing runtime parameters](../../LLM_Serving/aim-engine/deploy/README.md#customizing-runtime-parameters)),
or set temperature per request from the client.

Without `--deploy` the installer configures the client only and never mutates a
running model. The `AIMService` model is immutable once created, so switching
models means deleting and recreating it (see
[Switching or stopping the served model](../../LLM_Serving/aim-engine/deploy/README.md#switching-or-stopping-the-served-model)).

## Air-gapped and shared installs

On a trusted, internet-severed cluster, we pre-vendor a pinned rocBudAI checkout
and pass `--rocbudai-src /path/to/rocBudAI`. We also pre-stage `opencode` and the
`@ai-sdk/openai-compatible` package, and reach the endpoint on the trusted network
or by port-forward. An administrator who wants a shared, site-wide module runs the
same installer with `--prefix` pointed at a shared path and publishes the one
`module use` line in the site profile:

```bash
./install-rocbudai-aim.sh --prefix /shared/apps/rocbudai-aim --namespace <ns> --trust
module use /shared/apps/rocbudai-aim/modulefiles
```

## Files

- `install-rocbudai-aim.sh`: the installer described above.
- `patches/rocbudai-aim.patch`: the AIM backend branch for `rocbudai-tui`,
  `rocbudai-airgap-check` and the load hook, applied to the pinned rocBudAI ref.
- `no-gpu-addendum.md`: the instructions overlay appended when the agent host has
  no local GPU (forbids `rocbudai-bench`; mandates dispatched GPU execution).
