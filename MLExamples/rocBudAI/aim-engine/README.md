# rocBudAI over AIM Engine

| At a glance | |
| --- | --- |
| Model | large ~120B, e.g. gpt-oss-120b (served remotely) |
| Engine | AIM Engine (vLLM on Kubernetes) |
| Needs | a cluster already running AIM Engine; `kubectl` or an endpoint; no local GPU |
| Setup | minutes if a model is already serving; add the weight download if you deploy one |

This directory installs [rocBudAI](../README.md) as a thin client of a model already
served by AIM (AMD Inference Microservices) Engine: the vLLM, OpenAI-compatible
endpoint stood up by
[`LLM_Serving/aim-engine/deploy`](../../LLM_Serving/aim-engine/deploy/README.md). The
experience is stock rocBudAI, unchanged (see
[What every recipe shares](../README.md#what-every-recipe-shares)); only the model
moves. The model runs remotely on the cluster, and the agent runs from a login node
or a laptop with no GPU of its own.

## Requirements and setup time

> This flow does not implement a production airgap: nothing here blocks the agent's
> traffic, and `rocbudai-airgap-check` only probes whether the host is restricted.
> Privacy rests on the serving cluster being trusted (see
> [The trust boundary](#the-trust-boundary)).

Before starting you need:

- A Kubernetes cluster with AMD Instinct GPUs where AIM Engine is already installed
  (installing it needs cluster-admin rights; see levels 3 and 4 of
  [`LLM_Serving/aim-engine/deploy`](../../LLM_Serving/aim-engine/deploy/README.md)).
- A namespace with a capable model already serving, or the rights to create an
  `AIMService` so `--deploy` can stand one up. The model should be large and
  tools-strong (~120B class; smaller models follow the persona unreliably) with a
  context window that fits the ~28k-token persona. A `--max-model-len` of 32k stalls
  on prefill, so prefer 64k or more; the launcher warns when it is small.
- A way to reach the model: `kubectl` with a kubeconfig for a port-forward, or a
  routable endpoint passed with `--endpoint` (which needs no `kubectl`).
- A Slurm login node whose compute nodes share its filesystem, where GPU workloads
  run (see [Partition and persona](#partition-and-persona)).
- `opencode` on `PATH` (`npm i -g opencode-ai@1.14.28`); on first run opencode also
  fetches the `@ai-sdk/openai-compatible` provider. Pre-stage both on an
  internet-severed cluster.
- `git`, unless you pass `--rocbudai-src` with a local checkout.

With a model already serving, setup is just authenticating `kubectl` and running the
installer. Deploying a model at install time adds its image pull and weight download:
tens of GB for gpt-oss-120b, and over 600 GB for some catalog models.

## How it works (read only if interested in the technicalities)

`install-rocbudai-aim.sh` clones rocBudAI at a pinned ref, applies a small AIM backend
patch (`patches/rocbudai-aim.patch`), drops in the no-GPU addendum, and writes an Lmod
modulefile that reuses rocBudAI's load hook. It installs into a prefix in your home
directory and touches nothing system-wide.

The patch adds one backend branch keyed on `ROCBUDAI_BACKEND=aim`, leaving the ollama
path untouched. When that backend is active the launcher asserts the cluster is
trusted (and refuses otherwise), resolves the endpoint (a URL you pass or a `kubectl
port-forward` to the predictor Service) and verifies it is in-boundary, reads the
served model id and `max_model_len` from `GET /v1/models`, then writes `opencode.json`
for the AIM provider and launches raw opencode instead of `ollama launch`.

Because the model is a custom provider opencode has no metadata for, the launcher
declares its `max_model_len` as the model's `limit` in `opencode.json` (see
[What every recipe shares](../README.md#what-every-recipe-shares)); without it,
opencode can send a negative `max_tokens` once the window fills, which vLLM rejects.
Everything else is stock rocBudAI. Driving raw opencode (which honors
`OPENCODE_CONFIG`) means the recipe needs none of the global-`AGENTS.md` workaround the
ollama path requires.

## The trust boundary

With stock rocBudAI, an air-gap can be implemented to enforce a trust boundary.
With a local ollama model that boundary is the GPU node; over AIM it becomes the
cluster that serves the model, so the recipe treats trust as an explicit, verifiable
assertion.

The installer asks you to confirm (or takes `--trust`) that the cluster is trusted, and
bakes `ROCBUDAI_AIM_TRUSTED=1` into the modulefile only if so. At launch the patched
`rocbudai-tui` re-checks that flag and the in-boundary endpoint, refusing otherwise.
`rocbudai-airgap-check` verifies the rest on demand: the trust assertion and
in-boundary endpoint, opencode's phone-home suppression and `share: disabled`, an
advisory host-egress probe, and with `--deep` that the predictor Service is a ClusterIP
rather than a public LoadBalancer or NodePort.

## Quick start

Install once, then `module load rocbudai` from any project directory.

**Step 1: authenticate `kubectl`** (skip if you use `--endpoint`). The launcher
port-forwards to the predictor, so `kubectl` must be pointed at the cluster and logged
in first. On a headless login node the OIDC browser login redirects to a `localhost`
callback the node cannot open, so SSH in with that port forwarded (it must match
kubelogin's `--listen-address`, default `8000`) and open the printed URL in a laptop
browser:

```bash
ssh -L 8000:localhost:8000 <user>@<login-node>
# then, in that same session on the node:
export KUBECONFIG=/path/to/your/kubeconfig
kubectl get aimservices -n <your-namespace>     # confirm one is Ready
```

See [Supplying a kubeconfig](../../LLM_Serving/aim-engine/deploy/README.md#supplying-a-kubeconfig)
for the kubelogin and OIDC details. The launcher does not pick the model by name: it
reads the served id from the predictor's `/v1/models`. When a namespace has more than
one predictor it takes the first listed, so to pin a specific one, export
`ROCBUDAI_AIM_SERVICE=<predictor-svc>` before `module load`, or install with
`--endpoint` pointing straight at it.

**Step 2: install**, passing the namespace and the Slurm partition for GPU jobs:

```bash
./install-rocbudai-aim.sh --namespace <your-namespace> --submit-partition <gpu-partition>
```

With a routable endpoint in hand, skip `kubectl` entirely:

```bash
./install-rocbudai-aim.sh --endpoint http://localhost:8080/v1 --submit-partition <gpu-partition>
```

The installer prints the lines that finish setup:

```bash
module use $HOME/rocbudai-aim/modulefiles
cd <your-project-dir>          # the code you want rocBudAI to profile
module load rocbudai
```

Loading the module launches the TUI as every recipe does (see
[What every recipe shares](../README.md#what-every-recipe-shares)).

## Partition and persona

The intended topology is the model on Kubernetes and GPU jobs on a Slurm system; today
the agent runs on the HPC login node, which submits directly. Two knobs control this:

- **Submit partition.** The agent runs where you load the module (a login node with no
  local GPU), so the no-GPU addendum dispatches every profiling job through
  `rocbudai-submit` (which ships with the tree, so it is always on `PATH`).
  `rocbudai-submit` reads its target partition from `ROCBUDAI_SUBMIT_PARTITION`,
  falling back to the Slurm default, and `-p <partition>` overrides it per run.
  `--submit-partition` at install bakes it into the modulefile.
- **Persona.** The persona matches the GPU you are *profiling*, independent of the GPU
  serving the model, so one install drives every supported architecture. By default
  the launcher asks at the start of each fresh session (a resumed session keeps its
  persona). Pin a default with `--gfx-arch` at install, or answer the prompt
  non-interactively by exporting `ROCBUDAI_GFX_ARCH=<arch>` before `module load`.

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

## Serving the model at install time

If no model is serving yet, `--deploy` (with `--namespace`) stands one up and installs
the client in one run, delegating to
[`aim_deploy.sh --level 2`](../../LLM_Serving/aim-engine/deploy/level-2.md). It forwards
`--model-image`, `--max-model-len` and repeatable `--engine-arg KEY=VALUE`, and inherits
`AIM_ACCELERATOR_COUNT`, `AIM_ENGINE_ARGS` and `AIM_MAX_MODEL_LEN`. Without
`--model-image` it deploys a small smoke model, so name the one you want.
`AIM_ACCELERATOR_COUNT` must match a GPU count the image ships a profile for (a wrong
count reports `ProfileNotFound`); list models with `kubectl get aimclustermodels` and
read the exact tag and valid counts from `kubectl get aimclustermodel <name> -o yaml`.

Example, serving gpt-oss-120b on four GPUs at its full context window:

```bash
AIM_ACCELERATOR_COUNT=4 ./install-rocbudai-aim.sh --namespace <ns> --deploy \
    --submit-partition <gpu-partition> \
    --model-image amdenterpriseai/aim-openai-gpt-oss-120b:0.11.1 \
    --max-model-len 131072 \
    --engine-arg kv-cache-dtype=fp8 \
    --engine-arg gpu-memory-utilization=0.90
```

The image ships 1, 2, 4 and 8 GPU profiles. `131072` is gpt-oss-120b's native ceiling
(setting it higher makes the engine refuse to start) and matches the stock path's 128k
budget; omit `--max-model-len` to keep the image default (often only 32k, which compacts
sooner). The weights are already mxfp4, so the precision knob here is the fp8 KV cache.
Only scalar `key=value` engine args reach the installer; object-valued knobs (such as an
`override-generation-config` temperature) need `aim_deploy.sh --overrides-file` (see
[Customizing runtime parameters](../../LLM_Serving/aim-engine/deploy/README.md#customizing-runtime-parameters)),
or set temperature per request.

`--deploy` returns immediately, but the model still has to download and load its weights
(minutes for a large model, and some exceed 600 GB), so wait for Ready before
`module load` or the launcher exits with `no predictor Service found`:

```bash
kubectl wait --for=condition=Ready aimservice/<name> -n <ns> --timeout=2400s
```

If it stays `Progressing` after the download, check the predictor pod
(`kubectl logs -n <ns> <predictor-pod> --previous`): `Specified profile ID '...' not
found` means the tag does not ship the advertised profile, so pick one that is `READY`
in `kubectl get aimclustermodels`. If the service is Ready but the TUI rejects prompts
with `the model's context length is only <N> tokens`, redeploy with a larger
`--max-model-len`. See
[Verify and run an inference](../../LLM_Serving/aim-engine/deploy/level-2.md#verify-and-run-an-inference)
for the full readiness steps.

Without `--deploy` the installer configures the client only. An `AIMService` is immutable
once created, so switching models means deleting and recreating it (see
[Switching or stopping the served model](../../LLM_Serving/aim-engine/deploy/README.md#switching-or-stopping-the-served-model)).

## Air-gapped and shared installs

On a trusted, internet-severed cluster, pre-vendor a pinned rocBudAI checkout and pass
`--rocbudai-src /path/to/rocBudAI`, pre-stage `opencode` and the
`@ai-sdk/openai-compatible` package, and reach the endpoint on the trusted network. An
administrator wanting a shared, site-wide module runs the same installer with `--prefix`
on a shared path and publishes the one `module use` line:

```bash
./install-rocbudai-aim.sh --prefix /shared/apps/rocbudai-aim --namespace <ns> --trust
module use /shared/apps/rocbudai-aim/modulefiles
```

## Files

- `install-rocbudai-aim.sh`: the installer described above.
- `patches/rocbudai-aim.patch`: the AIM backend branch for `rocbudai-tui`,
  `rocbudai-airgap-check` and the load hook, applied to the pinned rocBudAI ref.
- `no-gpu-addendum.md`: the instructions overlay appended when the agent host has no
  local GPU, so GPU work is dispatched via `rocbudai-submit`.
