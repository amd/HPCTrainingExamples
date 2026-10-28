# rocBudAI

[rocBudAI](https://github.com/AMD-HPC/rocBudAI) is an AI profiling
assistant that is ROCm aware. It has knowledge on the AMD profiling tools
and runs an optimization loop where it profiles, optimizes, and re-evaluates, while
the user supervises and steers.
Technically speaking, it is an [opencode](https://opencode.ai) agent that runs from a terminal
and is driven by GPU-architecture personas. Through a short discovery interview, it
walks us through profiling and optimizing our own code with the ROCm profilers.

The rocBudAI repository tracks the development of the agent and its system-level
deployment on a cluster. According to the flow in the repo, a system administrator installs 
that deployment, which provides:

- a shared module and an ollama service;
- Slurm gating of the GPU allocations the agent runs in;
- an airgap that keeps code and prompts inside the cluster.

This directory shows how to try rocBudAI at a user level instead, without
administrator access, therefore **no air-gap is enforced** in the installation proceess, so 
please be aware of that as you use these recipes.
Each subdirectory installs the stock agent into your home
directory and changes only where the model is served. The intent of this exercise
is to show how to hook rocBudAI to different inference providers so that users and sistes
can pick their preferred way of using this agentic workflow based on their users needs.
Many agentic tools exist out there and many new keep appearing, but what's important
is that users and sites have the option to pick what is most suitable for them.

Concerning what is shown here, the agent and its workflow are identical across 
all subdirectories; only the trust boundary and the model differ. The behavior they share is collected in
[What every recipe shares](#what-every-recipe-shares) below, so each subdirectory
documents only what is unique to it.

- [`ollama`](ollama/README.md): runs a small model (Laguna XS) with an ollama
  daemon we start ourselves on a GPU node without sys admin level changes to the job scheduler.
- [`aim-engine`](aim-engine/README.md): runs the agent as a thin client of a large
  model served by [`LLM_Serving/aim-engine`](../LLM_Serving/aim-engine/README.md),
  on a cluster we assume is trusted and in-boundary. The agent then also runs from
  a login node or a laptop without a local GPU, as long as a KUBECONFIG file is provided.
- [`atom`](atom/README.md): runs the agent against a frontier model (Kimi-K3)
  served on a cluster by the AMD [ATOM](https://github.com/ROCm/ATOM) engine.
  For the Kimi K3 model to fit in one node, we need MI355X. The model runs on a compute node
  and the agent on the login node, so like `aim-engine` it needs no local GPU, but model, weights and agent
  all stay on one HPC system.
- [`muse`](muse/README.md): runs the agent against a frontier model
  (Muse-Spark-Open) served on a cluster by mainline
  [vLLM](https://github.com/vllm-project/vllm) built from source on an MI355X node.
  Like `atom` the model runs on a compute node and the agent on the login node with
  no local GPU, and model, weights and agent all stay on one HPC system; it differs
  in the serving engine.

Note that the recipes above could be modified so that the agent is setup to run on a user's laptop
and then can ssh into a cluster and submit jobs from there. For simplicity, this is not what currently
the above recipes are doing.

## What every recipe shares

All recipes differ only in the model and where it is served; the agent and
its workflow are the same. The common parts are described once here, and each
subdirectory links back rather than repeating them.

The agent is unmodified rocBudAI with the same terminal UI: a
session picker when the project directory holds prior sessions, then the
welcome banner and the seven-question discovery interview. `rocbudai --continue`
resumes the last session in the directory and `rocbudai --new` forces a fresh one;
session naming, the idle auto-nudge and the resume hints behave as upstream. How
that UI is launched is the one thing that varies, and each recipe says so.

A session writes `AGENTS.md`, `opencode.json` and `.rocbudai-*`
into the project directory we launch from, so we add those to its `.gitignore`.

When the agent runs where there is no GPU, a login
node or a laptop, it cannot profile locally. The recipe then appends
`no-gpu-addendum.md` to opencode's instructions, overriding the persona: the agent
dispatches every build, run and profiler job through `rocbudai-submit` to Slurm
instead of running it locally. For a recipe that assumes a GPU is present (the Ollama recipe)
the addendum is omitted and the stock local flow applies.

Two knobs behave the same way in every recipe. The GPU jobs the agent dispatches run
through `rocbudai-submit`, which targets the partition in `ROCBUDAI_SUBMIT_PARTITION`
(the Slurm default when unset) and takes `-p <partition>` per run; recipes that bake a
partition at install do so from an installer flag. The persona matches the GPU you are
profiling, not the one serving the model, and is chosen with `--gfx-arch` (or
`ROCBUDAI_GFX_ARCH` for a single shell); each recipe states whether it pins a default
or asks per fresh session.

A remotely served model reaches opencode through a custom provider
it has no metadata for, so the launcher declares the served context window as that
model's `limit` in `opencode.json`. That is what lets opencode auto-compact the
history and cap output tokens instead of sending a request the engine rejects.

As mentioned above, none of these recipes enforce any
airgap or Slurm gating like what can be achieved with a stock rocBudAI instalation. A deployment that
must keep code private should follows the stock install in
[`INSTALL.md`](https://github.com/AMD-HPC/rocBudAI/blob/main/INSTALL.md) or be appropriately
handled by the sys admins of your systems. Each
recipe states its own trust boundary in its own README.
